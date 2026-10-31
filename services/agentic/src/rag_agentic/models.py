"""Wire contracts for both agents.

One envelope shape crosses every boundary — ExampleApp -> orchestrator, and
orchestrator -> document-search — so a payload can be replayed against either
runtime unchanged, and a local in-process call is byte-identical to a remote
``InvokeAgentRuntime`` call.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr, model_validator

# Caller spellings we accept for each field, compared with case, underscores,
# hyphens and spaces removed. ExampleApp has sent `queryText`, `QueryText` and the
# long-standing misspelling `BussinessObject`; rejecting those bought nothing but
# failed integrations, so they are normalised here rather than argued about.
_FIELD_SYNONYMS: dict[str, str] = {
    "query": "query",
    "querytext": "query",
    "question": "query",
    "baseurl": "base_url",
    "businessobject": "business_object",
    "bussinessobject": "business_object",  # sic -- ExampleApp's spelling
    "usercode": "user_code",
    "sessionid": "session_id",
    "objectcodes": "object_codes",
    "objectcode": "object_codes",
    "operation": "operation",
    "mode": "operation",
    "featurecode": "feature_code",
    "feature": "feature_code",
    "history": "history",
    "messages": "history",
    "conversationhistory": "history",
}


def _canonical_key(key: str) -> str:
    """A caller's field name reduced to letters and digits, lowercased."""
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


class HistoryTurn(BaseModel):
    """One earlier exchange, as the caller recorded it."""

    model_config = ConfigDict(extra="ignore")

    role: str = "user"
    content: str = ""


def expand_history(raw: Any) -> Any:
    """Accept either shape of history the caller may hold.

    ExampleApp stores a turn as one row -- ``{"QueryText": ..., "Response": ...}`` --
    while Converse wants separate user and assistant messages. A paired row becomes
    two turns, in that order; a row already shaped as {role, content} passes through.
    """
    # A caller with no history may send null rather than omit the field; .NET
    # serialises an empty nullable list that way. Treat it as "no history", not
    # as a malformed request -- it would otherwise 400 the FIRST message of every
    # conversation, which is the one case guaranteed to have no history.
    if raw is None:
        return []
    if not isinstance(raw, list):
        return raw
    out: list[Any] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        lowered = {str(k).lower().replace("_", ""): v for k, v in item.items()}
        if "role" in lowered:
            out.append(item)
            continue
        asked = lowered.get("querytext") or lowered.get("query") or lowered.get("question")
        answered = lowered.get("response") or lowered.get("answer") or lowered.get("responsetext")
        if asked:
            out.append({"role": "user", "content": asked})
        if answered:
            out.append({"role": "assistant", "content": answered})
    return out


# Earlier turns are re-read on EVERY model round of the new turn, so they cost
# latency directly. Past this many characters the oldest are dropped -- a caller
# sending its whole transcript should not make the current question slow.
HISTORY_CHAR_BUDGET = 12_000


def to_converse_history(turns: list[HistoryTurn]) -> list[dict[str, Any]]:
    """Earlier turns as a Converse transcript the model will accept.

    Bedrock requires strictly alternating roles starting with ``user``, and the
    current question is appended as a user turn after this. A caller's transcript
    will not always satisfy that -- a dropped turn, a retried message, an assistant
    turn first after trimming -- and a malformed transcript is a 400 from Bedrock,
    i.e. a failed answer. So it is normalised here rather than trusted:
    consecutive same-role turns collapse, a leading assistant turn is dropped, and
    a trailing user turn is dropped because the new question takes its place.
    """
    cleaned: list[tuple[str, str]] = []
    for turn in turns:
        role = "assistant" if str(turn.role).strip().lower().startswith("a") else "user"
        text = str(turn.content or "").strip()
        if not text:
            continue
        if cleaned and cleaned[-1][0] == role:
            cleaned[-1] = (role, cleaned[-1][1] + "\n\n" + text)
        else:
            cleaned.append((role, text))

    # Newest first while trimming, so the oldest context is what gets dropped.
    kept: list[tuple[str, str]] = []
    budget = HISTORY_CHAR_BUDGET
    for role, text in reversed(cleaned):
        if budget - len(text) < 0:
            break
        budget -= len(text)
        kept.append((role, text))
    kept.reverse()

    while kept and kept[0][0] != "user":
        kept.pop(0)
    while kept and kept[-1][0] != "assistant":
        kept.pop()
    return [{"role": r, "content": [{"text": t}]} for r, t in kept]


class AgentRequest(BaseModel):
    """What ExampleApp posts to the orchestrator's ``/invocations``.

    ``base_url`` is the tenancy key: it identifies which ExampleApp deployment the
    caller is on, and the resolver turns it into (environment, tenant) — which in
    turn selects the S3 subfolder holding that tenant's vectors, keyword index and
    metadata. It is NOT free-form context for the model.
    """

    # Unknown fields are ignored, and ExampleApp's PascalCase spellings are accepted
    # alongside the snake_case ones (UserCode / BusinessObject).
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _accept_caller_spellings(cls, data: Any) -> Any:
        """Map however the caller spelled a field onto the canonical name.

        A field already supplied under its canonical name always wins, so a payload
        carrying both `query` and `queryText` cannot be reordered into a different
        meaning.
        """
        if not isinstance(data, dict):
            return data
        renamed: dict[str, Any] = {}
        for key, value in data.items():
            target = _FIELD_SYNONYMS.get(_canonical_key(key))
            renamed[target or key] = value
        # Canonical spellings take precedence over anything mapped onto them.
        for key, value in data.items():
            if key in cls.model_fields:
                renamed[key] = value
        if "history" in renamed:
            renamed["history"] = expand_history(renamed["history"])
        # A phase-1 lookup asks which object codes exist; there is no question to
        # answer, so it need not carry one.
        if str(renamed.get("operation") or "").strip().lower() == "objectcodes":
            renamed.setdefault("query", "")
        return renamed

    base_url: str = Field(
        validation_alias=AliasChoices("base_url", "baseUrl", "BaseUrl"),
        description="Caller's ExampleApp base URL, e.g. https://foxtrot.acmedev.com/",
    )
    query: str = Field(
        validation_alias=AliasChoices("query", "Query"),
        description="The end user's question, verbatim",
    )
    # Optional hard pre-filter. When present the retrieval tools are constrained to
    # documents carrying this business object *before* ranking — it is enforced in
    # the tool layer, never left to the model's discretion.
    business_object: str | None = Field(
        default=None,
        validation_alias=AliasChoices("business_object", "BusinessObject", "businessObject"),
        description="Optional BusinessObject pre-filter",
    )
    # Who asked, as ExampleApp knows them. Carried through both agents and echoed back
    # so a request can be traced end to end in the logs and in the caller's audit.
    # It does NOT narrow what is searched: tenancy comes from base_url, and any
    # per-user document permissions would have to be enforced against the index.
    user_code: str | None = Field(
        default=None,
        validation_alias=AliasChoices("user_code", "UserCode", "userCode"),
        description="Calling user's ExampleApp UserCode",
    )
    # AgentCore requires >= 33 chars; the runtime layer pads/generates when absent.
    session_id: str | None = Field(
        default=None, validation_alias=AliasChoices("session_id", "sessionId", "SessionId")
    )
    # The documents this user may see, as ExampleApp's own permissions decided.
    # Comma-separated object codes, normally obtained from an `operation:
    # "objectCodes"` call and then filtered by ExampleApp against the user's ACL.
    # Enforced exactly like business_object: merged into every query before
    # ranking, and never widenable by the model.
    object_codes: str | None = Field(
        default=None,
        validation_alias=AliasChoices("object_codes", "objectCodes", "ObjectCodes"),
        description="ACL-filtered object codes the user may see (comma-separated)",
    )
    # Earlier turns of this conversation, oldest first, excluding the current
    # Query. Used to resolve references ("it", "that one") -- never as a source of
    # facts, which always come from a fresh search in the current scope.
    history: list[HistoryTurn] = Field(
        default_factory=list,
        validation_alias=AliasChoices("history", "History", "messages", "conversationHistory"),
    )
    # Which capability ExampleApp is invoking. When given, the orchestrator routes
    # straight to that specialist instead of asking a model to decide -- removing
    # both a round trip and the chance of routing differently on identical input.
    feature_code: str | None = Field(
        default=None,
        validation_alias=AliasChoices("feature_code", "featureCode", "FeatureCode"),
        description="Capability to invoke, e.g. DOCUMENT_SEARCH",
    )
    # "objectCodes" asks only which object codes exist for this BusinessObject --
    # a metadata lookup with no model call. Anything else is a normal question.
    operation: str | None = Field(
        default=None, validation_alias=AliasChoices("operation", "Operation", "mode")
    )
    # ExampleApp reserves credits before calling us and passes the reservation here.
    # With one present the agent only settles it -- it never calls CheckAndReserve,
    # so it is not a gate and cannot refuse a request.
    reservation_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "reservation_id", "reservationId", "ReservationId", "ReservationID"
        ),
        description="Reservation._id from ExampleApp's CheckAndReserve",
    )
    # The caller's ExampleApp IAM bearer token, forwarded so credits can be reserved
    # and settled as the real user. SecretStr so it cannot be printed by accident:
    # a token in a log line is a usable credential for its five-minute life.
    # Never echoed back -- `AgentResponse` has no field for it.
    access_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "access_token", "accessToken", "AccessToken", "authorization", "Authorization"
        ),
        description="ExampleApp IAM bearer token, used only for the Licensing credit calls",
        repr=False,
        exclude=True,
    )
    # NOTE: how many passages are retrieved is deployment configuration
    # (AGENTIC_TOP_K), not something a caller sets per request.


class Citation(BaseModel):
    """One grounded source behind an answer."""

    document_id: str
    document_name: str = ""
    page: int | None = None
    snippet: str = ""
    score: float = 0.0
    facets: dict[str, str] = Field(default_factory=dict)
    download_url: str | None = None


class DocumentSummary(BaseModel):
    """One row of a document listing.

    Listings are answered from the metadata store directly, so this carries EVERY
    match (up to the envelope cap), not just the handful the model quoted. It is
    how a caller renders "594 documents" as 594 rows without the model having to
    write 594 names into its prose.
    """

    document_id: str
    document_name: str = ""
    expiry_date: str | None = None
    # Same shape as Citation.facets: object_code, business_object, status, ...
    facets: dict[str, str] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """One step of the agent loop — the audit trail ExampleApp can surface or log."""

    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    # Compact result summary (row counts, ids) rather than the full payload, so a
    # trace stays cheap to log and safe to return to the browser.
    summary: str = ""
    took_ms: float = 0.0
    error: str | None = None


class Usage(BaseModel):
    """Token accounting, summed across every model turn in the loop."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    # Titan tokens for embedding the QUERY of each search this request ran. Billed
    # separately from the Claude model tokens, hence kept separate.
    embedding_tokens: int = 0
    model: str = ""
    # Which agent spent it. Only meaningful on a sub-agent's entry, where it is
    # the one thing a credit record cannot otherwise tell you: a SubAgents row
    # carries a ModelId, but the same model serves every specialist.
    agent_id: str = ""
    # How many model calls this usage covers. One per Converse round, summed by
    # `add`, so a delegated request reports the whole run's calls. Licensing bills
    # this as BatchCallCount.
    llm_calls: int = 0

    @property
    def total_tokens(self) -> int:
        """Input + output. Cache reads are already counted inside input_tokens."""
        return self.input_tokens + self.output_tokens

    def add(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            embedding_tokens=self.embedding_tokens + other.embedding_tokens,
            model=other.model or self.model,
            agent_id=other.agent_id or self.agent_id,
            llm_calls=self.llm_calls + other.llm_calls,
        )


class AgentResponse(BaseModel):
    """What an agent returns. Identical shape from both runtimes."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    # Which agent actually served the turn — the orchestrator's routing decision,
    # echoed back so the caller can tell "answered directly" from "delegated".
    route: Literal["document_search", "direct", "error"] = "direct"
    environment: str = ""
    tenant: str = ""
    business_object: str | None = None
    # Echoed back so the caller can tie an answer to the user who asked.
    user_code: str | None = None
    # Every document matching the last listing the agent ran, independent of
    # `citations` (which are the sources behind the prose, capped for readability).
    # Empty when the turn did not list anything.
    documents: list[DocumentSummary] = Field(default_factory=list)
    # Phase-1 answer: the object codes available for the requested BusinessObject,
    # comma-separated, for ExampleApp to filter against the user's ACL.
    object_codes: str = ""
    # How many documents matched that listing in total. None when nothing listed.
    # May exceed len(documents) when the result set is larger than the cap.
    total_matching: int | None = None
    trace: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    # One entry per specialist this request delegated to. `usage` above stays the
    # total for the whole run; this is the breakdown Licensing bills as SubAgents.
    sub_usage: list[Usage] = Field(default_factory=list)
    session_id: str = ""

    # ---- telemetry, flattened for the caller's logs -----------------------------
    # The same numbers as `usage`, repeated at the top level so a caller can log a
    # row without reaching into a nested object. Filled in by the runtime, which is
    # the only layer that sees the whole request including the delegation hop.
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Cached prompt prefix: reads are billed at a discount, writes at a premium.
    # Both are already inside input_tokens -- reported separately so a caller can
    # price a request without re-deriving them.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    # Titan tokens for embedding the query of each search this request ran.
    embedding_tokens: int = 0
    model_id: str = ""
    # Which runtime answered, for correlating a log line with CloudWatch.
    agent_id: str = ""
    # Wall time for the whole request as the runtime measured it.
    duration_ms: float = 0.0
