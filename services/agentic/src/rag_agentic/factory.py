"""Construct the component graph for whichever agent this container is running.

One image, two roles. ``AGENTIC_ROLE`` decides which agent gets built and, for the
orchestrator, how it reaches the search agent (a separate AgentCore Runtime in
production, or in-process for local dev and tests).

Pass ``model``/``embedder``/``search_client`` to inject fakes -- the whole graph
runs offline that way, with no AWS calls at all.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

from rag_search.embedder import Embedder, FakeEmbedder, TitanEmbedder

from rag_agentic.agents.orchestrator import OrchestratorAgent
from rag_agentic.agents.search_agent import DocumentSearchAgent
from rag_agentic.config import AgenticSettings, AgentRole, SearchTransport
from rag_agentic.converse import BedrockConverseModel, ConverseModel
from rag_agentic.credit_gate import CreditGate
from rag_agentic.document_urls import DocumentUrlBuilder, RegionalPresigners
from rag_agentic.index_store import TenantIndexStore
from rag_agentic.licensing import (
    ClientCredentialsToken,
    ForwardedToken,
    HttpLicensingClient,
    TokenProvider,
)
from rag_agentic.models import AgentRequest, AgentResponse
from rag_agentic.search_client import (
    AgentCoreSearchClient,
    InProcessSearchClient,
    SearchAgentClient,
)
from rag_agentic.tenancy import TenantResolver
from rag_core.config import Settings, get_settings
from rag_core.di import Container
from rag_core.logging import get_logger

_log = get_logger("rag_agentic.factory")


class Agent:
    """Structural type shared by both agents."""

    async def handle(self, request: AgentRequest) -> AgentResponse:  # pragma: no cover
        raise NotImplementedError


@dataclass(slots=True)
class AgenticComponents:
    settings: AgenticSettings
    role: AgentRole
    agent: Any
    resolver: TenantResolver
    model: ConverseModel
    index_store: TenantIndexStore | None = None
    search_client: SearchAgentClient | None = None
    # Meters the run against ExampleApp Licensing. Inert unless licensing is
    # configured, so the handler calls it unconditionally.
    credit_gate: CreditGate | None = None
    _closeables: list[Any] = field(default_factory=list)

    async def aclose(self) -> None:
        for resource in reversed(self._closeables):
            with contextlib.suppress(Exception):  # best-effort shutdown
                await resource.aclose()
        self._closeables.clear()


def build_components(
    settings: AgenticSettings | None = None,
    core_settings: Settings | None = None,
    *,
    container: Container | None = None,
    model: ConverseModel | None = None,
    embedder: Embedder | None = None,
    search_client: SearchAgentClient | None = None,
) -> AgenticComponents:
    settings = settings or AgenticSettings()
    core_settings = core_settings or get_settings()
    container = container or Container(core_settings)
    offline = settings.embed_provider.lower() == "fake"
    closeables: list[Any] = []

    if model is None:
        model = BedrockConverseModel(
            core_settings.bedrock,
            model_id=settings.model,
            guardrail_id=settings.guardrail_id,
            guardrail_version=settings.guardrail_version,
            guardrail_trace=settings.guardrail_trace,
        )
        closeables.append(model)

    resolver = TenantResolver(settings)

    # The orchestrator needs no index of its own -- it never retrieves. Building
    # one anyway would make it download tenant snapshots it will never read.
    needs_index = settings.role is AgentRole.DOCUMENT_SEARCH or (
        settings.role is AgentRole.ORCHESTRATOR
        and settings.search_transport is SearchTransport.IN_PROCESS
    )

    index_store: TenantIndexStore | None = None
    search_agent: DocumentSearchAgent | None = None
    if needs_index:
        if embedder is None:
            embedder = (
                FakeEmbedder(dim=256)
                if offline
                else TitanEmbedder(container.bedrock(), dim=settings.embed_dim)
            )
            closeables.append(embedder)
        # The container owns the S3/Bedrock client lifecycles; closing it (below)
        # tears down every AWS client the embedder and index store share.
        index_store = TenantIndexStore(settings, container.s3())
        # Links are signed per source-bucket region (see document_urls).
        presigners = RegionalPresigners(core_settings.aws)
        closeables.append(presigners)
        closeables.append(index_store)
        closeables.append(container)
        search_agent = DocumentSearchAgent(
            settings=settings,
            model=model,
            embedder=embedder,
            index_store=index_store,
            resolver=resolver,
            # Validates the URL mode/template now, so a bad config fails at startup.
            document_urls=DocumentUrlBuilder(
                settings, container.s3(), presigner_for_region=presigners
            ),
        )

    if settings.role is AgentRole.DOCUMENT_SEARCH:
        assert search_agent is not None  # guaranteed by needs_index above
        _log.info("agentic_components_built", role=settings.role.value, offline=offline)
        return AgenticComponents(
            settings=settings,
            role=settings.role,
            agent=search_agent,
            resolver=resolver,
            model=model,
            index_store=index_store,
            _closeables=closeables,
        )

    if search_client is None:
        if settings.search_transport is SearchTransport.IN_PROCESS:
            assert search_agent is not None
            search_client = InProcessSearchClient(search_agent)
        else:
            search_client = AgentCoreSearchClient(settings)
        closeables.append(search_client)

    credit_gate = _build_credit_gate(settings, closeables)

    orchestrator = OrchestratorAgent(
        settings=settings,
        model=model,
        search_client=search_client,
        # Fail fast on an unknown base_url, before spending any model tokens.
        resolve_tenant=resolver.resolve,
    )
    _log.info(
        "agentic_components_built",
        role=settings.role.value,
        transport=settings.search_transport.value,
        offline=offline,
    )
    return AgenticComponents(
        settings=settings,
        role=settings.role,
        agent=orchestrator,
        resolver=resolver,
        model=model,
        index_store=index_store,
        search_client=search_client,
        credit_gate=credit_gate,
        _closeables=closeables,
    )


def _build_credit_gate(settings: AgenticSettings, closeables: list[Any]) -> CreditGate:
    """Meter runs against ExampleApp Licensing, when it is configured.

    Returns an inert gate when it is not, rather than None, so the request handler
    has one code path instead of two.
    """
    if not settings.licensing_enabled or not settings.licensing_base_url:
        if settings.licensing_enabled:
            # Enabled with nowhere to call is a deployment mistake worth naming.
            _log.warning("licensing_enabled_without_base_url")
        return CreditGate(None, settings)

    if settings.role is not AgentRole.ORCHESTRATOR:
        # Only the entry point meters. The orchestrator forwards the whole request
        # to its specialist, ReservationId included, so a specialist that also
        # settled would close the same reservation twice and bill one user request
        # as two. Enforced here rather than by remembering which runtime gets the
        # environment variable.
        _log.info("licensing_not_metered_by_this_role", role=settings.role.value)
        return CreditGate(None, settings)

    tokens: TokenProvider
    if settings.licensing_token_endpoint and settings.licensing_client_id:
        tokens = ClientCredentialsToken(
            token_endpoint=settings.licensing_token_endpoint,
            client_id=settings.licensing_client_id,
            client_secret=settings.licensing_client_secret.get_secret_value(),
            scope=settings.licensing_scope,
            timeout_s=settings.licensing_timeout_s,
        )
    else:
        # Forwarded token: ExampleApp sends the signed-in user's IAM token with the
        # request, so no client secret lives in this runtime.
        tokens = ForwardedToken()

    client = HttpLicensingClient(
        base_url=settings.licensing_base_url,
        tokens=tokens,
        region=settings.region,
        timeout_s=settings.licensing_timeout_s,
        send_user_prompt=settings.licensing_send_user_prompt,
    )
    closeables.append(client)
    _log.info(
        "licensing_enabled",
        base_url=settings.licensing_base_url,
        feature_code=settings.licensing_feature_code,
        auth="client_credentials" if isinstance(tokens, ClientCredentialsToken) else "forwarded",
        fail_open=settings.licensing_fail_open,
    )
    return CreditGate(client, settings)
