"""Async SQS client for the ingestion pipeline.

SQS decouples discovery from processing: the ingestion service enqueues work and
returns; a fleet of workers consumes at their own pace, giving natural
back-pressure and horizontal scaling. This client provides long-polling receive,
batched send/delete (10x fewer API calls), and visibility-timeout extension for
long-running processing.
"""

from __future__ import annotations

from typing import Any

from botocore.exceptions import BotoCoreError, ClientError
from pydantic import Field

from rag_core.aws.session import AwsSessionFactory
from rag_core.logging import get_logger
from rag_core.models.base import DomainModel
from rag_core.resilience import RetryableError, async_retry

_log = get_logger("rag_core.aws.sqs")


class SQSMessage(DomainModel):
    """A received SQS message with the handle needed to delete/extend it."""

    message_id: str
    receipt_handle: str
    body: str
    attributes: dict[str, str] = Field(default_factory=dict)
    receive_count: int = 1


def _wrap(exc: Exception) -> Exception:
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        # Almost all SQS faults are transient at this layer; surface as retryable.
        return RetryableError(f"SQS error {code}: {exc}")
    if isinstance(exc, BotoCoreError):
        return RetryableError(f"SQS transport error: {exc}")
    return exc


class SQSClient:
    def __init__(self, factory: AwsSessionFactory) -> None:
        self._factory = factory
        self._client: Any | None = None
        self._cm: Any | None = None

    async def __aenter__(self) -> SQSClient:
        await self._ensure()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _ensure(self) -> Any:
        if self._client is None:
            self._cm = self._factory.client("sqs")
            self._client = await self._cm.__aenter__()
        return self._client

    async def aclose(self) -> None:
        if self._cm is not None:
            await self._cm.__aexit__(None, None, None)
            self._cm = None
            self._client = None

    @async_retry(max_attempts=5)
    async def send(
        self, queue_url: str, body: str, *, attributes: dict[str, str] | None = None
    ) -> str:
        """Send one message; returns the SQS MessageId."""
        client = await self._ensure()
        kwargs: dict[str, Any] = {"QueueUrl": queue_url, "MessageBody": body}
        if attributes:
            kwargs["MessageAttributes"] = {
                k: {"DataType": "String", "StringValue": v} for k, v in attributes.items()
            }
        try:
            resp = await client.send_message(**kwargs)
            return resp["MessageId"]
        except (ClientError, BotoCoreError) as exc:
            raise _wrap(exc) from exc

    @async_retry(max_attempts=5)
    async def send_batch(self, queue_url: str, bodies: list[str]) -> list[str]:
        """Send up to 10 messages in one call. Returns failed entry ids (if any)."""
        client = await self._ensure()
        if not bodies:
            return []
        if len(bodies) > 10:
            raise ValueError("SQS send_batch accepts at most 10 messages")
        entries = [{"Id": str(i), "MessageBody": b} for i, b in enumerate(bodies)]
        try:
            resp = await client.send_message_batch(QueueUrl=queue_url, Entries=entries)
            failed = [f["Id"] for f in resp.get("Failed", [])]
            if failed:
                _log.warning("sqs_batch_partial_failure", failed=len(failed))
            return failed
        except (ClientError, BotoCoreError) as exc:
            raise _wrap(exc) from exc

    @async_retry(max_attempts=5)
    async def receive(
        self,
        queue_url: str,
        *,
        max_messages: int = 10,
        wait_time_s: int = 20,
        visibility_timeout_s: int = 300,
    ) -> list[SQSMessage]:
        """Long-poll for messages. Empty list means no work this cycle."""
        client = await self._ensure()
        try:
            resp = await client.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=wait_time_s,
                VisibilityTimeout=visibility_timeout_s,
                AttributeNames=["ApproximateReceiveCount"],
                MessageAttributeNames=["All"],
            )
        except (ClientError, BotoCoreError) as exc:
            raise _wrap(exc) from exc

        out: list[SQSMessage] = []
        for m in resp.get("Messages", []):
            attrs = {k: v["StringValue"] for k, v in m.get("MessageAttributes", {}).items()}
            out.append(
                SQSMessage(
                    message_id=m["MessageId"],
                    receipt_handle=m["ReceiptHandle"],
                    body=m["Body"],
                    attributes=attrs,
                    receive_count=int(
                        m.get("Attributes", {}).get("ApproximateReceiveCount", "1")
                    ),
                )
            )
        return out

    @async_retry(max_attempts=5)
    async def delete(self, queue_url: str, receipt_handle: str) -> None:
        """Acknowledge a message (delete after successful processing)."""
        client = await self._ensure()
        try:
            await client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
        except (ClientError, BotoCoreError) as exc:
            raise _wrap(exc) from exc

    @async_retry(max_attempts=4)
    async def extend_visibility(
        self, queue_url: str, receipt_handle: str, *, seconds: int
    ) -> None:
        """Extend a message's visibility timeout for long-running processing."""
        client = await self._ensure()
        try:
            await client.change_message_visibility(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=seconds,
            )
        except (ClientError, BotoCoreError) as exc:
            raise _wrap(exc) from exc
