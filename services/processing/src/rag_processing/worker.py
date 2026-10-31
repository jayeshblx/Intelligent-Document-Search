"""SQS consumer that drives the ProcessingService.

Reliability model:
* Long-poll receive with N concurrent consumers (back-pressured by SQS visibility).
* A visibility *heartbeat* extends the timeout while a long job (e.g. Textract on a
  300-page scan) runs, so the message isn't redelivered mid-flight.
* Success -> delete. Retryable failure -> leave the message (SQS redelivers; the
  queue's redrive policy moves it to the DLQ after maxReceiveCount). Permanent
  failure (poison) -> route to DLQ explicitly and delete, so one bad document
  never blocks the queue.
"""

from __future__ import annotations

import asyncio
import contextlib

from rag_core.aws import SQSClient, SQSMessage
from rag_core.config import Settings
from rag_core.logging import get_logger
from rag_core.models import DocumentEvent, PipelineMessage
from rag_core.observability import counter
from rag_core.resilience import NonRetryableError
from rag_processing.config import ProcessingSettings
from rag_processing.service import ProcessingService

_log = get_logger("rag_processing.worker")

_HANDLED = counter("processing_worker_messages_total", "Worker message outcomes", ("outcome",))


class ProcessingWorker:
    def __init__(
        self,
        *,
        sqs: SQSClient,
        service: ProcessingService,
        core_settings: Settings,
        settings: ProcessingSettings,
    ) -> None:
        self._sqs = sqs
        self._service = service
        self._core = core_settings
        self._settings = settings
        self._stop = asyncio.Event()
        self._queue = core_settings.sqs.ingestion_queue_url
        self._dlq = core_settings.sqs.ingestion_dlq_url
        self._visibility = core_settings.sqs.visibility_timeout_s

    async def run(self) -> None:
        """Start consumers and block until ``stop`` is called."""
        if not self._queue:
            raise RuntimeError("no input queue configured (SQS__INGESTION_QUEUE_URL)")
        n = self._settings.worker_concurrency
        _log.info("worker_started", concurrency=n, queue=self._queue)
        consumers = [asyncio.create_task(self._consume(), name=f"consumer-{i}") for i in range(n)]
        try:
            await self._stop.wait()
        finally:
            for c in consumers:
                c.cancel()
            await asyncio.gather(*consumers, return_exceptions=True)
            _log.info("worker_stopped")

    async def stop(self) -> None:
        self._stop.set()

    async def _consume(self) -> None:
        while not self._stop.is_set():
            try:
                messages = await self._sqs.receive(
                    self._queue,
                    max_messages=self._core.sqs.max_messages,
                    wait_time_s=self._core.sqs.wait_time_s,
                    visibility_timeout_s=self._visibility,
                )
            except Exception as exc:
                _log.warning("receive_error", error=str(exc))
                await asyncio.sleep(1.0)
                continue
            for message in messages:
                await self._handle(message)

    async def _handle(self, message: SQSMessage) -> None:
        try:
            envelope = PipelineMessage[DocumentEvent].model_validate_json(message.body)
        except Exception as exc:
            _log.error("unparseable_message", error=str(exc))
            await self._to_dlq(message, reason=f"unparseable: {exc}")
            _HANDLED.labels(outcome="poison").inc()
            return

        heartbeat = asyncio.create_task(self._heartbeat(message))
        try:
            await self._service.process_event(envelope.payload)
            await self._sqs.delete(self._queue, message.receipt_handle)
            _HANDLED.labels(outcome="ok").inc()
        except NonRetryableError as exc:
            _log.error("permanent_failure", document_id=envelope.payload.document_id, error=str(exc))
            await self._to_dlq(message, reason=str(exc))
            _HANDLED.labels(outcome="poison").inc()
        except Exception as exc:
            # Too many attempts -> stop the bleeding by DLQ-ing; else let SQS redeliver.
            if message.receive_count >= self._settings.poison_max_receive:
                _log.error("max_retries_exceeded", receive_count=message.receive_count, error=str(exc))
                await self._to_dlq(message, reason=f"max retries: {exc}")
                _HANDLED.labels(outcome="dlq").inc()
            else:
                _log.warning("retryable_failure", receive_count=message.receive_count, error=str(exc))
                _HANDLED.labels(outcome="retry").inc()
                # Do not delete: SQS will redeliver after the visibility timeout.
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    async def _heartbeat(self, message: SQSMessage) -> None:
        """Periodically extend visibility so long jobs aren't redelivered."""
        interval = max(1, self._visibility // 2)
        try:
            while True:
                await asyncio.sleep(interval)
                await self._sqs.extend_visibility(
                    self._queue, message.receipt_handle, seconds=self._visibility
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.debug("heartbeat_error", error=str(exc))

    async def _to_dlq(self, message: SQSMessage, *, reason: str) -> None:
        """Send a message to the DLQ (if configured) and remove it from the queue."""
        if self._dlq:
            try:
                await self._sqs.send(self._dlq, message.body, attributes={"reason": reason[:256]})
            except Exception as exc:
                _log.error("dlq_send_failed", error=str(exc))
                return  # leave on queue for redrive rather than dropping
        await self._sqs.delete(self._queue, message.receipt_handle)
