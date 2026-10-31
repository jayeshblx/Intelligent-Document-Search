"""Tests for SQSClient error handling, batching, and message lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from botocore.exceptions import BotoCoreError, ClientError

from rag_core.aws.session import AwsSessionFactory
from rag_core.aws.sqs import SQSClient, SQSMessage
from rag_core.resilience import RetryableError


@pytest.fixture
def mock_factory() -> AwsSessionFactory:
    return MagicMock(spec=AwsSessionFactory)


@pytest.fixture
def mock_client() -> AsyncMock:
    return AsyncMock()


async def test_sqs_send_success(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    mock_client.send_message.return_value = {"MessageId": "msg-123"}
    
    sqs = SQSClient(mock_factory)
    async with sqs:
        msg_id = await sqs.send("http://queue", "test body")
    
    assert msg_id == "msg-123"
    mock_client.send_message.assert_awaited_once()


async def test_sqs_send_with_attributes(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    mock_client.send_message.return_value = {"MessageId": "msg-456"}
    
    sqs = SQSClient(mock_factory)
    async with sqs:
        await sqs.send("http://queue", "body", attributes={"key1": "val1", "key2": "val2"})
    
    call_args = mock_client.send_message.call_args
    attrs = call_args[1]["MessageAttributes"]
    assert attrs["key1"]["StringValue"] == "val1"
    assert attrs["key2"]["StringValue"] == "val2"
    assert attrs["key1"]["DataType"] == "String"


async def test_sqs_send_error_wrapped(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    mock_client.send_message.side_effect = ClientError(
        {"Error": {"Code": "QueueDoesNotExist"}},
        "SendMessage"
    )
    
    sqs = SQSClient(mock_factory)
    with pytest.raises(RetryableError, match="QueueDoesNotExist"):
        async with sqs:
            await sqs.send("http://bad-queue", "body")


async def test_sqs_send_batch_success(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    mock_client.send_message_batch.return_value = {"Successful": [], "Failed": []}
    
    sqs = SQSClient(mock_factory)
    async with sqs:
        failed = await sqs.send_batch("http://queue", ["msg1", "msg2", "msg3"])
    
    assert failed == []
    call_args = mock_client.send_message_batch.call_args
    entries = call_args[1]["Entries"]
    assert len(entries) == 3
    assert entries[0]["MessageBody"] == "msg1"


async def test_sqs_send_batch_partial_failure(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    mock_client.send_message_batch.return_value = {
        "Successful": [{"Id": "0"}],
        "Failed": [{"Id": "1"}, {"Id": "2"}]
    }
    
    sqs = SQSClient(mock_factory)
    async with sqs:
        failed = await sqs.send_batch("http://queue", ["m1", "m2", "m3"])
    
    assert failed == ["1", "2"]


async def test_sqs_send_batch_empty_list(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    
    sqs = SQSClient(mock_factory)
    async with sqs:
        failed = await sqs.send_batch("http://queue", [])
    
    assert failed == []
    mock_client.send_message_batch.assert_not_awaited()


async def test_sqs_send_batch_limit_enforced(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    
    sqs = SQSClient(mock_factory)
    with pytest.raises(ValueError, match="at most 10"):
        async with sqs:
            await sqs.send_batch("http://queue", ["msg"] * 11)


async def test_sqs_receive_messages(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    mock_client.receive_message.return_value = {
        "Messages": [
            {
                "MessageId": "m1",
                "ReceiptHandle": "r1",
                "Body": "body1",
                "Attributes": {"ApproximateReceiveCount": "2"},
                "MessageAttributes": {"attr1": {"StringValue": "val1"}}
            },
            {
                "MessageId": "m2",
                "ReceiptHandle": "r2",
                "Body": "body2",
                "Attributes": {"ApproximateReceiveCount": "1"},
                "MessageAttributes": {}
            }
        ]
    }
    
    sqs = SQSClient(mock_factory)
    async with sqs:
        messages = await sqs.receive("http://queue", max_messages=10, wait_time_s=20, visibility_timeout_s=300)
    
    assert len(messages) == 2
    assert messages[0].message_id == "m1"
    assert messages[0].body == "body1"
    assert messages[0].receive_count == 2
    assert messages[0].attributes == {"attr1": "val1"}
    assert messages[1].receive_count == 1


async def test_sqs_receive_no_messages(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    mock_client.receive_message.return_value = {}
    
    sqs = SQSClient(mock_factory)
    async with sqs:
        messages = await sqs.receive("http://queue")
    
    assert messages == []


async def test_sqs_delete_message(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    mock_client.delete_message.return_value = {}
    
    sqs = SQSClient(mock_factory)
    async with sqs:
        await sqs.delete("http://queue", "receipt-handle-123")
    
    mock_client.delete_message.assert_awaited_once()
    call_args = mock_client.delete_message.call_args
    assert call_args[1]["ReceiptHandle"] == "receipt-handle-123"


async def test_sqs_extend_visibility(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    mock_client.change_message_visibility.return_value = {}
    
    sqs = SQSClient(mock_factory)
    async with sqs:
        await sqs.extend_visibility("http://queue", "receipt-456", seconds=600)
    
    call_args = mock_client.change_message_visibility.call_args
    assert call_args[1]["VisibilityTimeout"] == 600


async def test_sqs_transport_error_wrapped(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    mock_factory.client = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)))
    mock_client.receive_message.side_effect = BotoCoreError()
    
    sqs = SQSClient(mock_factory)
    with pytest.raises(RetryableError, match="transport error"):
        async with sqs:
            await sqs.receive("http://queue")


async def test_sqs_context_manager_cleanup(mock_factory: AwsSessionFactory, mock_client: AsyncMock) -> None:
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock()
    mock_factory.client = MagicMock(return_value=cm)
    
    sqs = SQSClient(mock_factory)
    async with sqs:
        pass
    
    cm.__aexit__.assert_awaited_once()
