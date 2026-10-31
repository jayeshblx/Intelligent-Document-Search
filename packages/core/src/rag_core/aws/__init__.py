"""AWS adapters (S3, SQS, Bedrock).

All clients are async (aioboto3) for high-throughput pipelines, share one tuned
``botocore`` config (adaptive retries + connection pooling), and translate
botocore exceptions into the platform's retryable/non-retryable taxonomy so the
``rag_core.resilience`` layer can make correct retry decisions.

Each client is an async context manager and also exposes ``aclose()`` for use
with an application-level lifecycle (FastAPI lifespan / Temporal worker).
"""

from __future__ import annotations

from rag_core.aws.bedrock import BedrockClient, EmbeddingResult, GenerationResult
from rag_core.aws.s3 import S3Client, S3ObjectSummary
from rag_core.aws.session import AwsSessionFactory
from rag_core.aws.sqs import SQSClient, SQSMessage

__all__ = [
    "AwsSessionFactory",
    "BedrockClient",
    "EmbeddingResult",
    "GenerationResult",
    "S3Client",
    "S3ObjectSummary",
    "SQSClient",
    "SQSMessage",
]
