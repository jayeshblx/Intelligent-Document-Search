"""Idempotently provision local AWS resources in LocalStack for dev.

Creates the S3 bucket and the ingestion queue + DLQ (with a redrive policy) so the
full write path can run on a laptop. Optionally uploads sample documents.

Prereqs: `docker compose up -d localstack` (see docker-compose.yml).

Usage
-----
    uv run python scripts/bootstrap_localstack.py
    uv run python scripts/bootstrap_localstack.py --upload ./samples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from rag_core.config import get_settings


def _clients() -> tuple[object, object]:
    s = get_settings()
    common = {
        "region_name": s.aws.region,
        "endpoint_url": s.aws.endpoint_url,
        "aws_access_key_id": s.aws.access_key_id or "test",
        "aws_secret_access_key": s.aws.secret_access_key or "test",
        "config": Config(retries={"max_attempts": 3}),
    }
    return boto3.client("s3", **common), boto3.client("sqs", **common)


def _ensure_bucket(s3: object, bucket: str, region: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)  # type: ignore[attr-defined]
        print(f"  bucket exists: {bucket}")
    except ClientError:
        kwargs = {"Bucket": bucket}
        if region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**kwargs)  # type: ignore[attr-defined]
        print(f"  created bucket: {bucket}")


def _queue_url_to_name(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def _ensure_queues(sqs: object, main_url: str, dlq_url: str) -> None:
    dlq_name = _queue_url_to_name(dlq_url)
    main_name = _queue_url_to_name(main_url)

    dlq = sqs.create_queue(QueueName=dlq_name)  # type: ignore[attr-defined]
    dlq_attrs = sqs.get_queue_attributes(  # type: ignore[attr-defined]
        QueueUrl=dlq["QueueUrl"], AttributeNames=["QueueArn"]
    )
    dlq_arn = dlq_attrs["Attributes"]["QueueArn"]
    print(f"  created DLQ: {dlq_name}")

    redrive = json.dumps({"deadLetterTargetArn": dlq_arn, "maxReceiveCount": "5"})
    sqs.create_queue(  # type: ignore[attr-defined]
        QueueName=main_name,
        Attributes={"RedrivePolicy": redrive, "VisibilityTimeout": "300"},
    )
    print(f"  created queue: {main_name} (redrive -> {dlq_name}, maxReceiveCount=5)")


def _upload(s3: object, bucket: str, prefix: str, folder: Path) -> None:
    count = 0
    for path in folder.rglob("*"):
        if path.is_file():
            key = prefix + path.name
            s3.upload_file(str(path), bucket, key)  # type: ignore[attr-defined]
            count += 1
            print(f"  uploaded: s3://{bucket}/{key}")
    print(f"  uploaded {count} file(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap LocalStack resources")
    parser.add_argument("--upload", type=Path, default=None, help="folder of sample docs to upload")
    args = parser.parse_args()

    settings = get_settings()
    s3, sqs = _clients()

    print("Bootstrapping LocalStack...")
    _ensure_bucket(s3, settings.s3.bucket, settings.aws.region)
    if settings.sqs.ingestion_queue_url and settings.sqs.ingestion_dlq_url:
        _ensure_queues(sqs, settings.sqs.ingestion_queue_url, settings.sqs.ingestion_dlq_url)
    else:
        print("  (skipping queues: SQS__INGESTION_QUEUE_URL/DLQ_URL not set)")

    if args.upload:
        _upload(s3, settings.s3.bucket, settings.s3.prefix, args.upload)

    print("Done.")


if __name__ == "__main__":
    main()
