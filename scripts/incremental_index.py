"""Run the scheduled index jobs locally -- the same code the Lambda runs.

    uv run python scripts/incremental_index.py index --target FOXTROT
    uv run python scripts/incremental_index.py purge --target BRAVO --dry-run
    uv run python scripts/incremental_index.py index --target FOXTROT --minutes 10

Targets come from INDEX_TARGETS (JSON), e.g. in PowerShell::

    $env:INDEX_TARGETS = '{"FOXTROT": {"collection": "FOXTROT_DocumentSearch_DEV", "metadata_database": "FOXTROT_DEV"},
                           "BRAVO": {"collection": "BRAVO_DocumentSearch_DEV", "metadata_database": "BRAVO_DEV"}}'

Bucket, prefix and tenant codes default to each collection's manifest. Mongo and AWS
credentials come from .env as for every other script.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from rag_search.index_jobs import run_job

from rag_core.logging import configure_logging


def main() -> None:
    configure_logging(level="INFO", json_logs=False, service_name="index-jobs-cli")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job", choices=["index", "purge"])
    parser.add_argument("--target", required=True, help="a key of INDEX_TARGETS, e.g. FOXTROT or BRAVO")
    parser.add_argument("--dry-run", action="store_true", help="purge: report only, delete nothing")
    parser.add_argument(
        "--minutes", type=float, default=0, help="index: stop starting new documents after this long"
    )
    args = parser.parse_args()
    deadline = time.monotonic() + args.minutes * 60 if args.minutes else None
    result = asyncio.run(run_job(args.job, args.target, deadline=deadline, dry_run=args.dry_run))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
