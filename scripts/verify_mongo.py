"""Verify the MongoDB metadata join: does OriginalFileName match the S3 basenames?

Connects to Mongo, preloads AttachmentMetaData, lists the S3 foxtrot/ objects, and
reports the match rate + a few sample joins with their extracted facets.
"""

import asyncio
from collections import Counter

from rag_search.config import SearchSettings
from rag_search.mongo_source import MongoMetadataSource, basename, facets_from_mongo

from rag_core.config import get_settings
from rag_core.di import Container


async def main() -> None:
    core = get_settings()
    ss = SearchSettings()
    if not ss.mongo_uri:
        print("SEARCH_MONGO_URI is not set")
        return

    mongo = MongoMetadataSource(
        ss.mongo_uri,
        database=ss.mongo_database,
        collection=ss.mongo_collection,
        join_field=ss.mongo_join_field,
        tls_ca_file=ss.mongo_tls_ca_file,
    )
    print(f"connecting to {ss.mongo_database}.{ss.mongo_collection} ...")
    n = await mongo.preload()
    print(f"loaded {n} Mongo docs keyed by {ss.mongo_join_field}\n")

    container = Container(core)
    s3 = container.s3()
    total = matched = 0
    facet_key_counts: Counter[str] = Counter()
    samples: list[tuple[str, dict[str, str]]] = []
    try:
        async for obj in s3.list_objects(core.s3.bucket, core.s3.prefix):
            total += 1
            doc = mongo.lookup(obj.key)
            if doc:
                matched += 1
                facets = facets_from_mongo(doc)
                for k in facets:
                    facet_key_counts[k] += 1
                if len(samples) < 6:
                    samples.append((basename(obj.key), {k: str(v) for k, v in facets.items()}))
    finally:
        await mongo.aclose()
        await container.aclose()

    rate = (matched / total * 100) if total else 0
    print(f"S3 objects scanned: {total}")
    print(f"matched to Mongo:   {matched}  ({rate:.1f}%)\n")
    print("facet coverage (docs with each facet):")
    for k, c in facet_key_counts.most_common():
        print(f"  {k:18} {c}")
    print("\nsample joins:")
    for name, facets in samples:
        print(f"  {name}")
        for k, v in facets.items():
            print(f"       {k} = {v}")


asyncio.run(main())
