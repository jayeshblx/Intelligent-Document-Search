# Scheduled index jobs (EventBridge Scheduler → Lambda)

Keeps the per-tenant Atlas collections (`FOXTROT_DocumentSearch_DEV`, `BRAVO_DocumentSearch_DEV`)
in step with S3 and the `AttachmentMetaData` catalogue, without full re-indexing.

| Schedule | Event | What it does |
|---|---|---|
| `index-foxtrot`, `index-bravo` — every 30 min | `{"job": "index", "target": "FOXTROT"}` | New S3 documents → metadata check → embed + index |
| `purge-foxtrot`, `purge-bravo` — daily 02:00 UTC | `{"job": "purge", "target": "FOXTROT"}` | Remove documents marked `DeleteMark=1` |

Code: [`rag_search/incremental.py`](../../services/search/src/rag_search/incremental.py)
(logic), [`rag_search/index_jobs.py`](../../services/search/src/rag_search/index_jobs.py)
(Lambda handler + wiring), [`scripts/incremental_index.py`](../../scripts/incremental_index.py) (same jobs, run locally).

## Incremental indexing, per run

1. **List** the tenant's S3 prefix. Candidates = objects with `LastModified` after the
   previous run (10 min overlap), plus files still **pending** from earlier runs.
2. **Skip** images (no text), files already in the collection, other tenants' files
   (`TenantCode`), and soft-deleted ones.
3. **Metadata check** — look up only the candidates in `AttachmentMetaData`
   (`GeneratedFileName`/`OriginalFileName` `$in` queries; never the whole collection).
4. **Has metadata** → extract, chunk, embed (Titan), write chunks + vectors + metadata.
   **No metadata yet** → parked as *pending* and retried every run; dropped after
   `INDEX_PENDING_MAX_AGE_D` days (default 7) if a record never appears.
5. Save the watermark. A run nearing the 15-minute Lambda limit stops starting new
   documents and parks the rest, so the next run carries on.

State lives in `EX_DOCUMENTSEARCH_<ENV>.<collection>_IndexState`: the watermark,
the pending files, and a run lock, so two overlapping invocations cannot double-embed.
The first run looks back `INDEX_INITIAL_LOOKBACK_H` hours (default 24).

A file that fails to extract is not retried every 30 minutes (that would re-download
a broken PDF forever); it is retried when it changes in S3 again. Failures are in the
result's `errors` and the logs.

## DeleteMark purge, per run

Reads this tenant's `DeleteMark = 1` records (1 / true / "1"), and removes every indexed
document with that file name from the collection: chunks, vectors and metadata. Another
tenant's DeleteMark on a shared file name is ignored. No S3 or Bedrock calls.

## Deploy

```powershell
# 1. Secret with the Atlas URI (read on FOXTROT_DEV/BRAVO_DEV, read/write on EX_DOCUMENTSEARCH_DEV)
aws secretsmanager create-secret --name exampleapp/docsearch/mongo-uri --secret-string "mongodb+srv://..."

# 2. Image -> ECR (arm64, from the repo root)
aws ecr create-repository --repository-name exampleapp-index-jobs
docker build -f deploy/index-jobs/Dockerfile --platform linux/arm64 --provenance=false `
  --secret id=extra_ca,src=certs/ca-bundle.pem -t <acct>.dkr.ecr.<region>.amazonaws.com/exampleapp-index-jobs:v1 .
docker push <acct>.dkr.ecr.<region>.amazonaws.com/exampleapp-index-jobs:v1

# 3. Stack
aws cloudformation deploy --stack-name exampleapp-docsearch-dev `
  --template-file deploy/index-jobs/template.yaml --capabilities CAPABILITY_IAM `
  --parameter-overrides `
    ImageUri=<acct>.dkr.ecr.<region>.amazonaws.com/exampleapp-index-jobs:v1 `
    MongoUriSecretArn=arn:aws:secretsmanager:...:secret:exampleapp/docsearch/mongo-uri-xxxx `
    SourceBucketArns=arn:aws:s3:::example-index-bucket `
    BravoMetadataDatabase=BRAVO_DEV

# 4. One-off test run
aws lambda invoke --function-name exampleapp-docsearch-dev-index-jobs `
  --payload '{"job":"purge","target":"FOXTROT","dry_run":true}' --cli-binary-format raw-in-base64-out out.json
```

Bucket, prefix and tenant codes per target come from each collection's **manifest**
(`scripts/mongo_index.py manifest ...`); check them with `mongo_index.py status`.
To override, add `"bucket"`, `"prefix"`, `"tenant_codes"` to that target in the
function's `INDEX_TARGETS`.

## Before going live

- **Atlas network access** — Lambda has no fixed IP. Either run it in private subnets
  behind a NAT gateway whose Elastic IP is on the Atlas access list, or use Atlas
  PrivateLink (`SubnetIds`/`SecurityGroupIds` parameters). In a VPC it also needs a
  route to S3, Bedrock, Textract and Secrets Manager (NAT or VPC endpoints).
- **Catalogue indexes** — the metadata check is an `$in` on `GeneratedFileName` and
  `OriginalFileName`, and the purge filters on `DeleteMark`. Ask the catalogue owner
  for indexes on those three fields if they don't exist yet.
- **Supplier names** — the full re-index votes one supplier name per object code across
  the corpus; an incremental run votes only within its own batch, so a new document can
  carry a different spelling until the next full re-index.
- Try it locally first: `uv run python scripts/incremental_index.py purge --target FOXTROT --dry-run`.
