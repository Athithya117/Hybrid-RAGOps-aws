# Quick decision map (one-line)

Pick `FLOW_LOG_MODE` first (`none` / `cloudwatch` / `s3`). All other logging-related vars only matter for `cloudwatch` or `s3`. Network vars control VPC/subnet/NAT behavior and must match your AZ planning.

---

# All environment variables — meaning, accepted values, default, and when to choose them

(Format: NAME — accepted values — default — what it does — when to choose / examples / caveats)

### Stack / naming / basic

* `PULUMI_STACK` — string — default: Pulumi stack name (pulumi.get_stack()) — selects the Pulumi stack context.
  When to set: use different stacks for dev/staging/prod (e.g., `dev`, `staging`, `prod`).

* `TAG_PREFIX` — string — default: `pulumi-<stack>` — prefix used to name AWS resources.
  When to set: set to your org/team prefix for clearer tags (e.g., `acme-platform`).

### Availability zones & VPC/subnet

* `MULTI_AZ_DEPLOYMENT` — boolean (`1/true/yes` or `0/false`) — default: `false` — whether to use multiple AZs.
  When to choose: `true` for production high-availability; `false` for quick dev or low-cost test.

* `AZ_COUNT` — integer — default: `3` if `MULTI_AZ=true` else `1` — number of AZs to create subnets in.
  When to choose: typical prod = `2` or `3` (region capacity dependent). For dev set `1`.

* `VPC_CIDR` — CIDR string — default: `10.0.0.0/16` — the VPC network address range.
  When to choose: pick a non-overlapping CIDR for your environment. Larger orgs usually use /16; for many clusters choose /16 or /20 depending on subnet plan.

* `PUBLIC_SUBNET_CIDRS` — CSV of CIDRs — default: auto-generated if unset — explicit public subnet CIDRs per AZ.
  When to set: if you need deterministic addresses (e.g., IP allow-lists) or align to your IP management. Otherwise leave empty to auto-generate.

* `PRIVATE_SUBNET_CIDRS` — CSV of CIDRs — default: auto-generated if unset — private subnet CIDRs per AZ.
  When to set: same reasoning as public.

### NAT / routing

* `NO_NAT` — boolean — default: `false` — if `true`, private subnets get no NAT egress route.
  When to choose: use `true` for isolated environments that must not access internet; otherwise `false` for normal operation.

* `NAT_SINGLE` — boolean — default: `false` — if `true` then single NAT gateway (cheaper, single point-of-failure); else one NAT per AZ.
  When to choose: dev/test use `true` to save cost. Prod use `false` (one NAT per AZ) for resilience.

### VPC endpoints

* `CREATE_VPC_ENDPOINTS` — boolean — default: `true` — create common VPC endpoints.
  When to choose: set `true` to reduce egress & use AWS private endpoints; `false` to avoid resources in test stacks.

* `VPC_ENDPOINT_SERVICES` — CSV — default: `["s3","ecr.api","ecr.dkr","ssm","sts"]` — services for which to create endpoints.
  When to choose: add `secretsmanager` or `ec2` if your apps need them privately. Keep `s3` for S3 access without NAT.

### Flow logging selection (primary)

* `FLOW_LOG_MODE` — enum: `none` | `cloudwatch` | `s3` — default: if not set the file uses legacy logic and defaults to `cloudwatch`.
  When to choose:

  * `none`: cost-conscious environments; no flow logs created. Good for dev-only when you don’t need network audit data.
  * `cloudwatch`: quick, real-time debugging; good for development and short retention production troubleshooting. Choose this for fast on-call debugging.
  * `s3`: required for long-term analytics and Athena queries; supports direct Parquet delivery and optional Glue crawler for cataloging.

> **Rule**: choose *exactly one* of the three. Most production analytics use `s3`, ops teams often keep `cloudwatch` for recent troubleshooting.

### S3-specific options (only for `FLOW_LOG_MODE=s3`)

* `FLOW_LOG_S3_BUCKET` — string (bucket name OR arn `arn:aws:s3:::bucket`) — default: `None` — if set, use this existing bucket rather than creating one.
  When to choose: set this if your org manages central logging buckets (recommended in enterprise).

* `FLOW_LOG_S3_CREATE` — boolean — default: `true` in our file (but recommended *off* for prod centrally-managed buckets) — whether the module should create the S3 bucket.
  When to choose: use `true` only in small orgs or when you own the bucket lifecycle; for enterprise multi-stack setups set `false` and provide `FLOW_LOG_S3_BUCKET` with an existing audited bucket.

* `FLOW_LOG_S3_CREATE_NAME` — string — default: `<TAG_PREFIX>-<stack>-vpc-flow-logs` — bucket name used when auto-creating.
  When to choose: override for naming policy or if your naming must follow company convention.

* `FLOW_LOG_S3_PREFIX` (fetched automatically) — string — default: `AWSLogs/<account>/vpcflowlogs/` — the prefix under the bucket where raw AWS flow logs land.
  When to choose: leave default unless you require a different layout. Note: Athena/Glue behavior depends on this value.

* `FLOW_LOG_S3_TRANSITION_DAYS` — int — default: `30` — transition to IA after N days.
  When to choose: common to move to IA or Glacier over time; compliance or cost goals change these numbers.

* `FLOW_LOG_S3_EXPIRE_DAYS` — int — default: `365` — object expiration days.
  When to choose: retention policy driven by compliance. Many orgs keep 90–365 days; security teams may require multi-year retention.

* `FLOW_LOG_S3_ACCESS_LOGGING` — boolean — default: `false` — whether to enable S3 server-access-logging for the log bucket.
  When to choose: enable for auditability if you worry about who modifies bucket contents; access logs themselves should go to a different bucket.

* `FLOW_LOG_S3_ACCESS_BUCKET` — string — default: `None` — bucket for access logs (required if access logging true).
  When to choose: supply a central bucket for access logs or allow the module to create a small one (but centralization is safer).

### S3 Server-Side Encryption / KMS

* `FLOW_LOG_SSE_ALGORITHM` — enum: `aws:kms` | `aes256` — default: `aws:kms` — SSE algorithm for bucket encryption.
  When to choose: `aws:kms` for compliance / audit requirements; `aes256` when you prefer S3-managed encryption (no KMS key management).

* `FLOW_LOG_KMS_CREATE` — boolean — default: `false` — whether to create a CMK for the logs (the module can create a key).
  When to choose: set `true` if you want a dedicated key for logs and you accept key rotation/policy responsibilities. In enterprise, CMKs often managed centrally — prefer `false` and pass `FLOW_LOG_KMS_ARN`.

* `FLOW_LOG_KMS_ARN` — string — default: `None` — use an existing CMK by ARN.
  When to choose: recommended for organizations with centralized KMS/Key policies. If provided, module will use this existing key.

**Caveat:** If `FLOW_LOG_SSE_ALGORITHM=aws:kms` and neither `FLOW_LOG_KMS_CREATE` nor `FLOW_LOG_KMS_ARN` is set, the configuration will fail.

### CloudWatch-specific

* `FLOW_LOG_CW_LOG_GROUP` — string — default: `/aws/vpc/flowlogs/<stack>` — name of the CloudWatch Log Group created.
  When to choose: override if you have naming conventions.

* `FLOW_LOG_CW_RETENTION_DAYS` — int — default: `14` — retention in days for CloudWatch logs.
  When to choose: CloudWatch logs cost more per GB stored; typical short retention = 7–30 days for operational debugging. For compliance increase retention or export to S3.

### Glue / Athena helpers

* `CREATE_GLUE_CRAWLER` — boolean — default: `true` — create a Glue crawler to discover partitions in S3 layout.
  When to choose: enable when you want quick partition discovery for Athena; good short-term choice.

* `GLUE_CRAWLER_SCHEDULE` — cron expression — default: `cron(0 * ? * * *)` (hourly) — schedule for the crawler.
  When to choose: hourly is common for logs; for high-volume setups you might run it more frequently or use partition projection to avoid crawlers.

* `CREATE_ATHENA` — boolean — default: `true` — creates Athena named query (DDL) pointing at parquet target.
  When to choose: helpful to export DDL and let analysts run it; safe to enable.

* `ATHENA_DB_NAME` — string — default: `vpc_flow_logs_<stack>` — Glue/Athena database name.
  When to choose: set per-environment DB for isolation.

* `ATHENA_TABLE_NAME` — string — default: `vpc_flow_parquet` — table name for parquet dataset.
  When to choose: set consistent name across environments or for multi-team catalog.

* `ATHENA_OUTPUT_BUCKET` — string — default: `None` — optional Athena query output location (if you have a centralized results bucket).
  When to choose: set to central bucket for auditability.

### Safety & cost guards

* `FLOW_LOG_MAX_DAILY_BYTES` — int bytes — default: `1 GiB` (1 * 1024^3) — threshold to trigger cost guard/alerts (module currently exports, integrate alerts separately).
  When to choose: tune to your expected ingestion; reduce for dev, larger for prod.

### Older/backwards compat (still supported)

* `ENABLE_FLOW_LOGS` — boolean — older toggle; if set and `FLOW_LOG_MODE` absent it maps to `cloudwatch`.
* `FLOW_LOG_DEST` — string — older name, also maps to `cloudwatch`/`s3` — legacy support.

---

# Practical recommended choices (quick presets)

### Local dev / quick tests

* `PULUMI_STACK=dev`
* `MULTI_AZ_DEPLOYMENT=false`, `AZ_COUNT=1`
* `NO_NAT=false` (or `true` if you want no internet)
* `FLOW_LOG_MODE=cloudwatch`
* `FLOW_LOG_CW_RETENTION_DAYS=7`
  Why: low-cost, fast debugging, nothing large created.

### Staging (small scale)

* `PULUMI_STACK=staging`
* `MULTI_AZ_DEPLOYMENT=true`, `AZ_COUNT=2`
* `NAT_SINGLE=true` (save cost but HA reduced)
* `FLOW_LOG_MODE=s3`, `FLOW_LOG_S3_CREATE=true`, `FLOW_LOG_S3_CREATE_NAME=acme-staging-vpc-logs`
* `FLOW_LOG_SSE_ALGORITHM=aes256` (unless compliance requires KMS)
* `CREATE_GLUE_CRAWLER=true` (start with crawler), `CREATE_ATHENA=true`
  Why: keep data for analytics while limiting operational complexity.

### Production (analytics + compliance)

* `PULUMI_STACK=prod`
* `MULTI_AZ_DEPLOYMENT=true`, `AZ_COUNT=3`
* `NAT_SINGLE=false` (one NAT per AZ)
* `FLOW_LOG_MODE=s3`, `FLOW_LOG_S3_CREATE=false`, `FLOW_LOG_S3_BUCKET=org-central-logs` (central audited bucket)
* `FLOW_LOG_SSE_ALGORITHM=aws:kms`, `FLOW_LOG_KMS_ARN=arn:aws:kms:...` (use central key)
* `CREATE_GLUE_CRAWLER=false` (if you use direct Parquet delivery + partition projection) or `CREATE_GLUE_CRAWLER=true` to discover partitions; `CREATE_ATHENA=true`
  Why: long-term analytics, compliance, cost control via Parquet + partitions, central key/bucket management.

---

# Security, compliance & operational caveats

* **KMS**: providing your own KMS (`FLOW_LOG_KMS_ARN`) is preferred in enterprises. If the module creates a CMK (`FLOW_LOG_KMS_CREATE=true`), you must manage key policies and rotations.
* **Bucket creation**: `FLOW_LOG_S3_CREATE=true` creates a bucket in the stack account. In multi-account orgs, prefer a central logging account/bucket and set `FLOW_LOG_S3_BUCKET` to that name.
* **Access controls**: the module creates bucket policies and IAM roles; review them in `pulumi preview` for principle and resource scoping.
* **Cost**: CloudWatch costs scale with ingestion; S3 + Athena has cheaper per-query cost if you use Parquet and partitioning. Glue crawler runs have modest cost; full ETL jobs (if you later reintroduce them) incur Glue/EMR cost.
* **Partitioning**: for Athena queries, Parquet + partition by `region/year/month/day` (or use partition projection) is the most cost-efficient. Merely relying on Glue crawler over raw gz may still be expensive.
* **Cross-account**: if logs will be delivered from another account, you must adapt bucket policy with `aws:SourceAccount` and SourceArn conditions (the module assumes same-account by default).

---

# Short pitfalls checklist (things that commonly break)

* Setting `FLOW_LOG_MODE=s3` but forgetting to set either `FLOW_LOG_S3_BUCKET` or `FLOW_LOG_S3_CREATE=true` → stack fails.
* Choosing `aws:kms` without providing `FLOW_LOG_KMS_ARN` or enabling `FLOW_LOG_KMS_CREATE` → stack fails.
* Using `FLOW_LOG_S3_CREATE=true` in a multi-stack org without naming policy → many buckets created and management burden.
* Leaving CloudWatch retention high for many GB/day → high bill. Use small retention or export to S3/parquet for archive.
* Not tuning crawler frequency or Athena partitioning → queries scan more data than necessary.

---

# Quick decision cheat-sheet

* You need short-term debugging and low infra churn → `FLOW_LOG_MODE=cloudwatch`, small retention.
* You need historical analytics, security forensics, compliance → `FLOW_LOG_MODE=s3` + Parquet + partitioning + KMS + Athena.
* You want cheap, ad-hoc analytics quickly (low volume) → `s3` + Glue Crawler (no ETL initially), but plan to move to Parquet/direct delivery for scale.
