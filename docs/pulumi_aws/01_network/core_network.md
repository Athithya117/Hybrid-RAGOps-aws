# Assumptions & upstream/downstream dependencies (concise)

* This explanation assumes you are running the `infra/pulumi_aws/core_network.py` Pulumi program we created earlier with AWS credentials available (environment or profile) and Pulumi pinned to the provider versions you use in CI.
* Upstream: Pulumi runtime, `pulumi_aws` provider, AWS account/region, optional existing S3/KMS resources.
* Downstream: `eks_cluster`, `nodegroup`, ALB/Cognito manifests or other modules will read the exported outputs (`vpc_id`, `public_subnet_ids`, `flow_log`, etc.). They expect the exports to be stable and to appear after `pulumi up` completes.

---

# Short summary of what the file does (one sentence)

`core_network.py` is a single Pulumi program that **creates a VPC + subnets + NAT/route tables + endpoints**, and optionally **configures VPC Flow Logs** (CloudWatch or S3), plus **Glue crawler + Glue ETL job + Athena helper** to convert and catalog S3 flow logs for efficient analytics.

---

# High-level structure (module sections)

1. Config helpers — functions that read env vars and Pulumi config with defaults.
2. Input initialization & validation — all envs set up at top with sane defaults, fail-fast checks.
3. AZ discovery and subnet auto-generation — creates public/private CIDRs if user left them empty.
4. VPC + subnets + IGW + public/ private route tables — deterministic creation per AZ.
5. NAT gateway(s) logic — either single NAT or per-AZ NATs; supports `NO_NAT`.
6. VPC endpoints — gateway (S3) + interface endpoints (ECR, SSM, STS, etc.).
7. Logging infra — S3 bucket (optional create), KMS (optional create), bucket policy, lifecycle.
8. FlowLog resource — CloudWatch or S3 Flow Log creation, with least-privilege IAM role for CloudWatch.
9. Glue crawler + Glue ETL job + Glue trigger — crawler to discover partitions, ETL to convert raw gz → partitioned Parquet, scheduled Glue trigger.
10. Athena named query helper — DDL to create partitioned Parquet table (exported as a named query).
11. Pulumi exports — stable outputs (`vpc_id`, `public_subnet_ids`, `flow_log` map, etc.).

---

# In-depth runtime control flow (what happens during `pulumi preview` and `pulumi up`)

> Important concept: Pulumi programs run *twice* conceptually — once to compute the resource graph during preview (synchronous Python execution, producing `Output` placeholders), and then the provider performs the actual apply to create resources. The program must therefore avoid doing destructive or non-idempotent side effects during evaluation; it should only declare resources and transform `Output`s safely.

### 1) Program start — Python process runs (preview & apply)

* Pulumi invokes `core_network.py`. The Python interpreter executes top-to-bottom. This includes:

  * Reading environment variables and `pulumi.Config()` via `_env_*` helpers. These are plain synchronous operations.
  * Validating values (CIDR parsing, AZ counts, mutually exclusive flags). If a validation fails, `pulumi.RunError` is raised and Pulumi aborts the run (preview fails). This is **intentional fail-fast** behaviour.
  * Calling `aws.get_availability_zones()` and `aws.get_region()` and `aws.get_caller_identity()` (data sources). These are synchronous data-source calls that happen at program runtime and return concrete values (not `Output`s). They are used for AZ selection and account/region values.

### 2) Construct the resource graph (declaration)

* The program declares Pulumi `Resource` objects: `aws.ec2.Vpc(...)`, `aws.ec2.Subnet(...)`, `aws.ec2.RouteTable(...)`, `aws.ec2.NatGateway(...)`, `aws.ec2.VpcEndpoint(...)`, `aws.s3.Bucket(...)` (optionally), `aws.kms.Key(...)` (optionally), `aws.ec2.FlowLog(...)`, `aws.iam.Role(...)` etc.
* Many fields of resources are plain strings or `pulumi.Output`s. When the program needs to combine `Output`s with strings, it uses `pulumi.Output.apply()` or `pulumi.Output.all().apply()` to produce derived strings safely — e.g., building `s3://<bucket>/<prefix>` is done inside an `apply` when bucket name is an Output.
* The code uses `pulumi.ResourceOptions(depends_on=...)` for resources that must not be created until others exist (e.g., FlowLog depends on bucket/policy/KMS where necessary). This forces ordering in the apply phase.

### 3) Preview stage

* Pulumi computes a preview of what will change. Because resource declarations were created, Pulumi compares the desired state with the current state and shows a plan. No resources are created yet.
* If some resource arguments depend on remote data (e.g., `get_availability_zones()`), those values are already included and shown in preview.
* Any `apply` blocks are not executed in the provider; they are only executed in the Pulumi engine when necessary to compute dependent arguments — but resource creation waits until `pulumi up`.

### 4) Apply stage

* When you run `pulumi up` and confirm, Pulumi will instruct the provider to create resources in the required order.
* AWS resource creation occurs via API calls: VPC → subnets → IGW → route tables → NATs (with EIP allocations) → private route tables → VPC endpoints → S3/KMS buckets/keys if configured → IAM role/policies → FlowLog → Glue resources.
* `depends_on` ensures the create order for resources that must be present before others (e.g., ensure bucket & policy exist before FlowLog creation).
* For AWS items that take time to reach consistent state (EIP/NAT provisioning, Glue job state), Pulumi waits until the provider resources reach a created state or times out based on provider defaults.

### 5) Post-apply

* Pulumi writes stack outputs (the `pulumi.export(...)` values) into the stack state. Downstream modules or CI scripts can `pulumi stack output --json` to consume them.
* Note: the Glue ETL job may not have run yet (if scheduled) — the ETL job runs on schedule after the Glue trigger executes. Glue crawler may take minutes to finish its first run; Flow Log objects typically land in S3 with a delay (delivery is not instantaneous).

---

# Detailed explanation of major code paths & resources

## Config & Validation

* `_env_str`, `_env_bool`, `_env_int`, `_env_list` read env vars first, then fall back to Pulumi config keys, then defaults. This gives flexibility for local env overrides and CI-defined `pulumi config`.
* Important validations:

  * `AZ_COUNT` must be >= 1 and <= available AZs.
  * Provided subnet CIDRs must be contained within `VPC_CIDR`.
  * If `FLOW_LOG_MODE=="s3"`, then either `FLOW_LOG_S3_BUCKET` or `FLOW_LOG_S3_CREATE=true` must be set. If KMS is chosen, either `FLOW_LOG_KMS_ARN` or `FLOW_LOG_KMS_CREATE=true` must be set.
* Fail-fast: invalid configuration raises `pulumi.RunError` during plan time.

## Subnet auto-generation

* If user leaves `PUBLIC_SUBNET_CIDRS` or `PRIVATE_SUBNET_CIDRS` empty, `_auto_generate_subnets()` splits the `VPC_CIDR` into sufficiently many /subnets by progressively increasing prefix length until it can produce `AZ_COUNT*2` subnets. It then assigns even-indexed subnets as public and odd-indexed as private. This deterministic logic simplifies multi-AZ topologies.

## VPC, subnets, route tables, NAT

* Creates a VPC with DNS hostname/support enabled.
* For each AZ:

  * Creates a public subnet (maps public IPs).
  * Creates a private subnet.
  * Associates the public subnet with the shared public route table (internet route via IGW).
* NAT logic:

  * If `NO_NAT` is true: private subnets get route tables without 0.0.0.0/0 entries (explicit egress is absent).
  * If `NAT_SINGLE` is true: a single NAT Gateway is created in the first public subnet and used by all private route tables (cost cheaper, single point of egress).
  * Otherwise: one NAT Gateway per public subnet with each private RT pointing to the NAT in its AZ (higher availability; higher cost).
* EIP resources are created for NATs as needed.

## VPC Endpoints

* If `CREATE_VPC_ENDPOINTS` is true:

  * Creates a **Gateway** VPC Endpoint for S3 and attaches it to the route tables.
  * Creates **Interface** endpoints for allowed services (ECR, SSM, STS, Secrets Manager, EC2) — each gets a security group that allows full traffic from VPC CIDR for simplicity; you may tighten these later.

## Logging: S3 bucket + KMS

* If `FLOW_LOG_MODE == "s3"`:

  * If `FLOW_LOG_S3_CREATE` is true, it creates an S3 bucket named `FLOW_LOG_S3_CREATE_NAME` with server-side encryption configuration:

    * If `FLOW_LOG_SSE_ALGORITHM == 'aws:kms'` and `FLOW_LOG_KMS_CREATE` true, it creates an AWS KMS CMK with a restricted key policy that **allows the Flow Logs service principal** (`vpc-flow-logs.amazonaws.com`) and your account root to use it. (Key policy is carefully built with `aws:SourceAccount` guard.)
    * If `FLOW_LOG_SSE_ALGORITHM == 'aes256'`, it configures AES256 SSE.
  * Lifecycle rules are attached (transition to STANDARD_IA after `FLOW_LOG_S3_TRANSITION_DAYS`, expire after `FLOW_LOG_S3_EXPIRE_DAYS`).
  * Optional access logging bucket can be created and enabled.
  * A bucket policy is created allowing the Flow Logs service principal to `s3:PutObject` into the specific prefix and with `Condition` limiting by `aws:SourceAccount`. This reduces the exposed write vector.

### Why bucket policy + kms policy matter

* Flow Logs writes to the bucket as an AWS service principal. The bucket policy must permit that specific principal and limit writes to the `AWSLogs/<acct>/vpcflowlogs/` prefix and to the account. KMS keys must allow the flow logs service to generate data keys so encrypted objects can be written.

## Flow Log resource creation

* CloudWatch: creates an IAM role for `vpc-flow-logs.amazonaws.com` with an inline policy limited to `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents` on the target log group ARN. Creates the CloudWatch Log Group with the desired retention, then creates an `aws.ec2.FlowLog` resource pointed at CloudWatch and supplying the role ARN.
* S3: constructs the destination ARN `arn:aws:s3:::bucket` (as a `pulumi.Output` if necessary), then creates `aws.ec2.FlowLog` with `log_destination_type="s3"`, `log_destination` pointing to the bucket ARN. The `FlowLog` resource has a `depends_on` array including the bucket and KMS key when created to avoid race conditions.

> Note: Flow log delivery is eventual; AWS batches flow log delivery to S3/CloudWatch and that may take minutes. Also, Flow Log into S3 uses compressed files under a date hierarchy.

## Glue Crawler (discovery)

* If `CREATE_GLUE_CRAWLER` is true and S3 logging is enabled:

  * Creates a Glue Catalog database (if `CREATE_ATHENA` or crawler enabled).
  * Creates a Glue `Crawler` that targets the S3 raw logs prefix (the `AWSLogs/<acct>/vpcflowlogs/<region>/` path). The crawler inspects folders and files and writes table metadata to the Glue Data Catalog. Use this to let Athena prune partitions by path (after crawler runs).
  * Optionally schedules the crawler via `Aws::Glue::CrawlerSchedule` (a cron expression) e.g., hourly.

## Glue ETL job (convert raw gz -> partitioned Parquet)

* If `CREATE_GLUE_ETL` is true:

  * Creates (or uploads) a PySpark script to S3 that:

    * Reads raw text/gz logs (line by line).
    * Splits whitespace into canonical VPC Flow Log fields.
    * Casts types, derives `year/month/day` from the `start` epoch and computes `region`.
    * Writes Parquet output partitioned by `region/year/month/day` to a `vpcflowlogs_parquet/` prefix.
  * Creates a Glue job resource with an IAM role allowing read from the raw prefix, write to the Parquet prefix, access to Glue/Athena, and KMS if needed.
  * Schedules a Glue Trigger (cron-like) to run the ETL (e.g., daily) or you can run it ad-hoc.

### Why ETL → Parquet and partitioning?

* Raw gz text scans are expensive. Parquet is columnar and dramatically reduces Athena scanned bytes and cost. Partitioning by time (and region) allows Athena partition pruning to read only the relevant files for a query.

## Athena named query (DDL helper)

* The program creates an Athena `NamedQuery` containing a DDL `CREATE EXTERNAL TABLE` pointing at the Parquet target (or a template that the user can run). This is a convenience for operators to run the query in the Athena console and create the partitioned table. In practice you may run CTAS (CREATE TABLE AS SELECT) to convert data or the Glue ETL job will write Parquet directly and update the Glue Catalog.

## Pulumi exports

* `vpc_id`, `public_subnet_ids`, `private_subnet_ids`, `nat_gateway_ids`, `route_table_ids`, `vpc_endpoint_ids`.
* `flow_log` — a map containing `mode`, `cloudwatch_log_group`, `s3_bucket` (as an `Output` if created), `s3_prefix`, `kms_key_arn`, `flow_log_resource_id`, `glue_db`, `glue_crawler`, `glue_job`, `glue_trigger`, `athena_named_query_id`. Downstream code uses these outputs to patch manifests, secrets, or to set ARNs in Kubernetes manifests.

---

# Important runtime caveats, race conditions, and how the file handles them

1. **Mixing `Output` and strings**

   * The code uses `pulumi.Output.apply()` when combining runtime `Output`s (like `log_bucket.bucket`) with strings (e.g., building `s3://bucket/prefix`) to avoid preview/runtime errors.

2. **Resource creation ordering**

   * `bucket -> bucket policy -> FlowLog` can race if FlowLog is created before the bucket policy exists. The code uses `opts=pulumi.ResourceOptions(depends_on=[...])` for the FlowLog resource when the bucket/key is created by the same program.

3. **KMS grants & key policy correctness**

   * KMS key policies are tricky. The module sets a key policy that grants usage to the Flow Logs service principal and to the account root (via `aws:SourceAccount`), and also creates grants when necessary. Still, different AWS accounts and organizations might need extra policy statements.

4. **Glue crawler / ETL timing**

   * The crawler may take several minutes to detect partitions. ETL jobs take time depending on data volume. The Pulumi apply completes when resources are created; data ingestion / catalog population may still be in progress.

5. **IAM least-privilege tradeoffs**

   * For simplicity and to avoid subtle permission errors, some Glue policies use broader permissions (e.g., `glue:*`) — in production you should tighten these to exact actions and resources.

6. **Delivery latency of flow logs**

   * Flow logs may be delayed (batches) — do not expect immediate files in S3 immediately after `FlowLog` creation.

---

# How to *inspect* what's happening at runtime (debug checklist)

Run these commands after `pulumi up` if something looks wrong:

* `pulumi stack output --json | jq .` — see exported outputs and confirm `flow_log` contents.
* For S3: `aws s3 ls s3://<bucket>/<prefix>/ --recursive --human-readable --summarize | head` — check whether objects exist.
* For FlowLogs: `aws ec2 describe-flow-logs --filter Name=resource-id,Values=<vpc-id>` — check FlowLog status.
* For CloudWatch: `aws logs describe-log-groups --log-group-name-prefix "/aws/vpc/flowlogs"` and `aws logs get-log-events` / CloudWatch Logs Insights in console.
* For Glue crawler/job: `aws glue get-crawler --name <crawler>` and `aws glue get-job-runs --job-name <job>` to see last run details.
* For Athena: run the named query or go to console to inspect the table and partitions.

---

# Single, smart, non-exit diagnostic command (copy-paste)

This single script aggregates the critical runtime context Pulumi and AWS artifacts. It is safe to paste and run (it does not `set -e` so it won't exit early). It prints human-friendly sections you can paste back into the conversation for diagnosis.

```bash
#!/usr/bin/env bash
# single diagnostic command: gather Pulumi+AWS context for core_network debugging
echo "=== TIMESTAMP ==="; date -u
echo
echo "=== AWS Caller Identity ==="; aws sts get-caller-identity --output json || true
echo
echo "=== AWS Region (env / config) ==="; echo "AWS_REGION=$AWS_REGION"; aws configure get region || true
echo
echo "=== Pulumi version & stack ==="; pulumi version || true; echo "Current stack:"; pulumi stack || true
echo
echo "=== Pulumi config (core_network relevant keys) ==="
pulumi config --show-secrets || true
echo
echo "=== Python & pip packages (pulumi libs) ==="
python3 -V || true
python3 -c "import pulumi,pulumi_aws; print('pulumi',pulumi.__version__,'pulumi_aws',pulumi_aws.__version__)" 2>/dev/null || true
echo
echo "=== Existing VPCs (non-default) ==="
aws ec2 describe-vpcs --filters Name=isDefault,Values=false --output json | jq -r '.Vpcs[] | {VpcId:.VpcId,CidrBlock:.CidrBlock,Tags:.Tags}' || true
echo
echo "=== S3 buckets likely relevant (names) ==="
if pulumi stack output flow_log >/dev/null 2>&1; then
  pulumi stack output --json | jq -r '.flow_log.s3_bucket // "no s3_bucket exported"' || true
fi
aws s3api list-buckets --output json | jq -r '.Buckets[] | .Name' | sed -n '1,50p' || true
echo
echo "=== KMS keys (list) ==="
aws kms list-keys --output json | jq -r '.Keys[] | .KeyId' | sed -n '1,20p' || true
echo
echo "=== Glue DBs (if any) ==="
aws glue get-databases --output json | jq -r '.DatabaseList[] | .Name' | sed -n '1,50p' || true
echo
echo "=== EC2 Flow Logs (recent) ==="
aws ec2 describe-flow-logs --output json | jq -r '.FlowLogs[] | {FlowLogId:.FlowLogId,ResourceId:.ResourceId,LogDestinationType:.LogDestinationType,LogGroupName:.LogGroupName,LogDestination:.LogDestination,LogFormat:.LogFormat,CreationTime:.CreationTime,FlowLogStatus:.FlowLogStatus}' | sed -n '1,50p' || true
echo
echo "=== Glue jobs (list) ==="
aws glue get-jobs --output json | jq -r '.Jobs[] | .Name' | sed -n '1,50p' || true
echo
echo "=== End of diagnostic ==="
```

* Copy/paste into a terminal with AWS credentials & Pulumi logged in. The script prints stack config, Pulumi version, AWS account, list of VPCs, flows, S3 buckets and Glue/KMS artifacts — everything useful to triage network & logging infra.

---

# Common failure modes & how to fix them

1. **`FlowLog` fails to write to S3**: check bucket policy, KMS grants & `aws:SourceAccount` conditions; ensure FlowLog resource uses the exact bucket ARN and that the Flow Logs service principal is allowed by both bucket policy and KMS key policy. Use the `describe-flow-logs` AWS CLI command to see `FlowLogStatus`.
2. **Glue job fails**: check IAM role attached to glue job for `s3:GetObject` on the raw prefix and `s3:PutObject` on parquet prefix; check logs in CloudWatch Logs (Glue job logs appear there).
3. **Athena queries scan too much data**: ensure ETL writes partitioned Parquet; run `MSCK REPAIR TABLE <table>` or have the crawler run to add partitions. Prefer partitioned Parquet to raw gz.
4. **Pulumi preview shows destructive changes**: inspect stack config and resource names; if you changed bucket naming defaults, Pulumi may try to replace buckets — avoid accidental bucket rename by setting explicit `FLOW_LOG_S3_CREATE_NAME`.
5. **Provider API mismatch errors**: pin Pulumi / pulumi-aws versions in `requirements.txt`.

---

# Suggested runtime observability & test checks to add to CI

* After apply, run a smoke check:

  * confirm `aws ec2 describe-flow-logs` shows the FlowLog with `FlowLogStatus=ACTIVE`.
  * if S3 logging, wait up to 10 minutes and confirm at least one object under `s3://<bucket>/<prefix>/` (or that the bucket exists). Use the `diagnostic` script above.
* Add Pulumi unit tests (Pulumi Mocks) for the core resource graph to ensure fields are as expected for each `FLOW_LOG_MODE`.
* Add cost guard alarms to watch for high CloudWatch ingestion or S3 PUT rates.

---

# Quick mental model for engineers

* `core_network.py` declares *what* you want; Pulumi will do *how* and *when*.
* The program performs *declarative* resource creation; side-effects (like Glue ETL runs or Flow Log files landing) happen after creation and may be delayed.
* Treat S3 + Glue/ETL as an eventual consistency pipeline — the program sets up the pipeline, but ingestion and catalog population are asynchronous.

---

# PRO TIPs (actionable)

1. For *production*, disable `FLOW_LOG_S3_CREATE` in the module and instead create a centralized, audited logging bucket via a separate stack. Use the module only to create FlowLog objects that point to that bucket.
2. Use Parquet + partitioned layout (region/year/month/day) and enable partition projection if you have very large data volumes — this removes the need for frequent crawlers.
3. In CI, run the diagnostic script after `pulumi up` to capture stack outputs and resource states automatically into the PR for reviewers.
4. Keep `FLOW_LOG_S3_CREATE_NAME` stable between runs to avoid accidental bucket replacements.
5. Add strict IAM least-privilege policy linting (policy-as-code checks) to prevent accidentally granting `*` permissions to Glue/Jobs/Key usage.

---

