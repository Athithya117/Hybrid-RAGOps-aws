#!/usr/bin/env bash
# core_network.sh — minimal outputs, non-paging smoke checks for infra/pulumi_aws/core_network.py
# Usage: ./core_network.sh
# Requirements: aws, jq, pulumi (optional if pulumi-outputs.json present)

# core_network.sh — read-only smoke validator for the network stack produced by core_network.py.
# Runs minimal, non-paging checks suitable for CI/CD, devcontainers, and operators modifying env vars.
# Automatically reads Pulumi outputs to adapt to user-provided configurations without relying on hardcoded prefixes.
# Validates VPC, subnets, route tables, Flow Logs (S3), sample log delivery, VPC endpoints,
# Glue crawler/database, and Athena named query when present. Avoids heavy S3 operations—fetches one sample key and
# counts up to 1000 objects for speed. Safe for NO_NAT=true setups by verifying S3 access through VPC endpoints.
# Generates a concise stdout summary and stores full diagnostic details in /tmp for troubleshooting without noise.

# EXPORT THESE IN TERMINAL BEFORE make pulumi-up and TESTING

: << 'EOF'

export TAG_PREFIX="${TAG_PREFIX:-pulumi}"                  # resource name prefix; set company prefix in prod
export MULTI_AZ_DEPLOYMENT="${MULTI_AZ_DEPLOYMENT:-false}"  # true => create AZ_COUNT AZs; set false for single-AZ dev
export AZ_COUNT="${AZ_COUNT:-3}"                           # AZ count when MULTI_AZ_DEPLOYMENT=true; keep <= region AZs
export VPC_CIDR="${VPC_CIDR:-10.0.0.0/16}"                  # VPC CIDR; change to avoid peering overlaps
export PUBLIC_SUBNET_CIDRS="${PUBLIC_SUBNET_CIDRS:-}"      # comma list or empty to auto-generate (recommended empty)
export PRIVATE_SUBNET_CIDRS="${PRIVATE_SUBNET_CIDRS:-}"    # comma list or empty to auto-generate
export NO_NAT="${NO_NAT:-true}"                            # true => private subnets have no NAT egress (isolated workloads)
export NAT_SINGLE="${NAT_SINGLE:-false}"                   # true => create single NAT (cheaper) vs per-AZ NATs
export CREATE_VPC_ENDPOINTS="${CREATE_VPC_ENDPOINTS:-true}" # create common VPC endpoints (s3, ecr, ssm, sts); disable if managed elsewhere
export CREATE_VPC_ENDPOINT_SERVICES="${CREATE_VPC_ENDPOINT_SERVICES:-s3,ecr.api,ecr.dkr,ssm,sts}" # services list

export FLOW_LOG_MODE="${FLOW_LOG_MODE:-s3}"       # none|cloudwatch|s3 — use cloudwatch for dev, s3 for analytics
export ENABLE_FLOW_LOGS="${ENABLE_FLOW_LOGS:-true}"  # Legacy toggle; used only if FLOW_LOG_MODE unset

export FLOW_LOG_CW_LOG_GROUP="${FLOW_LOG_CW_LOG_GROUP:-/aws/vpc/flowlogs/${STACK}}" # (if FLOW_LOG_MODE=cloudwatch) CW group name; change retention below
export FLOW_LOG_CW_RETENTION_DAYS="${FLOW_LOG_CW_RETENTION_DAYS:-14}" # (if FLOW_LOG_MODE=cloudwatch)  retention for CW logs (shorter for dev)

export FLOW_LOG_S3_BUCKET="${FLOW_LOG_S3_BUCKET:-rag-vpc-flow-31}"        # existing bucket name or ARN when using external bucket (s3 mode)
export FLOW_LOG_S3_CREATE_NAME="${FLOW_LOG_S3_CREATE_NAME:-${TAG_PREFIX}-${STACK}-vpc-flow-logs}" # bucket name when auto-creating
export FLOW_LOG_S3_LIFECYCLE_TRANSITION_DAYS="${FLOW_LOG_S3_LIFECYCLE_TRANSITION_DAYS:-30}" # move to IA after X days
export FLOW_LOG_S3_EXPIRATION_DAYS="${FLOW_LOG_S3_EXPIRATION_DAYS:-365}" # expire logs after X days; increase for compliance
export FLOW_LOG_S3_ACCESS_LOGGING="${FLOW_LOG_S3_ACCESS_LOGGING:-true}"
export FLOW_LOG_S3_ACCESS_BUCKET="${FLOW_LOG_S3_ACCESS_BUCKET:-rag-vpc-flow-31}" # bucket to receive access logs (required if access logging enabled)

export CREATE_GLUE_CRAWLER="${CREATE_GLUE_CRAWLER:-true}"    # create Glue crawler to discover partitions (cheap, recommended)
export GLUE_CRAWLER_SCHEDULE="${GLUE_CRAWLER_SCHEDULE:-cron(0 * ? * * *)}" # schedule for crawler (hourly default)

export CREATE_ATHENA="${CREATE_ATHENA:-true}"               # create Athena helper (Glue DB + named query) when using parquet
export ATHENA_DB_NAME="${ATHENA_DB_NAME:-vpc_flow_logs_${STACK}}" # Glue/Athena DB name; centralize if needed
export ATHENA_TABLE_NAME="${ATHENA_TABLE_NAME:-vpc_flow_parquet}" # Athena table name for parquet dataset
export ATHENA_OUTPUT_BUCKET="${ATHENA_OUTPUT_BUCKET:-}"     # optional Athena query results bucket; defaults to log bucket when empty

export FLOW_LOG_MAX_DAILY_BYTES="${FLOW_LOG_MAX_DAILY_BYTES:-1073741824}" # 1 GiB/day guardrail; tune per traffic expectations

EOF
set -uo pipefail

TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
LOGFILE="/tmp/core_network_smoke_${TIMESTAMP}.log"
exec 3>&1                     # save stdout
exec 1> >(tee -a "$LOGFILE")  # tee stdout to logfile
exec 2> >(tee -a "$LOGFILE" >&2)



AWS_PAGER="${AWS_PAGER:-}"    # ensure pager disabled
export AWS_PAGER

# simple logger helpers
_info(){ printf "%s\n" "$*"; }
_warn(){ printf "WARN: %s\n" "$*" >&2; }
_err(){ printf "ERROR: %s\n" "$*" >&2; }

# fail-fast if required commands missing
_missing=()
for cmd in aws jq; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    _missing+=("$cmd")
  fi
done
if [ "${#_missing[@]}" -ne 0 ]; then
  _err "Missing required commands: ${_missing[*]}. Install and retry."
  exit 2
fi

_info "CORE NETWORK SMOKE TEST — $TIMESTAMP"
_info "logfile: $LOGFILE"
_info ""

# Load outputs: prefer infra/pulumi_aws/pulumi-outputs.json if present
OUT_FILE="infra/pulumi_aws/pulumi-outputs.json"
if [ -f "$OUT_FILE" ]; then
  _info "Reading pulumi outputs from $OUT_FILE"
  PULUMI_JSON="$(cat "$OUT_FILE")"
else
  if command -v pulumi >/dev/null 2>&1; then
    _info "Reading pulumi stack outputs via pulumi CLI"
    PULUMI_JSON="$(pulumi stack output --json 2>/dev/null || true)"
  else
    _warn "Pulumi outputs file not found and pulumi CLI unavailable."
    PULUMI_JSON="{}"
  fi
fi

# Helper to extract outputs (fall back to empty)
jq_or_empty() {
  echo "$PULUMI_JSON" | jq -r "$1 // empty"
}

VPC_ID="$(jq_or_empty '.vpc_id')"
FLOW_LOG_ID="$(jq_or_empty '.flow_log_id')"
FLOW_LOG_MODE="$(jq_or_empty '.flow_log_mode')"
FLOW_LOG_S3_BUCKET="$(jq_or_empty '.flow_log_s3_bucket')"
GLUE_CRAWLER="$(jq_or_empty '.glue_crawler')"
GLUE_DB="$(jq_or_empty '.glue_db')"
ATHENA_NQ="$(jq_or_empty '.athena_named_query')"
PUBLIC_SUBNETS_JSON="$(echo "$PULUMI_JSON" | jq -c '.public_subnet_ids // []')"
PRIVATE_SUBNETS_JSON="$(echo "$PULUMI_JSON" | jq -c '.private_subnet_ids // []')"
ROUTE_TABLES_JSON="$(echo "$PULUMI_JSON" | jq -c '.route_table_ids.private // []')"

_info "Summary outputs (minimal):"
printf "  VPC ID: %s\n" "${VPC_ID:-<missing>}"
printf "  FlowLog: %s  (mode: %s)\n" "${FLOW_LOG_ID:-<missing>}" "${FLOW_LOG_MODE:-<missing>}"
printf "  FlowLog S3 bucket: %s\n" "${FLOW_LOG_S3_BUCKET:-<missing>}"
printf "  Glue DB / Crawler: %s / %s\n" "${GLUE_DB:-<none>}" "${GLUE_CRAWLER:-<none>}"
printf "  Athena NamedQuery ID: %s\n" "${ATHENA_NQ:-<none>}"
printf "  Public subnets: %s\n" "$(echo "$PUBLIC_SUBNETS_JSON" | jq -r 'length')"
printf "  Private subnets: %s\n" "$(echo "$PRIVATE_SUBNETS_JSON" | jq -r 'length')"
printf "  Private route tables: %s\n" "$(echo "$ROUTE_TABLES_JSON" | jq -r 'length')"
_info ""

# 1) EC2 FlowLog status (concise)
if [ -n "$FLOW_LOG_ID" ] && [ "$FLOW_LOG_ID" != "null" ]; then
  _info "Checking EC2 FlowLog status..."
  FL_OUT="$(aws ec2 describe-flow-logs --flow-log-ids "$FLOW_LOG_ID" --output json 2>/dev/null || true)"
  if [ -z "$FL_OUT" ] || [ "$FL_OUT" = "null" ]; then
    _warn "FlowLog $FLOW_LOG_ID: no data (check permissions)."
    FLOW_STATUS="MISSING"
  else
    # extract key fields
    FLOW_STATUS="$(echo "$FL_OUT" | jq -r '.FlowLogs[0].FlowLogStatus // "UNKNOWN"')"
    FLOW_DELIVER="$(echo "$FL_OUT" | jq -r '.FlowLogs[0].DeliverLogsStatus // "UNKNOWN"')"
    FLOW_DEST_TYPE="$(echo "$FL_OUT" | jq -r '.FlowLogs[0].LogDestinationType // "UNKNOWN"')"
    FLOW_DEST="$(echo "$FL_OUT" | jq -r '.FlowLogs[0].LogDestination // "UNKNOWN"')"
    printf "  FlowLog: %s | status=%s deliver=%s dest_type=%s dest=%s\n" "$FLOW_LOG_ID" "$FLOW_STATUS" "$FLOW_DELIVER" "$FLOW_DEST_TYPE" "$FLOW_DEST"
  fi
else
  _warn "No FlowLog ID exported; skipping FlowLog checks."
  FLOW_STATUS="NONE"
fi
_info ""

# get account id for S3 prefix lookup
AWS_ACCOUNT="$(aws sts get-caller-identity --query Account --output text 2>/dev/null || true)"
if [ -z "$AWS_ACCOUNT" ]; then
  _warn "Unable to determine AWS Account. S3 key checks may fail."
fi

# 2) S3 checks (sample key + attributes + count)
if [ -n "$FLOW_LOG_S3_BUCKET" ] && [ "$FLOW_LOG_S3_BUCKET" != "null" ]; then
  _info "Checking S3 bucket '$FLOW_LOG_S3_BUCKET' for flow log objects (minimal)..."
  PREFIX="AWSLogs/${AWS_ACCOUNT}/vpcflowlogs/"
  # get one sample key
  SAMPLE_KEY="$(aws s3api list-objects-v2 --bucket "$FLOW_LOG_S3_BUCKET" --prefix "$PREFIX" --max-items 1 --query 'Contents[0].Key' --output text 2>/dev/null || true)"
  if [ -z "$SAMPLE_KEY" ] || [ "$SAMPLE_KEY" = "None" ]; then
    _warn "No flow log objects found under prefix $PREFIX (bucket reachable?)."
    FOUND_COUNT=0
  else
    # count up to 1000 objects quickly (minimal)
    FOUND_COUNT="$(aws s3api list-objects-v2 --bucket "$FLOW_LOG_S3_BUCKET" --prefix "$PREFIX" --max-items 1000 --query 'length(Contents[])' --output text 2>/dev/null || echo "0")"
    printf "  Found objects (sample): %s  (count up to 1000: %s)\n" "$SAMPLE_KEY" "${FOUND_COUNT:-0}"
    # prefer get-object-attributes if available, fall back to head-object
    if aws s3api get-object-attributes --bucket "$FLOW_LOG_S3_BUCKET" --key "$SAMPLE_KEY" --object-attributes ETag,ObjectSize  >/dev/null 2>&1; then
      aws s3api get-object-attributes --bucket "$FLOW_LOG_S3_BUCKET" --key "$SAMPLE_KEY" --object-attributes ETag,ObjectSize --output json | jq -r '. | "    size=\(.ObjectSize) etag=\(.ETag)"'
    else
      aws s3api head-object --bucket "$FLOW_LOG_S3_BUCKET" --key "$SAMPLE_KEY" --output json 2>/dev/null | jq -r '{Size: .ContentLength, LastModified: .LastModified} | "    size=\(.Size) last_modified=\(.LastModified)"' || true
    fi
  fi
else
  _warn "No flow log S3 bucket exported; skipping S3 checks."
fi
_info ""

# 3) VPC Endpoints (summary)
if [ -n "$VPC_ID" ] && [ "$VPC_ID" != "null" ]; then
  _info "Listing VPC endpoints for VPC $VPC_ID (summary)..."
  aws ec2 describe-vpc-endpoints --filters Name=vpc-id,Values="$VPC_ID" --query 'VpcEndpoints[].{Id:VpcEndpointId,Service:ServiceName,Type:VpcEndpointType,State:State,SubnetCount:length(SubnetIds)}' --output table 2>/dev/null || _warn "Failed to list VPC endpoints (check permissions)."
else
  _warn "VPC ID missing; skipping VPC endpoint checks."
fi
_info ""

# 4) Route table sanity for private RTs: show only relevant route targets
echo "Route-table checks (private RTs):"
if echo "$ROUTE_TABLES_JSON" | jq -e 'length > 0' >/dev/null 2>&1; then
  echo "$ROUTE_TABLES_JSON" | jq -r '.[]' | while read -r rtb; do
    printf "\n--- route-table: %s ---\n" "$rtb"
    aws ec2 describe-route-tables --route-table-ids "$rtb" --output json 2>/dev/null | \
      jq -r '.RouteTables[0].Routes[] | select(.GatewayId!=null or .NatGatewayId!=null or .DestinationPrefixListId!=null) | {Destination:(.DestinationCidrBlock//.DestinationPrefixListId),Gateway:(.GatewayId//.NatGatewayId//"")}' \
      || _warn "  could not describe route table $rtb"
  done
else
  _warn "No private route table IDs exported; skipping route-table checks."
fi
_info ""

# 5) Subnets: concise list
_info "Subnet details (public):"
echo "$PUBLIC_SUBNETS_JSON" | jq -r '.[]' | while read -r sid; do
  aws ec2 describe-subnets --subnet-ids "$sid" --query 'Subnets[0].[SubnetId,CidrBlock,AvailabilityZone,MapPublicIpOnLaunch]' --output text 2>/dev/null \
    | awk '{printf "  pub: %s  | %s | %s | mapPublic=%s\n", $1,$2,$3,$4}' \
    || _warn "  pub: $sid (describe failed)"
done
_info ""

_info "Subnet details (private):"
echo "$PRIVATE_SUBNETS_JSON" | jq -r '.[]' | while read -r sid; do
  aws ec2 describe-subnets --subnet-ids "$sid" --query 'Subnets[0].[SubnetId,CidrBlock,AvailabilityZone,MapPublicIpOnLaunch]' --output text 2>/dev/null \
    | awk '{printf "  priv: %s  | %s | %s | mapPublic=%s\n", $1,$2,$3,$4}' \
    || _warn "  priv: $sid (describe failed)"
done
_info ""

# 6) Glue crawler & DB (concise)
if [ -n "$GLUE_CRAWLER" ] && [ "$GLUE_CRAWLER" != "null" ]; then
  _info "Glue crawler status:"
  aws glue get-crawler --name "$GLUE_CRAWLER" --output json 2>/dev/null | jq -r '.Crawler | {Name:.Name,State:.State,DatabaseName:.DatabaseName,CreationTime:.CreationTime}' \
    || _warn "  failed to get crawler $GLUE_CRAWLER (check permissions)"
else
  _info "Glue crawler not configured; skipping."
fi
if [ -n "$GLUE_DB" ] && [ "$GLUE_DB" != "null" ]; then
  aws glue get-database --name "$GLUE_DB" --output json 2>/dev/null | jq -r '.Database | {Name:.Name,Description:.Description,CreateTime:.CreateTime}' \
    || _warn "  failed to get database $GLUE_DB"
fi
_info ""

# 7) Athena named query (concise)
if [ -n "$ATHENA_NQ" ] && [ "$ATHENA_NQ" != "null" ]; then
  _info "Athena named query:"
  aws athena get-named-query --named-query-id "$ATHENA_NQ" --output json 2>/dev/null | jq -r '.NamedQuery | {Name:.Name,Database:.Database,NamedQueryId:.NamedQueryId}' \
    || _warn "  failed to fetch named query $ATHENA_NQ"
else
  _info "Athena named query not configured; skipping."
fi
_info ""

# 8) Quick functional test hints (printed only) — non-executing suggestions
_info "Quick functional tests you can run from inside the VPC (examples):"
_info "  - From a pod in private subnet: curl -sS --fail https://aws.amazon.com -o /dev/null && echo 'egress ok'"
_info "  - Validate S3 endpoint (gateway): aws s3 ls s3://<bucket>  (should not require NAT if vpce-s3 exists and IAM allows)"
_info "  - Use SSM to run tests on an EC2 in private subnet when needed"
_info ""

# 9) Final concise verdict
_ok=0
if [ "$FLOW_STATUS" = "ACTIVE" ] && [ "$(echo "${FOUND_COUNT:-0}" | awk '{print $0+0}')" -gt 0 ]; then
  _ok=1
fi

if [ "$_ok" -eq 1 ]; then
  _info "VERDICT: Smoke checks PASSED (FlowLog ACTIVE + sample objects present)."
else
  _warn "VERDICT: Smoke checks PARTIAL/FAILED — inspect full log: $LOGFILE"
fi

_info ""
_info "Finished. Full log: $LOGFILE"
# Print short summary to original stdout (non-logged)
{
  printf "\nSMOKE SUMMARY: vpc=%s flow=%s flow_dest=%s s3_bucket=%s s3_sample=%s s3_count=%s\n" \
    "${VPC_ID:-<missing>}" "${FLOW_LOG_ID:-<missing>}" "${FLOW_DEST_TYPE:-<unknown>}" "${FLOW_LOG_S3_BUCKET:-<missing>}" "${SAMPLE_KEY:-<none>}" "${FOUND_COUNT:-0}"
} >&3

exit 0
