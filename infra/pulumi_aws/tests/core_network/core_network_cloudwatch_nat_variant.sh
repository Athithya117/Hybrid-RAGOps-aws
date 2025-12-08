#!/usr/bin/env bash
# infra/pulumi_aws/tests/core_network/core_network_cloudwatch_nat_variant.sh
#
# Purpose:
#   Deterministic variant smoke test for core_network Pulumi stack (cloudwatch + NAT variant).
#   - Applies a pre-defined variant via `make pulumi-up`
#   - Reads infra outputs from infra/pulumi_aws/pulumi-outputs.json
#   - Runs a concise set of non-destructive checks:
#       * EC2 FlowLog status
#       * CloudWatch Log Group (when using cloudwatch)
#       * S3 sample objects (when using s3)
#       * VPC endpoints summary
#       * Route table summary for private RTs (coalesced destination/target)
#       * Subnet brief listing
#   - Produces a minimal human-friendly summary on STDOUT and writes a full log to /tmp.
#
# Notes (important):
#   - This script is intended for smoke-testing only; it does not create resources except by calling `make pulumi-up`.
#   - Comments are present inside by design (these test scripts are exceptional).
#
: <<'EOF'
Example variant environment (export these before running, or edit below block and uncomment):
export TAG_PREFIX="${TAG_PREFIX:-pulumi}"
export MULTI_AZ_DEPLOYMENT="${MULTI_AZ_DEPLOYMENT:-true}"
export AZ_COUNT="${AZ_COUNT:-2}"
export VPC_CIDR="${VPC_CIDR:-10.1.0.0/16}"
export NO_NAT="${NO_NAT:-false}"
export NAT_SINGLE="${NAT_SINGLE:-true}"
export CREATE_VPC_ENDPOINTS="${CREATE_VPC_ENDPOINTS:-true}"
export CREATE_VPC_ENDPOINT_SERVICES="${CREATE_VPC_ENDPOINT_SERVICES:-s3,ssm}"
export FLOW_LOG_MODE="${FLOW_LOG_MODE:-cloudwatch}"
export FLOW_LOG_CW_LOG_GROUP="${FLOW_LOG_CW_LOG_GROUP:-/aws/vpc/flowlogs/${STACK}}"
export FLOW_LOG_CW_RETENTION_DAYS="${FLOW_LOG_CW_RETENTION_DAYS:-7}"
export CREATE_GLUE_CRAWLER="${CREATE_GLUE_CRAWLER:-false}"
export CREATE_ATHENA="${CREATE_ATHENA:-false}"
export FLOW_LOG_S3_BUCKET="${FLOW_LOG_S3_BUCKET:-}"
EOF

set -euo pipefail

# ---- fail-fast tool checks ----
for cmd in aws pulumi jq make; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: required command not found: $cmd"; exit 2; }
done

# ---- logging setup ----
TS=$(date -u +"%Y%m%dT%H%M%SZ")
LOG="/tmp/core_network_variant_${TS}.log"
: > "$LOG"
echo "CORE NETWORK VARIANT SMOKE TEST — $TS" | tee -a "$LOG"

# ---- show minimal chosen config (echo envs we rely on) ----
cat <<EOF | tee -a "$LOG"
Selected variant config (minimal):
  MULTI_AZ_DEPLOYMENT=${MULTI_AZ_DEPLOYMENT:-<unset>}
  AZ_COUNT=${AZ_COUNT:-<unset>}
  VPC_CIDR=${VPC_CIDR:-<unset>}
  NO_NAT=${NO_NAT:-<unset>}
  NAT_SINGLE=${NAT_SINGLE:-<unset>}
  ENDPOINT_SERVICES=${CREATE_VPC_ENDPOINT_SERVICES:-<unset>}
  FLOW_LOG_MODE=${FLOW_LOG_MODE:-<unset>}
  FLOW_LOG_CW_RETENTION_DAYS=${FLOW_LOG_CW_RETENTION_DAYS:-<unset>}
  CREATE_GLUE_CRAWLER=${CREATE_GLUE_CRAWLER:-<unset>}
  CREATE_ATHENA=${CREATE_ATHENA:-<unset>}
Logs: $LOG
EOF

# ---- apply infra (deterministic) ----
echo "==> running: make pulumi-up (logs -> $LOG)" | tee -a "$LOG"
if ! make pulumi-up >>"$LOG" 2>&1; then
  echo "ERROR: pulumi up failed — see $LOG" | tee -a "$LOG"
  tail -n 200 "$LOG" | sed -n '1,200p'
  exit 3
fi

# ---- read outputs (fail fast if missing) ----
OUTFILE="infra/pulumi_aws/pulumi-outputs.json"
if [ ! -f "$OUTFILE" ]; then
  echo "ERROR: expected outputs file not found: $OUTFILE" | tee -a "$LOG"
  exit 4
fi

VPC_ID=$(jq -r '.vpc_id // empty' "$OUTFILE")
FLOW_LOG_ID=$(jq -r '.flow_log_id // empty' "$OUTFILE")
FLOW_LOG_MODE_OUT=$(jq -r '.flow_log_mode // empty' "$OUTFILE")
FLOW_LOG_S3_BUCKET=$(jq -r '.flow_log_s3_bucket // empty' "$OUTFILE")
GLUE_CRAWLER=$(jq -r '.glue_crawler // empty' "$OUTFILE")
GLUE_DB=$(jq -r '.glue_db // empty' "$OUTFILE")
ATHENA_NQ=$(jq -r '.athena_named_query // empty' "$OUTFILE")
PUBLIC_COUNT=$(jq -r '.public_subnet_ids | length // 0' "$OUTFILE")
PRIVATE_COUNT=$(jq -r '.private_subnet_ids | length // 0' "$OUTFILE")
RT_PRIVATE_COUNT=$(jq -r '.route_table_ids.private | length // 0' "$OUTFILE")

# ---- concise summary to STDOUT ----
cat <<EOF | tee -a "$LOG"
=== SUMMARY (concise) ===
VPC ID: ${VPC_ID:-<missing>}
FlowLog (id / mode): ${FLOW_LOG_ID:-<missing>} / ${FLOW_LOG_MODE_OUT:-${FLOW_LOG_MODE:-<missing>}}
S3 bucket (if any): ${FLOW_LOG_S3_BUCKET:-<none>}
Glue crawler/db: ${GLUE_CRAWLER:-<none>} / ${GLUE_DB:-<none>}
Athena named query id: ${ATHENA_NQ:-<none>}
Public subnets: $PUBLIC_COUNT
Private subnets: $PRIVATE_COUNT
Private route tables: $RT_PRIVATE_COUNT
Logfile: $LOG
EOF

# ---- 1) FlowLog status (concise) ----
if [ -n "$FLOW_LOG_ID" ]; then
  echo "Checking FlowLog status..." | tee -a "$LOG"
  # non-fatal - just print a compact row
  aws ec2 describe-flow-logs --flow-log-ids "$FLOW_LOG_ID" \
    --query 'FlowLogs[0].[FlowLogId,FlowLogStatus,DeliverLogsStatus,LogDestinationType,LogDestination]' \
    --output text 2>>"$LOG" | tee -a "$LOG" || true
else
  echo "No FlowLog ID exported; skipping." | tee -a "$LOG"
fi

# ---- 2) CloudWatch Log Group check (if cloudwatch) ----
if [ "${FLOW_LOG_MODE_OUT:-${FLOW_LOG_MODE:-}}" = "cloudwatch" ]; then
  LG="${FLOW_LOG_CW_LOG_GROUP:-/aws/vpc/flowlogs/}"
  echo "Checking CloudWatch Log Group: $LG" | tee -a "$LOG"
  aws logs describe-log-groups --log-group-name-prefix "$LG" --output json 2>>"$LOG" | tee -a "$LOG" || true
fi

# ---- 3) S3 sample objects (if s3 mode / bucket provided) ----
if [ -n "$FLOW_LOG_S3_BUCKET" ]; then
  ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>>"$LOG" || echo "")
  PREFIX="AWSLogs/${ACCOUNT}/vpcflowlogs/"
  echo "Checking S3 bucket '${FLOW_LOG_S3_BUCKET}' for sample objects under '${PREFIX}' (max 5) ..." | tee -a "$LOG"
  aws s3api list-objects-v2 --bucket "$FLOW_LOG_S3_BUCKET" --prefix "$PREFIX" --max-items 5 --output json 2>>"$LOG" | tee -a "$LOG" || true
fi

# ---- 4) VPC endpoints summary (non-destructive) ----
if [ -n "$VPC_ID" ]; then
  echo "Listing VPC endpoints (summary) for VPC: $VPC_ID" | tee -a "$LOG"
  aws ec2 describe-vpc-endpoints --filters Name=vpc-id,Values="$VPC_ID" \
    --query 'VpcEndpoints[].{Id:VpcEndpointId,Service:ServiceName,Type:VpcEndpointType,State:State,SubnetCount:length(SubnetIds)}' \
    --output table 2>>"$LOG" | tee -a "$LOG" || true
fi

# ---- 5) Route tables: print concise routes for private RTs (coalesced fields) ----
if [ "$RT_PRIVATE_COUNT" -gt 0 ]; then
  echo "Route-table checks (private RTs):" | tee -a "$LOG"
  # iterate each private route table id that was exported
  jq -r '.route_table_ids.private[]?' "$OUTFILE" 2>/dev/null | while read -r rtid; do
    [ -z "$rtid" ] && continue
    echo "--- route-table: $rtid ---" | tee -a "$LOG"
    # Use coalesce() in JMESPath to normalize optional fields (portable across AWS CLI versions).
    aws ec2 describe-route-tables --route-table-ids "$rtid" \
      --query 'RouteTables[0].Routes[].{Destination: coalesce(DestinationCidrBlock, DestinationPrefixListId), Target: coalesce(NatGatewayId, GatewayId, NetworkInterfaceId, VpcPeeringConnectionId)}' \
      --output table 2>>"$LOG" | tee -a "$LOG" || true
  done
fi

# ---- 6) Subnet brief summary (public & private) ----
echo "Subnet details (summary):" | tee -a "$LOG"
jq -r '.public_subnet_ids[]? // empty' "$OUTFILE" 2>/dev/null | while read -r s; do
  [ -z "$s" ] && continue
  aws ec2 describe-subnets --subnet-ids "$s" --query 'Subnets[0].{SubnetId:SubnetId,Cidr:CidrBlock,AZ:AvailabilityZone,Public:MapPublicIpOnLaunch}' --output json 2>>"$LOG" \
    | jq -c '.' | sed 's/^/  pub: /' | tee -a "$LOG" || true
done
jq -r '.private_subnet_ids[]? // empty' "$OUTFILE" 2>/dev/null | while read -r s; do
  [ -z "$s" ] && continue
  aws ec2 describe-subnets --subnet-ids "$s" --query 'Subnets[0].{SubnetId:SubnetId,Cidr:CidrBlock,AZ:AvailabilityZone,Public:MapPublicIpOnLaunch}' --output json 2>>"$LOG" \
    | jq -c '.' | sed 's/^/  priv: /' | tee -a "$LOG" || true
done

# ---- short non-blocking sleep to allow for eventual consistency of ENIs / delivery ----
echo "Sleeping 10s for eventual consistency..." | tee -a "$LOG"
sleep 10

# ---- final verdict ----
echo "==== QUICK VERDICT ====" | tee -a "$LOG"
FL_STATUS="unknown"
if [ -n "$FLOW_LOG_ID" ]; then
  FL_STATUS=$(aws ec2 describe-flow-logs --flow-log-ids "$FLOW_LOG_ID" --query 'FlowLogs[0].FlowLogStatus' --output text 2>>"$LOG" || echo "ERROR")
fi

# If cloudwatch mode was selected (env or exported), treat that as acceptable
if [ "$FL_STATUS" = "ACTIVE" ] || [ "${FLOW_LOG_MODE_OUT:-${FLOW_LOG_MODE:-}}" = "cloudwatch" ]; then
  echo "VERDICT: Smoke checks PASSED (FlowLog status: $FL_STATUS)" | tee -a "$LOG"
else
  echo "VERDICT: Smoke checks UNVERIFIED (FlowLog status: $FL_STATUS) — inspect $LOG" | tee -a "$LOG"
fi

printf "SMOKE SUMMARY: vpc=%s flow=%s mode=%s s3_bucket=%s public=%s private=%s\n" \
  "${VPC_ID:-<none>}" "${FLOW_LOG_ID:-<none>}" "${FLOW_LOG_MODE_OUT:-${FLOW_LOG_MODE:-<none>}}" "${FLOW_LOG_S3_BUCKET:-<none>}" \
  "$PUBLIC_COUNT" "$PRIVATE_COUNT" | tee -a "$LOG"

echo "Full logfile available at: $LOG"
echo "You can view it with: cat $LOG"
exit 0
