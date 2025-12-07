#!/usr/bin/env bash
set -euo pipefail
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  echo "ERROR: do not source this file. Run it: bash $0" >&2
  return 1 2>/dev/null || exit 1
fi

export PROJECT_DIR="${PROJECT_DIR:-infra/pulumi_aws}"       # pulumi project dir; change if repo layout differs
export VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/venv}"         # virtualenv path used by pulumi_setup.sh
export REQ_FILE="${REQ_FILE:-${PROJECT_DIR}/requirements.txt}" # pip requirements file for bootstrapping
export AWS_REGION="${AWS_REGION:-ap-south-1}"              # AWS region for infra; change per target region
export S3_BUCKET="${S3_BUCKET:-e2e-rag-42}"                # global artifacts bucket (used to create pulumi state bucket if needed)
export PULUMI_S3_BUCKET="${PULUMI_STATE_BUCKET:-rag-pulumi-state31}"  # Pulumi backend S3 bucket; change to central bucket in prod
export S3_PREFIX="${S3_PREFIX:-pulumi/}"                   # S3 prefix used by Pulumi backend
export PULUMI_STATE_BUCKET="${PULUMI_STATE_BUCKET:-rag-pulumi-state31}" # Pulumi state bucket alias (keeps names readable)
export PULUMI_STATE_PREFIX="${PULUMI_STATE_PREFIX:-${S3_PREFIX}}" # Pulumi state prefix alias
export DDB_TABLE="${DDB_TABLE:-pulumi-state-locks}"        # DynamoDB table for Pulumi state locking; change if shared
export PULUMI_STACK="${PULUMI_STACK:-prod}"                # default stack name; override per environment (dev/stage/prod)
export STACK="${STACK:-${PULUMI_STACK}}"                  # convenience alias used by scripts & code
export PULUMI_CONFIG_PASSPHRASE="${PULUMI_CONFIG_PASSPHRASE:-password}" # pulumi config passphrase (use secure secret in prod)
export PULUMI_ORG="${PULUMI_ORG:-}"                       # pulumi cloud org (empty when using S3 backend)
export PULUMI_BINARY_PATH="${PULUMI_BINARY_PATH:-}"       # override pulumi CLI path if needed
export PULUMI_CREDS_FILE="${PULUMI_CREDS_FILE:-/tmp/pulumi-ci-credentials.json}" # local creds file for automation
export PULUMI_AUTOINIT="${PULUMI_AUTOINIT:-true}"         # auto bootstrap pulumi backend resources when running setup
export ENABLE_PULUMI_AUTOINIT="${ENABLE_PULUMI_AUTOINIT:-true}" # duplicate guard for scripts that read different var
export POLICY_NAME="${POLICY_NAME:-PulumiStateAccessPolicy}" # IAM policy name for Pulumi state access (creation)
export FORCE_DELETE="${FORCE_DELETE:-true}"               # allow force delete of created infra in destroy scripts
export FORCE_DELETE_BUCKET="${FORCE_DELETE_BUCKET:-false}" # only true to allow programmatic S3 bucket deletion (dangerous)
export PIP_BREAK_SYSTEM_PACKAGES_FLAG="${PIP_BREAK_SYSTEM_PACKAGES_FLAG:---no-input}" # pip flag for noninteractive installs
export PULUMI_LOGIN_URL="${PULUMI_LOGIN_URL:-s3://${PULUMI_S3_BUCKET}/${S3_PREFIX}}" # pulumi login url (S3 backend default)
export PULUMI_PYTHON_CMD="${PULUMI_PYTHON_CMD:-${VENV_DIR}/bin/python}" # python used to run pulumi programs in the venv
export PULUMI_CREDS_JSON="${PULUMI_CREDS_JSON:-}"          # optional JSON creds for pulumi (CI) if using cloud backend
export PULUMI_IAM_USER="${PULUMI_IAM_USER:-}"              # optional IAM user name to grant pulumi-state access (if creating)
export PYTHON_BIN="${PYTHON_BIN:-python3}"                 # system python binary fallback used by helper scripts

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
export ENABLE_FLOW_LOGS="${ENABLE_FLOW_LOGS:-false}"      # legacy toggle; used only if FLOW_LOG_MODE unset
export FLOW_LOG_CW_LOG_GROUP="${FLOW_LOG_CW_LOG_GROUP:-/aws/vpc/flowlogs/${STACK}}" # CW group name; change retention below
export FLOW_LOG_CW_RETENTION_DAYS="${FLOW_LOG_CW_RETENTION_DAYS:-14}" # retention for CW logs (shorter for dev)

export FLOW_LOG_S3_BUCKET="${FLOW_LOG_S3_BUCKET:-rag-vpc-flow-31}"        # existing bucket name or ARN when using external bucket (s3 mode)
export FLOW_LOG_S3_CREATE="${FLOW_LOG_S3_CREATE:-false}"  # true => auto-create bucket in same stack (careful in prod)
export FLOW_LOG_S3_CREATE_NAME="${FLOW_LOG_S3_CREATE_NAME:-${TAG_PREFIX}-${STACK}-vpc-flow-logs}" # bucket name when auto-creating
export FLOW_LOG_S3_PREFIX="${FLOW_LOG_S3_PREFIX:-AWSLogs/<ACCOUNT>/vpcflowlogs/}" # delivery prefix; change only if transforming layout
export FLOW_LOG_S3_LIFECYCLE_TRANSITION_DAYS="${FLOW_LOG_S3_LIFECYCLE_TRANSITION_DAYS:-30}" # move to IA after X days
export FLOW_LOG_S3_EXPIRATION_DAYS="${FLOW_LOG_S3_EXPIRATION_DAYS:-365}" # expire logs after X days; increase for compliance
export FLOW_LOG_S3_ACCESS_LOGGING="${FLOW_LOG_S3_ACCESS_LOGGING:-false}" # enable S3 server access logs for the log bucket
export FLOW_LOG_S3_ACCESS_BUCKET="${FLOW_LOG_S3_ACCESS_BUCKET:-}" # bucket to receive access logs (required if access logging enabled)

export FLOW_LOG_SSE_ALGORITHM="${FLOW_LOG_SSE_ALGORITHM:-AES256}" # AES256|aws:kms — choose aws:kms for compliance
export FLOW_LOG_KMS_CREATE="${FLOW_LOG_KMS_CREATE:-false}"   # true => create CMK for log bucket (requires approvals)
export FLOW_LOG_KMS_ARN="${FLOW_LOG_KMS_ARN:-}"              # existing CMK ARN to use (mutually exclusive with CREATE true)

export CREATE_GLUE_CRAWLER="${CREATE_GLUE_CRAWLER:-true}"    # create Glue crawler to discover partitions (cheap, recommended)
export GLUE_CRAWLER_SCHEDULE="${GLUE_CRAWLER_SCHEDULE:-cron(0 * ? * * *)}" # schedule for crawler (hourly default)
export CREATE_GLUE_ETL="${CREATE_GLUE_ETL:-false}"          # true => create Glue ETL job to convert raw->parquet (costly; default OFF)
export GLUE_ETL_SCHEDULE="${GLUE_ETL_SCHEDULE:-cron(0 2 * * ? *)}" # ETL schedule (daily 02:00 UTC)
export GLUE_ETL_DPU="${GLUE_ETL_DPU:-10}"                   # Glue DPUs for ETL job; increase for big volumes
export GLUE_SCRIPT_S3_PREFIX="${GLUE_SCRIPT_S3_PREFIX:-glue-scripts/}" # where ETL script is uploaded within scripts bucket

export CREATE_ATHENA="${CREATE_ATHENA:-true}"               # create Athena helper (Glue DB + named query) when using parquet
export ATHENA_DB_NAME="${ATHENA_DB_NAME:-vpc_flow_logs_${STACK}}" # Glue/Athena DB name; centralize if needed
export ATHENA_TABLE_NAME="${ATHENA_TABLE_NAME:-vpc_flow_parquet}" # Athena table name for parquet dataset
export ATHENA_OUTPUT_BUCKET="${ATHENA_OUTPUT_BUCKET:-}"     # optional Athena query results bucket; defaults to log bucket when empty

export FLOW_LOG_MAX_DAILY_BYTES="${FLOW_LOG_MAX_DAILY_BYTES:-1073741824}" # 1 GiB/day guardrail; tune per traffic expectations
export ZSTD_COMPRESSION_LEVEL=3 

export AVOID_DOMAIN="${AVOID_DOMAIN:-true}"                # if true, skip creating DNS/certs (useful for learners / freenom)



if [ -z "$PYTHON_BIN" ]; then
  for p in python3.12 python3.11 python3.10 python3; do
    if command -v "$p" >/dev/null 2>&1; then
      PYTHON_BIN="$p"
      break
    fi
  done
fi
if [ -z "$PYTHON_BIN" ]; then
  echo "ERROR: no python3 interpreter found (tried python3.12, python3.11, python3.10, python3)" >&2
  exit 11
fi
abs_path() {
  local p="$1"
  if command -v realpath >/dev/null 2>&1; then
    realpath -m "$p"
  elif command -v readlink >/dev/null 2>&1; then
    readlink -f "$p" || "$PYTHON_BIN" -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$p"
  else
    "$PYTHON_BIN" -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$p"
  fi
}
PROJECT_DIR="$(abs_path "$PROJECT_DIR")"
VENV_DIR="$(abs_path "$VENV_DIR")"
REQ_FILE="$(abs_path "$REQ_FILE")"
mkdir -p "$PROJECT_DIR"
mkdir -p "${PROJECT_DIR}/.pulumi-logs"
out_json="${PROJECT_DIR}/pulumi-outputs.json"
[ -f "$out_json" ] || printf '{}' >"$out_json" || true
out_exports="${PROJECT_DIR}/pulumi-exports.sh"
[ -f "$out_exports" ] || { printf '#!/usr/bin/env bash\n# pulumi exports placeholder\n' >"$out_exports" || true; chmod +x "$out_exports" || true; }
prog="$(basename "$0")"
usage() {
  cat <<EOF
Usage: $prog [--create|--delete] [--force] [--preview] [--preview-and-up] [-h|--help]
  --create            create backend + venv + pulumi up (or preview)
  --delete            destroy stack and remove pulumi prefix artifacts
  --force             with --delete also remove pulumi prefix for all stacks
  --preview           run pulumi preview only (no up)
  --preview-and-up    run preview and, if successful, pulumi up
EOF
}
MODE="" FORCE_FLAG=false PREVIEW=false PREVIEW_AND_UP=false
while [ $# -gt 0 ]; do
  case "$1" in
    --create) MODE="create"; shift;;
    --delete) MODE="delete"; shift;;
    --force) FORCE_FLAG=true; shift;;
    --preview) PREVIEW=true; shift;;
    --preview-and-up) PREVIEW_AND_UP=true; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done
[ -n "$MODE" ] || { echo "ERROR: must pass --create or --delete" >&2; usage; exit 2; }
log() { printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
die() { echo "ERROR: $*" >&2; exit "${2:-1}"; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command '$1' not found" 10; }
TMPS=()
cleanup() { for f in "${TMPS[@]:-}"; do [ -f "$f" ] && rm -f "$f"; done; }
trap cleanup EXIT
retry() {
  local tries=${1:-5}; shift
  local delay=${1:-1}; shift
  local i=0 rc=0
  while [ $i -lt $tries ]; do
    set +e
    "$@"
    rc=$?
    set -e
    [ $rc -eq 0 ] && return 0
    i=$((i+1))
    sleep $delay
    delay=$((delay * 2))
  done
  return $rc
}
require_cmd aws
require_cmd curl
if ! aws sts get-caller-identity >/dev/null 2>&1; then
  die "AWS credentials not configured or not working (aws sts get-caller-identity failed)" 20
fi
create_bucket_if_missing() {
  local bucket="$1"
  log "s3: ensure bucket exists: ${bucket} (region=${AWS_REGION})"
  if retry 6 1 aws s3api head-bucket --bucket "$bucket" >/dev/null 2>&1; then
    log "s3: bucket exists"
  else
    if [ "$AWS_REGION" = "us-east-1" ]; then
      aws s3api create-bucket --bucket "$bucket" >/dev/null 2>&1 || log "s3: create returned non-zero"
    else
      aws s3api create-bucket --bucket "$bucket" --create-bucket-configuration LocationConstraint="$AWS_REGION" >/dev/null 2>&1 || log "s3: create returned non-zero"
    fi
    retry 8 2 aws s3api head-bucket --bucket "$bucket" >/dev/null 2>&1 || log "s3: head-bucket still failing (continuing)"
  fi
  aws s3api put-bucket-versioning --bucket "$bucket" --versioning-configuration Status=Enabled >/dev/null 2>&1 || true
  aws s3api put-bucket-encryption --bucket "$bucket" --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}' >/dev/null 2>&1 || true
  aws s3api put-bucket-lifecycle-configuration --bucket "$bucket" --lifecycle-configuration "{\"Rules\":[{\"ID\":\"pulumi-noncurrent-expire\",\"Prefix\":\"${S3_PREFIX}\",\"Status\":\"Enabled\",\"NoncurrentVersionExpiration\":{\"NoncurrentDays\":365}}]}" >/dev/null 2>&1 || true
  log "s3: bucket prepared (prefix=${S3_PREFIX})"
}
delete_s3_objects() {
  local bucket="$1" prefix="${2:-}"
  log "s3-delete: deleting objects in s3://${bucket}/${prefix}"
  while :; do
    local rv count objs tmp
    if [ -n "$prefix" ]; then
      rv="$(aws s3api list-object-versions --bucket "$bucket" --prefix "$prefix" --output json 2>/dev/null || echo '{}')"
    else
      rv="$(aws s3api list-object-versions --bucket "$bucket" --output json 2>/dev/null || echo '{}')"
    fi
    count=$(command -v jq >/dev/null 2>&1 && jq -r '[.Versions[], .DeleteMarkers[]] | length' <<<"$rv" 2>/dev/null || "$PYTHON_BIN" - <<PY
import sys,json
try:
  r=json.load(sys.stdin)
  c=sum(len(r.get(k,[])) for k in ("Versions","DeleteMarkers"))
  print(c)
except Exception:
  print(0)
PY
)
    if [ -z "$count" ] || [ "$count" = "0" ]; then break; fi
    objs=$(command -v jq >/dev/null 2>&1 && jq -c '[.Versions[]?, .DeleteMarkers[]?] | map({Key:.Key,VersionId:.VersionId})' <<<"$rv" || "$PYTHON_BIN" - <<PY
import sys,json
r=json.load(sys.stdin)
arr=[]
for k in ("Versions","DeleteMarkers"):
  for it in r.get(k,[]):
    arr.append({"Key":it.get("Key"), "VersionId": it.get("VersionId")})
print(json.dumps(arr))
PY
)
    tmp="$(mktemp)"; TMPS+=("$tmp")
    printf '{"Objects":%s}' "$objs" >"$tmp"
    aws s3api delete-objects --bucket "$bucket" --delete "file://$tmp" >/dev/null 2>&1 || true
    rm -f "$tmp" || true
    sleep 1
  done
  log "s3-delete: done for s3://${bucket}/${prefix}"
}
create_dynamodb_if_missing() {
  local table="$1"
  log "ddb: ensure table ${table}"
  if aws dynamodb describe-table --table-name "$table" >/dev/null 2>&1; then
    log "ddb: exists"
  else
    set +e
    aws dynamodb create-table --table-name "$table" \
      --attribute-definitions AttributeName=LockID,AttributeType=S \
      --key-schema AttributeName=LockID,KeyType=HASH \
      --billing-mode PAY_PER_REQUEST --region "$AWS_REGION" >/dev/null 2>&1
    rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
      aws dynamodb wait table-exists --table-name "$table" --region "$AWS_REGION" >/dev/null 2>&1 || true
      log "ddb: created and ACTIVE"
    else
      log "ddb: create returned non-zero (continuing)"
    fi
  fi
  aws dynamodb update-time-to-live --table-name "$table" --time-to-live-specification "Enabled=true,AttributeName=Expires" >/dev/null 2>&1 || true
}
delete_dynamodb_table_if_exists() {
  local table="$1"
  if aws dynamodb describe-table --table-name "$table" >/dev/null 2>&1; then
    aws dynamodb delete-table --table-name "$table" --region "$AWS_REGION" >/dev/null 2>&1 || true
    aws dynamodb wait table-not-exists --table-name "$table" --region "$AWS_REGION" || true
    log "ddb-delete: table deleted or attempted"
  else
    log "ddb-delete: table not found; skipping"
  fi
}
get_account_id() { aws sts get-caller-identity --query Account --output text 2>/dev/null || true; }
wait_for_policy_arn() {
  local name="$1" tries=8 delay=1 arn=""
  for i in $(seq 1 $tries); do
    arn="$(aws iam list-policies --scope Local --query "Policies[?PolicyName=='${name}'].Arn" --output text 2>/dev/null || true)"
    [ -n "$arn" ] && { echo "$arn"; return 0; }
    sleep "$delay"
    delay=$((delay * 2))
  done
  return 1
}
ensure_policy() {
  local bucket="$1" table="$2" name="$3"
  log "iam: ensure policy ${name}"
  local existing
  existing="$(aws iam list-policies --scope Local --query "Policies[?PolicyName=='${name}'].Arn" --output text || true)"
  if [ -n "$existing" ]; then
    log "iam: policy exists $existing"
    echo "$existing"; return 0
  fi
  local acct; acct="$(get_account_id || true)"
  local tmp
  tmp="$(mktemp)"; TMPS+=("$tmp")
  cat >"$tmp" <<JSON
{
  "Version":"2012-10-17",
  "Statement":[
    {
      "Effect":"Allow",
      "Action":["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket","s3:GetBucketVersioning","s3:PutBucketVersioning"],
      "Resource":["arn:aws:s3:::${bucket}","arn:aws:s3:::${bucket}/*"]
    },
    {
      "Effect":"Allow",
      "Action":["dynamodb:GetItem","dynamodb:PutItem","dynamodb:DeleteItem","dynamodb:UpdateItem","dynamodb:Query","dynamodb:Scan","dynamodb:ConditionCheckItem"],
      "Resource":["arn:aws:dynamodb:${AWS_REGION}:${acct}:table/${table}"]
    }
  ]
}
JSON
  aws iam create-policy --policy-name "$name" --policy-document "file://$tmp" >/dev/null 2>&1 || true
  local arn
  arn="$(wait_for_policy_arn "$name" || true)"
  log "iam: policy ARN: ${arn:-not-found}"
  echo "$arn"
}
create_iam_user_if_requested() {
  local user="${1:-}" policy_arn="${2:-}" creds_file="${3:-}"
  if [ -z "$user" ]; then log "iam: no IAM user requested; skipping"; return 0; fi
  log "iam: ensure user $user"
  aws iam create-user --user-name "$user" >/dev/null 2>&1 || true
  if [ -n "$policy_arn" ]; then aws iam attach-user-policy --user-name "$user" --policy-arn "$policy_arn" >/dev/null 2>&1 || true; fi
  if [ -z "$(aws iam list-access-keys --user-name "$user" --query 'AccessKeyMetadata[].AccessKeyId' --output text || true)" ]; then
    aws iam create-access-key --user-name "$user" >"$creds_file"
    chmod 600 "$creds_file" || true
    log "iam: created access key at $creds_file"
  else
    log "iam: user has access keys; not creating new one"
  fi
}
delete_policy_and_user_idempotent() {
  local policy_name="$1" user="$2"
  local existing
  existing="$(aws iam list-policies --scope Local --query "Policies[?PolicyName=='${policy_name}'].Arn" --output text || true)"
  if [ -n "$existing" ]; then
    for u in $(aws iam list-entities-for-policy --policy-arn "$existing" --query 'PolicyUsers[].UserName' --output text || true); do aws iam detach-user-policy --user-name "$u" --policy-arn "$existing" || true; done
    for r in $(aws iam list-entities-for-policy --policy-arn "$existing" --query 'PolicyRoles[].RoleName' --output text || true); do aws iam detach-role-policy --role-name "$r" --policy-arn "$existing" || true; done
    for v in $(aws iam list-policy-versions --policy-arn "$existing" --query 'Versions[?IsDefaultVersion==`false`].VersionId' --output text || true); do aws iam delete-policy-version --policy-arn "$existing" --version-id "$v" || true; done
    aws iam delete-policy --policy-arn "$existing" || true
    log "iam-delete: policy delete attempted"
  else
    log "iam-delete: policy not found; skipping"
  fi
  if [ -n "$user" ]; then
    if aws iam get-user --user-name "$user" >/dev/null 2>&1; then
      for k in $(aws iam list-access-keys --user-name "$user" --query 'AccessKeyMetadata[].AccessKeyId' --output text || true); do aws iam delete-access-key --user-name "$user" --access-key-id "$k" || true; done
      for a in $(aws iam list-attached-user-policies --user-name "$user" --query 'AttachedPolicies[].PolicyArn' --output text || true); do aws iam detach-user-policy --user-name "$user" --policy-arn "$a" || true; done
      for ip in $(aws iam list-user-policies --user-name "$user" --query 'PolicyNames[]' --output text || true); do aws iam delete-user-policy --user-name "$user" --policy-name "$ip" || true; done
      aws iam delete-user --user-name "$user" || true
      log "iam-delete: user delete attempted"
    else
      log "iam-delete: user not found; skipping"
    fi
  fi
}
ensure_pulumi_cli() {
  if command -v pulumi >/dev/null 2>&1; then return 0; fi
  if [ -x "${PULUMI_BINARY_PATH:-}" ]; then export PATH="$(dirname "$PULUMI_BINARY_PATH"):$PATH"; fi
  if ! command -v pulumi >/dev/null 2>&1; then
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL https://get.pulumi.com | sh >/dev/null 2>&1 || die "pulumi install failed"
      export PATH="$HOME/.pulumi/bin:$PATH"
    else
      die "pulumi CLI not found and cannot auto-install (curl missing)" 11
    fi
  fi
  for i in 1 3; do
    if pulumi version >/dev/null 2>&1; then
      log "pulumi: $(pulumi version)"
      return 0
    fi
    sleep 1
  done
  die "pulumi not responding after install" 11
}
create_venv_and_install() {
  mkdir -p "$(dirname "$VENV_DIR")"
  if [ ! -d "$VENV_DIR" ]; then
    log "venv: creating venv at $VENV_DIR using $PYTHON_BIN"
    "$PYTHON_BIN" -m venv "$VENV_DIR" || die "creating venv failed"
  fi
  VENV_PY="${VENV_DIR}/bin/python"
  VENV_PIP="${VENV_DIR}/bin/pip"
  if [ ! -x "$VENV_PIP" ]; then
    "$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
  fi
  log "venv: upgrading pip setuptools wheel in venv (quiet)"
  set +e
  "$VENV_PY" -m pip install -q --disable-pip-version-check --upgrade pip setuptools wheel >/dev/null 2>&1
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    if "$VENV_PY" -m pip --version 2>&1 | grep -qi "externally-managed-environment"; then
      cat >&2 <<MSG
ERROR: pip failed due to "externally-managed-environment" (PEP 668).
Remediation:
  1) Install system venv support OR
  2) set PYTHON_BIN to an unmanaged python and re-run.
MSG
      exit 13
    fi
    die "pip upgrade in venv failed (rc=$rc)."
  fi
  mkdir -p "${PROJECT_DIR}/.pulumi-logs"
  if [ -f "$REQ_FILE" ]; then
    log "venv: installing packages from ${REQ_FILE} into venv (quiet)"
    LOGF="${PROJECT_DIR}/.pulumi-logs/venv-pip-$(date -u +%s).log"
    set +e
    "$VENV_PY" -m pip install -q --disable-pip-version-check -r "$REQ_FILE" >"$LOGF" 2>&1
    rc=$?
    set -e
    if [ $rc -ne 0 ]; then
      log "venv: pip install failed; tail of log follows:"
      tail -n 200 "$LOGF" >&2 || true
      die "pip install -r ${REQ_FILE} failed (rc=$rc)."
    fi
  else
    log "venv: installing default packages into venv (quiet)"
    LOGF="${PROJECT_DIR}/.pulumi-logs/venv-pip-default-$(date -u +%s).log"
    set +e
    "$VENV_PY" -m pip install -q --disable-pip-version-check pulumi==3.209.0 pulumi-aws==7.12.0 boto3 awscli >"$LOGF" 2>&1
    rc=$?
    set -e
    if [ $rc -ne 0 ]; then
      tail -n 200 "$LOGF" >&2 || true
      die "pip install default packages failed (rc=$rc)."
    fi
  fi
  source "${VENV_DIR}/bin/activate"
  export PULUMI_PYTHON_CMD="${VENV_PY}"
  log "venv: ready ($VENV_DIR) with python $("$VENV_PY" --version 2>/dev/null || true)"
}
activate_venv_if_exists() {
  mkdir -p "$(dirname "$VENV_DIR")" || true
  if [ -d "$VENV_DIR" ]; then
    source "${VENV_DIR}/bin/activate"
    export PULUMI_PYTHON_CMD="${VENV_DIR}/bin/python"
    log "venv: activated and PULUMI_PYTHON_CMD=${PULUMI_PYTHON_CMD}"
  fi
}
find_pulumi_entrypoint() {
  local pd="$PROJECT_DIR"
  local pd_name
  pd_name="$(awk -F: '/^name[[:space:]]*:/ {gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2; exit}' "${PROJECT_DIR}/Pulumi.yaml" 2>/dev/null || true)"
  local candidates=("${pd}/__main__.py" "${pd}/${pd_name}/__main__.py" "${pd}/${pd_name}.py" "${pd}/main.py" "${pd}/__init__.py")
  for f in "${candidates[@]}"; do [ -f "$f" ] && { printf '%s' "$f"; return 0; }; done
  return 1
}
ensure_valid_entrypoint_exists() {
  if ep="$(find_pulumi_entrypoint)"; then
    if ! "$PYTHON_BIN" -m py_compile "$ep" >/dev/null 2>&1; then
      die "Pulumi entrypoint '$ep' exists but contains syntax errors. Fix and re-run."
    fi
    log "pulumi: entrypoint found and valid: $ep"
    return 0
  fi
  die "__main__.py or other Python Pulumi entrypoint missing in ${PROJECT_DIR}; add one and re-run."
}
get_pulumi_project_name() {
  local pd="${PROJECT_DIR}/Pulumi.yaml"
  [ -f "$pd" ] && awk -F: '/^name[[:space:]]*:/ {gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2; exit}' "$pd" || true
}
verify_stack_selected() {
  pulumi stack >/dev/null 2>&1 || return 1
  return 0
}
pulumi_select_or_init_stack() {
  local stack="$1"
  for attempt in 1 6; do
    export PULUMI_PYTHON_CMD="${VENV_DIR}/bin/python"
    if pulumi stack select "$stack" >/dev/null 2>&1; then
      log "pulumi: selected existing stack '$stack'"
      return 0
    fi
    sleep $((attempt))
  done
  PROJECT_NAME="$(get_pulumi_project_name || true)"
  candidates=("$stack")
  [ -n "${PROJECT_NAME:-}" ] && candidates+=("${PROJECT_NAME}/${stack}")
  [ -n "${PULUMI_ORG:-}" ] && [ -n "${PROJECT_NAME:-}" ] && candidates+=("${PULUMI_ORG}/${PROJECT_NAME}/${stack}")
  for c in "${candidates[@]}"; do
    [ -z "$c" ] && continue
    for attempt in 1 4; do
      log "pulumi: trying stack init '$c' (attempt $attempt)"
      set +e
      export PULUMI_PYTHON_CMD="${VENV_DIR}/bin/python"
      pulumi stack init "$c" >/dev/null 2>&1
      rc=$?
      set -e
      if [ $rc -eq 0 ]; then
        pulumi stack select "$c" >/dev/null 2>&1 || true
        verify_stack_selected || true
        log "pulumi: created and selected '$c'"
        return 0
      fi
      sleep $((attempt))
    done
  done
  log "pulumi: fallback -> attempting non-interactive 'pulumi new python --yes --force'"
  ensure_pulumi_cli
  set +e
  export PULUMI_PYTHON_CMD="${VENV_DIR}/bin/python"
  pulumi new python --yes --force >/dev/null 2>&1
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then die "unable to select or init pulumi stack '${stack}' and pulumi new failed"; fi
  pulumi stack init "$stack" >/dev/null 2>&1 || true
  pulumi stack select "$stack" >/dev/null 2>&1 || true
  verify_stack_selected || die "fallback created stack but verification failed"
  log "pulumi: fallback created and selected stack '$stack'"
  return 0
}
pulumi_preview_and_capture() {
  local logdir="${PROJECT_DIR}/.pulumi-logs"; mkdir -p "$logdir"
  local logf="${logdir}/pulumi-preview-$(date -u +%s).log"
  : >"$logf"
  export PULUMI_PYTHON_CMD="${VENV_DIR}/bin/python"
  if pulumi preview --diff --non-interactive >"$logf" 2>&1; then
    log "pulumi: preview succeeded (log: $logf)"; return 0
  else
    log "pulumi: preview failed; last 200 lines of $logf" >&2
    tail -n 200 "$logf" >&2 || true
    return 2
  fi
}
pulumi_up_and_capture() {
  local logdir="${PROJECT_DIR}/.pulumi-logs"; mkdir -p "$logdir"
  local logf="${logdir}/pulumi-up-$(date -u +%s).log"
  : >"$logf"
  export PULUMI_PYTHON_CMD="${VENV_DIR}/bin/python"
  if pulumi up --yes >"$logf" 2>&1; then
    log "pulumi: up succeeded (log: $logf)"; return 0
  else
    log "pulumi: up failed; last 200 lines of $logf" >&2
    tail -n 200 "$logf" >&2 || true
    return 3
  fi
}
write_stack_outputs() {
  local out_json="${PROJECT_DIR}/pulumi-outputs.json"
  local out_sh="${PROJECT_DIR}/pulumi-exports.sh"
  mkdir -p "${PROJECT_DIR}/.pulumi-logs" || true
  export PULUMI_PYTHON_CMD="${VENV_DIR}/bin/python"
  set +e
  pulumi stack output --json >"${out_json}.tmp" 2>/dev/null
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    log "pulumi: could not get stack outputs (rc=${rc}); writing empty outputs file"
    printf '{}' >"${out_json}.tmp" || true
  fi
  mv "${out_json}.tmp" "$out_json" || true
  if [ -s "$out_json" ] && command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    "$PYTHON_BIN" - "$out_json" "$out_sh" <<'PY'
import json,sys,os
json_fn = sys.argv[1]
out_fn = sys.argv[2]
try:
    with open(json_fn) as f:
        data = json.load(f)
except Exception:
    data = {}
tmp = out_fn + ".tmp"
with open(tmp, "w") as o:
    o.write("#!/usr/bin/env bash\n# pulumi exports generated\n")
    for k, v in data.items():
        key = "PULUMI_" + k.upper().replace("-", "_")
        if isinstance(v, str):
            val = v
        else:
            import json as _j
            val = _j.dumps(v)
        val = val.replace('"', '\\"')
        o.write(f'export {key}="{val}"\n')
os.replace(tmp, out_fn)
PY
  else
    printf '#!/usr/bin/env bash\n# pulumi exports placeholder\n' >"$out_sh" || true
  fi
  chmod +x "$out_sh" >/dev/null 2>&1 || true
  log "pulumi: outputs written to $out_json and $out_sh"
}
pulumi_login_and_run() {
  ensure_pulumi_cli
  export AWS_DYNAMODB_LOCK_TABLE="$DDB_TABLE"
  [ -n "$PULUMI_CONFIG_PASSPHRASE" ] && export PULUMI_CONFIG_PASSPHRASE
  export PULUMI_PYTHON_CMD="${VENV_DIR}/bin/python"
  log "pulumi: login ${PULUMI_LOGIN_URL} (PULUMI_PYTHON_CMD=${PULUMI_PYTHON_CMD})"
  pulumi login "${PULUMI_LOGIN_URL}" >/dev/null 2>&1 || log "pulumi: login returned non-zero (continuing)"
  if [ ! -d "$PROJECT_DIR" ]; then die "project dir $PROJECT_DIR not found" 13; fi
  pushd "$PROJECT_DIR" >/dev/null || exit 1
  ensure_valid_entrypoint_exists
  activate_venv_if_exists
  pulumi_select_or_init_stack "$STACK"
  pulumi config set aws:region "$AWS_REGION" >/dev/null 2>&1 || true
  for e in $(env | awk -F= '/^PULUMI_CONFIG_/{print $1}'); do
    val="$(printenv "$e")"; key="${e#PULUMI_CONFIG_}"; key_lc="$(echo "$key" | tr '[:upper:]' '[:lower:]')"
    pulumi config set "$key_lc" "$val" >/dev/null 2>&1 || true
  done
  local up_rc=0
  if [ "$PREVIEW" = true ]; then
    pulumi_preview_and_capture || up_rc=$?
    write_stack_outputs
    popd >/dev/null || true
    if [ $up_rc -ne 0 ]; then die "pulumi preview failed (see logs)"; else return 0; fi
  fi
  if [ "$PREVIEW_AND_UP" = true ]; then
    pulumi_preview_and_capture || { log "pulumi: preview failed; aborting up"; up_rc=$?; }
    if [ $up_rc -ne 0 ]; then
      write_stack_outputs
      popd >/dev/null || true
      die "pulumi preview failed; aborting up"
    fi
  fi
  if [ "$PREVIEW" != true ]; then
    pulumi_up_and_capture || up_rc=$?
    write_stack_outputs
    popd >/dev/null || true
    if [ $up_rc -ne 0 ]; then
      die "pulumi up failed; inspect logs in ${PROJECT_DIR}/.pulumi-logs" 1
    fi
    return 0
  fi
}
pulumi_destroy_stack_if_exists_noninteractive() {
  ensure_pulumi_cli
  if [ ! -d "$PROJECT_DIR" ]; then log "pulumi: project dir ${PROJECT_DIR} not found; skipping destroy"; return; fi
  pushd "$PROJECT_DIR" >/dev/null || return
  activate_venv_if_exists
  export PULUMI_PYTHON_CMD="${VENV_DIR}/bin/python"
  if pulumi stack select "$STACK" >/dev/null 2>&1; then
    pulumi destroy --yes >/dev/null 2>&1 || true
    pulumi stack rm --yes >/dev/null 2>&1 || true
    log "pulumi: stack destroyed/removed"
  else
    PROJECT_NAME="$(get_pulumi_project_name || true)"
    if [ -n "${PROJECT_NAME:-}" ]; then
      for candidate in "${PROJECT_NAME}/${STACK}" "${PULUMI_ORG:-}/${PROJECT_NAME}/${STACK}"; do
        if pulumi stack select "$candidate" >/dev/null 2>&1; then
          pulumi destroy --yes >/dev/null 2>&1 || true
          pulumi stack rm --yes >/dev/null 2>&1 || true
          log "pulumi: stack ${candidate} destroyed/removed"
        fi
      done
    fi
    log "pulumi: stack ${STACK} not present; skipping"
  fi
  popd >/dev/null || true
}
cleanup_local_outputs() {
  local out_json="${PROJECT_DIR}/pulumi-outputs.json"
  local out_sh="${PROJECT_DIR}/pulumi-exports.sh"
  local pulumi_dir="${PROJECT_DIR}/.pulumi"
  log "cleanup-local: removing $out_json , $out_sh , and $pulumi_dir (if present)"
  rm -f "$out_json" "$out_sh" || true
  rm -rf "$pulumi_dir" || true
}
log "Using project dir: ${PROJECT_DIR}"
log "Using Pulumi backend bucket: ${PULUMI_S3_BUCKET}"
log "Using data S3 bucket: ${S3_BUCKET}"
log "Using python interpreter: ${PYTHON_BIN}"
if [ "$MODE" = "create" ]; then
  log "=== CREATE MODE ==="
  create_bucket_if_missing "$PULUMI_S3_BUCKET"
  if [ -n "$S3_BUCKET" ] && [ "$S3_BUCKET" != "$PULUMI_S3_BUCKET" ]; then
    create_bucket_if_missing "$S3_BUCKET"
  fi
  create_dynamodb_if_missing "$DDB_TABLE"
  POLICY_ARN="$(ensure_policy "$PULUMI_S3_BUCKET" "$DDB_TABLE" "$POLICY_NAME" || true)"
  create_iam_user_if_requested "${PULUMI_IAM_USER:-}" "$POLICY_ARN" "$PULUMI_CREDS_FILE"
  log "waiting briefly for IAM propagation..."
  sleep 3
  create_venv_and_install
  if [ ! -f "${PROJECT_DIR}/Pulumi.yaml" ]; then
    cat >"${PROJECT_DIR}/Pulumi.yaml" <<YAML
name: ${STACK}-project
runtime: python
description: Minimal project created by pulumi_setup.sh
YAML
    log "pulumi-project: wrote ${PROJECT_DIR}/Pulumi.yaml"
  else
    log "pulumi-project: Pulumi.yaml exists; leaving"
  fi
  if [ ! -f "$REQ_FILE" ]; then
    cat >"$REQ_FILE" <<'REQ'
pulumi==3.209.0
pulumi-aws==7.12.0
boto3
REQ
    log "pulumi-project: wrote $REQ_FILE"
  else
    log "pulumi-project: requirements.txt exists; leaving"
  fi
  pulumi_login_and_run
  log "ensure: $out_json and $out_exports present"
  log "CREATE complete"
  exit 0
fi
if [ "$MODE" = "delete" ]; then
  log "=== DELETE MODE ==="
  if [ "$FORCE_FLAG" = true ] || [ "$FORCE_DELETE" = "true" ]; then
    log "[delete] FORCE mode enabled; pulumi prefixes will be removed for all stacks (bucket preserved)"
  fi
  pulumi_destroy_stack_if_exists_noninteractive
  delete_s3_objects "$PULUMI_S3_BUCKET" "${S3_PREFIX}${STACK}"
  delete_s3_objects "$PULUMI_S3_BUCKET" "$S3_PREFIX"
  delete_dynamodb_table_if_exists "$DDB_TABLE"
  delete_policy_and_user_idempotent "$POLICY_NAME" "${PULUMI_IAM_USER:-}"
  cleanup_local_outputs
  log "DELETE complete (bucket preserved)"
  exit 0
fi
exit 0
