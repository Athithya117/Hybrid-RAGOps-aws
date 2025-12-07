#!/usr/bin/env bash
# infra/pulumi_aws/pulumi_setup.sh
# Safe, idempotent Pulumi helper suitable for repeated --delete / --create cycles.
# Usage:
#   bash infra/pulumi_aws/pulumi_setup.sh --create
#   bash infra/pulumi_aws/pulumi_setup.sh --delete --force
#   bash infra/pulumi_aws/pulumi_setup.sh --delete --force --delete-bucket
set -euo pipefail

# Prevent sourcing
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  echo "ERROR: do not source this file. Run it: bash $0" >&2
  return 1 2>/dev/null || exit 1
fi

export PROJECT_DIR="${PROJECT_DIR:-infra/pulumi_aws}"
export VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/venv}"
export REQ_FILE="${REQ_FILE:-${PROJECT_DIR}/requirements.txt}"
export AWS_REGION="${AWS_REGION:-ap-south-1}"

export PULUMI_S3_BUCKET="${PULUMI_S3_BUCKET:-e2e-rag-42}"
export S3_BUCKET="${S3_BUCKET:-${PULUMI_S3_BUCKET}}"
export S3_PREFIX="${S3_PREFIX:-pulumi/}"
export PULUMI_STATE_BUCKET="${PULUMI_STATE_BUCKET:-${PULUMI_S3_BUCKET}}"
export PULUMI_STATE_PREFIX="${PULUMI_STATE_PREFIX:-${S3_PREFIX}}"
export DDB_TABLE="${DDB_TABLE:-pulumi-state-locks}"

export PULUMI_STACK="${PULUMI_STACK:-prod}"
export STACK="${STACK:-${PULUMI_STACK}}"
export PULUMI_CONFIG_PASSPHRASE="${PULUMI_CONFIG_PASSPHRASE:-password}"
export PULUMI_ORG="${PULUMI_ORG:-}"
export PULUMI_BINARY_PATH="${PULUMI_BINARY_PATH:-}"
export PULUMI_CREDS_FILE="${PULUMI_CREDS_FILE:-/tmp/pulumi-ci-credentials.json}"
export PULUMI_AUTOINIT="${PULUMI_AUTOINIT:-true}"
export POLICY_NAME="${POLICY_NAME:-PulumiStateAccessPolicy}"

export FORCE_DELETE="${FORCE_DELETE:-true}"
export FORCE_DELETE_BUCKET="${FORCE_DELETE_BUCKET:-false}"

export ENABLE_PULUMI_AUTOINIT="${ENABLE_PULUMI_AUTOINIT:-true}"
export PIP_BREAK_SYSTEM_PACKAGES_FLAG="${PIP_BREAK_SYSTEM_PACKAGES_FLAG:---no-input}"
export PULUMI_LOGIN_URL="${PULUMI_LOGIN_URL:-s3://${S3_BUCKET}/${S3_PREFIX}}"
export PULUMI_PYTHON_CMD="${PULUMI_PYTHON_CMD:-${VENV_DIR}/bin/python}"

# ---- Minimal-Cost Defaults Below ----
export AVOID_DOMAIN="${AVOID_DOMAIN:-true}"
export MULTI_AZ_DEPLOYMENT="${MULTI_AZ_DEPLOYMENT:-false}"
export AZ_COUNT="${AZ_COUNT:-3}"
export VPC_CIDR="${VPC_CIDR:-10.0.0.0/16}"
export PUBLIC_SUBNET_CIDRS="${PUBLIC_SUBNET_CIDRS:-}"
export PRIVATE_SUBNET_CIDRS="${PRIVATE_SUBNET_CIDRS:-}"

# >>> COST MINIMAL DEFAULTS <<<
export NO_NAT="${NO_NAT:-true}"                         # ✔ No NAT Gateways (major cost saving)
export NAT_SINGLE="${NAT_SINGLE:-false}"                # irrelevant when NO_NAT=true
export CREATE_VPC_ENDPOINTS="${CREATE_VPC_ENDPOINTS:-true}"  
export CREATE_VPC_ENDPOINT_SERVICES="${CREATE_VPC_ENDPOINT_SERVICES:-s3,ecr.api,ecr.dkr,ssm,sts}"

export ENABLE_FLOW_LOGS="${ENABLE_FLOW_LOGS:-false}"    # ✔ avoid CloudWatch ingestion charges
export FLOW_LOG_DEST="${FLOW_LOG_DEST:-cloudwatch}"
export FLOW_LOG_S3_BUCKET="${FLOW_LOG_S3_BUCKET:-}"

export TAG_PREFIX="${TAG_PREFIX:-pulumi}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  for p in python3.12 python3.11 python3.10 python3; do
    if command -v "$p" >/dev/null 2>&1; then PYTHON_BIN="$p"; break; fi
  done
fi
if [ -z "$PYTHON_BIN" ]; then echo "ERROR: python3 not found" >&2; exit 11; fi

# -------------------------
# Normalize to absolute paths early
# -------------------------
PROJECT_DIR="$(cd "${PROJECT_DIR}" 2>/dev/null && pwd || echo "${PROJECT_DIR}")"
if [[ "${VENV_DIR}" != /* ]]; then VENV_DIR="$(pwd)/${VENV_DIR}"; fi
REQ_FILE="$(cd "$(dirname "${REQ_FILE}")" 2>/dev/null && printf '%s\n' "$(pwd)/$(basename "${REQ_FILE}")" || echo "${REQ_FILE}")"

# -------------------------
# Globals & helpers
# -------------------------
TMPS=()
cleanup_tmp() { for f in "${TMPS[@]:-}"; do [ -f "$f" ] && rm -f "$f"; done; }
trap cleanup_tmp EXIT

EXIT_CODE=0
on_exit() { echo "SCRIPT_EXIT=${EXIT_CODE}"; }
trap on_exit EXIT

retry() {
  local tries=${1:-5}; shift
  local delay=${1:-1}; shift || true
  local i=0 rc=0
  while [ $i -lt $tries ]; do
    set +e
    "$@"
    rc=$?
    set -e
    [ $rc -eq 0 ] && return 0
    i=$((i+1))
    sleep $delay
    delay=$((delay*2))
  done
  return $rc
}

require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "required command '$1' not found" >&2; EXIT_CODE=10; exit $EXIT_CODE; } }

require_cmd aws
require_cmd curl

if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "ERROR: AWS credentials not configured (aws sts get-caller-identity failed)" >&2
  EXIT_CODE=20; exit $EXIT_CODE
fi

# -------------------------
# S3 / DDB helpers
# -------------------------
create_bucket_if_missing() {
  local bucket="$1"
  echo "[`date -u +%Y-%m-%dT%H:%M:%SZ`] s3: ensure bucket exists: ${bucket} (region=${AWS_REGION})"
  if retry 6 1 aws s3api head-bucket --bucket "$bucket" >/dev/null 2>&1; then
    echo "s3: bucket exists"
  else
    if [ "$AWS_REGION" = "us-east-1" ]; then
      aws s3api create-bucket --bucket "$bucket" >/dev/null 2>&1 || true
    else
      aws s3api create-bucket --bucket "$bucket" --create-bucket-configuration LocationConstraint="$AWS_REGION" >/dev/null 2>&1 || true
    fi
    retry 8 2 aws s3api head-bucket --bucket "$bucket" >/dev/null 2>&1 || true
  fi
  aws s3api put-bucket-versioning --bucket "$bucket" --versioning-configuration Status=Enabled >/dev/null 2>&1 || true
  aws s3api put-bucket-encryption --bucket "$bucket" --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}' >/dev/null 2>&1 || true
  aws s3api put-bucket-lifecycle-configuration --bucket "$bucket" --lifecycle-configuration "{\"Rules\":[{\"ID\":\"pulumi-noncurrent-expire\",\"Prefix\":\"${S3_PREFIX}\",\"Status\":\"Enabled\",\"NoncurrentVersionExpiration\":{\"NoncurrentDays\":365}}]}" >/dev/null 2>&1 || true
  echo "s3: bucket prepared (prefix=${S3_PREFIX})"
}

delete_s3_objects() {
  local bucket="$1" prefix="${2:-}"
  if [ -z "$prefix" ]; then echo "[`date -u +%Y-%m-%dT%H:%M:%SZ`] s3-delete: deleting objects in s3://${bucket}/ (EMPTY PREFIX)"; else echo "[`date -u +%Y-%m-%dT%H:%M:%SZ`] s3-delete: deleting objects in s3://${bucket}/${prefix}"; fi
  while :; do
    local rv count objs tmp
    if [ -n "$prefix" ]; then
      rv="$(aws s3api list-object-versions --bucket "$bucket" --prefix "$prefix" --output json 2>/dev/null || echo '{}')"
    else
      rv="$(aws s3api list-object-versions --bucket "$bucket" --output json 2>/dev/null || echo '{}')"
    fi
    count="$("$PYTHON_BIN" - <<PY
import sys,json
try:
  r=json.load(sys.stdin)
  print(sum(len(r.get(k,[])) for k in ("Versions","DeleteMarkers")))
except Exception:
  print(0)
PY
    <<<"$rv")"
    if [ -z "$count" ] || [ "$count" = "0" ]; then break; fi
    objs="$("$PYTHON_BIN" - <<PY
import sys,json
r=json.load(sys.stdin)
arr=[]
for k in ("Versions","DeleteMarkers"):
  for it in r.get(k,[]):
    arr.append({"Key":it.get("Key"), "VersionId": it.get("VersionId")})
print(json.dumps(arr))
PY
    <<<"$rv")"
    tmp="$(mktemp)"; TMPS+=("$tmp")
    printf '{"Objects":%s}' "$objs" >"$tmp"
    set +e; aws s3api delete-objects --bucket "$bucket" --delete "file://$tmp" >/dev/null 2>&1 || true; set -e
    rm -f "$tmp" || true
    sleep 1
  done
  echo "s3-delete: done for s3://${bucket}/${prefix}"
}

empty_and_delete_bucket_force() {
  local bucket="$1"
  echo "[`date -u +%Y-%m-%dT%H:%M:%SZ`] s3-delete-all: force-empty & delete s3://${bucket}"
  aws s3api list-objects-v2 --bucket "$bucket" --max-items 50 --query 'Contents[].Key' --output text 2>/dev/null || true
  delete_s3_objects "$bucket" ""
  set +e; aws s3api delete-bucket --bucket "$bucket" --region "$AWS_REGION" >/dev/null 2>&1 || true; set -e
  echo "s3-delete-all: bucket delete attempted"
}

create_dynamodb_if_missing() {
  local table="$1"
  echo "ddb: ensure table ${table}"
  if aws dynamodb describe-table --table-name "$table" >/dev/null 2>&1; then echo "ddb: exists"
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
      echo "ddb: created and ACTIVE"
    else
      echo "ddb: create returned non-zero (continuing)"
    fi
  fi
  aws dynamodb update-time-to-live --table-name "$table" --time-to-live-specification "Enabled=true,AttributeName=Expires" >/dev/null 2>&1 || true
}

delete_dynamodb_table_if_exists() {
  local table="$1"
  if aws dynamodb describe-table --table-name "$table" >/dev/null 2>&1; then
    aws dynamodb delete-table --table-name "$table" --region "$AWS_REGION" >/dev/null 2>&1 || true
    aws dynamodb wait table-not-exists --table-name "$table" --region "$AWS_REGION" || true
    echo "ddb-delete: table deleted or attempted"
  else
    echo "ddb-delete: table not found; skipping"
  fi
}

get_account_id() { aws sts get-caller-identity --query Account --output text 2>/dev/null || true; }

wait_for_policy_arn() {
  local name="$1" tries=8 delay=1 arn=""
  for i in $(seq 1 $tries); do
    arn="$(aws iam list-policies --scope Local --query "Policies[?PolicyName=='${name}'].Arn" --output text 2>/dev/null || true)"
    [ -n "$arn" ] && { echo "$arn"; return 0; }
    sleep "$delay"; delay=$((delay*2))
  done
  return 1
}

ensure_policy() {
  local bucket="$1" table="$2" name="$3"
  echo "iam: ensure policy ${name}"
  local existing
  existing="$(aws iam list-policies --scope Local --query "Policies[?PolicyName=='${name}'].Arn" --output text || true)"
  if [ -n "$existing" ]; then
    echo "iam: policy exists $existing"
    echo "$existing"; return 0
  fi
  local acct; acct="$(get_account_id || true)"
  local tmp; tmp="$(mktemp)"; TMPS+=("$tmp")
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
  local arn; arn="$(wait_for_policy_arn "$name" || true)"
  echo "iam: policy ARN: ${arn:-not-found}"
  echo "$arn"
}

delete_policy_idempotent() {
  local policy_name="$1"
  local existing
  existing="$(aws iam list-policies --scope Local --query "Policies[?PolicyName=='${policy_name}'].Arn" --output text || true)"
  if [ -n "$existing" ]; then
    for r in $(aws iam list-entities-for-policy --policy-arn "$existing" --query 'PolicyRoles[].RoleName' --output text || true); do aws iam detach-role-policy --role-name "$r" --policy-arn "$existing" || true; done
    for v in $(aws iam list-policy-versions --policy-arn "$existing" --query 'Versions[?IsDefaultVersion==`false`].VersionId' --output text || true); do aws iam delete-policy-version --policy-arn "$existing" --version-id "$v" || true; done
    aws iam delete-policy --policy-arn "$existing" || true
    echo "iam-delete: policy delete attempted"
  else
    echo "iam-delete: policy not found; skipping"
  fi
}

# -------------------------
# Pulumi helpers
# -------------------------
ensure_pulumi_cli() {
  if command -v pulumi >/dev/null 2>&1; then return 0; fi
  if command -v curl >/dev/null 2>&1; then curl -fsSL https://get.pulumi.com | sh; export PATH="$HOME/.pulumi/bin:$PATH"; else echo "pulumi CLI not found and cannot auto-install (curl missing)" >&2; EXIT_CODE=11; exit $EXIT_CODE; fi
  for i in 1 3; do
    if pulumi version >/dev/null 2>&1; then echo "pulumi: $(pulumi version)"; return 0; fi
    sleep 1
  done
  echo "pulumi not responding after install" >&2; EXIT_CODE=11; exit $EXIT_CODE
}

create_venv_and_install() {
  mkdir -p "$(dirname "${VENV_DIR}")"
  if [ ! -d "$VENV_DIR" ]; then
    echo "venv: creating venv at $VENV_DIR using $PYTHON_BIN"
    "$PYTHON_BIN" -m venv "$VENV_DIR" >/dev/null 2>&1 || { echo "ERROR: creating venv failed" >&2; EXIT_CODE=12; exit $EXIT_CODE; }
  fi

  VENV_PY="${VENV_DIR}/bin/python"
  VENV_PIP="${VENV_DIR}/bin/pip"

  if [ ! -x "$VENV_PIP" ]; then
    "$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
  fi

  echo "venv: upgrading pip setuptools wheel in venv"
  set +e
  "$VENV_PY" -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then echo "pip upgrade in venv failed (rc=$rc)." >&2; EXIT_CODE=13; exit $EXIT_CODE; fi

  if [ -f "$REQ_FILE" ]; then
    echo "venv: installing packages from ${REQ_FILE}"
    set +e
    "$VENV_PY" -m pip install -r "$REQ_FILE" -q >/dev/null 2>&1
    rc=$?
    set -e
    if [ $rc -ne 0 ]; then echo "pip install -r ${REQ_FILE} failed (rc=$rc)." >&2; EXIT_CODE=14; exit $EXIT_CODE; fi
  else
    echo "venv: requirements.txt missing; installing pulumi basics"
    "$VENV_PY" -m pip install -q pulumi pulumi-aws boto3 awscli >/dev/null 2>&1 || true
  fi

  # activate venv
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  echo "venv: ready ($VENV_DIR) with python $("$VENV_DIR/bin/python" --version 2>/dev/null || true)"
}

activate_venv_if_exists() {
  if [ -d "$VENV_DIR" ]; then
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
    echo "venv: activated and PULUMI_PYTHON_CMD=${VENV_DIR}/bin/python"
  fi
}

# Find entrypoint for Pulumi program
find_pulumi_entrypoint() {
  local pd="${PROJECT_DIR}"
  local pd_name
  pd_name="$(awk -F: '/^name[[:space:]]*:/ {gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2; exit}' "${PROJECT_DIR}/Pulumi.yaml" 2>/dev/null || true)"
  local candidates=( "${pd}/__main__.py" "${pd}/${pd_name}/__main__.py" "${pd}/${pd_name}.py" "${pd}/main.py" "${pd}/__init__.py" )
  for f in "${candidates[@]}"; do
    [ -f "$f" ] && { printf '%s' "$f"; return 0; }
  done
  return 1
}

ensure_valid_entrypoint_exists() {
  local interpreter="${VENV_DIR}/bin/python"
  if [ ! -x "$interpreter" ]; then interpreter="${PYTHON_BIN}"; fi
  if ep="$(find_pulumi_entrypoint)"; then
    if ! "$interpreter" -m py_compile "$ep" >/dev/null 2>&1; then
      echo "Pulumi entrypoint '$ep' exists but fails compilation with $interpreter" >&2
      EXIT_CODE=15; exit $EXIT_CODE
    fi
    echo "pulumi: entrypoint found and valid: $ep"
    return 0
  fi
  echo "__main__.py or other Pulumi entrypoint missing in ${PROJECT_DIR}; add one and re-run." >&2
  EXIT_CODE=16; exit $EXIT_CODE
}

# Use stack init/select; do not call pulumi new in automation
pulumi_select_or_init_stack() {
  local stack="$1"
  ensure_pulumi_cli
  for attempt in 1 6; do
    if pulumi stack select "$stack" --cwd "$PROJECT_DIR" >/dev/null 2>&1; then
      echo "pulumi: selected existing stack '$stack'"; return 0
    fi
    sleep $((attempt))
  done

  echo "pulumi: attempting stack init '$stack' from existing project (non-interactive)"
  set +e
  pulumi stack init "$stack" --cwd "$PROJECT_DIR" >/tmp/pulumi_stack_init_out.txt 2>/tmp/pulumi_stack_init_err.txt
  rc=$?
  set -e
  if [ $rc -eq 0 ]; then
    pulumi stack select "$stack" --cwd "$PROJECT_DIR" >/dev/null 2>&1 || true
    echo "pulumi: created and selected '$stack'"
    return 0
  fi

  echo "pulumi stack init failed; stdout:"
  sed -n '1,200p' /tmp/pulumi_stack_init_out.txt || true
  echo "pulumi stack init stderr:"
  sed -n '1,400p' /tmp/pulumi_stack_init_err.txt || true
  return 1
}

pulumi_preview_and_capture() {
  local logdir="${PROJECT_DIR}/.pulumi-logs"; mkdir -p "$logdir"
  local logf="${logdir}/pulumi-preview-$(date -u +%s).log"
  : >"$logf"
  if pulumi preview --diff --non-interactive --cwd "$PROJECT_DIR" >"$logf" 2>&1; then
    echo "pulumi: preview succeeded (log: $logf)"; return 0
  else
    echo "pulumi: preview failed; tailing log" >&2
    tail -n 200 "$logf" >&2 || true
    return 2
  fi
}

pulumi_up_and_capture() {
  local logdir="${PROJECT_DIR}/.pulumi-logs"; mkdir -p "$logdir"
  local logf="${logdir}/pulumi-up-$(date -u +%s).log"
  : >"$logf"
  if pulumi up --yes --cwd "$PROJECT_DIR" >"$logf" 2>&1; then
    echo "pulumi: up succeeded (log: $logf)"; return 0
  else
    echo "pulumi: up failed; tailing log" >&2
    tail -n 200 "$logf" >&2 || true
    return 3
  fi
}

write_stack_outputs() {
  local out_json="${PROJECT_DIR}/pulumi-outputs.json"
  local out_sh="${PROJECT_DIR}/pulumi-exports.sh"
  mkdir -p "${PROJECT_DIR}/.pulumi-logs" || true
  set +e
  pulumi stack output --json --cwd "$PROJECT_DIR" >"${out_json}.tmp" 2>/dev/null
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then printf '{}' >"${out_json}.tmp" || true; fi
  mv "${out_json}.tmp" "$out_json" || true

  if [ -s "$out_json" ] && command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    "$PYTHON_BIN" - "$out_json" "$out_sh" <<'PY'
import json,sys,os
json_fn = sys.argv[1]
out_fn = sys.argv[2]
try:
    with open(json_fn) as f: data = json.load(f)
except Exception:
    data = {}
tmp = out_fn + ".tmp"
with open(tmp, "w") as o:
    o.write("#!/usr/bin/env bash\n# pulumi exports generated\n")
    for k,v in data.items():
        key = "PULUMI_" + k.upper().replace("-", "_")
        if isinstance(v, str): val = v
        else: import json as _j; val = _j.dumps(v)
        val = val.replace('"', '\\"')
        o.write(f'export {key}="{val}"\n')
os.replace(tmp, out_fn)
PY
  else
    printf '#!/usr/bin/env bash\n# pulumi exports placeholder\n' >"$out_sh" || true
  fi
  chmod +x "$out_sh" >/dev/null 2>&1 || true
}

backup_stack_if_exists() {
  local stack="$1"; local backup_dir="${PROJECT_DIR}/backups"; mkdir -p "$backup_dir"
  set +e
  pulumi stack ls --cwd "$PROJECT_DIR" --json >/dev/null 2>&1
  rc=$?
  set -e
  if [ $rc -eq 0 ]; then
    set +e
    pulumi stack select "$stack" --cwd "$PROJECT_DIR" >/dev/null 2>&1
    rc2=$?
    set -e
    if [ $rc2 -eq 0 ]; then
      local outfile="${backup_dir}/${stack}-export-$(date -u +%Y%m%dT%H%M%SZ).json"
      echo "Backing up current stack to ${outfile}"
      set +e
      pulumi stack export --cwd "$PROJECT_DIR" --stack "$stack" >"$outfile" 2>/dev/null || true
      set -e
    fi
  fi
}

# Login: logout first, ensure bucket exists then login; if login fails for missing bucket, return non-zero
pulumi_login_and_run() {
  ensure_pulumi_cli
  export AWS_DYNAMODB_LOCK_TABLE="$DDB_TABLE"
  [ -n "${PULUMI_CONFIG_PASSPHRASE:-}" ] && export PULUMI_CONFIG_PASSPHRASE

  echo "pulumi: logout (clear cached backend)"; set +e; pulumi logout >/dev/null 2>&1 || true; set -e

  echo "pulumi: login s3://${S3_BUCKET}/${S3_PREFIX}"
  set +e
  pulumi login "s3://${S3_BUCKET}/${S3_PREFIX}" >/tmp/pulumi_login_out.txt 2>/tmp/pulumi_login_err.txt
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo "pulumi login failed; stderr:"
    sed -n '1,200p' /tmp/pulumi_login_err.txt || true
    return $rc
  fi

  activate_venv_if_exists
  ensure_valid_entrypoint_exists
  pulumi_select_or_init_stack "$STACK" || return 1

  pulumi config set aws:region "$AWS_REGION" --cwd "$PROJECT_DIR" >/dev/null 2>&1 || true

  if [ "${PREVIEW:-false}" = "true" ]; then pulumi_preview_and_capture || return $?; write_stack_outputs; return 0; fi
  if [ "${PREVIEW_AND_UP:-false}" = "true" ]; then pulumi_preview_and_capture || { echo "pulumi preview failed; aborting up" >&2; return 2; }; fi

  pulumi_up_and_capture || { write_stack_outputs; return $?; }
  write_stack_outputs
  return 0
}

pulumi_destroy_stack_if_exists_noninteractive() {
  ensure_pulumi_cli
  activate_venv_if_exists

  # If backend missing, skip destroy gracefully
  set +e
  pulumi login "s3://${S3_BUCKET}/${S3_PREFIX}" >/tmp/pulumi_login_out.txt 2>/tmp/pulumi_login_err.txt
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo "pulumi login for destroy failed (backend may be absent); skipping destroy."
    sed -n '1,200p' /tmp/pulumi_login_err.txt || true
    return 0
  fi

  set +e
  pulumi stack select "$STACK" --cwd "$PROJECT_DIR" >/dev/null 2>&1
  rc=$?
  set -e
  if [ $rc -eq 0 ]; then
    backup_stack_if_exists "$STACK"
    pulumi destroy --yes --cwd "$PROJECT_DIR" >/dev/null 2>&1 || true
    pulumi stack rm --yes --cwd "$PROJECT_DIR" >/dev/null 2>&1 || true
    echo "pulumi: stack destroyed/removed"
    return 0
  fi

  PROJECT_NAME="$(awk -F: '/^name[[:space:]]*:/ {gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2; exit}' "${PROJECT_DIR}/Pulumi.yaml" 2>/dev/null || true)"
  if [ -n "${PROJECT_NAME:-}" ]; then
    for candidate in "${PROJECT_NAME}/${STACK}" "${PULUMI_ORG:-}/${PROJECT_NAME}/${STACK}"; do
      set +e
      pulumi stack select "$candidate" --cwd "$PROJECT_DIR" >/dev/null 2>&1
      rc2=$?
      set -e
      if [ $rc2 -eq 0 ]; then
        backup_stack_if_exists "$candidate"
        pulumi destroy --yes --cwd "$PROJECT_DIR" >/dev/null 2>&1 || true
        pulumi stack rm --yes --cwd "$PROJECT_DIR" >/dev/null 2>&1 || true
        echo "pulumi: stack ${candidate} destroyed/removed"
      fi
    done
  fi

  echo "pulumi: stack ${STACK} not present; skipping"
  return 0
}

cleanup_local_outputs() {
  local out_json="${PROJECT_DIR}/pulumi-outputs.json"
  local out_sh="${PROJECT_DIR}/pulumi-exports.sh"
  local pulumi_dir="${PROJECT_DIR}/.pulumi"
  echo "cleanup-local: removing $out_json , $out_sh , and $pulumi_dir (if present)"
  rm -f "$out_json" "$out_sh" || true
  rm -rf "$pulumi_dir" || true
  if [ "${FORCE_CLEAN_VENV:-false}" = "true" ]; then
    echo "cleanup-local: removing venv ${VENV_DIR}"
    rm -rf "${VENV_DIR}" || true
  else
    echo "cleanup-local: preserving venv ${VENV_DIR}"
  fi
}

# -------------------------
# CLI parsing
# -------------------------
prog="$(basename "$0")"
usage() {
  cat <<EOF
Usage: $prog [--create|--delete] [--force] [--delete-bucket] [--preview] [--preview-and-up] [-h|--help]
  --create            create backend + venv + pulumi up
  --delete            destroy stack and remove backend prefixes by default
  --force             force prefix deletion (no interactive prompt)
  --delete-bucket     ALSO remove entire S3 bucket (DANGEROUS) -- must be explicit
  --preview           run pulumi preview only (no up)
  --preview-and-up    run preview and then up if preview ok
EOF
}

MODE="" FORCE_FLAG=false PREVIEW=false PREVIEW_AND_UP=false DELETE_BUCKET_FLAG=false
while [ $# -gt 0 ]; do
  case "$1" in
    --create) MODE="create"; shift;;
    --delete) MODE="delete"; shift;;
    --force) FORCE_FLAG=true; shift;;
    --delete-bucket) DELETE_BUCKET_FLAG=true; shift;;
    --preview) PREVIEW=true; shift;;
    --preview-and-up) PREVIEW_AND_UP=true; shift;;
    -h|--help) usage; EXIT_CODE=0; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; EXIT_CODE=2; exit $EXIT_CODE;;
  esac
done
[ -n "$MODE" ] || { echo "ERROR: must pass --create or --delete" >&2; usage; EXIT_CODE=2; exit $EXIT_CODE; }

echo "[`date -u +%Y-%m-%dT%H:%M:%SZ`] Using project dir: ${PROJECT_DIR}"
echo "[`date -u +%Y-%m-%dT%H:%M:%SZ`] Using S3 bucket: ${S3_BUCKET}"
echo "[`date -u +%Y-%m-%dT%H:%M:%SZ`] Using python interpreter: ${PYTHON_BIN}"

# -------------------------
# Main flows
# -------------------------
if [ "$MODE" = "create" ]; then
  echo "=== CREATE MODE ==="
  # ensure bucket + ddb + policy exist (idempotent)
  create_bucket_if_missing "$S3_BUCKET"
  create_dynamodb_if_missing "$DDB_TABLE"
  POLICY_ARN="$(ensure_policy "$S3_BUCKET" "$DDB_TABLE" "$POLICY_NAME" || true)"
  echo "waiting briefly for IAM propagation..."
  sleep 2

  create_venv_and_install

  if [ ! -f "${PROJECT_DIR}/Pulumi.yaml" ]; then
    cat >"${PROJECT_DIR}/Pulumi.yaml" <<YAML
name: pulumi_aws
runtime: python
description: Minimal Pulumi program
YAML
    echo "pulumi-project: wrote ${PROJECT_DIR}/Pulumi.yaml"
  else
    echo "pulumi-project: Pulumi.yaml exists; leaving"
  fi

  if [ ! -f "$REQ_FILE" ]; then
    cat >"$REQ_FILE" <<'REQ'
pulumi>=3.0.0,<4.0.0
REQ
    echo "pulumi-project: wrote $REQ_FILE"
  else
    echo "pulumi-project: requirements.txt exists; leaving"
  fi

  # Attempt login and run (if login fails due to bucket missing, create bucket then retry)
  if ! pulumi_login_and_run; then
    echo "pulumi: login/run failed; attempting to create bucket and retry"
    create_bucket_if_missing "$S3_BUCKET"
    if ! pulumi_login_and_run; then
      echo "pulumi: second login/run attempt failed; aborting"
      EXIT_CODE=17; exit $EXIT_CODE
    fi
  fi

  echo "CREATE complete"
  EXIT_CODE=0
  exit $EXIT_CODE
fi

if [ "$MODE" = "delete" ]; then
  echo "=== DELETE MODE ==="
  if [ "$FORCE_FLAG" = true ] || [ "${FORCE_DELETE:-false}" = "true" ]; then
    echo "[delete] FORCE mode enabled; prefixes will be removed"
  fi
  if [ "$DELETE_BUCKET_FLAG" = true ] || [ "${FORCE_DELETE_BUCKET:-false}" = "true" ]; then
    echo "[delete] WARNING: full S3 bucket deletion requested"; WANT_BUCKET_DELETE=true
  else
    WANT_BUCKET_DELETE=false
  fi

  pulumi_destroy_stack_if_exists_noninteractive

  if [ "$FORCE_FLAG" = true ] || [ "${FORCE_DELETE:-false}" = "true" ]; then
    delete_s3_objects "$S3_BUCKET" "${S3_PREFIX}${STACK}"
    delete_s3_objects "$S3_BUCKET" "$S3_PREFIX"
  else
    echo "info: not in FORCE mode; only removing stack prefix if present"
    delete_s3_objects "$S3_BUCKET" "${S3_PREFIX}${STACK}" || true
  fi

  delete_dynamodb_table_if_exists "$DDB_TABLE"
  delete_policy_idempotent "$POLICY_NAME"

  if [ "$WANT_BUCKET_DELETE" = true ]; then
    empty_and_delete_bucket_force "$S3_BUCKET"
  else
    echo "info: S3 bucket preserved; full deletion not performed"
  fi

  cleanup_local_outputs

  echo "DELETE complete"
  EXIT_CODE=0
  exit $EXIT_CODE
fi

echo "Unrecognized mode; aborting" >&2
EXIT_CODE=1
exit $EXIT_CODE
