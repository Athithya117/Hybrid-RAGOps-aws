#!/usr/bin/env bash
# infra/pulumi_aws/force_wipe_pulumi_backend.sh
# Safe, idempotent force-wipe helper for Pulumi S3 backend, DynamoDB lock table, and an IAM policy.
# Defaults to DRY_RUN=1 (no destructive actions). Set DRY_RUN=0 to execute.
set -euo pipefail

# --------- Config (edit or override via environment) ----------
BUCKET="${BUCKET:-e2e-rag-system-42}"
PREFIX="${PREFIX:-pulumi/}"
DDB_TABLE="${DDB_TABLE:-pulumi-state-locks}"
IAM_POLICY_NAME="${IAM_POLICY_NAME:-PulumiStateAccessPolicy}"
PROJECT_DIR="${PROJECT_DIR:-infra/pulumi_aws}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/venv}"
AWS_REGION="${AWS_REGION:-ap-south-1}"

# Safety knobs
DRY_RUN="${DRY_RUN:-1}"                 # 1 => no destructive actions, 0 => execute
ALLOW_DELETE_BUCKET="${ALLOW_DELETE_BUCKET:-false}" # must be "true" to delete entire bucket
FORCE_REMOVE_LOCAL_VENV="${FORCE_REMOVE_LOCAL_VENV:-false}"

# --------- Helpers ----------
log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
err() { printf '[%s] ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || { err "required command '$1' not found"; exit 10; } }

# minimal requirements
require_cmd aws
require_cmd python3 || require_cmd python || { err "python3 (or python) required"; exit 11; }

# optional but helpful
if command -v jq >/dev/null 2>&1; then
  USE_JQ=1
else
  USE_JQ=0
fi

# cleanup temp files on exit
TMPS=()
_cleanup() { for f in "${TMPS[@]:-}"; do [ -f "$f" ] && rm -f "$f"; done; }
trap _cleanup EXIT

# --------- Utility: run if not dry run ----------
do_or_dry() {
  if [ "$DRY_RUN" != "0" ]; then
    log "[DRY_RUN] $*"
  else
    log "EXEC: $*"
    eval "$@"
  fi
}

# --------- Attempt graceful pulumi destroy (best-effort) ----------
try_pulumi_destroy() {
  if command -v pulumi >/dev/null 2>&1 && [ -d "$PROJECT_DIR" ]; then
    log "Attempting graceful pulumi destroy/select (best-effort)"
    pushd "$PROJECT_DIR" >/dev/null || return
    set +e
    stacks_json="$(pulumi stack ls --json 2>/dev/null || true)"
    set -e
    if [ -n "$stacks_json" ]; then
      if [ "$USE_JQ" -eq 1 ]; then
        mapfile -t stacks < <(echo "$stacks_json" | jq -r '.[].name')
      else
        # python fallback
        mapfile -t stacks < <(python3 - <<PY
import sys, json
try:
  arr=json.load(sys.stdin)
  for it in arr:
    print(it.get("name",""))
except Exception:
  pass
PY
<<<"$stacks_json")
      fi
      for st in "${stacks[@]:-}"; do
        [ -z "$st" ] && continue
        log "Pulumi: attempting destroy for stack: $st"
        if [ "$DRY_RUN" = "0" ]; then
          pulumi stack select "$st" --non-interactive || true
          pulumi destroy --yes || true
          pulumi stack rm --yes || true
        fi
      done
    else
      log "No pulumi stacks discoverable in $PROJECT_DIR (or pulumi CLI not configured)."
    fi
    popd >/dev/null || true
  else
    log "pulumi CLI not present or project dir missing; skipping graceful pulumi destroy."
  fi
}

# --------- Delete S3 objects (all versions & delete markers) under a prefix ----------
delete_s3_prefix() {
  local bucket="$1" prefix="$2"
  log "Preparing to delete all object versions and delete markers under s3://${bucket}/${prefix}"
  if [ "$DRY_RUN" != "0" ]; then
    log "[DRY_RUN] would iterate and delete versions under ${bucket}/${prefix}"
    return 0
  fi

  # Loop until no Versions / DeleteMarkers are returned
  while :; do
    set +e
    resp="$(aws s3api list-object-versions --bucket "$bucket" --prefix "${prefix}" --output json 2>/dev/null || echo '{}')"
    set -e

    if [ "$USE_JQ" -eq 1 ]; then
      objs_json="$(echo "$resp" | jq -c '[.Versions[]?, .DeleteMarkers[]?] | map({Key:.Key, VersionId:.VersionId})')"
    else
      objs_json="$(python3 - <<PY
import sys,json
try:
  r=json.load(sys.stdin)
  arr=[]
  for k in ("Versions","DeleteMarkers"):
    for it in r.get(k,[]):
      arr.append({"Key": it.get("Key"), "VersionId": it.get("VersionId")})
  print(json.dumps(arr))
except Exception:
  print("[]")
PY
<<<"$resp")"
    fi

    # if empty, break
    if [ -z "$objs_json" ] || [ "$objs_json" = "[]" ] || [ "$objs_json" = "null" ]; then
      log "No object versions/delete markers found under ${bucket}/${prefix}."
      break
    fi

    # aws delete-objects supports up to 1000 entries per request. We'll send as single request since we listed a page.
    tmpf="$(mktemp)"
    TMPS+=("$tmpf")
    printf '{"Objects":%s}' "$objs_json" >"$tmpf"
    # Attempt delete
    aws s3api delete-objects --bucket "$bucket" --delete "file://$tmpf" >/dev/null 2>&1 || true
    rm -f "$tmpf" || true
    # repeat until no objects remain
    sleep 1
  done

  log "Deletion of objects under ${bucket}/${prefix} completed (or attempted)."
}

# --------- Delete entire bucket (opt-in) ----------
delete_entire_bucket() {
  local bucket="$1"
  if [ "$ALLOW_DELETE_BUCKET" != "true" ]; then
    log "ALLOW_DELETE_BUCKET not true; skipping deletion of bucket ${bucket}"
    return 0
  fi
  if [ "$DRY_RUN" != "0" ]; then
    log "[DRY_RUN] would empty and delete bucket ${bucket}"
    return 0
  fi

  log "Emptying entire bucket ${bucket} ..."
  delete_s3_prefix "$bucket" ""
  log "Deleting bucket ${bucket} ..."
  aws s3api delete-bucket --bucket "$bucket" --region "${AWS_REGION}" >/dev/null 2>&1 || true
  log "Bucket delete attempted for ${bucket}."
}

# --------- Delete DynamoDB table ----------
delete_dynamodb() {
  local table="$1"
  log "Deleting DynamoDB table ${table} (if present)"
  if [ "$DRY_RUN" != "0" ]; then
    log "[DRY_RUN] would delete dynamodb table ${table}"
    return 0
  fi
  if aws dynamodb describe-table --table-name "$table" >/dev/null 2>&1; then
    aws dynamodb delete-table --table-name "$table" --region "${AWS_REGION}" >/dev/null 2>&1 || true
    aws dynamodb wait table-not-exists --table-name "$table" --region "${AWS_REGION}" || true
    log "DynamoDB table ${table} deletion attempted."
  else
    log "DynamoDB table ${table} not found; skipping."
  fi
}

# --------- Delete local IAM policy (best-effort) ----------
delete_iam_policy() {
  local policy_name="$1"
  log "Attempting to delete IAM local policy ${policy_name} (best-effort)."
  if [ "$DRY_RUN" != "0" ]; then
    log "[DRY_RUN] would remove IAM policy ${policy_name} and detach references"
    return 0
  fi
  arn="$(aws iam list-policies --scope Local --query "Policies[?PolicyName=='${policy_name}'].Arn" --output text 2>/dev/null || true)"
  if [ -n "$arn" ]; then
    # detach from users and roles
    for u in $(aws iam list-entities-for-policy --policy-arn "$arn" --query 'PolicyUsers[].UserName' --output text 2>/dev/null || echo ""); do
      aws iam detach-user-policy --user-name "$u" --policy-arn "$arn" >/dev/null 2>&1 || true
    done
    for r in $(aws iam list-entities-for-policy --policy-arn "$arn" --query 'PolicyRoles[].RoleName' --output text 2>/dev/null || echo ""); do
      aws iam detach-role-policy --role-name "$r" --policy-arn "$arn" >/dev/null 2>&1 || true
    done
    # delete non-default versions
    for v in $(aws iam list-policy-versions --policy-arn "$arn" --query 'Versions[?IsDefaultVersion==`false`].VersionId' --output text 2>/dev/null || echo ""); do
      aws iam delete-policy-version --policy-arn "$arn" --version-id "$v" >/dev/null 2>&1 || true
    done
    aws iam delete-policy --policy-arn "$arn" >/dev/null 2>&1 || true
    log "IAM policy ${policy_name} deletion attempted."
  else
    log "IAM policy ${policy_name} not found; skipping."
  fi
}

# --------- Local cleanup ----------
cleanup_local() {
  log "Performing local cleanup of ${PROJECT_DIR}/.pulumi and outputs."
  if [ "$DRY_RUN" != "0" ]; then
    log "[DRY_RUN] would rm -rf ${PROJECT_DIR}/.pulumi ${PROJECT_DIR}/pulumi-outputs.json ${PROJECT_DIR}/pulumi-exports.sh"
    if [ "$FORCE_REMOVE_LOCAL_VENV" = "true" ]; then
      log "[DRY_RUN] would rm -rf ${VENV_DIR}"
    fi
    return 0
  fi
  rm -rf "${PROJECT_DIR}/.pulumi" "${PROJECT_DIR}/pulumi-outputs.json" "${PROJECT_DIR}/pulumi-exports.sh" || true
  if [ "$FORCE_REMOVE_LOCAL_VENV" = "true" ]; then
    rm -rf "${VENV_DIR}" || true
  fi
  log "Local cleanup done."
}

# --------- Main flow ----------
log "Starting force-wipe (DRY_RUN=${DRY_RUN}, ALLOW_DELETE_BUCKET=${ALLOW_DELETE_BUCKET})"

# 1) Graceful pulumi destroy
try_pulumi_destroy

# 2) Delete Pulumi prefix objects
delete_s3_prefix "$BUCKET" "$PREFIX"

# 3) Optionally delete entire bucket (opt-in)
if [ "$ALLOW_DELETE_BUCKET" = "true" ]; then
  delete_entire_bucket "$BUCKET"
fi

# 4) Delete DynamoDB table
delete_dynamodb "$DDB_TABLE"

# 5) Delete IAM policy
delete_iam_policy "$IAM_POLICY_NAME"

# 6) Local cleanup
cleanup_local

log "Force-wipe finished. If DRY_RUN=1, re-run with DRY_RUN=0 to execute (and set ALLOW_DELETE_BUCKET=true to remove entire bucket)."
