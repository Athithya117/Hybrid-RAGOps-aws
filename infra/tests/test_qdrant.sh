#!/usr/bin/env bash
# scripts/run_qdrant_e2e.sh
# Deterministic generator + manifests apply script for qdrant backup E2E testing.
# - Derives S3 endpoint from region if not provided.
# - Supports LocalStack (USE_LOCALSTACK=1).
# - Runs infra/generators/gen_qdrant.py to emit manifests.
# - Applies safe manifests only (skip values.yaml; apply CronJob only if schedule looks valid).
set -euo pipefail

# ------------------ Defaults (override via env) ------------------
USE_LOCALSTACK=${USE_LOCALSTACK:-1}              # 1 = use LocalStack, 0 = real AWS
ENV=${ENV:-STAGING}                              # STAGING (creates k8s secret from env) or PROD (expect IRSA)
MANIFESTS_DIR=${MANIFESTS_DIR:-infra/manifests/qdrant}
PYTHON=${PYTHON:-python3}
KUBECTL=${KUBECTL:-kubectl}
HELM=${HELM:-helm}
AWS=${AWS:-aws}
JQ=${JQ:-jq}
CURL=${CURL:-curl}

# Qdrant / backup defaults (change as required)
QDRANT_NAMESPACE=${QDRANT_NAMESPACE:-qdrant}
QDRANT_RELEASE=${QDRANT_RELEASE:-qdrant}
QDRANT_IMAGE_TAG=${QDRANT_IMAGE_TAG:-v1.16.1}
QDRANT_REPLICAS=${QDRANT_REPLICAS:-3}
BACKUP_S3_REGION=${BACKUP_S3_REGION:-us-east-1}
BACKUP_S3_BUCKET=${BACKUP_S3_BUCKET:-qdrant-test-bucket}
BACKUP_S3_PREFIX=${BACKUP_S3_PREFIX:-qdrant/backups}
BACKUP_COMPRESSION=${BACKUP_COMPRESSION:-zstd}
BACKUP_SCHEDULE=${BACKUP_SCHEDULE:-"0 */6 * * *"}   # generator may override; empty -> skip cronjob
BACKUP_IMAGE=${BACKUP_IMAGE:-athithya5354/qdrant-backup:v2}
IRSA_ROLE_ARN=${IRSA_ROLE_ARN:-""}
QDRANT__SERVICE__API_KEY=${QDRANT__SERVICE__API_KEY:-test-api-key}

# ------------------ Derived S3 endpoint ------------------
if [ -n "${BACKUP_S3_ENDPOINT:-}" ]; then
  S3_ENDPOINT="${BACKUP_S3_ENDPOINT}"
else
  if [ "${USE_LOCALSTACK:-0}" -eq 1 ]; then
    S3_ENDPOINT="http://localhost:4566"
  else
    # Build a region-based endpoint (simple canonical form)
    if [ "${BACKUP_S3_REGION}" = "us-east-1" ]; then
      S3_ENDPOINT="https://s3.amazonaws.com"
    else
      S3_ENDPOINT="https://s3.${BACKUP_S3_REGION}.amazonaws.com"
    fi
  fi
fi

echo "Running with:"
echo "  ENV=$ENV USE_LOCALSTACK=$USE_LOCALSTACK"
echo "  S3 endpoint = $S3_ENDPOINT"
echo "  S3 bucket   = $BACKUP_S3_BUCKET (prefix=$BACKUP_S3_PREFIX, region=$BACKUP_S3_REGION)"
echo "  manifests   = $MANIFESTS_DIR"
echo

# ------------------ Sanity checks ------------------
for tool in "$PYTHON" "$KUBECTL" "$AWS" "$CURL" "$JQ"; do
  if ! command -v "${tool%% *}" >/dev/null 2>&1; then
    echo "ERROR: required binary '${tool%% *}' not found on PATH" >&2
    exit 2
  fi
done

# If using LocalStack, make sure it's reachable
if [ "${USE_LOCALSTACK:-0}" -eq 1 ]; then
  echo "Checking LocalStack S3 endpoint $S3_ENDPOINT ..."
  if ! curl -sS --fail --max-time 3 "$S3_ENDPOINT" >/dev/null 2>&1; then
    echo "WARN: LocalStack endpoint $S3_ENDPOINT not reachable. If you want to use real AWS, set USE_LOCALSTACK=0 and provide AWS creds." >&2
  fi
fi

# ------------------ Export envs needed by generator ------------------
export ENV MANIFESTS_DIR QDRANT_NAMESPACE QDRANT_RELEASE QDRANT_IMAGE_TAG QDRANT_REPLICAS
export BACKUP_S3_BUCKET BACKUP_S3_PREFIX BACKUP_S3_REGION BACKUP_S3_ENDPOINT="$S3_ENDPOINT" BACKUP_COMPRESSION BACKUP_SCHEDULE BACKUP_IMAGE IRSA_ROLE_ARN QDRANT__SERVICE__API_KEY

# If using localstack, also export AWS test creds if not present (safe local-only defaults)
if [ "${USE_LOCALSTACK:-0}" -eq 1 ]; then
  export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-test}
  export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-test}
  export AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN:-}
fi

# ------------------ Ensure S3 bucket exists ------------------
echo "Ensuring S3 bucket exists: $BACKUP_S3_BUCKET ..."
AWS_OPTS=()
if [ -n "$S3_ENDPOINT" ]; then
  # only pass endpoint-url flag when using LocalStack (or custom endpoint)
  AWS_OPTS+=(--endpoint-url "$S3_ENDPOINT")
fi

# head-bucket may fail on some S3 endpoints; try create-if-missing
set +e
if "${AWS}" "${AWS_OPTS[@]}" s3api head-bucket --bucket "$BACKUP_S3_BUCKET" >/dev/null 2>&1; then
  echo "Bucket already exists: $BACKUP_S3_BUCKET"
else
  echo "Creating bucket: $BACKUP_S3_BUCKET"
  if "${AWS}" "${AWS_OPTS[@]}" s3api create-bucket --bucket "$BACKUP_S3_BUCKET" --region "$BACKUP_S3_REGION" >/dev/null 2>&1; then
    echo "Bucket created: $BACKUP_S3_BUCKET"
  else
    echo "Warning: could not create head-bucket (it may already exist or require special region/auth). Continuing..."
  fi
fi
set -e

# ------------------ Run generator (deterministic) ------------------
echo "Running generator: python3 infra/generators/gen_qdrant.py"
"$PYTHON" infra/generators/gen_qdrant.py
echo "Generator completed — manifests should be at: $MANIFESTS_DIR"
echo

# ------------------ Apply safe manifests ------------------
# We will apply:
#  - clusterrole*.yaml
#  - clusterrolebinding*.yaml
#  - secret*.yaml
#  - serviceaccount*.yaml
#  - backup-job.yaml (one-off job that runs snapshot immediately)
#  - backup-cronjob.yaml only if schedule looks valid (non-empty & 5 fields)
#  Skip values.yaml explicitly (Helm input).
echo "Applying safe manifests from $MANIFESTS_DIR (skipping values.yaml) ..."

apply_file() {
  local f="$1"
  echo "  -> kubectl apply -f $f"
  if ! $KUBECTL apply -f "$f"; then
    echo "  ERROR applying $f (continuing). See kubectl output above."
  fi
}

# helper to check cron schedule looks like 5 fields (very basic)
cron_is_valid() {
  local s="$1"
  # empty or all whitespace -> invalid
  if [ -z "${s// }" ]; then return 1; fi
  # count fields (allow >5 — advanced cron expressions may be supported but we keep simple)
  local count
  count=$(awk -v RS=' ' 'END{print NR}' <<<"$s" 2>/dev/null || echo 0)
  [ "$count" -ge 5 ] && return 0 || return 1
}

# apply RBAC + secret + SA
for pattern in clusterrole*.yaml clusterrolebinding*.yaml secret*.yaml serviceaccount*.yaml; do
  for f in "$MANIFESTS_DIR"/$pattern; do
    [ -f "$f" ] || continue
    if [[ "$(basename "$f")" == "secret-sample.yaml" ]] && [ "$ENV" = "STAGING" ]; then
      # secret-sample.yaml may be placeholder; apply real secret was already created by generator if AWS creds present
      echo "  (note) applying secret-sample.yaml to create sample secret"
    fi
    apply_file "$f"
  done
done

# Apply backup-job.yaml (if exists) - this is a synchronous one-shot job typically safe to apply.
if [ -f "$MANIFESTS_DIR/backup-job.yaml" ]; then
  echo "Applying backup-job.yaml (one-shot job)"
  apply_file "$MANIFESTS_DIR/backup-job.yaml"
fi

# Apply CronJob only when schedule looks valid
cron_schedule=""
# detect schedule if present inside manifest (simple extraction)
if [ -f "$MANIFESTS_DIR/backup-cronjob.yaml" ]; then
  cron_schedule=$(sed -n '1,120p' "$MANIFESTS_DIR/backup-cronjob.yaml" | sed -n 's/^[[:space:]]*schedule:[[:space:]]*"\?\(.*\)"\?/\1/p' | head -n1 || true)
fi
if [ -n "$cron_schedule" ] && cron_is_valid "$cron_schedule"; then
  echo "Applying backup-cronjob.yaml (schedule='$cron_schedule')"
  apply_file "$MANIFESTS_DIR/backup-cronjob.yaml"
else
  echo "Skipping CronJob: schedule not present or invalid ('$cron_schedule')"
fi

echo
echo "Apply completed. IMPORTANT: Do NOT kubectl apply values.yaml; use Helm for chart installs."
echo "To install/upgrade qdrant (helm) use your standard CI workflow / Helm + values.yaml (ArgoCD handles that in prod)."

# Print a short deterministic checklist for next steps
cat <<EOF

NEXT (manual/deterministic) steps to validate backups:

1) Ensure Qdrant statefulset/pods are running (expected replicas = $QDRANT_REPLICAS):
   $KUBECTL -n $QDRANT_NAMESPACE get sts,pods -l app.kubernetes.io/name=qdrant -o wide

2) Create a collection + insert points via port-forward:
   $KUBECTL -n $QDRANT_NAMESPACE port-forward svc/$QDRANT_RELEASE 6333:6333 >/dev/null 2>&1 &
   sleep 2
   curl -s -X PUT "http://localhost:6333/collections/test_col" -H "Content-Type: application/json" -H "api-key: $QDRANT__SERVICE__API_KEY" -d '{\"vectors\":{\"size\":16,\"distance\":\"Dot\"}}' | $JQ .
   curl -s -X PUT "http://localhost:6333/collections/test_col/points?wait=true" -H "Content-Type: application/json" -H "api-key: $QDRANT__SERVICE__API_KEY" -d '{\"points\":[{\"id\":1,\"vector\":['$(printf '0.01,%.0s' {1..15})'0.01]}]}' | $JQ .

3) Trigger per-pod snapshots (port-forward each pod's http port locally and POST /collections/test_col/snapshots?wait=true).
4) Poll the S3 bucket for objects:
   ${AWS} ${USE_LOCALSTACK:+--endpoint-url $S3_ENDPOINT} s3api list-objects-v2 --bucket $BACKUP_S3_BUCKET --prefix $BACKUP_S3_PREFIX --query 'Contents[].Key' --output json

EOF

# deterministic exit
exit 0
