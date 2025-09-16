#!/usr/bin/env bash
set -euo pipefail
CMD="${1:-up}"
STACK="${STACK:-dev}"
S3_BUCKET="${S3_BUCKET:-}"
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "$S3_BUCKET" ]; then
  echo "ERROR: S3_BUCKET must be set. e.g. export S3_BUCKET=my-pulumi-bucket"
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required to parse pulumi outputs. Install jq and retry."
  exit 2
fi
VENV_DIR="${PROJECT_DIR}/.venv"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
echo "Installing pulumi and dependencies into venv: $VENV_DIR"
pip install --upgrade pip >/dev/null 2>&1 || true
if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
  pip install -r "${PROJECT_DIR}/requirements.txt" >/dev/null 2>&1 || true
fi
cd "$PROJECT_DIR"
if ! aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null; then
  echo "S3 bucket $S3_BUCKET does not exist or is not accessible."
  echo "Attempting to create bucket (requires create bucket IAM permission)..."
  if [ "$AWS_REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$S3_BUCKET"
  else
    aws s3api create-bucket --bucket "$S3_BUCKET" --create-bucket-configuration LocationConstraint="$AWS_REGION"
  fi
  echo "Created or attempted to create s3://$S3_BUCKET"
fi
pulumi login "s3://$S3_BUCKET/pulumi/"
if pulumi stack select "$STACK" >/dev/null 2>&1; then
  pulumi stack select "$STACK"
else
  pulumi stack select --create "$STACK"
fi
pulumi config set aws:region "$AWS_REGION" --stack "$STACK"
if [ "$CMD" = "up" ]; then
  pulumi up --yes --stack "$STACK"
elif [ "$CMD" = "destroy" ]; then
  pulumi destroy --yes --stack "$STACK"
elif [ "$CMD" = "preview" ]; then
  pulumi preview --stack "$STACK"
else
  echo "Unknown command: $CMD"
  exit 2
fi
if [ "$CMD" = "up" ]; then
  OUT_JSON="$(pulumi stack output --stack "$STACK" --json)"
  echo "$OUT_JSON" > "${PROJECT_DIR}/pulumi-outputs.json"
  jq -r '
    to_entries[]
    | "export PULUMI_" + (.key | ascii_upcase) + "="
      + (
          if (.value | type) == "array" then
            (.value | join(",")) | @sh
          else
            .value | @sh
          end
        )
  ' <<<"$OUT_JSON" > "${PROJECT_DIR}/pulumi-exports.sh"
  chmod +x "${PROJECT_DIR}/pulumi-exports.sh"
  cat "${PROJECT_DIR}/pulumi-exports.sh"
  echo ""
  echo "[INFO] Run 'eval \$(cat ${PROJECT_DIR}/pulumi-exports.sh)' to import these into your shell"
fi
