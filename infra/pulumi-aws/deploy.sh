#!/usr/bin/env bash
set -euo pipefail

# Usage: ./deploy.sh up
# Commands: up | destroy | preview
CMD="${1:-up}"
STACK="${STACK:-dev}"
S3_BUCKET="${S3_BUCKET:-}"     # required: bucket name for pulumi state
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$S3_BUCKET" ]; then
  echo "ERROR: S3_BUCKET must be set. e.g. export S3_BUCKET=my-pulumi-bucket"
  exit 2
fi

# ensure jq is present (used to export pulumi outputs)
if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required to parse pulumi outputs. Install jq and retry."
  exit 2
fi

# create venv and install deps (idempotent)
VENV_DIR="${PROJECT_DIR}/.venv"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
echo "Installing pulumi and dependencies into venv: $VENV_DIR"
pip install --upgrade pip >/dev/null 2>&1 || true
pip install -r "${PROJECT_DIR}/requirements.txt" >/dev/null 2>&1 || true

cd "$PROJECT_DIR"

# Ensure S3 backend is accessible. Optionally create the bucket if it doesn't exist.
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

# Pulumi login to S3 backend
pulumi login "s3://$S3_BUCKET/pulumi/"

# create/select stack non-interactively
if pulumi stack select "$STACK" >/dev/null 2>&1; then
  pulumi stack select "$STACK"
else
  pulumi stack select --create "$STACK"
fi

# Set required AWS region config for stack (non-interactive)
pulumi config set aws:region "$AWS_REGION" --stack "$STACK"

# Allow overriding some project-specific environment configs by exporting env vars
# (these are read by networking.py via os.environ; set before calling script)
# e.g. export MY_SSH_CIDR="203.0.113.42/32" ; export PUBLIC_SUBNET_CIDRS="10.0.1.0/24,10.0.2.0/24"

# Run Pulumi command
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

# --- Only for "up": capture outputs into JSON and emit shell-friendly exports ---
if [ "$CMD" = "up" ]; then
  OUT_JSON="$(pulumi stack output --stack "$STACK" --json)"
  echo "$OUT_JSON" > "${PROJECT_DIR}/pulumi-outputs.json"

  # Convert outputs to exports with prefix PULUMI_ and uppercase keys.
  # If a value is an array, join it with commas, then shell-escape; otherwise shell-escape the value.
  # This avoids invalid constructs like: export FOO='a' 'b'

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
  echo "[INFO] Run 'eval \$(cat infra/pulumi-aws/pulumi-exports.sh)' to import these into your shell"
fi

