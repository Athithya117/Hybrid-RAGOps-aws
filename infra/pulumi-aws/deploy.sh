#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   S3_BUCKET=your-bucket STACK=prod AWS_REGION=ap-south-1 bash infra/pulumi-aws/deploy.sh [up|destroy|preview]

CMD="${1:-up}"
STACK="${STACK:-dev}"
S3_BUCKET="${S3_BUCKET:-}"
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
PROJECT_DIR="infra/pulumi-aws"

if [ -z "$S3_BUCKET" ]; then
  echo "ERROR: S3_BUCKET must be set. e.g. export S3_BUCKET=my-pulumi-bucket"
  exit 2
fi

# Ensure project dir exists and resolve absolute path so we don't accidentally create nested paths.
mkdir -p "$PROJECT_DIR"
PROJECT_DIR_ABS="$(cd "$PROJECT_DIR" && pwd)"
OUTPUT_JSON="$PROJECT_DIR_ABS/pulumi-outputs.json"
OUTPUT_SH="$PROJECT_DIR_ABS/pulumi-exports.sh"
VENV_DIR="$PROJECT_DIR_ABS/.venv"

if ! command -v jq >/dev/null 2>&1; then
  echo "WARNING: jq not found. Script will still run but exports generation will be limited."
fi

# create venv if missing
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "Installing pulumi and dependencies into venv: $VENV_DIR"
python -m pip install --upgrade pip >/dev/null 2>&1 || true
if [ -f "${PROJECT_DIR_ABS}/requirements.txt" ]; then
  python -m pip install -r "${PROJECT_DIR_ABS}/requirements.txt" >/dev/null 2>&1 || true
else
  python -m pip install pulumi pulumi-aws pulumi-tls >/dev/null 2>&1 || true
fi

cd "$PROJECT_DIR_ABS"

# Ensure the S3 bucket exists (attempt to create if not)
if ! aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null; then
  echo "S3 bucket $S3_BUCKET does not exist or is not accessible. Attempting to create..."
  if [ "$AWS_REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$S3_BUCKET" >/dev/null 2>&1 || true
  else
    aws s3api create-bucket --bucket "$S3_BUCKET" --create-bucket-configuration LocationConstraint="$AWS_REGION" >/dev/null 2>&1 || true
  fi
  echo "Created or attempted to create s3://$S3_BUCKET"
fi

pulumi login "s3://$S3_BUCKET/pulumi/" >/dev/null 2>&1 || true

# Select or create stack (non-fatal on intermediate failures)
if pulumi stack select "$STACK" >/dev/null 2>&1; then
  pulumi stack select "$STACK"
else
  pulumi stack select --create "$STACK" >/dev/null 2>&1 || true
fi

pulumi config set aws:region "$AWS_REGION" --stack "$STACK" >/dev/null 2>&1 || true

# Try a refresh (non-fatal)
pulumi refresh --stack "$STACK" --yes >/dev/null 2>&1 || true

# Run requested command
if [ "$CMD" = "up" ]; then
  pulumi up --stack "$STACK" --yes
elif [ "$CMD" = "destroy" ]; then
  pulumi destroy --stack "$STACK" --yes || true

  # remove generated artifacts (no-fail)
  rm -f "${PROJECT_DIR_ABS}/pulumi-aws-rag-prod-ec2-key.pem" || true
  rm -f "$OUTPUT_SH" || true
  rm -f "$OUTPUT_JSON" || true

elif [ "$CMD" = "preview" ]; then
  pulumi preview --stack "$STACK"
else
  echo "Unknown command: $CMD"
  exit 2
fi

#
# Produce pulumi-outputs.json and pulumi-exports.sh (safe, non-failing)
#
TMP_OUT="$(mktemp)"
# Try to capture pulumi outputs; do not fail the script if pulumi fails here.
pulumi stack output --stack "$STACK" --json > "$TMP_OUT" 2>/dev/null || true

# If the tmp file contains valid JSON and is non-empty, move it into place; otherwise create a minimal {}
if [ -s "$TMP_OUT" ] && command -v jq >/dev/null 2>&1 && jq -e . "$TMP_OUT" >/dev/null 2>&1; then
  mv "$TMP_OUT" "$OUTPUT_JSON"
else
  # create a minimal empty JSON file
  printf '{ }\n' > "$OUTPUT_JSON"
  rm -f "$TMP_OUT" || true
fi

# Generate exports script if possible (fallback to stub)
if command -v jq >/dev/null 2>&1 && jq -e . "$OUTPUT_JSON" >/dev/null 2>&1; then
  {
    printf '#!/usr/bin/env bash\n'
    printf '# Generated pulumi exports (safe to source)\n'
    # Convert keys to uppercase and sanitize dashes to underscores.
    jq -r '
      to_entries
      | map({
          k: ("PULUMI_" + (.key | ascii_upcase | gsub("-"; "_"))),
          v: (.value | if type=="string" then . else tostring end)
        })
      | .[] | "export \(.k)=\"" + ( .v | gsub("\""; "\\\"") ) + "\""
    ' "$OUTPUT_JSON"
  } > "$OUTPUT_SH" || {
    # fallback stub if for some reason generation fails
    printf '#!/usr/bin/env bash\n# pulumi-exports generation failed; stub created\n' > "$OUTPUT_SH"
  }
else
  printf '#!/usr/bin/env bash\n# pulumi outputs unavailable (jq missing or invalid JSON). File created as stub.\n' > "$OUTPUT_SH"
fi

chmod +x "$OUTPUT_SH" || true
echo "Wrote: $OUTPUT_JSON"
echo "Wrote: $OUTPUT_SH"

# deactivate venv if active
if [ -n "${VIRTUAL_ENV:-}" ]; then
  deactivate || true
fi

# done
exit 0
