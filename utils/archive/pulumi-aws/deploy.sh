#!/usr/bin/env bash
set -euo pipefail
CMD="${1:-up}"
STACK="${STACK:-prod}"
PROJECT_DIR="infra/pulumi-aws"
if [ -z "$S3_BUCKET" ]; then
  echo "ERROR: S3_BUCKET must be set. e.g. export S3_BUCKET=my-pulumi-bucket" >&2
  exit 2
fi
mkdir -p "$PROJECT_DIR"
PROJECT_DIR_ABS="$(cd "$PROJECT_DIR" && pwd)"
OUTPUT_JSON="$PROJECT_DIR_ABS/pulumi-outputs.json"
OUTPUT_SH="$PROJECT_DIR_ABS/pulumi-exports.sh"
VENV_DIR="$PROJECT_DIR_ABS/.venv"
CPU_AMI_ARCH="${CPU_AMI_ARCH:-arm}"
case "$CPU_AMI_ARCH" in
  amd) ARCH="amd64" ;;
  arm) ARCH="arm64" ;;
  *) echo "ERROR: CPU_AMI_ARCH must be 'amd' or 'arm' (got '$CPU_AMI_ARCH')" >&2; exit 3 ;;
esac
REGION="${AWS_REGION}"
if [ -z "$REGION" ]; then
  REGION="$(aws configure get region || true)"
fi
if [ -z "$REGION" ]; then
  echo "ERROR: AWS region not set. Export AWS_REGION or configure aws CLI." >&2
  exit 2
fi
if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: aws CLI not found in PATH" >&2
  exit 5
fi
OWNER="099720109477"
PATTERNS=(
  "ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-${ARCH}-server-*"
  "ubuntu/images/hvm-ssd-gp3/ubuntu-jammy-22.04-${ARCH}-server-*"
  "ubuntu/images/hvm-ssd/ubuntu-jammy*${ARCH}*server*"
)

AMI=""
for p in "${PATTERNS[@]}"; do
  CANDIDATE="$(aws ec2 describe-images --region "$REGION" --owners "$OWNER" --filters "Name=name,Values=$p" --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text 2>/dev/null || true)"
  if [ -n "$CANDIDATE" ] && [ "$CANDIDATE" != "None" ] && [ "$CANDIDATE" != "null" ]; then
    AMI="$CANDIDATE"
    break
  fi
done

if [ -z "$AMI" ]; then
  echo "ERROR: No Canonical Ubuntu 22.04 AMI found in region $REGION for arch $ARCH" >&2
  exit 4
fi

export BASE_AMI_ID="$AMI"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip >/dev/null 2>&1 || true
if [ -f "${PROJECT_DIR_ABS}/requirements.txt" ]; then
  python -m pip install -r "${PROJECT_DIR_ABS}/requirements.txt" >/dev/null 2>&1 || true
else
  python -m pip install pulumi pulumi-aws pulumi-tls >/dev/null 2>&1 || true
fi
cd "$PROJECT_DIR_ABS" || exit 1
if ! aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null; then
  if [ "$AWS_REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$S3_BUCKET" >/dev/null 2>&1 || true
  else
    aws s3api create-bucket --bucket "$S3_BUCKET" --create-bucket-configuration LocationConstraint="$AWS_REGION" >/dev/null 2>&1 || true
  fi
fi
pulumi login "s3://$S3_BUCKET/pulumi/" >/dev/null 2>&1 || true
if pulumi stack select "$STACK" >/dev/null 2>&1; then
  pulumi stack select "$STACK"
else
  pulumi stack select --create "$STACK" >/dev/null 2>&1 || true
fi
pulumi config set aws:region "$AWS_REGION" --stack "$STACK" >/dev/null 2>&1 || true
pulumi refresh --stack "$STACK" --yes >/dev/null 2>&1 || true
if [ "$CMD" = "up" ]; then
  MULTI_AZ_RAW="${MULTI_AZ_WEAVIATE_DEPLOYMENT:-false}"
  MULTI_AZ="$(echo "$MULTI_AZ_RAW" | tr '[:upper:]' '[:lower:]')"
  if [ "$MULTI_AZ" = "true" ] || [ "$MULTI_AZ" = "1" ] || [ "$MULTI_AZ" = "yes" ]; then
    if [ -z "${WEAVIATE_PRIVATE_IPS:-}" ]; then
      echo "ERROR: MULTI_AZ_WEAVIATE_DEPLOYMENT is true but WEAVIATE_PRIVATE_IPS is not set" >&2
      exit 2
    fi
    if [ -n "${WEAVIATE_PRIVATE_IP:-}" ]; then
      echo "ERROR: MULTI_AZ=true but WEAVIATE_PRIVATE_IP (single IP) is set; remove it and use WEAVIATE_PRIVATE_IPS" >&2
      exit 2
    fi
    IFS=',' read -r -a IP_ARR <<< "${WEAVIATE_PRIVATE_IPS}"
    if [ -n "${PULUMI_PUBLIC_SUBNET_COUNT:-}" ]; then
      if [ "${#IP_ARR[@]}" -ne "$PULUMI_PUBLIC_SUBNET_COUNT" ]; then
        echo "ERROR: PULUMI_PUBLIC_SUBNET_COUNT=${PULUMI_PUBLIC_SUBNET_COUNT} but WEAVIATE_PRIVATE_IPS has ${#IP_ARR[@]} entries" >&2
        exit 2
      fi
    fi
    pulumi config set multi_az_weaviate_deployment true --stack "$STACK" >/dev/null 2>&1 || true
    pulumi config set weaviate_private_ips "${WEAVIATE_PRIVATE_IPS}" --stack "$STACK" >/dev/null 2>&1 || true
    pulumi config rm weaviate_private_ip --stack "$STACK" >/dev/null 2>&1 || true
  else
    if [ -n "${WEAVIATE_PRIVATE_IPS:-}" ]; then
      echo "ERROR: MULTI_AZ_WEAVIATE_DEPLOYMENT is false but WEAVIATE_PRIVATE_IPS is set; unset it or set MULTI_AZ=true" >&2
      exit 2
    fi
    if [ -n "${WEAVIATE_PRIVATE_IP:-}" ]; then
      pulumi config set multi_az_weaviate_deployment false --stack "$STACK" >/dev/null 2>&1 || true
      pulumi config set weaviate_private_ip "${WEAVIATE_PRIVATE_IP}" --stack "$STACK" >/dev/null 2>&1 || true
      pulumi config rm weaviate_private_ips --stack "$STACK" >/dev/null 2>&1 || true
    else
      pulumi config set multi_az_weaviate_deployment false --stack "$STACK" >/dev/null 2>&1 || true
      pulumi config rm weaviate_private_ips --stack "$STACK" >/dev/null 2>&1 || true
      pulumi config rm weaviate_private_ip --stack "$STACK" >/dev/null 2>&1 || true
    fi
  fi
  pulumi up --stack "$STACK" --yes
fi

elif [ "$CMD" = "destroy" ]; then
  pulumi destroy --stack "$STACK" --yes || true
  rm -f "$PROJECT_DIR_ABS/pulumi-aws-rag-prod-ec2-key.pem" || true
  rm -f "$OUTPUT_SH" || true
  if [ -f "$OUTPUT_JSON" ]; then
    rm -f "$OUTPUT_JSON" || true
  fi
  S_LINE='if [ -f "/workspace/infra/pulumi-aws/pulumi-exports.sh" ]; then source "/workspace/infra/pulumi-aws/pulumi-exports.sh"; fi'
  for f in ~/.bashrc ~/.profile; do
    if [ -f "$f" ]; then
      grep -F -v "$S_LINE" "$f" > "${f}.pulumi.tmp" || true
      mv "${f}.pulumi.tmp" "$f" || true
    fi
  done
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    deactivate || true
  fi
  exit 0
elif [ "$CMD" = "preview" ]; then
  pulumi preview --stack "$STACK"
else
  echo "Unknown command: $CMD" >&2
  exit 2
fi
TMP_OUT="$(mktemp)"
pulumi stack output --stack "$STACK" --json > "$TMP_OUT" 2>/dev/null || true
if [ -s "$TMP_OUT" ] && command -v jq >/dev/null 2>&1 && jq -e . "$TMP_OUT" >/dev/null 2>&1; then
  mv "$TMP_OUT" "$OUTPUT_JSON"
else
  printf '{ }\n' > "$OUTPUT_JSON"
  rm -f "$TMP_OUT" || true
fi
if [ "$CMD" = "up" ]; then
  if command -v jq >/dev/null 2>&1 && jq -e . "$OUTPUT_JSON" >/dev/null 2>&1; then
    {
      printf '%s\n' '#!/usr/bin/env bash'
      printf '%s\n' '# Generated pulumi exports'
      jq -r '
        to_entries
        | map({
            k: ("PULUMI_" + (.key | ascii_upcase | gsub("-"; "_"))),
            v: (.value | if type=="string" then . else tostring end)
          })
        | .[] | "export \(.k)=\"" + ( .v | gsub("\""; "\\\"") ) + "\""
      ' "$OUTPUT_JSON"
    } > "$OUTPUT_SH" || {
      printf '%s\n' '#!/usr/bin/env bash' > "$OUTPUT_SH"
    }
  else
    printf '%s\n' '#!/usr/bin/env bash' > "$OUTPUT_SH"
  fi
  if [ -f "$OUTPUT_SH" ]; then
    chmod +x "$OUTPUT_SH" || true
  fi
  printf '\n\n'
  printf '%s\n' "[INFO] Run the following command block to persist env variables (idempotent):"
  printf '\n'
  cat <<'BLOCK'
for f in ~/.bashrc ~/.profile; do
  grep -qxF 'if [ -f "/workspace/infra/pulumi-aws/pulumi-exports.sh" ]; then source "/workspace/infra/pulumi-aws/pulumi-exports.sh"; fi' "$f" || \
  echo 'if [ -f "/workspace/infra/pulumi-aws/pulumi-exports.sh" ]; then source "/workspace/infra/pulumi-aws/pulumi-exports.sh"; fi' >> "$f"
done && exec bash -l
BLOCK
  printf '\n\n'
fi
if [ -n "${VIRTUAL_ENV:-}" ]; then
  deactivate || true
fi
exit 0
