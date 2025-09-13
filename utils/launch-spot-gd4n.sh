#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

AWS_REGION=${AWS_REGION:-ap-south-1}
ARCH=${ARCH:-x86_64}   # set to "arm64" if you want Graviton-based AMIs

echo "Searching for latest Deep Learning GPU AMI for Ubuntu 22.04 in region: $AWS_REGION (arch=$ARCH)"

# Primary: search public AMIs owned by Amazon with name matching Ubuntu 22.04 Deep Learning GPU
read -r AMI_ID AMI_NAME <<EOF
$(aws ec2 describe-images \
  --region "${AWS_REGION}" \
  --owners amazon \
  --filters \
    "Name=name,Values=*Deep Learning*GPU*Ubuntu*22.04*" \
    "Name=architecture,Values=${ARCH}" \
    "Name=state,Values=available" \
  --query 'Images | sort_by(@, &CreationDate)[-1].{ImageId:ImageId,Name:Name}' \
  --output text 2>/dev/null || true)
EOF

if [ -z "${AMI_ID:-}" ] || [ "${AMI_ID:-}" = "None" ]; then
  echo "EC2 image search did not return an AMI. Trying SSM Parameter Store fallback..."
  # List parameters and pick the first that looks like a GPU ubuntu-22.04 DLAMI
  PARAM=$(aws ssm get-parameters-by-path \
    --region "${AWS_REGION}" \
    --path /aws/service/deeplearning/ami/ \
    --query "Parameters[?contains(Name, 'ubuntu-22.04') && contains(Name, 'gpu')].[Name]" \
    --output text 2>/dev/null | head -n1 || true)

  if [ -n "$PARAM" ]; then
    AMI_ID=$(aws ssm get-parameter --name "$PARAM" --region "${AWS_REGION}" --query "Parameter.Value" --output text)
    AMI_NAME="$PARAM (from SSM)"
  fi
fi

if [ -z "${AMI_ID:-}" ] || [ "${AMI_ID:-}" = "None" ]; then
  echo "ERROR: Could not locate a DLAMI (Deep Learning GPU AMI) for Ubuntu 22.04 automatically."
  echo "Options:"
  echo "  * Verify AWS CLI permissions (ec2:DescribeImages, ssm:GetParametersByPath, ssm:GetParameter)."
  echo "  * Pick an AMI manually in the console and set AMI_ID yourself."
  exit 2
fi

echo "Found AMI_ID: ${AMI_ID}"
echo "Found AMI_NAME: ${AMI_NAME:-(unknown)}"
# export for downstream usage
export AMI_ID
