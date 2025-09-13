#!/usr/bin/env bash
# packer-build.sh
set -euo pipefail

# Load .env
if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
else
  echo "Please create a .env file (copy .env.example -> .env and edit)."
  exit 1
fi

# Validate required vars
: "${APP1_IMAGE:?APP1_IMAGE must be set in .env (can be empty if not used)}"
: "${APP2_IMAGE:?APP2_IMAGE must be set in .env (can be empty if not used)}"
: "${AWS_REGION:?AWS_REGION must be set in .env}"
: "${AMI_NAME_PREFIX:=ray-custom-ami}"
: "${PACKER_INSTANCE_TYPE:=t3.micro}"

echo "→ Running packer build (region=$AWS_REGION, instance_type=$PACKER_INSTANCE_TYPE)"
packer init ray-custom-ami.pkr.hcl

packer build \
  -var "region=${AWS_REGION}" \
  -var "instance_type=${PACKER_INSTANCE_TYPE}" \
  -var "ami_name=${AMI_NAME_PREFIX}" \
  -var "source_ami_owner=${PACKER_SOURCE_AMI_OWNER:-099720109477}" \
  -var "source_ami_filter=${PACKER_SOURCE_AMI_FILTER:-ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*}" \
  -var "app1_image=${APP1_IMAGE}" \
  -var "app2_image=${APP2_IMAGE}" \
  ray-custom-ami.pkr.hcl
