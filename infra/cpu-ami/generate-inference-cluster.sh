#!/usr/bin/env bash
# generate-ray-config.sh
set -euo pipefail

# Load .env
if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
else
  echo "Please create a .env from .env.example and fill values."
  exit 1
fi


# Determine AMI: prefer CUSTOM_AMI_ID from .env; allow arg override
if [ "${1:-}" != "" ]; then
  CUSTOM_AMI="$1"
elif [ -n "${CUSTOM_AMI_ID:-}" ]; then
  CUSTOM_AMI="$CUSTOM_AMI_ID"
else
  echo "Usage: $0 <ami-id>  OR set CUSTOM_AMI_ID in .env"
  echo "You can run packer-build.sh and paste the AMI id into .env CUSTOM_AMI_ID."
  exit 1
fi

# Get Pulumi outputs (expects you are in Pulumi project directory and stack selected)
STACK_JSON=$(pulumi stack output --json)
if [ -z "$STACK_JSON" ]; then
  echo "pulumi stack output returned empty. Ensure Pulumi stack is selected and 'pulumi up' run."
  exit 1
fi

SECURITY_GROUP=$(echo "$STACK_JSON" | jq -r .security_group_id)
HEAD_SUBNET=$(echo "$STACK_JSON" | jq -r .public_subnet_ids[0])
WORKER_SUBNET=$(echo "$STACK_JSON" | jq -r .public_subnet_ids[1])

# Basic validation
if [ -z "$SECURITY_GROUP" ] || [ "$SECURITY_GROUP" = "null" ]; then
  echo "Security group not found in Pulumi outputs."
  exit 1
fi

# Default values fallback
RAY_CLUSTER_NAME=${RAY_CLUSTER_NAME:-ray-inference-cpu-cluster}
RAY_SSH_USER=${RAY_SSH_USER:-ubuntu}
RAY_SSH_PRIVATE_KEY=${RAY_SSH_PRIVATE_KEY:-$SSH_KEY_PATH}
RAY_HEAD_INSTANCE_TYPE=${RAY_HEAD_INSTANCE_TYPE:-t3.medium}
RAY_WORKER_INSTANCE_TYPE=${RAY_WORKER_INSTANCE_TYPE:-t3.large}
RAY_MAX_WORKERS=${RAY_MAX_WORKERS:-5}

cat > ray-inference-cpu-cluster.yaml <<EOF
cluster_name: ${RAY_CLUSTER_NAME}

provider:
  type: aws
  region: ${AWS_REGION}
  use_internal_ips: True

auth:
  ssh_user: ${RAY_SSH_USER}
  ssh_private_key: ${RAY_SSH_PRIVATE_KEY}

docker:
  container_name: ray_container
  pull_before_run: False

available_node_types:

  head_node:
    node_config:
      InstanceType: ${RAY_HEAD_INSTANCE_TYPE}
      SubnetId: ${HEAD_SUBNET}
      SecurityGroupIds: ["${SECURITY_GROUP}"]
      ImageId: "${CUSTOM_AMI}"
    resources: {"CPU": 2}
    min_workers: 0
    max_workers: 0
    docker:
      image: ray-base
      container_name: ray_head

  app1_worker:
    node_config:
      InstanceType: ${RAY_WORKER_INSTANCE_TYPE}
      SubnetId: ${WORKER_SUBNET}
      SecurityGroupIds: ["${SECURITY_GROUP}"]
      ImageId: "${CUSTOM_AMI}"
    resources: {"CPU": 2, "app1": 1}
    min_workers: 0
    max_workers: ${RAY_MAX_WORKERS}
    docker:
      image: app1
      container_name: ray_app1
      pull_before_run: False

  app2_worker:
    node_config:
      InstanceType: ${RAY_WORKER_INSTANCE_TYPE}
      SubnetId: ${WORKER_SUBNET}
      SecurityGroupIds: ["${SECURITY_GROUP}"]
      ImageId: "${CUSTOM_AMI}"
    resources: {"CPU": 2, "app2": 1}
    min_workers: 0
    max_workers: ${RAY_MAX_WORKERS}
    docker:
      image: app2
      container_name: ray_app2
      pull_before_run: False

head_node_type: head_node

setup_commands:
  - pip install -U ray

head_start_ray_commands:
  - ray stop
  - ray start --head --port=6379 --dashboard-host=0.0.0.0

worker_start_ray_commands:
  - ray stop
  - ray start --address=\$RAY_HEAD_IP:6379
EOF

echo " Generated ray-inference-cpu-cluster.yaml (AMI: $CUSTOM_AMI)"
