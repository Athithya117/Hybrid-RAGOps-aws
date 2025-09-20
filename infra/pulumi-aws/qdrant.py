# qdrant.py — Pulumi module: single ENI + Launch Template + ASG constrained to one subnet/AZ for deterministic ray and qdrant connection
from __future__ import annotations
import base64
import json
import os
from typing import List, Optional
import pulumi
import pulumi_aws as aws
from pulumi import Config, ResourceOptions

cfg = Config()

# Optional networking stack reference (same pattern you already use)
networking_stack = cfg.get("networking_stack") or os.environ.get("NETWORKING_STACK")

# Resolve a single subnet (we will constrain ASG to this subnet/AZ)
subnet_id: Optional[str] = None
subnet_az: Optional[str] = None
qdrant_sg_id: Optional[str] = None
ec2_key_name: Optional[str] = None

if networking_stack:
    ref = pulumi.StackReference(networking_stack)
    subnet_id = ref.get_output("public_subnet_id_0")
    subnet_az = ref.get_output("public_subnet_az_0")
    qdrant_sg_id = ref.get_output("qdrant_security_group_id")
    ec2_key_name = ref.get_output("ec2_key_name")
else:
    subnet_id = cfg.get("subnet_id") or os.environ.get("PULUMI_PUBLIC_SUBNET_ID_0")
    subnet_az = cfg.get("subnet_az") or os.environ.get("PULUMI_PUBLIC_SUBNET_AZ_0")
    qdrant_sg_id = cfg.get("qdrant_sg_id") or os.environ.get("PULUMI_QDRANT_SECURITY_GROUP_ID") or os.environ.get("QDRANT_SG_ID")
    ec2_key_name = cfg.get("ec2_key_name") or os.environ.get("PULUMI_EC2_KEY_NAME") or os.environ.get("EC2_KEY_NAME")

if not subnet_id:
    raise RuntimeError("Missing subnet_id / PULUMI_PUBLIC_SUBNET_ID_0 (or provide networking_stack)")

# AMI / bucket / instance config (use env or Pulumi config)
ami_id = cfg.get("ami_id") or os.environ.get("BASE_AMI_ID")
if not ami_id:
    raise RuntimeError("BASE_AMI_ID (Ubuntu AMI) not set for this region")

snapshot_s3_bucket = cfg.get("snapshot_s3_bucket") or os.environ.get("S3_BUCKET")
snapshot_s3_prefix = cfg.get("snapshot_s3_prefix") or os.environ.get("SNAPSHOT_S3_PREFIX") or "qdrant/backups/"

instance_type = cfg.get("instance_type") or os.environ.get("QDRANT_INSTANCE_TYPE") or "c8g.4xlarge"
launch_template_name = cfg.get("launch_template_name") or os.environ.get("LAUNCH_TEMPLATE_NAME") or "qdrant-launch-template"
instance_role_name = cfg.get("instance_role_name") or os.environ.get("INSTANCE_ROLE_NAME") or "qdrant-ec2-role"

qdrant_ebs_type = cfg.get("qdrant_ebs_type") or os.environ.get("QDRANT_EBS_TYPE") or "gp3"
qdrant_ebs_size = int(cfg.get("qdrant_ebs_size") or os.environ.get("QDRANT_EBS_SIZE") or "8")

_dot = cfg.get("delete_on_termination")
if _dot is None:
    delete_on_termination = os.environ.get("QDRANT_EBS_DELETE_ON_TERM", "true").lower() == "true"
else:
    delete_on_termination = str(_dot).lower() == "true"

if not snapshot_s3_bucket:
    raise RuntimeError("snapshot_s3_bucket / S3_BUCKET is required")

# IAM role & instance profile (same pattern)
assume_role_policy = aws.iam.get_policy_document(
    statements=[
        {
            "Effect": "Allow",
            "Principals": [{"Type": "Service", "Identifiers": ["ec2.amazonaws.com"]}],
            "Actions": ["sts:AssumeRole"],
        }
    ]
).json

role = aws.iam.Role(instance_role_name, assume_role_policy=assume_role_policy)

policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowS3SnapshotRead",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:ListBucket"],
            "Resource": [
                f"arn:aws:s3:::{snapshot_s3_bucket}",
                f"arn:aws:s3:::{snapshot_s3_bucket}/{snapshot_s3_prefix}*",
                f"arn:aws:s3:::{snapshot_s3_bucket}/latest_qdrant_backup.manifest.json",
            ],
        }
    ],
}
aws.iam.RolePolicy(f"{instance_role_name}-s3-read", role=role.id, policy=json.dumps(policy))
instance_profile = aws.iam.InstanceProfile(f"{instance_role_name}-profile", role=role.name)

# user_data (use your existing script; placeholders preserved)
_raw_user_data = """#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="/workspace/qdrant/data"
BACKUP_DIR="/workspace/qdrant/backups"
SNAPSHOT_DIR="/workspace/qdrant/backups/snapshots"
MANIFEST_S3="s3://__SNAPSHOT_S3_BUCKET__/latest_qdrant_backup.manifest.json"
export QDRANT__STORAGE__SNAPSHOTS_PATH="${SNAPSHOT_DIR}"
mkdir -p "${DATA_DIR}" "${BACKUP_DIR}" "${SNAPSHOT_DIR}"
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -y
  apt-get install -y --no-install-recommends jq awscli docker.io xfsprogs zstd python3-pip || true
elif command -v yum >/dev/null 2>&1; then
  yum install -y jq awscli docker xfsprogs zstd python3-pip || true
fi
python3 -m pip install --no-cache-dir boto3 || true
systemctl enable --now docker || true
# NVMe mount logic (unchanged)...
if ! mountpoint -q "${DATA_DIR}"; then
  DEV="$(lsblk -dn -o NAME,TYPE | awk '/nvme[0-9]+n1/ && $2=="disk"{print "/dev/"$1; exit}')"
  if [ -n "${DEV}" ]; then
    if ! blkid "${DEV}" >/dev/null 2>&1; then
      mkfs.xfs -f -L qdrant-data "${DEV}"
    fi
    UUID="$(blkid -s UUID -o value "${DEV}")"
    if ! grep -q "UUID=${UUID}" /etc/fstab; then
      printf "UUID=%s %s xfs defaults,noatime,nodiratime 0 2\n" "${UUID}" "${DATA_DIR}" >> /etc/fstab
    fi
    mkdir -p "${DATA_DIR}"
    mountpoint -q "${DATA_DIR}" || mount "${DATA_DIR}"
    chown -R 1000:1000 "${DATA_DIR}"
  fi
fi
# Restore logic (unchanged)...
if [ -f "${DATA_DIR}/.qdrant_initialized" ]; then
  echo "Qdrant data already initialized; skipping restore."
else
  TMP_MANIFEST="/tmp/latest_qdrant_backup.manifest.json"
  TGZ_S3URI=""
  if aws s3 cp "${MANIFEST_S3}" "${TMP_MANIFEST}" 2>/dev/null; then
    TGZ_S3URI="$(jq -r '.latest_snapshot_to_path // empty' "${TMP_MANIFEST}" 2>/dev/null || true)"
    if [ -z "${TGZ_S3URI}" ]; then
      KEYNAME="$(jq -r '.latest_snapshot // empty' "${TMP_MANIFEST}" 2>/dev/null || true)"
      if [ -n "${KEYNAME}" ]; then
        TGZ_S3URI="s3://__SNAPSHOT_S3_BUCKET__/__SNAPSHOT_S3_PREFIX__${KEYNAME}"
      fi
    fi
  fi
  if [ -n "${TGZ_S3URI}" ]; then
    if [[ "${TGZ_S3URI}" =~ ^s3://([^/]+)/(.*)$ ]]; then
      BUCKET="${BASH_REMATCH[1]}"
      KEY="${BASH_REMATCH[2]}"
      LOCAL="/tmp/qdrant_snapshot.tar.zst"
      set +e
      aws s3 cp "s3://${BUCKET}/${KEY}" "${LOCAL}"
      rc=$?
      set -e
      if [ $rc -eq 0 ] && [ -f "${LOCAL}" ]; then
        mkdir -p "${SNAPSHOT_DIR}"
        tar -I 'zstd -d -T0' -xvf "${LOCAL}" -C "${SNAPSHOT_DIR}" || true
        rm -f "${LOCAL}"
        chown -R 1000:1000 "${SNAPSHOT_DIR}" "${DATA_DIR}" || true
      fi
    fi
  fi
fi
IMAGE_TAG="__IMAGE_TAG__"
docker pull "${IMAGE_TAG}" || true
DIGEST="$(docker inspect --format='{{index .RepoDigests 0}}' "${IMAGE_TAG}" 2>/dev/null || true)"
if [ -n "${DIGEST}" ]; then
  IMAGE="${DIGEST}"
else
  IMAGE="${IMAGE_TAG}"
fi
# Start qdrant (listens on 0.0.0.0). ENI must be attached prior to docker start.
docker run -d --name qdrant --restart unless-stopped -p 6333:6333 -v "${DATA_DIR}":/qdrant/storage -e QDRANT__HTTP__API_KEY="${QDRANT_API_KEY:-}" "${IMAGE}" || true
touch "${DATA_DIR}/.qdrant_initialized" || true
chown 1000:1000 "${DATA_DIR}/.qdrant_initialized" || true
echo "User-data finished."
"""

def make_user_data(bucket: str, prefix: str, image_tag: str = "qdrant/qdrant:v1.15.4") -> str:
    ud = _raw_user_data
    ud = ud.replace("__SNAPSHOT_S3_BUCKET__", bucket)
    ud = ud.replace("__SNAPSHOT_S3_PREFIX__", prefix)
    ud = ud.replace("__IMAGE_TAG__", image_tag)
    return ud

user_data_output = pulumi.Output.all(snapshot_s3_bucket, snapshot_s3_prefix).apply(
    lambda args: make_user_data(args[0], args[1])
)
user_data_b64 = user_data_output.apply(lambda s: base64.b64encode(s.encode()).decode())

# block device mapping
block_device_mappings = [
    aws.ec2.LaunchTemplateBlockDeviceMappingArgs(
        device_name="/dev/xvda",
        ebs=aws.ec2.LaunchTemplateBlockDeviceMappingEbsArgs(
            volume_size=int(qdrant_ebs_size),
            volume_type=str(qdrant_ebs_type),
            delete_on_termination=bool(delete_on_termination),
        ),
    )
]

# Determine deterministic IP for the single ENI
# Provide via env/config: QDRANT_PRIVATE_IP (recommended).
deterministic_ip = cfg.get("qdrant_private_ip") or os.environ.get("QDRANT_PRIVATE_IP")
if not deterministic_ip:
    # default convenience but you should set QDRANT_PRIVATE_IP explicitly for production
    deterministic_ip = "10.0.1.10"
    pulumi.log.warn("QDRANT_PRIVATE_IP not supplied; defaulting to 10.0.1.10 (override via env or pulumi config)")

# Create single ENI in chosen subnet with deterministic private IP
eni = aws.ec2.NetworkInterface(
    "qdrant-eni",
    subnet_id=subnet_id,
    private_ip=deterministic_ip,
    description="deterministic eni for qdrant (single-AZ)",
    security_groups=[qdrant_sg_id] if qdrant_sg_id else None,
    tags={"Name": "qdrant-eni"},
)

# Create Launch Template that attaches the ENI (device_index=0)
lt = aws.ec2.LaunchTemplate(
    launch_template_name,
    name=launch_template_name,
    image_id=ami_id,
    instance_type=instance_type,
    key_name=ec2_key_name,
    iam_instance_profile=aws.ec2.LaunchTemplateIamInstanceProfileArgs(name=instance_profile.name),
    user_data=user_data_b64,
    network_interfaces=[
        aws.ec2.LaunchTemplateNetworkInterfacesArgs(
            device_index=0,
            network_interface_id=eni.id,
        )
    ],
    block_device_mappings=block_device_mappings,
    tag_specifications=[
        aws.ec2.LaunchTemplateTagSpecificationArgs(resource_type="instance", tags={"Name": "qdrant-node"})
    ],
)

# Create a single-ASG constrained to the same subnet/AZ (desired_capacity=1)
asg = aws.autoscaling.Group(
    "qdrant-asg-single-az",
    desired_capacity=1,
    min_size=1,
    max_size=1,
    launch_template=aws.autoscaling.GroupLaunchTemplateArgs(id=lt.id, version="$Latest"),
    vpc_zone_identifiers=[subnet_id],  # constrain to the single subnet/AZ
    health_check_type="EC2",
    health_check_grace_period=900,
    termination_policies=["Default"],
    tags=[aws.autoscaling.GroupTagArgs(key="Name", value="qdrant-asg", propagate_at_launch=True)],
    opts=ResourceOptions(depends_on=[instance_profile]),
)

# Exports for pulumi-exports.sh and Ray consumers
pulumi.export("qdrant_eni_id", eni.id)
pulumi.export("qdrant_eni_private_ip", eni.private_ip)
pulumi.export("qdrant_launch_template_id", lt.id)
pulumi.export("qdrant_asg_name", asg.name)
pulumi.export("qdrant_url", pulumi.Output.concat("http://", eni.private_ip, ":6333"))
pulumi.export("qdrant_private_ip", eni.private_ip)

# helpful: also export the subnet used (so operator knows AZ)
pulumi.export("qdrant_subnet_id", subnet_id)
pulumi.export("qdrant_subnet_az", subnet_az)
