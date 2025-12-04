from __future__ import annotations
import base64
import json
import os
from typing import List, Optional
import pulumi
import pulumi.asset as asset
import pulumi_aws as aws
from pulumi import Config, Output, ResourceOptions
cfg = Config()
try:
    import networking
except Exception:
    networking = None
subnet_ids: Optional[List[pulumi.Input[str]]] = None
subnet_azs: Optional[List[pulumi.Input[str]]] = None
weaviate_sg_id: Optional[pulumi.Input[str]] = None
ec2_key_name: Optional[pulumi.Input[str]] = None
if networking is not None:
    try:
        if hasattr(networking, "public_subnets") and networking.public_subnets:
            subnet_ids = [s.id for s in networking.public_subnets]
            subnet_azs = [s.availability_zone for s in networking.public_subnets]
    except Exception:
        pass
    try:
        weaviate_sg_id = getattr(networking, "weaviate_sg").id if hasattr(networking, "weaviate_sg") else getattr(networking, "weaviate_security_group_id", None)
    except Exception:
        pass
    try:
        ec2_key_name = getattr(networking, "aws_key").key_name if hasattr(networking, "aws_key") else getattr(networking, "ec2_key_name", None)
    except Exception:
        pass
networking_stack = cfg.get("networking_stack") or os.environ.get("NETWORKING_STACK")
if (not subnet_ids) and networking_stack:
    try:
        ref = pulumi.StackReference(networking_stack)
        subnet_ids = subnet_ids or ref.get_output("public_subnet_ids")
        subnet_azs = subnet_azs or ref.get_output("public_subnet_azs")
        weaviate_sg_id = weaviate_sg_id or ref.get_output("weaviate_security_group_id")
        ec2_key_name = ec2_key_name or ref.get_output("ec2_key_name")
    except Exception:
        pulumi.log.warn("Failed to read networking_stack outputs; falling back to config/env")
if not subnet_ids:
    env_subnets = os.environ.get("PULUMI_PUBLIC_SUBNET_IDS") or cfg.get("subnet_ids")
    if env_subnets:
        subnet_ids = [s.strip() for s in (env_subnets.split(",") if isinstance(env_subnets, str) else env_subnets)]
    else:
        single = os.environ.get("PULUMI_PUBLIC_SUBNET_ID_0") or cfg.get("subnet_id")
        if single:
            subnet_ids = [single]
if not subnet_azs:
    env_azs = os.environ.get("PULUMI_PUBLIC_SUBNET_AZS") or cfg.get("subnet_azs")
    if env_azs:
        subnet_azs = [a.strip() for a in (env_azs.split(",") if isinstance(env_azs, str) else env_azs)]
    else:
        single_az = os.environ.get("PULUMI_PUBLIC_SUBNET_AZ_0") or cfg.get("subnet_az")
        if single_az:
            subnet_azs = [single_az]
if not weaviate_sg_id:
    weaviate_sg_id = cfg.get("weaviate_sg_id") or os.environ.get("PULUMI_WEAVIATE_SECURITY_GROUP_ID") or os.environ.get("WEAVIATE_SG_ID")
if not ec2_key_name:
    ec2_key_name = cfg.get("ec2_key_name") or os.environ.get("PULUMI_EC2_KEY_NAME") or os.environ.get("EC2_KEY_NAME")
if not subnet_ids or len(subnet_ids) == 0:
    raise RuntimeError("Missing subnet_ids")
ami_id = cfg.get("ami_id") or os.environ.get("BASE_AMI_ID")
if not ami_id:
    raise RuntimeError("BASE_AMI_ID must be provided")
snapshot_s3_bucket = cfg.get("snapshot_s3_bucket") or os.environ.get("S3_BUCKET")
snapshot_s3_prefix = cfg.get("snapshot_s3_prefix") or os.environ.get("SNAPSHOT_S3_PREFIX") or "weaviate/backups/"
instance_type = cfg.get("instance_type") or os.environ.get("WEAVIATE_INSTANCE_TYPE") or "c8g.4xlarge"
launch_template_name = cfg.get("launch_template_name") or os.environ.get("LAUNCH_TEMPLATE_NAME") or "weaviate-launch-template"
instance_role_name = cfg.get("instance_role_name") or os.environ.get("INSTANCE_ROLE_NAME") or "weaviate-ec2-role"
ebs_type = cfg.get("weaviate_ebs_type") or os.environ.get("WEAVIATE_EBS_TYPE") or "gp3"
ebs_size = int(cfg.get("weaviate_ebs_size") or os.environ.get("WEAVIATE_EBS_SIZE") or "8")
_dot = cfg.get("delete_on_termination")
delete_on_termination = (os.environ.get("WEAVIATE_EBS_DELETE_ON_TERM", "true").lower() == "true") if (_dot is None) else (str(_dot).lower() == "true")
if not snapshot_s3_bucket:
    raise RuntimeError("snapshot_s3_bucket / S3_BUCKET is required")
multi_az_flag = os.environ.get("MULTI_AZ_WEAVIATE_DEPLOYMENT", "false").lower() in ("1", "true", "yes")
assume_role_policy = aws.iam.get_policy_document(statements=[{"Effect": "Allow","Principals": [{"Type": "Service", "Identifiers": ["ec2.amazonaws.com"]}],"Actions": ["sts:AssumeRole"],}]).json
role = aws.iam.Role(instance_role_name, assume_role_policy=assume_role_policy)
s3_resources = [f"arn:aws:s3:::{snapshot_s3_bucket}", f"arn:aws:s3:::{snapshot_s3_bucket}/*", f"arn:aws:s3:::{snapshot_s3_bucket}/{snapshot_s3_prefix}*"]
policy = {
    "Version": "2012-10-17",
    "Statement": [
        {"Sid": "AllowS3SnapshotRead","Effect": "Allow","Action": ["s3:GetObject","s3:ListBucket"],"Resource": s3_resources},
        {"Sid": "AllowEIPActions","Effect": "Allow","Action": ["ec2:AssociateAddress","ec2:DescribeAddresses","ec2:AllocateAddress","ec2:CreateTags"],"Resource": ["*"]}
    ],
}
aws.iam.RolePolicy(f"{instance_role_name}-s3-read", role=role.id, policy=json.dumps(policy))
instance_profile = aws.iam.InstanceProfile(f"{instance_role_name}-profile", role=role.name)
def make_user_data(bucket: str, prefix: str, region: str, eip_tag: str) -> str:
    raw = r"""#!/bin/bash
set -euo pipefail
REGION="__REGION__"
EIP_TAG_KEY="Name"
EIP_TAG_VAL="__EIP_TAG__"
S3_BUCKET="__S3_BUCKET__"
S3_MANIFEST_KEY="latest_weaviate_backup.manifest.json"
BACKUP_S3_PREFIX="__SNAPSHOT_PREFIX__"
WEAVIATE_IMAGE="cr.weaviate.io/weaviate:latest"
WEAVIATE_PORT_HTTP=8080
WEAVIATE_PORT_GRPC=50051
HOST_DATA_DIR="/workspace/weaviate/data"
CONTAINER_PERSISTENCE_PATH="/var/lib/weaviate"
EXTRA_ENVS=()
if command -v apt-get >/dev/null 2>&1; then apt-get update -y; apt-get install -y unzip curl jq docker.io || true; elif command -v yum >/dev/null 2>&1; then yum makecache -y; yum install -y unzip curl jq docker || true; else echo "no pkg mgr"; fi
systemctl enable docker || true
systemctl start docker || true
if ! command -v aws >/dev/null 2>&1; then TMPDIR=$(mktemp -d); curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "${TMPDIR}/awscliv2.zip"; unzip -q "${TMPDIR}/awscliv2.zip" -d "${TMPDIR}"; "${TMPDIR}/aws/install" --update || "${TMPDIR}/aws/install"; rm -rf "${TMPDIR}"; fi
export AWS_DEFAULT_REGION="$REGION"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
DEV="$(lsblk -dn -o NAME,TYPE | awk '/nvme[0-9]+n1/ && $2==\"disk\"{print \"/dev/\"$1; exit}')"
if [ -n "$DEV" ]; then
  if ! blkid "$DEV" >/dev/null 2>&1; then
    mkfs.xfs -f -L weaviate-data "$DEV"
  fi
  UUID="$(blkid -s UUID -o value "$DEV")"
  if ! grep -q "UUID=${UUID}" /etc/fstab; then
    printf "UUID=%s %s xfs defaults,noatime,nodiratime 0 2\n" "${UUID}" "${HOST_DATA_DIR}" >> /etc/fstab
  fi
  mkdir -p "${HOST_DATA_DIR}"
  mountpoint -q "${HOST_DATA_DIR}" || mount "${HOST_DATA_DIR}"
  chown -R 1000:1000 "${HOST_DATA_DIR}"
else
  mkdir -p "${HOST_DATA_DIR}"
  chown -R 1000:1000 "${HOST_DATA_DIR}"
fi
ALLOCATION_ID=$(aws ec2 describe-addresses --filters "Name=tag:${EIP_TAG_KEY},Values=${EIP_TAG_VAL}" --query 'Addresses[0].AllocationId' --output text || echo "None")
if [ -z "$ALLOCATION_ID" ] || [ "$ALLOCATION_ID" = "None" ] || [ "$ALLOCATION_ID" = "null" ]; then
  ALLOCATION_ID=$(aws ec2 allocate-address --domain vpc --query 'AllocationId' --output text)
  [ -n "$ALLOCATION_ID" ] && aws ec2 create-tags --resources "$ALLOCATION_ID" --tags Key="${EIP_TAG_KEY}",Value="${EIP_TAG_VAL}" || true
fi
for i in 1 2 3 4 5; do if aws ec2 associate-address --instance-id "$INSTANCE_ID" --allocation-id "$ALLOCATION_ID" --allow-reassociation --region "$REGION"; then break; else sleep 3; fi; done
EIP_PUBLIC_IP=$(aws ec2 describe-addresses --allocation-ids "$ALLOCATION_ID" --query 'Addresses[0].PublicIp' --output text --region "$REGION" || echo "")
echo "${EIP_PUBLIC_IP:-}" >/etc/weaviate_eip || true
if docker ps -a --format '{{.Names}}' | grep -qw weaviate; then docker rm -f weaviate || true; fi
ENV_OPTS=(-e PERSISTENCE_DATA_PATH="$CONTAINER_PERSISTENCE_PATH" -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED="false" -e DEFAULT_VECTORIZER_MODULE="none")
for e in "${EXTRA_ENVS[@]}"; do ENV_OPTS+=("$e"); done
docker run -d --name weaviate --restart unless-stopped -p 0.0.0.0:${WEAVIATE_PORT_HTTP}:8080 -p 0.0.0.0:${WEAVIATE_PORT_GRPC}:50051 -v "${HOST_DATA_DIR}:${CONTAINER_PERSISTENCE_PATH}" "${ENV_OPTS[@]}" "$WEAVIATE_IMAGE"
for i in {1..40}; do if curl -fsS "http://127.0.0.1:${WEAVIATE_PORT_HTTP}/v1/meta" >/dev/null 2>&1; then break; else sleep 5; fi; done
BACKUP_ID=""
if [ -n "$S3_BUCKET" ]; then
  if aws s3api head-object --bucket "$S3_BUCKET" --key "$S3_MANIFEST_KEY" >/dev/null 2>&1; then
    aws s3 cp "s3://$S3_BUCKET/$S3_MANIFEST_KEY" /tmp/weaviate_manifest.json --region "$REGION" || true
    BACKUP_ID=$(jq -r '.backup_id // .id // .path // empty' /tmp/weaviate_manifest.json || true)
    if [ -z "$BACKUP_ID" ]; then PATHVAL=$(jq -r '.path // empty' /tmp/weaviate_manifest.json || true); [ -n "$PATHVAL" ] && BACKUP_ID=$(basename "$PATHVAL"); fi
  fi
  if [ -z "$BACKUP_ID" ]; then
    LATEST_KEY=$(aws s3api list-objects-v2 --bucket "$S3_BUCKET" --prefix "$BACKUP_S3_PREFIX" --query 'sort_by(Contents,&LastModified)[-1].Key' --output text --region "$REGION" || echo "None")
    if [ -n "$LATEST_KEY" ] && [ "$LATEST_KEY" != "None" ]; then BACKUP_ID=$(basename "$LATEST_KEY"); fi
  fi
fi
if [ -n "${BACKUP_ID:-}" ]; then
  for i in {1..20}; do if curl -fsS "http://127.0.0.1:${WEAVIATE_PORT_HTTP}/v1/meta" >/dev/null 2>&1; then break; else sleep 5; fi; done
  curl -fsS -X POST "http://127.0.0.1:${WEAVIATE_PORT_HTTP}/v1/backups/s3/${BACKUP_ID}/restore" -H "Content-Type: application/json" -d '{"wait_for_completion":true}' || true
fi
exit 0
"""
    raw = raw.replace("__REGION__", region).replace("__EIP_TAG__", eip_tag).replace("__S3_BUCKET__", bucket).replace("__SNAPSHOT_PREFIX__", prefix)
    return raw
def create_unit(idx: int, subnet_id: pulumi.Input[str], subnet_az: Optional[pulumi.Input[str]], eip_tag: str):
    eip = aws.ec2.Eip(f"weaviate-eip-{idx}", tags={"Name": eip_tag})
    lt_name = f"{launch_template_name}-{idx}"
    user_data_b64 = Output.all(snapshot_s3_bucket, snapshot_s3_prefix).apply(lambda args: base64.b64encode(make_user_data(args[0], args[1], aws.config.region or "us-east-1", eip_tag).encode()).decode())
    block_device_mappings = [aws.ec2.LaunchTemplateBlockDeviceMappingArgs(device_name="/dev/xvda", ebs=aws.ec2.LaunchTemplateBlockDeviceMappingEbsArgs(volume_size=int(ebs_size), volume_type=str(ebs_type), delete_on_termination=bool(delete_on_termination)))]
    lt = aws.ec2.LaunchTemplate(f"weaviate-launch-template-{idx}", name=lt_name, image_id=ami_id, instance_type=instance_type, key_name=ec2_key_name, iam_instance_profile=aws.ec2.LaunchTemplateIamInstanceProfileArgs(name=instance_profile.name), user_data=user_data_b64, block_device_mappings=block_device_mappings, tag_specifications=[aws.ec2.LaunchTemplateTagSpecificationArgs(resource_type="instance", tags={"Name": f"weaviate-node-{idx}"})], opts=ResourceOptions(depends_on=[eip, instance_profile]))
    vpc_zone_identifiers = Output.from_input([subnet_id]).apply(lambda s: list(s) if s else [])
    asg = aws.autoscaling.Group(f"weaviate-asg-{idx}", desired_capacity=1, min_size=1, max_size=1, launch_template=aws.autoscaling.GroupLaunchTemplateArgs(id=lt.id, version="$Latest"), vpc_zone_identifiers=vpc_zone_identifiers, health_check_type="EC2", health_check_grace_period=900, termination_policies=["Default"], tags=[aws.autoscaling.GroupTagArgs(key="Name", value=f"weaviate-asg-{idx}", propagate_at_launch=True)], opts=ResourceOptions(depends_on=[instance_profile]))
    pulumi.export(f"weaviate_eip_allocation_id_{idx}", eip.allocation_id)
    pulumi.export(f"weaviate_eip_public_ip_{idx}", eip.public_ip)
    pulumi.export(f"weaviate_launch_template_id_{idx}", lt.id)
    pulumi.export(f"weaviate_asg_name_{idx}", asg.name)
    pulumi.export(f"weaviate_public_endpoint_{idx}", pulumi.Output.concat("http://", eip.public_ip, ":8080"))
    return {"eip": eip, "lt": lt, "asg": asg}
if multi_az_flag:
    provided_ips = os.environ.get("WEAVIATE_PRIVATE_IPS")
    if provided_ips:
        ip_list = [p.strip() for p in provided_ips.split(",") if p.strip()]
        if len(ip_list) != len(subnet_ids):
            raise RuntimeError("When MULTI_AZ_WEAVIATE_DEPLOYMENT=true provide same number of WEAVIATE_PRIVATE_IPS as subnets")
    units = []
    for idx, sn in enumerate(subnet_ids):
        tag = f"weaviate-eip-{idx}"
        units.append(create_unit(idx, sn, (subnet_azs[idx] if subnet_azs and idx < len(subnet_azs) else None), tag))
    pulumi.export("weaviate_eip_allocation_ids", [u["eip"].allocation_id for u in units])
    pulumi.export("weaviate_eip_public_ips", [u["eip"].public_ip for u in units])
    pulumi.export("weaviate_public_endpoints", [pulumi.Output.concat("http://", u["eip"].public_ip, ":8080") for u in units])
else:
    tag = "weaviate-eip"
    unit = create_unit(0, subnet_ids[0], (subnet_azs[0] if subnet_azs else None), tag)
    pulumi.export("weaviate_eip_allocation_id", unit["eip"].allocation_id)
    pulumi.export("weaviate_eip_public_ip", unit["eip"].public_ip)
    pulumi.export("weaviate_launch_template_id", unit["lt"].id)
    pulumi.export("weaviate_asg_name", unit["asg"].name)
    pulumi.export("weaviate_public_endpoint", pulumi.Output.concat("http://", unit["eip"].public_ip, ":8080"))
pulumi.export("weaviate_multi_az_enabled", multi_az_flag)
