# infra/pulumi-aws/qdrant.py
from __future__ import annotations
import base64
import json
import os
from typing import Optional
import pulumi
import pulumi_aws as aws
from pulumi import Config, ResourceOptions

cfg = Config()

# Prefer importing in-process networking module if available.
try:
    import networking  # type: ignore
except Exception:
    networking = None  # type: ignore

subnet_id: Optional[pulumi.Input[str]] = None
subnet_az: Optional[pulumi.Input[str]] = None
qdrant_sg_id: Optional[pulumi.Input[str]] = None
ec2_key_name: Optional[pulumi.Input[str]] = None

# 1) Prefer the module-level list that networking.py defines (public_subnets)
if networking is not None:
    # networking.public_subnets is a Python list of Subnet resources in networking.py
    if hasattr(networking, "public_subnets"):
        try:
            # use the first subnet (index 0)
            subnet_id = networking.public_subnets[0].id
        except Exception:
            # leave subnet_id as-is if something odd happens
            subnet_id = subnet_id
    # try module-level security group / key names
    try:
        qdrant_sg_id = networking.qdrant_sg.id
    except Exception:
        if hasattr(networking, "qdrant_security_group_id"):
            qdrant_sg_id = getattr(networking, "qdrant_security_group_id")
    try:
        ec2_key_name = networking.aws_key.key_name
    except Exception:
        if hasattr(networking, "ec2_key_name"):
            ec2_key_name = getattr(networking, "ec2_key_name")
    # also try subnet az from module if present
    if hasattr(networking, "public_subnets"):
        try:
            subnet_az = networking.public_subnets[0].availability_zone
        except Exception:
            subnet_az = subnet_az

# 2) If not found, try a configured external networking_stack (StackReference)
networking_stack = cfg.get("networking_stack") or os.environ.get("NETWORKING_STACK")
if subnet_id is None and networking_stack:
    try:
        ref = pulumi.StackReference(networking_stack)
        subnet_id = subnet_id or ref.get_output("public_subnet_id_0")
        subnet_az = subnet_az or ref.get_output("public_subnet_az_0")
        qdrant_sg_id = qdrant_sg_id or ref.get_output("qdrant_security_group_id")
        ec2_key_name = ec2_key_name or ref.get_output("ec2_key_name")
    except Exception:
        pass

# 3) Fallback to env / pulumi config names
if subnet_id is None:
    subnet_id = cfg.get("subnet_id") or os.environ.get("PULUMI_PUBLIC_SUBNET_ID_0") or os.environ.get("PUBLIC_SUBNET_ID_0")
    subnet_az = subnet_az or cfg.get("subnet_az") or os.environ.get("PULUMI_PUBLIC_SUBNET_AZ_0")
    qdrant_sg_id = qdrant_sg_id or cfg.get("qdrant_sg_id") or os.environ.get("PULUMI_QDRANT_SECURITY_GROUP_ID")

# 4) LAST RESORT: reference the same stack outputs (works if networking exported them)
if subnet_id is None:
    try:
        my_ref = pulumi.StackReference(pulumi.get_stack())
        subnet_id = my_ref.get_output("public_subnet_id_0")
        subnet_az = subnet_az or my_ref.get_output("public_subnet_az_0")
        qdrant_sg_id = qdrant_sg_id or my_ref.get_output("qdrant_security_group_id")
    except Exception:
        subnet_id = None

if subnet_id is None:
    raise RuntimeError(
        "Missing subnet_id — provide networking stack or set PULUMI_PUBLIC_SUBNET_ID_0 or pulumi config 'subnet_id'.\n"
        "Ensure networking.py defines `public_subnets` or export the subnet id via stack outputs."
    )

# --- remaining configuration (unchanged / abbreviated) ---
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

assume_role_policy = aws.iam.get_policy_document(
    statements=[{"Effect": "Allow", "Principals": [{"Type": "Service", "Identifiers": ["ec2.amazonaws.com"]}], "Actions": ["sts:AssumeRole"]}]
).json

role = aws.iam.Role(instance_role_name, assume_role_policy=assume_role_policy)
policy = {
    "Version": "2012-10-17",
    "Statement": [
        {"Sid": "AllowS3SnapshotRead", "Effect": "Allow",
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

# user_data (keep your full script here; truncated in this snippet)
_raw_user_data = """#!/usr/bin/env bash
set -euo pipefail
# ... your original bootstrap/restore script ...
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

# deterministic IP (pulumi config or env)
deterministic_ip = cfg.get("qdrant_private_ip") or os.environ.get("QDRANT_PRIVATE_IP") or "10.0.1.10"

# LaunchTemplate: network interface requests subnet + private_ip_address
lt_network_interface = aws.ec2.LaunchTemplateNetworkInterfaceArgs(
    device_index=0,
    subnet_id=subnet_id,
    private_ip_address=deterministic_ip,
    associate_public_ip_address=False,
    security_groups=[qdrant_sg_id] if qdrant_sg_id else None,
)

lt = aws.ec2.LaunchTemplate(
    launch_template_name,
    name=launch_template_name,
    image_id=ami_id,
    instance_type=instance_type,
    key_name=ec2_key_name,
    iam_instance_profile=aws.ec2.LaunchTemplateIamInstanceProfileArgs(name=instance_profile.name),
    user_data=user_data_b64,
    network_interfaces=[lt_network_interface],
    block_device_mappings=block_device_mappings,
    tag_specifications=[
        aws.ec2.LaunchTemplateTagSpecificationArgs(resource_type="instance", tags={"Name": "qdrant-node"})
    ],
)

# IMPORTANT: make vpc_zone_identifiers an Output list to avoid passing a literal null into the provider
vpc_zone_identifiers = pulumi.Output.all(subnet_id).apply(lambda sid: [sid[0]] if isinstance(sid, (list, tuple)) else [sid])

asg = aws.autoscaling.Group(
    "qdrant-asg-single-az",
    desired_capacity=1,
    min_size=1,
    max_size=1,
    launch_template=aws.autoscaling.GroupLaunchTemplateArgs(id=lt.id, version="$Latest"),
    vpc_zone_identifiers=vpc_zone_identifiers,
    health_check_type="EC2",
    health_check_grace_period=900,
    termination_policies=["Default"],
    tags=[aws.autoscaling.GroupTagArgs(key="Name", value="qdrant-asg", propagate_at_launch=True)],
    opts=ResourceOptions(depends_on=[instance_profile]),
)

pulumi.export("qdrant_launch_template_id", lt.id)
pulumi.export("qdrant_asg_name", asg.name)
pulumi.export("qdrant_private_ip", deterministic_ip)
pulumi.export("qdrant_url", pulumi.Output.concat("http://", deterministic_ip, ":6333"))
pulumi.export("qdrant_subnet_id", subnet_id)
pulumi.export("qdrant_subnet_az", subnet_az)
