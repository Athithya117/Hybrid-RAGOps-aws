# infra/pulumi-aws/qdrant.py
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

# If network module is available in-process (we expect __main__.py imports networking first)
try:
    import networking  # type: ignore
except Exception:
    networking = None  # type: ignore

# --- Network inputs resolution (robust) ---
subnet_ids: Optional[List[pulumi.Input[str]]] = None
subnet_azs: Optional[List[pulumi.Input[str]]] = None
qdrant_sg_id: Optional[pulumi.Input[str]] = None
ec2_key_name: Optional[pulumi.Input[str]] = None

# 1) prefer in-process networking (most convenient)
if networking is not None:
    try:
        if hasattr(networking, "public_subnets") and networking.public_subnets:
            subnet_ids = [s.id for s in networking.public_subnets]
            subnet_azs = [s.availability_zone for s in networking.public_subnets]
    except Exception:
        pass
    try:
        qdrant_sg_id = getattr(networking, "qdrant_sg").id if hasattr(networking, "qdrant_sg") else getattr(networking, "qdrant_security_group_id", None)
    except Exception:
        pass
    try:
        ec2_key_name = getattr(networking, "aws_key").key_name if hasattr(networking, "aws_key") else getattr(networking, "ec2_key_name", None)
    except Exception:
        pass

# 2) fallback to a stack reference (optional)
networking_stack = cfg.get("networking_stack") or os.environ.get("NETWORKING_STACK")
if (not subnet_ids) and networking_stack:
    try:
        ref = pulumi.StackReference(networking_stack)
        subnet_ids = subnet_ids or ref.get_output("public_subnet_ids")
        subnet_azs = subnet_azs or ref.get_output("public_subnet_azs")
        qdrant_sg_id = qdrant_sg_id or ref.get_output("qdrant_security_group_id")
        ec2_key_name = ec2_key_name or ref.get_output("ec2_key_name")
    except Exception:
        pulumi.log.warn("Failed to read networking_stack outputs; falling back to config/env")

# 3) fallback to env/config
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

if not qdrant_sg_id:
    qdrant_sg_id = cfg.get("qdrant_sg_id") or os.environ.get("PULUMI_QDRANT_SECURITY_GROUP_ID") or os.environ.get("QDRANT_SG_ID")
if not ec2_key_name:
    ec2_key_name = cfg.get("ec2_key_name") or os.environ.get("PULUMI_EC2_KEY_NAME") or os.environ.get("EC2_KEY_NAME")

if not subnet_ids or len(subnet_ids) == 0:
    raise RuntimeError("Missing subnet_ids (provide in-process networking or PULUMI_PUBLIC_SUBNET_IDS / PULUMI_PUBLIC_SUBNET_ID_0)")

# --- Core config & validation ---
ami_id = cfg.get("ami_id") or os.environ.get("BASE_AMI_ID")
if not ami_id:
    raise RuntimeError("BASE_AMI_ID (Ubuntu AMI) must be provided in environment or pulumi config")

snapshot_s3_bucket = cfg.get("snapshot_s3_bucket") or os.environ.get("S3_BUCKET")
snapshot_s3_prefix = cfg.get("snapshot_s3_prefix") or os.environ.get("SNAPSHOT_S3_PREFIX") or "qdrant/backups/"

instance_type = cfg.get("instance_type") or os.environ.get("QDRANT_INSTANCE_TYPE") or "c8g.4xlarge"
launch_template_name = cfg.get("launch_template_name") or os.environ.get("LAUNCH_TEMPLATE_NAME") or "qdrant-launch-template"
instance_role_name = cfg.get("instance_role_name") or os.environ.get("INSTANCE_ROLE_NAME") or "qdrant-ec2-role"

qdrant_ebs_type = cfg.get("qdrant_ebs_type") or os.environ.get("QDRANT_EBS_TYPE") or "gp3"
qdrant_ebs_size = int(cfg.get("qdrant_ebs_size") or os.environ.get("QDRANT_EBS_SIZE") or "8")

_dot = cfg.get("delete_on_termination")
delete_on_termination = (os.environ.get("QDRANT_EBS_DELETE_ON_TERM", "true").lower() == "true") if (_dot is None) else (str(_dot).lower() == "true")

if not snapshot_s3_bucket:
    raise RuntimeError("snapshot_s3_bucket / S3_BUCKET is required")

# MULTI-AZ toggle -- create one ENI+ASG per-subnet if true, otherwise single ENI+single ASG
multi_az_flag = os.environ.get("MULTI_AZ_QDRANT_DEPLOYMENT", "false").lower() in ("1", "true", "yes")

# IAM role & instance profile (to allow EC2 read from S3)
assume_role_policy = aws.iam.get_policy_document(
    statements=[{
        "Effect": "Allow",
        "Principals": [{"Type": "Service", "Identifiers": ["ec2.amazonaws.com"]}],
        "Actions": ["sts:AssumeRole"],
    }]
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

# --- User data / provisioning script (fully included, with a small wait-for-IP section) ---
_raw_user_data = r"""#!/usr/bin/env bash
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

# NVMe mount logic (common)
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

# Restore logic...
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

# Wait for the deterministic secondary ENI IP to show up on the instance before starting qdrant.
TARGET_IP="__QDRANT_PRIVATE_IP__"
if [ -n "$TARGET_IP" ]; then
  MAX_WAIT=600
  WAITED=0
  while ! ip -4 addr show | grep -w -q "${TARGET_IP}"; do
    echo "Waiting for ENI IP ${TARGET_IP} to appear on instance (waited ${WAITED}s)..."
    sleep 2
    WAITED=$((WAITED+2))
    if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
      echo "Timeout waiting for ENI IP ${TARGET_IP}; starting qdrant anyway."
      break
    fi
  done
fi

IMAGE_TAG="__IMAGE_TAG__"
docker pull "${IMAGE_TAG}" || true
DIGEST="$(docker inspect --format='{{index .RepoDigests 0}}' "${IMAGE_TAG}" 2>/dev/null || true)"
if [ -n "${DIGEST}" ]; then
  IMAGE="${DIGEST}"
else
  IMAGE="${IMAGE_TAG}"
fi

docker run -d --name qdrant --restart unless-stopped -p 6333:6333 -v "${DATA_DIR}":/qdrant/storage -e QDRANT__HTTP__API_KEY="${QDRANT_API_KEY:-}" "${IMAGE}" || true
touch "${DATA_DIR}/.qdrant_initialized" || true
chown 1000:1000 "${DATA_DIR}/.qdrant_initialized" || true
echo "User-data finished."
"""

def make_user_data(bucket: str, prefix: str, private_ip: str, image_tag: str = "qdrant/qdrant:v1.15.4") -> str:
    ud = _raw_user_data
    ud = ud.replace("__SNAPSHOT_S3_BUCKET__", bucket)
    ud = ud.replace("__SNAPSHOT_S3_PREFIX__", prefix)
    ud = ud.replace("__IMAGE_TAG__", image_tag)
    ud = ud.replace("__QDRANT_PRIVATE_IP__", private_ip or "")
    return ud

# block device mapping for root volume
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

# Helper: lambda code (same for single-ENI and per-AZ lambdas)
lambda_code = r"""
import json, os, time, boto3, traceback
ec2 = boto3.client('ec2')
asg = boto3.client('autoscaling')

ENI_ID = os.environ.get('ENI_ID')

def describe_eni(eni_id):
    return ec2.describe_network_interfaces(NetworkInterfaceIds=[eni_id])['NetworkInterfaces'][0]

def attach_eni(eni_id, instance_id):
    ni = describe_eni(eni_id)
    if ni.get('Attachment'):
        return True, 'already-attached'
    # attach as device index 1 (primary is 0)
    resp = ec2.attach_network_interface(NetworkInterfaceId=eni_id, InstanceId=instance_id, DeviceIndex=1)
    return True, resp.get('AttachmentId')

def detach_eni(eni_id):
    ni = describe_eni(eni_id)
    att = ni.get('Attachment')
    if not att:
        return True, 'not-attached'
    ec2.detach_network_interface(AttachmentId=att['AttachmentId'], Force=True)
    # wait briefly for detachment
    for _ in range(10):
        ni = describe_eni(eni_id)
        if not ni.get('Attachment'):
            return True, 'detached'
        time.sleep(1)
    return False, 'detach-timeout'

def complete_lifecycle(hook_name, asg_name, token, instance_id, result='CONTINUE'):
    kwargs = {'LifecycleHookName': hook_name, 'AutoScalingGroupName': asg_name, 'LifecycleActionResult': result}
    if token:
        kwargs['LifecycleActionToken'] = token
    else:
        kwargs['InstanceId'] = instance_id
    return asg.complete_lifecycle_action(**kwargs)

def handler(event, context):
    try:
        recs = event.get('Records', [])
        for r in recs:
            msg = r.get('Sns', {}).get('Message')
            if not msg:
                continue
            payload = json.loads(msg)
            transition = payload.get('LifecycleTransition')
            hook_name = payload.get('LifecycleHookName')
            asg_name = payload.get('AutoScalingGroupName')
            token = payload.get('LifecycleActionToken')
            instance_id = payload.get('EC2InstanceId')
            if transition and transition.endswith('EC2_INSTANCE_LAUNCHING'):
                try:
                    ok, info = attach_eni(ENI_ID, instance_id)
                    complete_lifecycle(hook_name, asg_name, token, instance_id, result='CONTINUE')
                except Exception:
                    traceback.print_exc()
                    try:
                        complete_lifecycle(hook_name, asg_name, token, instance_id, result='ABANDON')
                    except Exception:
                        pass
            elif transition and transition.endswith('EC2_INSTANCE_TERMINATING'):
                try:
                    ok, info = detach_eni(ENI_ID)
                except Exception:
                    traceback.print_exc()
                finally:
                    try:
                        complete_lifecycle(hook_name, asg_name, token, instance_id, result='CONTINUE')
                    except Exception:
                        pass
    except Exception:
        traceback.print_exc()
        raise
"""

# Utility to create one ENI + LT + ASG + lifecycle resources per-subnet
def create_qdrant_unit(idx: int, subnet_id: pulumi.Input[str], subnet_az: Optional[pulumi.Input[str]], deterministic_ip: Optional[str]):
    suffix = f"az{idx}"
    eni = aws.ec2.NetworkInterface(
        f"qdrant-eni-{idx}",
        subnet_id=subnet_id,
        private_ip=deterministic_ip,
        description=f"qdrant-eni-{suffix}",
        security_groups=[qdrant_sg_id] if qdrant_sg_id else None,
        tags={"Name": f"qdrant-eni-{suffix}", "qdrant": "true"},
    )

    # user-data base64 with this ENI.private_ip injected
    user_data_b64 = Output.all(snapshot_s3_bucket, snapshot_s3_prefix, eni.private_ip).apply(
        lambda args: base64.b64encode(make_user_data(args[0], args[1], args[2]).encode()).decode()
    )

    lt = aws.ec2.LaunchTemplate(
        f"qdrant-launch-template-{idx}",
        name=f"{launch_template_name}-{idx}",
        image_id=ami_id,
        instance_type=instance_type,
        key_name=ec2_key_name,
        iam_instance_profile=aws.ec2.LaunchTemplateIamInstanceProfileArgs(name=instance_profile.name),
        user_data=user_data_b64,
        block_device_mappings=block_device_mappings,
        tag_specifications=[aws.ec2.LaunchTemplateTagSpecificationArgs(resource_type="instance", tags={"Name": f"qdrant-node-{idx}"})],
    )

    # ASG constrained to this subnet only
    vpc_zone_identifiers = Output.from_input([subnet_id]).apply(lambda s: list(s) if s else [])

    asg = aws.autoscaling.Group(
        f"qdrant-asg-{idx}",
        desired_capacity=1,
        min_size=1,
        max_size=1,
        launch_template=aws.autoscaling.GroupLaunchTemplateArgs(id=lt.id, version="$Latest"),
        vpc_zone_identifiers=vpc_zone_identifiers,
        health_check_type="EC2",
        health_check_grace_period=900,
        termination_policies=["Default"],
        tags=[aws.autoscaling.GroupTagArgs(key="Name", value=f"qdrant-asg-{idx}", propagate_at_launch=True)],
        opts=ResourceOptions(depends_on=[instance_profile]),
    )

    # SNS topic, lifecycle hooks, Lambda handler for this ENI
    topic = aws.sns.Topic(f"qdrant-lifecycle-topic-{idx}", display_name=f"qdrant-lifecycle-topic-{idx}")

    # lifecycle role allowing autoscaling to publish to SNS
    lifecycle_role = aws.iam.Role(
        f"qdrant-lifecycle-role-{idx}",
        assume_role_policy=json.dumps({
            "Version": "2012-10-17",
            "Statement": [{"Action": "sts:AssumeRole", "Principal": {"Service": "autoscaling.amazonaws.com"}, "Effect": "Allow"}]
        }),
    )
    aws.iam.RolePolicy(f"qdrant-lifecycle-role-policy-{idx}", role=lifecycle_role.id, policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["sns:Publish"], "Resource": topic.arn}
        ]
    }))

    hook_launch = aws.autoscaling.LifecycleHook(
        f"qdrant-launch-hook-{idx}",
        autoscaling_group_name=asg.name,
        lifecycle_transition="autoscaling:EC2_INSTANCE_LAUNCHING",
        default_result="ABANDON",
        heartbeat_timeout=3600,
        notification_target_arn=topic.arn,
        role_arn=lifecycle_role.arn,
    )

    hook_term = aws.autoscaling.LifecycleHook(
        f"qdrant-terminate-hook-{idx}",
        autoscaling_group_name=asg.name,
        lifecycle_transition="autoscaling:EC2_INSTANCE_TERMINATING",
        default_result="CONTINUE",
        heartbeat_timeout=300,
        notification_target_arn=topic.arn,
        role_arn=lifecycle_role.arn,
    )

    # Lambda role/policy
    lambda_role = aws.iam.Role(
        f"qdrant-lambda-role-{idx}",
        assume_role_policy=json.dumps({
            "Version": "2012-10-17",
            "Statement": [{"Action": "sts:AssumeRole", "Principal": {"Service": "lambda.amazonaws.com"}, "Effect": "Allow"}]
        }),
    )
    aws.iam.RolePolicyAttachment(f"qdrant-lambda-logs-{idx}", role=lambda_role.name, policy_arn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")

    lambda_policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["ec2:DescribeInstances", "ec2:DescribeNetworkInterfaces", "ec2:AttachNetworkInterface", "ec2:DetachNetworkInterface"], "Resource": "*"},
            {"Effect": "Allow", "Action": ["autoscaling:CompleteLifecycleAction"], "Resource": "*"},
        ],
    }
    aws.iam.RolePolicy(f"qdrant-lambda-inline-policy-{idx}", role=lambda_role.id, policy=json.dumps(lambda_policy_doc))

    lambda_fn = aws.lambda_.Function(
        f"qdrant-lifecycle-handler-{idx}",
        runtime="python3.9",
        role=lambda_role.arn,
        handler="index.handler",
        code=asset.AssetArchive({"index.py": asset.StringAsset(lambda_code)}),
        environment=aws.lambda_.FunctionEnvironmentArgs(variables={"ENI_ID": eni.id}),
        opts=ResourceOptions(depends_on=[topic]),
    )

    # Allow SNS to invoke this lambda
    aws.lambda_.Permission(
        f"qdrant-lambda-sns-perm-{idx}",
        action="lambda:InvokeFunction",
        function=lambda_fn.name,
        principal="sns.amazonaws.com",
        source_arn=topic.arn
    )

    # Subscribe lambda to topic
    aws.sns.TopicSubscription(
        f"qdrant-lifecycle-sub-{idx}",
        topic=topic.arn,
        protocol="lambda",
        endpoint=lambda_fn.arn,
        opts=ResourceOptions(depends_on=[lambda_fn])
    )

    # Exports
    pulumi.export(f"qdrant_eni_id_{idx}", eni.id)
    pulumi.export(f"qdrant_eni_private_ip_{idx}", eni.private_ip)
    pulumi.export(f"qdrant_launch_template_id_{idx}", lt.id)
    pulumi.export(f"qdrant_asg_name_{idx}", asg.name)
    pulumi.export(f"qdrant_url_{idx}", pulumi.Output.concat("http://", eni.private_ip, ":6333"))

    return {"eni": eni, "lt": lt, "asg": asg, "topic": topic, "lambda": lambda_fn}

# --- Top-level creation logic ---
enis_created = []
lts = []
asgs = []

# Determine deterministic IP(s)
# If MULTI_AZ true, you can provide QDRANT_PRIVATE_IPS as comma-separated list matching subnets count.
if multi_az_flag:
    provided_ips = os.environ.get("QDRANT_PRIVATE_IPS")
    single_ip = os.environ.get("QDRANT_PRIVATE_IP")
    if provided_ips:
        ip_list = [p.strip() for p in provided_ips.split(",") if p.strip()]
        if len(ip_list) != len(subnet_ids):
            raise RuntimeError("When MULTI_AZ_QDRANT_DEPLOYMENT=true, provide same number of QDRANT_PRIVATE_IPS as subnets")
        deterministic_ips = ip_list
    elif single_ip:
        # auto-increment last octet across AZs (best effort)
        import ipaddress
        try:
            base = ipaddress.IPv4Address(single_ip)
            deterministic_ips = [str(ipaddress.IPv4Address(int(base) + i)) for i in range(len(subnet_ids))]
        except Exception as e:
            pulumi.log.warn(f"Could not auto-increment QDRANT_PRIVATE_IP: {e}; falling back to dynamic ENIs")
            deterministic_ips = [None] * len(subnet_ids)
    else:
        deterministic_ips = [None] * len(subnet_ids)

    # create unit per-subnet
    for idx, sn in enumerate(subnet_ids):
        create_qdrant_unit(idx, sn, (subnet_azs[idx] if subnet_azs and idx < len(subnet_azs) else None), deterministic_ips[idx])
else:
    # single ENI mode: use first subnet from list
    primary_subnet = subnet_ids[0]
    deterministic_ip = cfg.get("qdrant_private_ip") or os.environ.get("QDRANT_PRIVATE_IP") or None
    if not deterministic_ip:
        # it's allowed to be dynamic, but deterministic recommended
        pulumi.log.warn("QDRANT_PRIVATE_IP not supplied; ENI will be created without explicit private_ip and AWS will assign one")
    # create a single unit with index 0
    create_qdrant_unit(0, primary_subnet, (subnet_azs[0] if subnet_azs else None), deterministic_ip)

# aggregated exports (helpful)
pulumi.export("qdrant_multi_az_enabled", multi_az_flag)
# picks the first found private ip as primary deterministic endpoint (useful for Ray)
def pick_first_ip(args):
    arr = args
    if not arr:
        return ""
    return arr[0] if isinstance(arr[0], str) else arr[0]
pulumi.export("qdrant_private_ips", [e.private_ip for e in (pulumi.get_stack() and [] )] )  # placeholder; per-indexed exports above are authoritative

pulumi.log.info("Qdrant deployment created. If you used deterministic IPs, ensure the IPs are inside the subnet CIDRs and not in use.")
