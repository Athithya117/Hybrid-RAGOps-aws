#!/usr/bin/env python3
# pulumi_aws/core_network.py
"""
Finalized core_network.py for Pulumi v3.209.0 + pulumi-aws v7.12.0.
Creates:
  - VPC, public/private subnets (auto or explicit)
  - Internet Gateway + public route table
  - NAT Gateways (per-AZ or single) unless NO_NAT=true
  - VPC Endpoints (gateway for S3, interface for others)
  - VPC Flow Logs (CloudWatch or S3)
  - Example S3 bucket with aws.s3.BucketVersioning
Exports stable outputs used by downstream modules.

This file:
 - validates inputs and fails fast
 - uses deterministic naming via TAG_PREFIX + stack
 - avoids deprecated args (e.g., no vpc=True on Eip)
 - collects and exports route table IDs created in this module
"""

from __future__ import annotations
import os
import json
import ipaddress
from typing import List, Dict, Optional

import pulumi
import pulumi_aws as aws

cfg = pulumi.Config()

def _env_bool(env: str, cfg_key: Optional[str], default: bool) -> bool:
    raw = os.getenv(env)
    if raw is None and cfg_key:
        raw = cfg.get(cfg_key)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y")

def _env_int(env: str, cfg_key: Optional[str], default: int) -> int:
    raw = os.getenv(env)
    if raw is None and cfg_key:
        raw = cfg.get(cfg_key)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception as e:
        raise pulumi.RunError(f"Invalid integer for {env or cfg_key}: {raw}") from e

def _env_str(env: str, cfg_key: Optional[str], default: Optional[str]) -> Optional[str]:
    raw = os.getenv(env)
    if raw is None and cfg_key:
        raw = cfg.get(cfg_key)
    return raw if raw is not None else default

def _env_list(env: str, cfg_key: Optional[str], default: Optional[List[str]]) -> List[str]:
    raw = os.getenv(env)
    if raw is None and cfg_key:
        raw = cfg.get(cfg_key)
    if raw is None:
        return default or []
    if isinstance(raw, list):
        return raw
    return [p.strip() for p in str(raw).split(",") if p.strip()]

# -------------------------
# Config knobs (env OR pulumi config)
# -------------------------
AWS_REGION = _env_str("AWS_REGION", "aws:region", None) or aws.get_region().name
STACK = _env_str("PULUMI_STACK", "stack", pulumi.get_stack())

MULTI_AZ = _env_bool("MULTI_AZ_DEPLOYMENT", "multiAz", False)
AZ_COUNT = _env_int("AZ_COUNT", "azCount", 3 if MULTI_AZ else 1)
if AZ_COUNT < 1:
    raise pulumi.RunError("AZ_COUNT must be >= 1")

VPC_CIDR = _env_str("VPC_CIDR", "vpcCidr", "10.0.0.0/16")
PUBLIC_SUBNET_CIDRS = _env_list("PUBLIC_SUBNET_CIDRS", "publicSubnetCidrs", [])
PRIVATE_SUBNET_CIDRS = _env_list("PRIVATE_SUBNET_CIDRS", "privateSubnetCidrs", [])

NO_NAT = _env_bool("NO_NAT", "noNat", False)
NAT_SINGLE = _env_bool("NAT_SINGLE", "natSingle", False)

CREATE_VPC_ENDPOINTS = _env_bool("CREATE_VPC_ENDPOINTS", "createVpcEndpoints", True)
VPC_ENDPOINT_SERVICES = _env_list("VPC_ENDPOINT_SERVICES", "vpcEndpointServices", ["s3", "ecr.api", "ecr.dkr", "ssm", "sts"])

ENABLE_FLOW_LOGS = _env_bool("ENABLE_FLOW_LOGS", "enableFlowLogs", False)
FLOW_LOG_DEST = _env_str("FLOW_LOG_DEST", "flowLogDest", "cloudwatch")  # cloudwatch | s3
FLOW_LOG_S3_BUCKET = _env_str("FLOW_LOG_S3_BUCKET", "flowLogS3Bucket", None)

TAG_PREFIX = _env_str("TAG_PREFIX", "tagPrefix", f"pulumi-{STACK}")

# -------------------------
# Validation
# -------------------------
try:
    vpc_network = ipaddress.IPv4Network(VPC_CIDR)
except Exception:
    raise pulumi.RunError(f"Invalid VPC_CIDR: {VPC_CIDR!r}")

if PUBLIC_SUBNET_CIDRS and len(PUBLIC_SUBNET_CIDRS) != AZ_COUNT:
    raise pulumi.RunError("PUBLIC_SUBNET_CIDRS length must equal AZ_COUNT or be empty (auto-generation).")

if PRIVATE_SUBNET_CIDRS and len(PRIVATE_SUBNET_CIDRS) != AZ_COUNT:
    raise pulumi.RunError("PRIVATE_SUBNET_CIDRS length must equal AZ_COUNT or be empty (auto-generation).")

if ENABLE_FLOW_LOGS and FLOW_LOG_DEST == "s3" and not FLOW_LOG_S3_BUCKET:
    raise pulumi.RunError("FLOW_LOG_S3_BUCKET must be set when FLOW_LOG_DEST=s3")

if NAT_SINGLE and NO_NAT:
    pulumi.log.info("NAT_SINGLE ignored since NO_NAT=true")

# -------------------------
# Helpers
# -------------------------
def _name(logical: str) -> str:
    return f"{TAG_PREFIX}-{STACK}-{logical}"

def _auto_generate_subnets(vpc_net: ipaddress.IPv4Network, az_count: int) -> (List[str], List[str]):
    needed = az_count * 2
    for new_prefix in range(vpc_net.prefixlen + 1, 28):
        total = 2 ** (new_prefix - vpc_net.prefixlen)
        if total >= needed:
            subnets = list(vpc_net.subnets(new_prefix=new_prefix))
            public = [str(subnets[i * 2]) for i in range(az_count)]
            private = [str(subnets[i * 2 + 1]) for i in range(az_count)]
            return public, private
    raise pulumi.RunError("VPC_CIDR too small to auto-generate requested subnet count; choose larger VPC_CIDR")

# -------------------------
# Availability Zones
# -------------------------
azs = aws.get_availability_zones(state="available").names
if AZ_COUNT > len(azs):
    raise pulumi.RunError(f"AZ_COUNT {AZ_COUNT} > available AZs in region ({len(azs)}).")
azs = azs[:AZ_COUNT]

# Auto-generate subnets if not provided
if not PUBLIC_SUBNET_CIDRS or not PRIVATE_SUBNET_CIDRS:
    pub_auto, priv_auto = _auto_generate_subnets(vpc_network, AZ_COUNT)
    if not PUBLIC_SUBNET_CIDRS:
        PUBLIC_SUBNET_CIDRS = pub_auto
    if not PRIVATE_SUBNET_CIDRS:
        PRIVATE_SUBNET_CIDRS = priv_auto

# Validate subnets are inside VPC
for cidr in PUBLIC_SUBNET_CIDRS + PRIVATE_SUBNET_CIDRS:
    net = ipaddress.IPv4Network(cidr)
    if not net.subnet_of(vpc_network):
        raise pulumi.RunError(f"Subnet CIDR {cidr} is not inside VPC_CIDR {VPC_CIDR}")

# -------------------------
# VPC
# -------------------------
vpc = aws.ec2.Vpc(
    _name("vpc"),
    cidr_block=VPC_CIDR,
    enable_dns_hostnames=True,
    enable_dns_support=True,
    tags={"Name": _name("vpc"), "Environment": STACK},
)

# Internet Gateway
igw = aws.ec2.InternetGateway(
    _name("igw"),
    vpc_id=vpc.id,
    tags={"Name": _name("igw"), "Environment": STACK},
)

# Public route table (to IGW)
public_rt = aws.ec2.RouteTable(
    _name("public-rt"),
    vpc_id=vpc.id,
    routes=[aws.ec2.RouteTableRouteArgs(cidr_block="0.0.0.0/0", gateway_id=igw.id)],
    tags={"Name": _name("public-rt"), "Environment": STACK},
)

# Create subnets and associations
public_subnets: List[aws.ec2.Subnet] = []
private_subnets: List[aws.ec2.Subnet] = []
private_route_table_ids: List[pulumi.Output[str]] = []

for i, az in enumerate(azs):
    pub = aws.ec2.Subnet(
        _name(f"public-subnet-{i}"),
        vpc_id=vpc.id,
        cidr_block=PUBLIC_SUBNET_CIDRS[i],
        availability_zone=az,
        map_public_ip_on_launch=True,
        tags={"Name": _name(f"public-subnet-{i}"), "kubernetes.io/role/elb": "1", "Environment": STACK},
    )
    public_subnets.append(pub)
    aws.ec2.RouteTableAssociation(_name(f"public-rta-{i}"), subnet_id=pub.id, route_table_id=public_rt.id)

    priv = aws.ec2.Subnet(
        _name(f"private-subnet-{i}"),
        vpc_id=vpc.id,
        cidr_block=PRIVATE_SUBNET_CIDRS[i],
        availability_zone=az,
        map_public_ip_on_launch=False,
        tags={"Name": _name(f"private-subnet-{i}"), "kubernetes.io/role/internal-elb": "1", "Environment": STACK},
    )
    private_subnets.append(priv)

# -------------------------
# NAT Gateways and private route tables
# -------------------------
nat_eips: List[aws.ec2.Eip] = []
nat_gws: List[aws.ec2.NatGateway] = []

if not NO_NAT:
    if NAT_SINGLE:
        eip = aws.ec2.Eip(_name("nat-eip-0"), tags={"Name": _name("nat-eip-0")})
        nat_eips.append(eip)
        nat = aws.ec2.NatGateway(_name("natgw-0"), allocation_id=eip.id, subnet_id=public_subnets[0].id, tags={"Name": _name("natgw-0"), "Environment": STACK})
        nat_gws.append(nat)
        for idx, priv_sub in enumerate(private_subnets):
            rt = aws.ec2.RouteTable(_name(f"private-rt-{idx}"), vpc_id=vpc.id, routes=[aws.ec2.RouteTableRouteArgs(cidr_block="0.0.0.0/0", nat_gateway_id=nat.id)], tags={"Name": _name(f"private-rt-{idx}")})
            aws.ec2.RouteTableAssociation(_name(f"private-rta-{idx}"), subnet_id=priv_sub.id, route_table_id=rt.id)
            private_route_table_ids.append(rt.id)
    else:
        # NAT per AZ
        for i, pub_sub in enumerate(public_subnets):
            eip = aws.ec2.Eip(_name(f"nat-eip-{i}"), tags={"Name": _name(f"nat-eip-{i}")})
            nat_eips.append(eip)
            nat = aws.ec2.NatGateway(_name(f"natgw-{i}"), allocation_id=eip.id, subnet_id=pub_sub.id, tags={"Name": _name(f"natgw-{i}"), "Environment": STACK})
            nat_gws.append(nat)
        for idx, priv_sub in enumerate(private_subnets):
            nat_for = nat_gws[idx % len(nat_gws)]
            rt = aws.ec2.RouteTable(_name(f"private-rt-{idx}"), vpc_id=vpc.id, routes=[aws.ec2.RouteTableRouteArgs(cidr_block="0.0.0.0/0", nat_gateway_id=nat_for.id)], tags={"Name": _name(f"private-rt-{idx}")})
            aws.ec2.RouteTableAssociation(_name(f"private-rta-{idx}"), subnet_id=priv_sub.id, route_table_id=rt.id)
            private_route_table_ids.append(rt.id)
else:
    # If NO_NAT, create private route tables without default route (explicit)
    for idx, priv_sub in enumerate(private_subnets):
        rt = aws.ec2.RouteTable(_name(f"private-rt-{idx}"), vpc_id=vpc.id, tags={"Name": _name(f"private-rt-{idx}")})
        aws.ec2.RouteTableAssociation(_name(f"private-rta-{idx}"), subnet_id=priv_sub.id, route_table_id=rt.id)
        private_route_table_ids.append(rt.id)

# -------------------------
# S3 buckets + versioning (v7 uses aws.s3.BucketVersioning)
# -------------------------
PULUMI_STATE_BUCKET = _env_str("PULUMI_STATE_BUCKET", "pulumiStateBucket", f"{TAG_PREFIX}-{STACK}-pulumi-state")
snapshot_bucket = aws.s3.Bucket(_name("pulumi-state-bucket"), bucket=PULUMI_STATE_BUCKET, tags={"Name": _name("pulumi-state-bucket"), "Environment": STACK})

# Use explicit BucketVersioning resource per v7 docs
_ = aws.s3.BucketVersioning(
    _name("pulumi-state-bucket-versioning"),
    bucket=snapshot_bucket.id,
    versioning_configuration=aws.s3.BucketVersioningVersioningConfigurationArgs(status="Enabled"),
)

# -------------------------
# VPC Endpoints (gateway for s3, interface for others)
# -------------------------
vpc_endpoints: Dict[str, aws.ec2.VpcEndpoint] = {}
if CREATE_VPC_ENDPOINTS:
    # collect route table ids to attach gateway endpoints to (public + private)
    gateway_route_table_ids = [public_rt.id] + private_route_table_ids
    for svc in VPC_ENDPOINT_SERVICES:
        svc = svc.strip()
        if svc == "s3":
            ep = aws.ec2.VpcEndpoint(_name("vpce-s3"), vpc_id=vpc.id, service_name=f"com.amazonaws.{AWS_REGION}.s3", vpc_endpoint_type="Gateway", route_table_ids=gateway_route_table_ids, tags={"Name": _name("vpce-s3")})
            vpc_endpoints["s3"] = ep
        else:
            mapping = {
                "ecr.api": f"com.amazonaws.{AWS_REGION}.ecr.api",
                "ecr.dkr": f"com.amazonaws.{AWS_REGION}.ecr.dkr",
                "ssm": f"com.amazonaws.{AWS_REGION}.ssm",
                "sts": f"com.amazonaws.{AWS_REGION}.sts",
                "ec2": f"com.amazonaws.{AWS_REGION}.ec2",
                "secretsmanager": f"com.amazonaws.{AWS_REGION}.secretsmanager",
                "kms": f"com.amazonaws.{AWS_REGION}.kms",
            }
            if svc not in mapping:
                raise pulumi.RunError(f"Unsupported VPC endpoint service shorthand: {svc}")
            ep_sg = aws.ec2.SecurityGroup(_name(f"vpce-sg-{svc}"), vpc_id=vpc.id, description=f"SG for VPC Endpoint {svc}", ingress=[aws.ec2.SecurityGroupIngressArgs(protocol="-1", from_port=0, to_port=0, cidr_blocks=[vpc.cidr_block])], egress=[aws.ec2.SecurityGroupEgressArgs(protocol="-1", from_port=0, to_port=0, cidr_blocks=["0.0.0.0/0"])])
            ep = aws.ec2.VpcEndpoint(_name(f"vpce-{svc}"), vpc_id=vpc.id, service_name=mapping[svc], vpc_endpoint_type="Interface", subnet_ids=[s.id for s in private_subnets], security_group_ids=[ep_sg.id], tags={"Name": _name(f"vpce-{svc}")})
            vpc_endpoints[svc] = ep

# -------------------------
# VPC Flow Logs
# -------------------------
flow_log_id = ""
if ENABLE_FLOW_LOGS:
    if FLOW_LOG_DEST == "cloudwatch":
        role = aws.iam.Role(_name("flowlog-role"), assume_role_policy=json.dumps({"Version":"2012-10-17","Statement":[{"Action":"sts:AssumeRole","Principal":{"Service":"vpc-flow-logs.amazonaws.com"},"Effect":"Allow"}]}), tags={"Name": _name("flowlog-role")})
        aws.iam.RolePolicyAttachment(_name("flowlog-role-CW"), role=role.name, policy_arn="arn:aws:iam::aws:policy/CloudWatchLogsFullAccess")
        flow = aws.ec2.FlowLog(_name("vpc-flow-log"), resource_id=vpc.id, traffic_type="ALL", iam_role_arn=role.arn, log_destination_type="cloud-watch-logs", tags={"Name": _name("vpc-flow-log")})
        flow_log_id = flow.id
    else:
        flow = aws.ec2.FlowLog(_name("vpc-flow-log-s3"), resource_id=vpc.id, traffic_type="ALL", log_destination_type="s3", log_destination=f"arn:aws:s3:::{FLOW_LOG_S3_BUCKET}", tags={"Name": _name("vpc-flow-log-s3")})
        flow_log_id = flow.id

# -------------------------
# Exports
# -------------------------
pulumi.export("vpc_id", vpc.id)
pulumi.export("public_subnet_ids", [s.id for s in public_subnets])
pulumi.export("private_subnet_ids", [s.id for s in private_subnets])
pulumi.export("nat_gateway_ids", [ng.id for ng in nat_gws])
pulumi.export("route_table_ids", {"public": public_rt.id, "private": private_route_table_ids})
pulumi.export("vpc_endpoint_ids", {k: v.id for k, v in vpc_endpoints.items()})
pulumi.export("flow_log_id", flow_log_id or "")

# End of file
