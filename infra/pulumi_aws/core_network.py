#!/usr/bin/env python3
from __future__ import annotations
import os
import json
import ipaddress
from typing import List, Dict, Optional, Any

import pulumi
import pulumi_aws as aws

cfg = pulumi.Config()

def _env_str(env: str, cfg_key: Optional[str], default: Optional[str]) -> Optional[str]:
    v = os.getenv(env)
    if v is None and cfg_key:
        v = cfg.get(cfg_key)
    return v if v is not None else default

def _env_bool(env: str, cfg_key: Optional[str], default: bool) -> bool:
    v = os.getenv(env)
    if v is None and cfg_key:
        v = cfg.get(cfg_key)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y")

def _env_int(env: str, cfg_key: Optional[str], default: int) -> int:
    v = os.getenv(env)
    if v is None and cfg_key:
        v = cfg.get(cfg_key)
    if v is None:
        return default
    try:
        return int(v)
    except Exception as ex:
        raise pulumi.RunError(f"Invalid integer for {env or cfg_key}: {v}") from ex

def _env_list(env: str, cfg_key: Optional[str], default: Optional[List[str]]) -> List[str]:
    v = os.getenv(env)
    if v is None and cfg_key:
        v = cfg.get(cfg_key)
    if v is None:
        return default or []
    if isinstance(v, list):
        return v
    return [p.strip() for p in str(v).split(",") if p.strip()]

AWS_REGION = _env_str("AWS_REGION", "aws:region", None) or aws.get_region().region
STACK = _env_str("PULUMI_STACK", "stack", pulumi.get_stack())
TAG_PREFIX = _env_str("TAG_PREFIX", "tagPrefix", f"pulumi-{STACK}")

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
VPC_ENDPOINT_SERVICES = _env_list("VPC_ENDPOINT_SERVICES", "vpcEndpointServices", ["s3", "ecr.api", "ecr.dkr", "ssm", "sts", "ec2", "secretsmanager"])

FLOW_LOG_MODE = (_env_str("FLOW_LOG_MODE", "flowLogMode", "none") or "none").lower()
ENABLE_FLOW_LOGS = _env_bool("ENABLE_FLOW_LOGS", "enableFlowLogs", False)

FLOW_LOG_CW_LOG_GROUP = _env_str("FLOW_LOG_CW_LOG_GROUP", "flowLogCwLogGroup", f"/aws/vpc/flowlogs/{STACK}")
FLOW_LOG_CW_RETENTION_DAYS = _env_int("FLOW_LOG_CW_RETENTION_DAYS", "flowLogCwRetentionDays", 14)

FLOW_LOG_S3_BUCKET = _env_str("FLOW_LOG_S3_BUCKET", "flowLogS3Bucket", None)
FLOW_LOG_S3_CREATE = _env_bool("FLOW_LOG_S3_CREATE", "flowLogS3Create", False)
FLOW_LOG_S3_CREATE_NAME = _env_str("FLOW_LOG_S3_CREATE_NAME", "flowLogS3CreateName", f"{TAG_PREFIX}-{STACK}-vpc-flow-logs")
FLOW_LOG_S3_PREFIX = _env_str("FLOW_LOG_S3_PREFIX", "flowLogS3Prefix", f"AWSLogs/{aws.get_caller_identity().account_id}/vpcflowlogs/")
FLOW_LOG_S3_LIFECYCLE_TRANSITION_DAYS = _env_int("FLOW_LOG_S3_LIFECYCLE_TRANSITION_DAYS", "flowLogS3TransitionDays", 30)
FLOW_LOG_S3_EXPIRATION_DAYS = _env_int("FLOW_LOG_S3_EXPIRATION_DAYS", "flowLogS3ExpirationDays", 365)
FLOW_LOG_S3_ACCESS_LOGGING = _env_bool("FLOW_LOG_S3_ACCESS_LOGGING", "flowLogS3AccessLogging", False)
FLOW_LOG_S3_ACCESS_BUCKET = _env_str("FLOW_LOG_S3_ACCESS_BUCKET", "flowLogS3AccessBucket", None)

FLOW_LOG_SSE_ALGORITHM = (_env_str("FLOW_LOG_SSE_ALGORITHM", "flowLogSseAlgorithm", "AES256") or "AES256").upper()
FLOW_LOG_KMS_CREATE = _env_bool("FLOW_LOG_KMS_CREATE", "flowLogKmsCreate", False)
FLOW_LOG_KMS_ARN = _env_str("FLOW_LOG_KMS_ARN", "flowLogKmsArn", None)

CREATE_GLUE_CRAWLER = _env_bool("CREATE_GLUE_CRAWLER", "createGlueCrawler", False)
GLUE_CRAWLER_SCHEDULE = _env_str("GLUE_CRAWLER_SCHEDULE", "glueCrawlerSchedule", "cron(0 * ? * * *)")

CREATE_GLUE_ETL = _env_bool("CREATE_GLUE_ETL", "createGlueEtl", False)
GLUE_ETL_SCHEDULE = _env_str("GLUE_ETL_SCHEDULE", "glueEtlSchedule", "cron(0 2 * * ? *)")
GLUE_ETL_DPU = _env_int("GLUE_ETL_DPU", "glueEtlDpu", 2)
GLUE_SCRIPT_S3_PREFIX = _env_str("GLUE_SCRIPT_S3_PREFIX", "glueScriptS3Prefix", "glue-scripts/")

# added: zstd compression level env / pulumi config
ZSTD_COMPRESSION_LEVEL = _env_int("ZSTD_COMPRESSION_LEVEL", "zstdCompressionLevel", 3)

CREATE_ATHENA = _env_bool("CREATE_ATHENA", "createAthena", False)
ATHENA_DB_NAME = _env_str("ATHENA_DB_NAME", "athenaDbName", f"vpc_flow_logs_{STACK}")
ATHENA_TABLE_NAME = _env_str("ATHENA_TABLE_NAME", "athenaTableName", "vpc_flow_parquet")
ATHENA_OUTPUT_BUCKET = _env_str("ATHENA_OUTPUT_BUCKET", "athenaOutputBucket", None)

DESTINATION_FILE_FORMAT = _env_str("FLOW_LOG_FILE_FORMAT", "flowLogFileFormat", "parquet")
DESTINATION_PER_HOUR_PARTITION = _env_bool("FLOW_LOG_PER_HOUR_PARTITION", "flowLogPerHourPartition", True)

FLOW_LOG_MAX_DAILY_BYTES = _env_int("FLOW_LOG_MAX_DAILY_BYTES", "flowLogMaxDailyBytes", 1 * 1024 * 1024 * 1024)

EXISTING_VPC_ID = _env_str("EXISTING_VPC_ID", "existingVpcId", None)
EXISTING_PUBLIC_SUBNET_IDS = _env_list("EXISTING_PUBLIC_SUBNET_IDS", "existingPublicSubnetIds", [])
EXISTING_PRIVATE_SUBNET_IDS = _env_list("EXISTING_PRIVATE_SUBNET_IDS", "existingPrivateSubnetIds", [])

try:
    vpc_network = ipaddress.IPv4Network(VPC_CIDR)
except Exception:
    raise pulumi.RunError(f"Invalid VPC_CIDR: {VPC_CIDR!r}")

if PUBLIC_SUBNET_CIDRS and len(PUBLIC_SUBNET_CIDRS) != AZ_COUNT:
    raise pulumi.RunError("PUBLIC_SUBNET_CIDRS length must equal AZ_COUNT or be empty (auto-generation).")
if PRIVATE_SUBNET_CIDRS and len(PRIVATE_SUBNET_CIDRS) != AZ_COUNT:
    raise pulumi.RunError("PRIVATE_SUBNET_CIDRS length must equal AZ_COUNT or be empty (auto-generation).")
if FLOW_LOG_MODE == "s3" and not (FLOW_LOG_S3_BUCKET or FLOW_LOG_S3_CREATE):
    raise pulumi.RunError("FLOW_LOG_S3_BUCKET must be provided or FLOW_LOG_S3_CREATE=true when FLOW_LOG_MODE='s3'.")

def _name(logical: str) -> str:
    return f"{TAG_PREFIX}-{STACK}-{logical}"

def _auto_generate_subnets(vpc_net: ipaddress.IPv4Network, az_count: int):
    needed = az_count * 2
    for new_prefix in range(vpc_net.prefixlen + 1, 28):
        total = 2 ** (new_prefix - vpc_net.prefixlen)
        if total >= needed:
            subs = list(vpc_net.subnets(new_prefix=new_prefix))
            public = [str(subs[i * 2]) for i in range(az_count)]
            private = [str(subs[i * 2 + 1]) for i in range(az_count)]
            return public, private
    raise pulumi.RunError("VPC_CIDR too small to auto-generate requested subnet count; choose larger VPC_CIDR")

azs = aws.get_availability_zones(state="available").names
if AZ_COUNT > len(azs):
    raise pulumi.RunError(f"AZ_COUNT {AZ_COUNT} > available AZs in region ({len(azs)}).")
azs = azs[:AZ_COUNT]

public_subnets: List[aws.ec2.Subnet] = []
private_subnets: List[aws.ec2.Subnet] = []
private_route_table_ids: List[pulumi.Output[str]] = []

vpc: Optional[Any] = None
if EXISTING_VPC_ID:
    vpc_info = aws.ec2.get_vpc(id=EXISTING_VPC_ID)
    vpc = pulumi.Output.from_input({"id": vpc_info.id, "cidr_block": vpc_info.cidr_block})
    if not EXISTING_PUBLIC_SUBNET_IDS or not EXISTING_PRIVATE_SUBNET_IDS:
        raise pulumi.RunError("When EXISTING_VPC_ID is set you must provide EXISTING_PUBLIC_SUBNET_IDS and EXISTING_PRIVATE_SUBNET_IDS (comma list).")
    for i, sid in enumerate(EXISTING_PUBLIC_SUBNET_IDS):
        public_subnets.append(aws.ec2.Subnet(f"{_name('existing-public-subnet')}-{i}", id=sid))
    for i, sid in enumerate(EXISTING_PRIVATE_SUBNET_IDS):
        private_subnets.append(aws.ec2.Subnet(f"{_name('existing-private-subnet')}-{i}", id=sid))
else:
    if not PUBLIC_SUBNET_CIDRS or not PRIVATE_SUBNET_CIDRS:
        pub_auto, priv_auto = _auto_generate_subnets(vpc_network, AZ_COUNT)
        if not PUBLIC_SUBNET_CIDRS:
            PUBLIC_SUBNET_CIDRS = pub_auto
        if not PRIVATE_SUBNET_CIDRS:
            PRIVATE_SUBNET_CIDRS = priv_auto
    for cidr in PUBLIC_SUBNET_CIDRS + PRIVATE_SUBNET_CIDRS:
        net = ipaddress.IPv4Network(cidr)
        if not net.subnet_of(vpc_network):
            raise pulumi.RunError(f"Subnet CIDR {cidr} is not inside VPC_CIDR {VPC_CIDR}")
    vpc = aws.ec2.Vpc(
        _name("vpc"),
        cidr_block=VPC_CIDR,
        enable_dns_hostnames=True,
        enable_dns_support=True,
        tags={"Name": _name("vpc"), "Environment": STACK},
    )
    igw = aws.ec2.InternetGateway(
        _name("igw"),
        vpc_id=vpc.id,
        tags={"Name": _name("igw"), "Environment": STACK},
    )
    public_rt = aws.ec2.RouteTable(
        _name("public-rt"),
        vpc_id=vpc.id,
        routes=[aws.ec2.RouteTableRouteArgs(cidr_block="0.0.0.0/0", gateway_id=igw.id)],
        tags={"Name": _name("public-rt"), "Environment": STACK},
    )
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
        aws.ec2.RouteTableAssociation(
            _name(f"public-rta-{i}"),
            subnet_id=pub.id,
            route_table_id=public_rt.id,
        )
        priv = aws.ec2.Subnet(
            _name(f"private-subnet-{i}"),
            vpc_id=vpc.id,
            cidr_block=PRIVATE_SUBNET_CIDRS[i],
            availability_zone=az,
            map_public_ip_on_launch=False,
            tags={"Name": _name(f"private-subnet-{i}"), "kubernetes.io/role/internal-elb": "1", "Environment": STACK},
        )
        private_subnets.append(priv)

    nat_eips: List[aws.ec2.Eip] = []
    nat_gws: List[aws.ec2.NatGateway] = []

    if not NO_NAT:
        if NAT_SINGLE:
            eip = aws.ec2.Eip(_name("nat-eip-0"), vpc=True, tags={"Name": _name("nat-eip-0")})
            nat_eips.append(eip)
            nat = aws.ec2.NatGateway(_name("natgw-0"), allocation_id=eip.id, subnet_id=public_subnets[0].id, tags={"Name": _name("natgw-0"), "Environment": STACK})
            nat_gws.append(nat)
            for idx, priv_sub in enumerate(private_subnets):
                rt = aws.ec2.RouteTable(_name(f"private-rt-{idx}"), vpc_id=vpc.id, routes=[aws.ec2.RouteTableRouteArgs(cidr_block="0.0.0.0/0", nat_gateway_id=nat.id)], tags={"Name": _name(f"private-rt-{idx}")})
                aws.ec2.RouteTableAssociation(_name(f"private-rta-{idx}"), subnet_id=priv_sub.id, route_table_id=rt.id)
                private_route_table_ids.append(rt.id)
        else:
            for i, pub_sub in enumerate(public_subnets):
                eip = aws.ec2.Eip(_name(f"nat-eip-{i}"), vpc=True, tags={"Name": _name(f"nat-eip-{i}")})
                nat_eips.append(eip)
                nat = aws.ec2.NatGateway(_name(f"natgw-{i}"), allocation_id=eip.id, subnet_id=pub_sub.id, tags={"Name": _name(f"natgw-{i}"), "Environment": STACK})
                nat_gws.append(nat)
            for idx, priv_sub in enumerate(private_subnets):
                nat_for = nat_gws[idx % len(nat_gws)]
                rt = aws.ec2.RouteTable(_name(f"private-rt-{idx}"), vpc_id=vpc.id, routes=[aws.ec2.RouteTableRouteArgs(cidr_block="0.0.0.0/0", nat_gateway_id=nat_for.id)], tags={"Name": _name(f"private-rt-{idx}")})
                aws.ec2.RouteTableAssociation(_name(f"private-rta-{idx}"), subnet_id=priv_sub.id, route_table_id=rt.id)
                private_route_table_ids.append(rt.id)
    else:
        for idx, priv_sub in enumerate(private_subnets):
            rt = aws.ec2.RouteTable(_name(f"private-rt-{idx}"), vpc_id=vpc.id, tags={"Name": _name(f"private-rt-{idx}")})
            aws.ec2.RouteTableAssociation(_name(f"private-rta-{idx}"), subnet_id=priv_sub.id, route_table_id=rt.id)
            private_route_table_ids.append(rt.id)

vpc_endpoints: Dict[str, aws.ec2.VpcEndpoint] = {}

if CREATE_VPC_ENDPOINTS:
    gateway_route_table_ids: List[Any] = []
    if not EXISTING_VPC_ID:
        gateway_route_table_ids = [public_rt.id] + private_route_table_ids
    mapping = {
        "ecr.api": f"com.amazonaws.{AWS_REGION}.ecr.api",
        "ecr.dkr": f"com.amazonaws.{AWS_REGION}.ecr.dkr",
        "ssm": f"com.amazonaws.{AWS_REGION}.ssm",
        "sts": f"com.amazonaws.{AWS_REGION}.sts",
        "ec2": f"com.amazonaws.{AWS_REGION}.ec2",
        "secretsmanager": f"com.amazonaws.{AWS_REGION}.secretsmanager",
    }
    for svc in VPC_ENDPOINT_SERVICES:
        svc = svc.strip()
        if svc == "s3":
            if gateway_route_table_ids:
                ep = aws.ec2.VpcEndpoint(
                    _name("vpce-s3"),
                    vpc_id=(vpc.id if not isinstance(vpc, dict) else vpc["id"]),
                    service_name=f"com.amazonaws.{AWS_REGION}.s3",
                    vpc_endpoint_type="Gateway",
                    route_table_ids=gateway_route_table_ids,
                    tags={"Name": _name("vpce-s3")},
                )
                vpc_endpoints["s3"] = ep
        else:
            if svc not in mapping:
                raise pulumi.RunError(f"Unsupported VPC endpoint service shorthand: {svc}")
            ep_sg = aws.ec2.SecurityGroup(
                _name(f"vpce-sg-{svc}"),
                vpc_id=(vpc.id if not isinstance(vpc, dict) else vpc["id"]),
                description=f"SG for VPC Endpoint {svc}",
                ingress=[aws.ec2.SecurityGroupIngressArgs(protocol="-1", from_port=0, to_port=0, cidr_blocks=[VPC_CIDR])],
                egress=[aws.ec2.SecurityGroupEgressArgs(protocol="-1", from_port=0, to_port=0, cidr_blocks=["0.0.0.0/0"])],
            )
            ep = aws.ec2.VpcEndpoint(
                _name(f"vpce-{svc}"),
                vpc_id=(vpc.id if not isinstance(vpc, dict) else vpc["id"]),
                service_name=mapping[svc],
                vpc_endpoint_type="Interface",
                subnet_ids=[s.id for s in private_subnets],
                security_group_ids=[ep_sg.id],
                tags={"Name": _name(f"vpce-{svc}")},
            )
            vpc_endpoints[svc] = ep

flow_log_id: Optional[pulumi.Output[str]] = None
kms_key: Optional[aws.kms.Key] = None
log_bucket: Optional[aws.s3.Bucket] = None
glue_db: Optional[aws.glue.CatalogDatabase] = None
glue_crawler: Optional[aws.glue.Crawler] = None
glue_job: Optional[aws.glue.Job] = None
athena_named_query: Optional[aws.athena.NamedQuery] = None

if (FLOW_LOG_MODE and FLOW_LOG_MODE != "none") or ENABLE_FLOW_LOGS:
    if FLOW_LOG_MODE == "cloudwatch" or (FLOW_LOG_MODE == "" and ENABLE_FLOW_LOGS):
        log_group = aws.cloudwatch.LogGroup(
            _name("vpc-flow-log-group"),
            name=FLOW_LOG_CW_LOG_GROUP,
            retention_in_days=FLOW_LOG_CW_RETENTION_DAYS,
            tags={"Name": _name("vpc-flow-log-group"), "Environment": STACK},
        )
        role = aws.iam.Role(
            _name("flowlog-role"),
            assume_role_policy=json.dumps({"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "vpc-flow-logs.amazonaws.com"}, "Action": "sts:AssumeRole"}]}),
            tags={"Name": _name("flowlog-role")},
        )
        aws.iam.RolePolicyAttachment(
            _name("flowlog-role-CW"),
            role=role.name,
            policy_arn="arn:aws:iam::aws:policy/CloudWatchLogsFullAccess",
        )
        flow = aws.ec2.FlowLog(
            _name("vpc-flow-log"),
            vpc_id=(vpc.id if not isinstance(vpc, dict) else vpc["id"]),
            traffic_type="ALL",
            log_destination=log_group.arn,
            log_destination_type="cloud-watch-logs",
            iam_role_arn=role.arn,
            opts=pulumi.ResourceOptions(depends_on=[log_group, role]),
        )
        flow_log_id = flow.id
    elif FLOW_LOG_MODE == "s3":
        account_id = aws.get_caller_identity().account_id
        if FLOW_LOG_S3_CREATE:
            bucket_name = f"{FLOW_LOG_S3_CREATE_NAME}-{account_id}-{STACK}"
            log_bucket = aws.s3.Bucket(
                _name("vpc-flow-logs-bucket"),
                bucket=bucket_name,
                tags={"Name": _name("vpc-flow-logs-bucket"), "Environment": STACK},
            )
            aws.s3.BucketPublicAccessBlock(
                _name("vpc-flow-logs-bucket-pab"),
                bucket=log_bucket.id,
                block_public_acls=True,
                block_public_policy=True,
                ignore_public_acls=True,
                restrict_public_buckets=True,
            )
            aws.s3.BucketVersioning(
                _name("vpc-flow-logs-bucket-versioning"),
                bucket=log_bucket.id,
                versioning_configuration={"status": "Enabled"},
            )
            if FLOW_LOG_SSE_ALGORITHM.lower() == "aws:kms":
                if FLOW_LOG_KMS_CREATE:
                    kms_key = aws.kms.Key(_name("flowlogs-kms"), description=_name("flowlogs-kms"), tags={"Name": _name("flowlogs-kms"), "Environment": STACK})
                    aws.s3.BucketServerSideEncryptionConfiguration(
                        _name("vpc-flow-logs-bucket-sse"),
                        bucket=log_bucket.id,
                        rules=[{"applyServerSideEncryptionByDefault": {"sseAlgorithm": "aws:kms", "kmsMasterKeyId": kms_key.arn}}],
                    )
                elif FLOW_LOG_KMS_ARN:
                    aws.s3.BucketServerSideEncryptionConfiguration(
                        _name("vpc-flow-logs-bucket-sse"),
                        bucket=log_bucket.id,
                        rules=[{"applyServerSideEncryptionByDefault": {"sseAlgorithm": "aws:kms", "kmsMasterKeyId": FLOW_LOG_KMS_ARN}}],
                    )
                else:
                    raise pulumi.RunError("FLOW_LOG_SSE_ALGORITHM=aws:kms requires FLOW_LOG_KMS_CREATE=true or FLOW_LOG_KMS_ARN set")
            else:
                aws.s3.BucketServerSideEncryptionConfiguration(
                    _name("vpc-flow-logs-bucket-sse-aes"),
                    bucket=log_bucket.id,
                    rules=[{"applyServerSideEncryptionByDefault": {"sseAlgorithm": "AES256"}}],
                )
            if FLOW_LOG_S3_ACCESS_LOGGING and FLOW_LOG_S3_ACCESS_BUCKET:
                access_bucket = aws.s3.Bucket(
                    _name("vpc-flow-logs-access-bucket"),
                    bucket=FLOW_LOG_S3_ACCESS_BUCKET,
                    acl="log-delivery-write",
                    tags={"Name": _name("vpc-flow-logs-access-bucket"), "Environment": STACK},
                )
                aws.s3.BucketLogging(
                    _name("vpc-flow-logs-bucket-logging"),
                    bucket=log_bucket.id,
                    target_bucket=access_bucket.id,
                    target_prefix="access/",
                )
            raw_bucket_output = log_bucket.bucket
            bucket_arn_output = log_bucket.arn
            policy = pulumi.Output.all(bucket_arn_output, account_id).apply(
                lambda args: json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Sid": "AllowPutObject",
                                "Effect": "Allow",
                                "Principal": {"Service": "vpc-flow-logs.amazonaws.com"},
                                "Action": ["s3:PutObject"],
                                "Resource": [f"{args[0]}/{FLOW_LOG_S3_PREFIX.rstrip('/') }/*"],
                                "Condition": {"StringEquals": {"aws:SourceAccount": args[1]}},
                            }
                        ],
                    }
                )
            )
            aws.s3.BucketPolicy(_name("vpc-flow-logs-bucket-policy"), bucket=log_bucket.id, policy=policy)
            dest_arn_output = log_bucket.arn
        else:
            bucket_input = FLOW_LOG_S3_BUCKET
            if bucket_input.startswith("arn:aws:s3:::"):
                dest_arn_output = bucket_input
            else:
                dest_arn_output = f"arn:aws:s3:::{bucket_input}"
            log_bucket = None
        destination_options = {}
        if DESTINATION_FILE_FORMAT:
            destination_options["fileFormat"] = DESTINATION_FILE_FORMAT
        if DESTINATION_PER_HOUR_PARTITION is not None:
            destination_options["perHourPartition"] = DESTINATION_PER_HOUR_PARTITION
        if destination_options:
            flow = aws.ec2.FlowLog(
                _name("vpc-flowlog-s3"),
                vpc_id=(vpc.id if not isinstance(vpc, dict) else vpc["id"]),
                traffic_type="ALL",
                log_destination=dest_arn_output,
                log_destination_type="s3",
                destination_options=destination_options,
                opts=pulumi.ResourceOptions(depends_on=[log_bucket] if log_bucket else None),
            )
        else:
            flow = aws.ec2.FlowLog(
                _name("vpc-flowlog-s3"),
                vpc_id=(vpc.id if not isinstance(vpc, dict) else vpc["id"]),
                traffic_type="ALL",
                log_destination=dest_arn_output,
                log_destination_type="s3",
                opts=pulumi.ResourceOptions(depends_on=[log_bucket] if log_bucket else None),
            )
        flow_log_id = flow.id

if FLOW_LOG_MODE == "s3" and (CREATE_GLUE_CRAWLER or CREATE_GLUE_ETL or CREATE_ATHENA):
    raw_bucket_output = None
    if log_bucket is not None:
        raw_bucket_output = log_bucket.bucket
    else:
        raw_bucket_output = FLOW_LOG_S3_BUCKET

    glue_db = aws.glue.CatalogDatabase(
        _name("vpc_flowlogs_db"),
        name=ATHENA_DB_NAME,
        description=f"Glue DB for VPC flow logs {STACK}",
    )

    glue_role = aws.iam.Role(
        _name("glue-role"),
        assume_role_policy=json.dumps({"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "glue.amazonaws.com"}, "Action": "sts:AssumeRole"}]}),
        tags={"Name": _name("glue-role"), "Environment": STACK},
    )

    s3_read_arn_output = pulumi.Output.from_input(raw_bucket_output).apply(lambda b: f"arn:aws:s3:::{b}/{FLOW_LOG_S3_PREFIX.rstrip('/') }/*")
    s3_bucket_arn_output = pulumi.Output.from_input(raw_bucket_output).apply(lambda b: f"arn:aws:s3:::{b}")
    parquet_prefix = "vpcflowlogs_parquet/"
    parquet_target_arn_output = pulumi.Output.from_input(raw_bucket_output).apply(lambda b: f"arn:aws:s3:::{b}/{parquet_prefix}*")
    parquet_bucket_arn_output = pulumi.Output.from_input(raw_bucket_output).apply(lambda b: f"arn:aws:s3:::{b}")

    def _glue_policy_json(s3_read, s3_bucket, parquet_target, parquet_bucket, kms_arn):
        statements = []
        statements.append({"Effect": "Allow", "Action": ["s3:GetObject", "s3:ListBucket"], "Resource": [s3_read, s3_bucket]})
        statements.append({"Effect": "Allow", "Action": ["s3:PutObject", "s3:PutObjectAcl", "s3:DeleteObject"], "Resource": [parquet_target]})
        statements.append({"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": [parquet_bucket]})
        statements.append({"Effect": "Allow", "Action": ["glue:*", "athena:*", "logs:*"], "Resource": ["*"]})
        if kms_arn:
            statements.append({"Effect": "Allow", "Action": ["kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey*"], "Resource": [kms_arn]})
        return json.dumps({"Version": "2012-10-17", "Statement": statements})

    kms_arn_for_policy = FLOW_LOG_KMS_ARN if FLOW_LOG_KMS_ARN else (kms_key.arn if kms_key is not None else None)
    glue_policy_output = pulumi.Output.all(s3_read_arn_output, s3_bucket_arn_output, parquet_target_arn_output, parquet_bucket_arn_output).apply(lambda args: _glue_policy_json(args[0], args[1], args[2], args[3], kms_arn_for_policy))
    aws.iam.RolePolicy(_name("glue-role-policy"), role=glue_role.id, policy=glue_policy_output)

    if CREATE_GLUE_CRAWLER:
        s3_target = pulumi.Output.from_input(raw_bucket_output).apply(lambda b: f"s3://{b.rstrip('/')}/{FLOW_LOG_S3_PREFIX.lstrip('/') }")
        def _make_crawler(path: str):
            kwargs = {
                "name": _name("vpc-flowlogs-crawler"),
                "role": glue_role.arn,
                "database_name": glue_db.name,
                "s3_targets": [{"path": path}],
                "schema_change_policy": {"update_behavior": "UPDATE_IN_DATABASE", "delete_behavior": "DEPRECATE_IN_DATABASE"},
                "tags": {"Name": _name("vpc-flowlogs-crawler"), "Environment": STACK},
            }
            if GLUE_CRAWLER_SCHEDULE:
                kwargs["schedule"] = GLUE_CRAWLER_SCHEDULE
            return aws.glue.Crawler(_name("vpc-flowlogs-crawler"), **kwargs)

        glue_crawler = s3_target.apply(lambda p: _make_crawler(p))

    if CREATE_GLUE_ETL:
        scripts_bucket = log_bucket if log_bucket is not None else aws.s3.Bucket(_name("glue-scripts-bucket"), acl="private", tags={"Name": _name("glue-scripts-bucket"), "Environment": STACK})

        glue_script = r"""import sys
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from pyspark.sql.functions import from_unixtime, year, month, dayofmonth
from awsglue.job import Job
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'SOURCE_S3_PATH', 'TARGET_S3_PATH', 'REGION', 'ZSTD_COMPRESSION_LEVEL'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
# ensure we set parquet compression to zstd and the zstd level
spark.conf.set("spark.sql.parquet.compression.codec", "zstd")
spark.conf.set("parquet.compression.codec.zstd.level", str(args.get('ZSTD_COMPRESSION_LEVEL', '3')))
df = spark.read.text(args['SOURCE_S3_PATH'])
cols = ['version','account_id','interface_id','srcaddr','dstaddr','srcport','dstport','protocol','packets','bytes','start','end','action','log_status']
df2 = df.selectExpr("split(value,'\\s+') as parts").select([f"parts[{i}] as {cols[i]}" for i in range(len(cols))])
df2 = df2.withColumn('start', df2['start'].cast('long'))
df2 = df2.withColumn('ts', from_unixtime(df2['start']))
df2 = df2.withColumn('year', year('ts')).withColumn('month', month('ts')).withColumn('day', dayofmonth('ts')).withColumn('region', args['REGION'])
df2.write.mode('append').partitionBy('region','year','month','day').parquet(args['TARGET_S3_PATH'])
job.commit()
"""

        script_obj = aws.s3.BucketObject(
            _name("glue-etl-script"),
            bucket=(scripts_bucket.id if isinstance(scripts_bucket, aws.s3.Bucket) else scripts_bucket),
            key=(GLUE_SCRIPT_S3_PREFIX.rstrip("/") + "/" + "vpc_flow_etl.py"),
            content=glue_script,
            content_type="text/x-python",
            acl="private",
        )

        script_location = pulumi.Output.all((scripts_bucket.bucket if isinstance(scripts_bucket, aws.s3.Bucket) else scripts_bucket), script_obj.key).apply(lambda args: f"s3://{args[0]}/{args[1]}")

        glue_job_role = aws.iam.Role(
            _name("glue-job-role"),
            assume_role_policy=json.dumps({"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "glue.amazonaws.com"}, "Action": "sts:AssumeRole"}]}),
            tags={"Name": _name("glue-job-role"), "Environment": STACK},
        )

        def _job_policy_json(script_arn, s3_read, s3_bucket, parquet_target, kms_arn):
            stm = []
            stm.append({"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": [script_arn]})
            stm.append({"Effect": "Allow", "Action": ["s3:GetObject", "s3:ListBucket"], "Resource": [s3_read, s3_bucket]})
            stm.append({"Effect": "Allow", "Action": ["s3:PutObject", "s3:PutObjectAcl", "s3:DeleteObject"], "Resource": [parquet_target]})
            stm.append({"Effect": "Allow", "Action": ["glue:*", "logs:*", "ec2:CreateNetworkInterface", "ec2:DeleteNetworkInterface", "ec2:DescribeNetworkInterfaces"], "Resource": ["*"]})
            if kms_arn:
                stm.append({"Effect": "Allow", "Action": ["kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey*"], "Resource": [kms_arn]})
            return json.dumps({"Version": "2012-10-17", "Statement": stm})

        kms_arn_for_job = FLOW_LOG_KMS_ARN if FLOW_LOG_KMS_ARN else (kms_key.arn if kms_key is not None else None)
        job_policy_output = pulumi.Output.all(script_obj.arn, s3_read_arn_output, s3_bucket_arn_output, parquet_target_arn_output).apply(lambda args: _job_policy_json(args[0], args[1], args[2], args[3], kms_arn_for_job))
        aws.iam.RolePolicy(_name("glue-job-inline-policy"), role=glue_job_role.id, policy=job_policy_output)

        target_parquet_path = pulumi.Output.from_input(raw_bucket_output).apply(lambda b: f"s3://{b.rstrip('/')}/{parquet_prefix}")
        default_args = {
            "--JOB_NAME": _name("vpc-flow-etl-job"),
            "--SOURCE_S3_PATH": pulumi.Output.from_input(raw_bucket_output).apply(lambda b: f"s3://{b.rstrip('/')}/{FLOW_LOG_S3_PREFIX.lstrip('/')}"),
            "--TARGET_S3_PATH": target_parquet_path,
            "--REGION": AWS_REGION,
            "--ZSTD_COMPRESSION_LEVEL": str(ZSTD_COMPRESSION_LEVEL),
        }

        glue_job = aws.glue.Job(
            _name("vpc-flow-etl-job"),
            name=_name("vpc-flow-etl-job"),
            role=glue_job_role.arn,
            glue_version="3.0",
            number_of_workers=GLUE_ETL_DPU,
            command={"name": "glueetl", "pythonVersion": "3", "scriptLocation": script_location},
            default_arguments=default_args,
            max_retries=1,
            tags={"Name": _name("vpc-flow-etl-job"), "Environment": STACK},
        )

        if GLUE_ETL_SCHEDULE:
            glue_trigger = aws.glue.Trigger(
                _name("vpc-flow-etl-trigger"),
                type="SCHEDULED",
                name=_name("vpc-flow-etl-trigger"),
                schedule=GLUE_ETL_SCHEDULE,
                actions=[{"job_name": glue_job.name}],
                tags={"Name": _name("vpc-flow-etl-trigger"), "Environment": STACK},
            )

    if CREATE_ATHENA:
        parquet_location = pulumi.Output.from_input(raw_bucket_output).apply(lambda b: f"s3://{b.rstrip('/')}/{parquet_prefix}")
        create_table_sql = parquet_location.apply(lambda loc: f"""
CREATE EXTERNAL TABLE IF NOT EXISTS {ATHENA_DB_NAME}.{ATHENA_TABLE_NAME} (
  version int,
  account_id string,
  interface_id string,
  srcaddr string,
  dstaddr string,
  srcport int,
  dstport int,
  protocol int,
  packets bigint,
  bytes bigint,
  start bigint,
  end bigint,
  action string,
  log_status string
)
PARTITIONED BY (region string, year int, month int, day int)
STORED AS PARQUET
LOCATION '{loc}';
""")
        # pulumi aws NamedQuery expects 'query' property (not 'query_string')
        athena_named_query = aws.athena.NamedQuery(
            _name("athena-create-parquet-table"),
            database=ATHENA_DB_NAME,
            query=create_table_sql,
            name=_name("athena-create-parquet-table"),
        )

pulumi.export("vpc_id", (vpc.id if not isinstance(vpc, dict) else vpc["id"]))
pulumi.export("public_subnet_ids", [s.id for s in public_subnets] if public_subnets else EXISTING_PUBLIC_SUBNET_IDS)
pulumi.export("private_subnet_ids", [s.id for s in private_subnets] if private_subnets else EXISTING_PRIVATE_SUBNET_IDS)
pulumi.export("route_table_ids", {"private": private_route_table_ids if private_route_table_ids else []})
pulumi.export("vpc_endpoint_ids", {k: v.id for k, v in vpc_endpoints.items()})
pulumi.export("flow_log_id", flow_log_id or "")
pulumi.export("flow_log_mode", FLOW_LOG_MODE)
pulumi.export("flow_log_s3_bucket", (log_bucket.bucket if log_bucket is not None else FLOW_LOG_S3_BUCKET))
pulumi.export("glue_db", (glue_db.name if glue_db is not None else None))
pulumi.export("glue_crawler", (glue_crawler.name if hasattr(glue_crawler, "name") else (glue_crawler if glue_crawler is not None else None)))
pulumi.export("glue_job", (glue_job.name if glue_job is not None else None))
pulumi.export("athena_named_query", (athena_named_query.id if athena_named_query is not None else None))
# export chosen ZSTD compression level
pulumi.export("zstd_compression_level", ZSTD_COMPRESSION_LEVEL)
