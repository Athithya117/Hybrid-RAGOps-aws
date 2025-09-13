# networking.py
"""
Pulumi networking stack for RAG platform (VPC + Security Groups).
- Uses environment variables only (no Pulumi.<stack>.yaml required).
- Creates a VPC, public subnets, public route table (IGW).
- Creates Ray SG and Qdrant SG.
- Allows Ray SG -> Qdrant on port 6333 and SSH (22) for debugging.
- Exports Pulumi outputs and prints shell-friendly `export ...` lines
  (these prints are emitted during `pulumi up` and can be captured).
"""

import os
import pulumi
import pulumi_aws as aws

# Config / env
config = pulumi.Config()
stack = pulumi.get_stack()
project = pulumi.get_project()

region = os.environ.get("AWS_REGION", aws.config.region or "us-east-1")
pulumi.log.info(f"Using region: {region}")

vpc_cidr = os.environ.get("VPC_CIDR", "10.0.0.0/16")
public_subnet_cidrs = os.environ.get(
    "PUBLIC_SUBNET_CIDRS", "10.0.1.0/24,10.0.2.0/24"
).split(",")
# Operator SSH CIDR (single deterministic entry)
my_ip = os.environ.get("MY_SSH_CIDR", "203.0.113.42/32")

tags_common = {"Project": project, "Stack": stack}

# VPC (with increased timeouts)
vpc = aws.ec2.Vpc(
    "ray-vpc",
    cidr_block=vpc_cidr,
    enable_dns_hostnames=True,
    enable_dns_support=True,
    tags={**tags_common, "Name": "ray-vpc"},
    opts=pulumi.ResourceOptions(custom_timeouts=vpc_timeouts),
)

# Internet Gateway (public networking only)
igw = aws.ec2.InternetGateway(
    "ray-igw",
    vpc_id=vpc.id,
    tags={**tags_common, "Name": "ray-igw"},
)

# Availability zones (resolve once)
azs = aws.get_availability_zones().names

# Public subnets (one per provided CIDR)
public_subnets = []
for i, cidr in enumerate(public_subnet_cidrs):
    az = azs[i % len(azs)]
    subnet = aws.ec2.Subnet(
        f"ray-public-{i}",
        vpc_id=vpc.id,
        cidr_block=cidr.strip(),
        map_public_ip_on_launch=True,  # public subnet (per request: avoid private subnets)
        availability_zone=az,
        tags={**tags_common, "Name": f"ray-public-{i}"},
    )
    public_subnets.append(subnet)

# Public Route Table and default route to IGW
public_rt = aws.ec2.RouteTable(
    "ray-public-rt",
    vpc_id=vpc.id,
    routes=[{"cidr_block": "0.0.0.0/0", "gateway_id": igw.id}],
    tags={**tags_common, "Name": "ray-public-rt"},
)

# Associate the public route table with each public subnet
for sn in public_subnets:
    aws.ec2.RouteTableAssociation(
        f"{sn._name}-assoc", subnet_id=sn.id, route_table_id=public_rt.id
    )

# Ray Security Group (indexing & inference workers)
ray_sg = aws.ec2.SecurityGroup(
    "ray-sg",
    vpc_id=vpc.id,
    description="Security group for Ray workers (indexing & inference)",
    ingress=[
        # operator SSH (restricted to single CIDR)
        {"protocol": "tcp", "from_port": 22, "to_port": 22, "cidr_blocks": [my_ip]},
        # intra-Ray cluster communications (allow from same SG)
        {"protocol": "tcp", "from_port": 6379, "to_port": 6379, "self": True},   # example redis
        {"protocol": "tcp", "from_port": 8265, "to_port": 8265, "self": True},   # Ray dashboard
        {"protocol": "tcp", "from_port": 10001, "to_port": 10001, "self": True}, # custom/worker port
    ],
    egress=[{"protocol": "-1", "from_port": 0, "to_port": 0, "cidr_blocks": ["0.0.0.0/0"]}],
    tags={**tags_common, "Name": "ray-sg"},
)

# Qdrant Security Group (restrict ingress to Ray SG)
qdrant_sg = aws.ec2.SecurityGroup(
    "qdrant-sg",
    vpc_id=vpc.id,
    description="Security group for Qdrant EC2 (allow port 6333 and SSH only from Ray SG)",
    ingress=[],  # explicit rules below referencing ray_sg
    egress=[{"protocol": "-1", "from_port": 0, "to_port": 0, "cidr_blocks": ["0.0.0.0/0"]}],
    tags={**tags_common, "Name": "qdrant-sg"},
)

# Allow Ray SG to access Qdrant on port 6333 (vector DB)
aws.ec2.SecurityGroupRule(
    "qdrant-allow-ray-6333",
    type="ingress",
    from_port=6333,
    to_port=6333,
    protocol="tcp",
    security_group_id=qdrant_sg.id,
    source_security_group_id=ray_sg.id,
    description="Allow Ray cluster access to Qdrant (port 6333)",
)

# Allow Ray SG to reach Qdrant over SSH for debugging (explicitly enabled)
aws.ec2.SecurityGroupRule(
    "qdrant-allow-ray-ssh",
    type="ingress",
    from_port=22,
    to_port=22,
    protocol="tcp",
    security_group_id=qdrant_sg.id,
    source_security_group_id=ray_sg.id,
    description="Allow SSH from Ray SG for debugging",
)

# Pulumi exports (standard outputs)
pulumi.export("vpc_id", vpc.id)
pulumi.export("public_subnet_ids", [s.id for s in public_subnets])
pulumi.export("public_subnet_azs", [s.availability_zone for s in public_subnets])
pulumi.export("ray_security_group_id", ray_sg.id)
pulumi.export("qdrant_security_group_id", qdrant_sg.id)

# Additionally print shell-friendly `export` lines during pulumi run.
# These prints are executed as the Outputs are resolved during the update/apply.
vpc.id.apply(lambda val: print(f'export PULUMI_VPC_ID="{val}"'))

pulumi.Output.all(*[s.id for s in public_subnets]).apply(
    lambda ids: print(f'export PULUMI_PUBLIC_SUBNET_IDS="{",".join(ids)}"')
)

pulumi.Output.all(*[s.availability_zone for s in public_subnets]).apply(
    lambda azs: print(f'export PULUMI_PUBLIC_SUBNET_AZS="{",".join(azs)}"')
)

ray_sg.id.apply(lambda val: print(f'export PULUMI_RAY_SG_ID="{val}"'))
qdrant_sg.id.apply(lambda val: print(f'export PULUMI_QDRANT_SG_ID="{val}"'))
