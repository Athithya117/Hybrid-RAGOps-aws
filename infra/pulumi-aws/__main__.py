import os
import pulumi
import pulumi_aws as aws

config = pulumi.Config()
stack = pulumi.get_stack()
project = pulumi.get_project()

region = os.environ.get("AWS_REGION", aws.config.region or "us-east-1")
pulumi.log.info(f"Using region: {region}")

vpc_cidr = os.environ.get("VPC_CIDR", "10.0.0.0/16")
public_subnet_cidrs = os.environ.get("PUBLIC_SUBNET_CIDRS", "10.0.1.0/24,10.0.2.0/24").split(",")
my_ip = os.environ.get("MY_SSH_CIDR", "203.0.113.42/32")
tags_common = {
    "Project": project,
    "Stack": stack,
}

# VPC
vpc = aws.ec2.Vpc(
    "ray-vpc",
    cidr_block=vpc_cidr,
    enable_dns_hostnames=True,
    enable_dns_support=True,
    tags={**tags_common, "Name": "ray-vpc"},
)

igw = aws.ec2.InternetGateway(
    "ray-igw",
    vpc_id=vpc.id,
    tags={**tags_common, "Name": "ray-igw"},
)

public_subnets = []
for i, cidr in enumerate(public_subnet_cidrs):
    az = f"{region}{chr(ord('a') + i)}"
    subnet = aws.ec2.Subnet(
        f"ray-public-{i}",
        vpc_id=vpc.id,
        cidr_block=cidr,
        map_public_ip_on_launch=True,
        availability_zone=az,
        tags={**tags_common, "Name": f"ray-public-{i}"},
    )
    public_subnets.append(subnet)

# Public route table
public_rt = aws.ec2.RouteTable(
    "ray-public-rt",
    vpc_id=vpc.id,
    routes=[{"cidr_block": "0.0.0.0/0", "gateway_id": igw.id}],
    tags={**tags_common, "Name": "ray-public-rt"},
)

for sn in public_subnets:
    aws.ec2.RouteTableAssociation(
        f"{sn._name}-assoc", subnet_id=sn.id, route_table_id=public_rt.id
    )


ray_sg = aws.ec2.SecurityGroup(
    "ray-sg",
    vpc_id=vpc.id,
    description="Security group for Ray cluster",
    ingress=[
        {"protocol": "tcp", "from_port": 22, "to_port": 22, "cidr_blocks": [my_ip]},
        {"protocol": "tcp", "from_port": 6379, "to_port": 6379, "self": True},
        {"protocol": "tcp", "from_port": 8265, "to_port": 8265, "self": True},
        {"protocol": "tcp", "from_port": 10001, "to_port": 10001, "self": True},
    ],
    egress=[
        {"protocol": "-1", "from_port": 0, "to_port": 0, "cidr_blocks": ["0.0.0.0/0"]}
    ],
    tags={**tags_common, "Name": "ray-sg"},
)

# Exports
pulumi.export("vpc_id", vpc.id)
pulumi.export("public_subnet_ids", [s.id for s in public_subnets])
pulumi.export("security_group_id", ray_sg.id)
