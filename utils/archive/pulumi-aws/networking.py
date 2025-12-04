import os
import stat
import pulumi
import pulumi_aws as aws
import pulumi_tls as tls
stack = pulumi.get_stack()
project = pulumi.get_project()
region = os.environ.get("AWS_REGION") or aws.config.region or "us-east-1"
vpc_cidr = os.environ.get("VPC_CIDR", "10.0.0.0/16")
public_subnet_cidrs = os.environ.get("PUBLIC_SUBNET_CIDRS", "10.0.1.0/24,10.0.2.0/24").split(",")
my_ip = os.environ.get("MY_SSH_CIDR", "0.0.0.0/0")
tags_common = {"Project": project, "Stack": stack}
vpc = aws.ec2.Vpc("ray-vpc", cidr_block=vpc_cidr, enable_dns_hostnames=True, enable_dns_support=True, tags={**tags_common, "Name": "ray-vpc"})
igw = aws.ec2.InternetGateway("ray-igw", vpc_id=vpc.id, tags={**tags_common, "Name": "ray-igw"})
azs = aws.get_availability_zones().names
public_subnets = []
for i, cidr in enumerate(public_subnet_cidrs):
    az = azs[i % len(azs)]
    subnet = aws.ec2.Subnet(f"ray-public-{i}", vpc_id=vpc.id, cidr_block=cidr.strip(), map_public_ip_on_launch=True, availability_zone=az, tags={**tags_common, "Name": f"ray-public-{i}"})
    public_subnets.append(subnet)
public_rt = aws.ec2.RouteTable("ray-public-rt", vpc_id=vpc.id, routes=[{"cidr_block": "0.0.0.0/0", "gateway_id": igw.id}], tags={**tags_common, "Name": "ray-public-rt"})
for idx, sn in enumerate(public_subnets):
    aws.ec2.RouteTableAssociation(f"ray-public-{idx}-assoc", subnet_id=sn.id, route_table_id=public_rt.id)
ray_sg = aws.ec2.SecurityGroup("ray-sg", vpc_id=vpc.id, description="ray-sg", ingress=[{"protocol": "tcp", "from_port": 22, "to_port": 22, "cidr_blocks": [my_ip]}, {"protocol": "tcp", "from_port": 6379, "to_port": 6379, "self": True}, {"protocol": "tcp", "from_port": 8265, "to_port": 8265, "self": True}, {"protocol": "tcp", "from_port": 10001, "to_port": 10001, "self": True}], egress=[{"protocol": "-1", "from_port": 0, "to_port": 0, "cidr_blocks": ["0.0.0.0/0"]}], tags={**tags_common, "Name": "ray-sg"})
weaviate_sg = aws.ec2.SecurityGroup("weaviate-sg", vpc_id=vpc.id, description="weaviate-sg", ingress=[], egress=[{"protocol": "-1", "from_port": 0, "to_port": 0, "cidr_blocks": ["0.0.0.0/0"]}], tags={**tags_common, "Name": "weaviate-sg"})
aws.ec2.SecurityGroupRule("weaviate-allow-ray-8080", type="ingress", from_port=8080, to_port=8080, protocol="tcp", security_group_id=weaviate_sg.id, source_security_group_id=ray_sg.id)
aws.ec2.SecurityGroupRule("weaviate-allow-ray-grpc", type="ingress", from_port=50051, to_port=50051, protocol="tcp", security_group_id=weaviate_sg.id, source_security_group_id=ray_sg.id)
aws.ec2.SecurityGroupRule("weaviate-allow-ray-ssh", type="ingress", from_port=22, to_port=22, protocol="tcp", security_group_id=weaviate_sg.id, source_security_group_id=ray_sg.id)
tls_key = tls.PrivateKey("ec2-key", algorithm="RSA", rsa_bits=4096)
aws_key = aws.ec2.KeyPair("rag-ec2-key", key_name=f"{project}-{stack}-ec2-key", public_key=tls_key.public_key_openssh, tags={**tags_common, "Name": f"{project}-{stack}-ec2-key"})
def _write_pem_to_disk(pem_str: str) -> str:
    script_dir = os.path.dirname(__file__) or os.getcwd()
    filename = os.path.join(script_dir, f"{project}-{stack}-ec2-key.pem")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    tmp = filename + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(pem_str)
    os.replace(tmp, filename)
    os.chmod(filename, stat.S_IRUSR | stat.S_IWUSR)
    return filename
pem_path = tls_key.private_key_pem.apply(lambda pk: _write_pem_to_disk(pk))
pulumi.export("vpc_id", vpc.id)
pulumi.export("public_subnet_ids", [s.id for s in public_subnets])
pulumi.export("public_subnet_azs", [s.availability_zone for s in public_subnets])
pulumi.export("ray_security_group_id", ray_sg.id)
pulumi.export("weaviate_security_group_id", weaviate_sg.id)
pulumi.export("ec2_key_name", aws_key.key_name)
pulumi.export("ec2_key_path", pulumi.Output.secret(pem_path))
for idx, sn in enumerate(public_subnets):
    pulumi.export(f"public_subnet_id_{idx}", sn.id)
    pulumi.export(f"public_subnet_az_{idx}", sn.availability_zone)
    sn.id.apply(lambda id_value, i=idx: print(f'export PULUMI_PUBLIC_SUBNET_ID_{i}=\"{id_value}\"'))
    sn.availability_zone.apply(lambda az_value, i=idx: print(f'export PULUMI_PUBLIC_SUBNET_AZ_{i}=\"{az_value}\"'))
vpc.id.apply(lambda val: print(f'export PULUMI_VPC_ID=\"{val}\"'))
ray_sg.id.apply(lambda val: print(f'export PULUMI_RAY_SG_ID=\"{val}\"'))
weaviate_sg.id.apply(lambda val: print(f'export PULUMI_WEAVIATE_SG_ID=\"{val}\"'))
aws_key.key_name.apply(lambda val: print(f'export PULUMI_EC2_KEY_NAME=\"{val}\"'))
pem_path.apply(lambda val: print(f'export PULUMI_EC2_KEY_PATH=\"{val}\"'))
pulumi.export("public_subnet_count", len(public_subnets))
