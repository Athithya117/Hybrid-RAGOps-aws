import json,botocore, boto3
import pulumi
from pulumi_aws import s3
from pulumi_command import local

def nuke_bucket(name):
    bucket = s3.Bucket(name, force_destroy=True)
    script = f"""
aws s3api list-object-versions --bucket {name} --output json | jq -r '.Versions[]?, .DeleteMarkers[]? | [.Key,.VersionId] | @tsv' | while IFS=$'\t' read -r k v; do aws s3api delete-object --bucket {name} --key "$k" --version-id "$v"; done || true
"""
    local.Command(f"nuke-{name}", create=script, delete=script, triggers=[name])
    return bucket

bucket_a = nuke_bucket("force-test-a")
bucket_b = nuke_bucket("force-test-b")
bucket_c = nuke_bucket("force-test-c")

pulumi.export("buckets", [bucket_a.id, bucket_b.id, bucket_c.id])
