"""
  python3 utils/s3_bucket.py --create
  python3 utils/s3_bucket.py --delete
"""

import os
import sys
import uuid
import boto3
from botocore.exceptions import ClientError

def create_local_folders(base_path="."):
    folders = [
        "data/raw/",
        "backups/dbs/qdrant/",
        "backups/dbs/arrangodb/"
    ]
    for folder in folders:
        path = os.path.join(base_path, folder)
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Created local folder: {path}")
        except OSError as e:
            print(f"Error creating local folder {path}: {e}", file=sys.stderr)

def get_bool_env(var_name: str, default: str = "true") -> bool:
    val = os.getenv(var_name, default).strip().lower()
    if val not in ("true", "false"):
        print(f"ERROR: {var_name} must be 'true' or 'false', got '{val}'", file=sys.stderr)
        sys.exit(1)
    return val == "true"

def create_s3_bucket():
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region     = os.environ.get("AWS_REGION")
    if not (access_key and secret_key and region):
        print("ERROR: Missing AWS credentials or region. "
              "Ensure AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION are set.", file=sys.stderr)
        sys.exit(1)

    block_public = get_bool_env("S3_BLOCK_PUBLIC_ACCESS", "true")
    bucket_name = os.environ.get(
        "S3_BUCKET",
        f"e2e-rag-bucket-{uuid.uuid4().hex[:8]}"
    )

    print(f"Creating S3 bucket: {bucket_name} in region: {region}")
    s3 = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region}
            )
        print("Bucket created or already exists.")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "BucketAlreadyOwnedByYou":
            print("Bucket already exists and is owned by you.")
        else:
            print(f"ERROR: Bucket creation failed: {e}", file=sys.stderr)
            sys.exit(1)

    pab_config = {
        "BlockPublicAcls":       block_public,
        "IgnorePublicAcls":      block_public,
        "BlockPublicPolicy":     block_public,
        "RestrictPublicBuckets": block_public,
    }
    try:
        s3.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration=pab_config
        )
        state = "blocked" if block_public else "unblocked"
        print(f"Public access {state} (S3_BLOCK_PUBLIC_ACCESS={block_public}).")
    except ClientError as e:
        print(f"WARNING: Could not set public access block: {e}", file=sys.stderr)

    return bucket_name, s3

def create_s3_folders(bucket_name, s3_client):
    prefixes = [
        "data/raw/",
        "pulumi/",
        "backups/dbs/qdrant/",
        "backups/dbs/arrangodb/"
    ]
    for prefix in prefixes:
        try:
            s3_client.put_object(Bucket=bucket_name, Key=prefix)
            print(f"Created S3 prefix: {prefix}")
        except ClientError as e:
            print(f"Error creating prefix {prefix}: {e}", file=sys.stderr)

def delete_s3_bucket():
    bucket_name = os.environ.get("S3_BUCKET")
    if not bucket_name:
        print("ERROR: Set S3_BUCKET env variable before deleting.", file=sys.stderr)
        sys.exit(1)

    print(f"Purging all objects and versions from bucket: {bucket_name}")
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    # Delete all object versions
    try:
        bucket.object_versions.delete()
    except ClientError as e:
        code = getattr(e, 'response', {}).get('Error', {}).get('Code', '')
        if code == "NoSuchBucket":
            print(f"WARNING: Bucket {bucket_name} does not exist.")
            return
        else:
            print(f"ERROR: Failed to delete object versions: {e}", file=sys.stderr)
            sys.exit(1)

    # Delete the bucket itself
    try:
        bucket.delete()
        print(f"S3 bucket {bucket_name} deleted.")
    except ClientError as e:
        print(f"ERROR: Failed to delete bucket {bucket_name}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if "--create" in sys.argv:
        print("=== Creating local folders ===")
        create_local_folders()
        print("\n=== Creating S3 bucket ===")
        bucket_name, s3_client = create_s3_bucket()

        # Export S3_BUCKET for downstream steps (in this process only)
        os.environ["S3_BUCKET"] = bucket_name
        print(f"\nS3_BUCKET set to: {bucket_name}")

        print("\n=== Creating S3 prefixes ===")
        create_s3_folders(bucket_name, s3_client)
        print("All done.")
    elif "--delete" in sys.argv:
        print("=== Deleting S3 bucket ===")
        delete_s3_bucket()
    else:
        print("Usage: python3 utils/s3_bucket.py --create | --delete", file=sys.stderr)
        sys.exit(1)
