"""
  python3 utils/s3_bucket.py --create
  python3 utils/s3_bucket.py --delete
"""

import os
import sys
import uuid
import boto3


def create_local_folders(base_path="."):
    folders = [
        "data/raw/",
        "backups/dbs/qdrant/",
        "backups/dbs/arrangodb/"
    ]
    for folder in folders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
        print(f"Created local folder: {path}")


def create_s3_bucket():
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region     = os.environ.get("AWS_REGION")
    # Use single env var S3_BUCKET, generate if not provided
    bucket_name = os.environ.get(
        "S3_BUCKET",
        f"e2e-rag-bucket-{uuid.uuid4().hex[:8]}"
    )

    if not access_key or not secret_key or not region:
        raise Exception(
            "Missing AWS credentials or region. "
            "Ensure AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION are set."
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
        print("Bucket created successfully.")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print("Bucket already exists and is owned by you.")
    except Exception as e:
        raise Exception(f"Bucket creation failed: {e}")

    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        }
    )
    print("Public access blocked.")
    return bucket_name, s3


def create_s3_folders(bucket_name, s3_client):
    prefixes = [
        "data/raw/",
        "pulumi/",
        "backups/dbs/qdrant/",
        "backups/dbs/arrangodb/"
    ]
    for prefix in prefixes:
        # zero-byte object to simulate folder
        s3_client.put_object(Bucket=bucket_name, Key=prefix)
        print(f"Created S3 prefix: {prefix}")


def delete_s3_bucket():
    bucket_name = os.environ.get("S3_BUCKET")
    if not bucket_name:
        raise Exception("Set S3_BUCKET env variable before deleting.")

    print(f"Purging all objects and versions from bucket: {bucket_name}")
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    # delete all versions and markers
    bucket.object_versions.delete()
    # delete bucket
    bucket.delete()
    print(f"S3 bucket {bucket_name} deleted.")


if __name__ == "__main__":
    if "--create" in sys.argv:
        print("Creating local folders...")
        create_local_folders()
        print("Local folders created.\n")

        bucket_name, s3_client = create_s3_bucket()
        # Export S3_BUCKET so that subsequent steps can use it
        os.environ["S3_BUCKET"] = bucket_name
        print(f"\nS3 Bucket ready: {bucket_name}")

        print("Creating S3 folder-like prefixes...")
        create_s3_folders(bucket_name, s3_client)
        print("S3 prefixes created.")

    elif "--delete" in sys.argv:
        delete_s3_bucket()

    else:
        print("Usage: python3 utils/s3_bucket.py --create | --delete")
        sys.exit(1)
