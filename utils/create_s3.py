import os
import boto3
import uuid

def create_local_folders(base_path="."):
    folders = [
        "data/raw/",
        "data/parsed/",
        "data/prechunked/",
        "data/latechunked/",
        "pulumi/",
        "backups/qdrant/",
        "backups/arrangodb/"
    ]
    for folder in folders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
        print(f"Created folder: {path}")

def create_s3_bucket():
    # Load environment variables
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_REGION", "us-east-1")
    bucket_name = os.environ.get("S3_BUCKET_NAME", f"e2e-rag-bucket-{uuid.uuid4().hex[:8]}")

    # Validate required values
    if not access_key or not secret_key:
        raise Exception("Missing AWS credentials. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")

    print(f"Creating S3 bucket: {bucket_name} in region: {region}")

    # Create S3 client
    s3 = boto3.client(
        's3',
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    # Create the bucket
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        print("Bucket created successfully.")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print("Bucket already exists and is owned by you.")
    except Exception as e:
        raise Exception(f"Bucket creation failed: {e}")

    # Block all public access
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

    return bucket_name

if __name__ == "__main__":
    print("Creating local folders...")
    create_local_folders()
    print("Local folders created.\n")

    bucket = create_s3_bucket()
    print(f"\nS3 Bucket ready: {bucket}")
