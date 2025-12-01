"""
USAGE:
  python3 utils/s3_buckets.py --create --region $AWS_REGION
  python3 utils/s3_buckets.py --delete $S3_BUCKET --region "$AWS_REGION" 
  python3 utils/s3_buckets.py --delete $BACKUP_S3_BUCKET --region "$AWS_REGION" 
HEADLESS:
  python3 utils/s3_buckets.py --create --region $AWS_REGION
  python3 utils/s3_buckets.py --delete $S3_BUCKET --region "$AWS_REGION" --force 
  python3 utils/s3_buckets.py --delete $BACKUP_S3_BUCKET --region "$AWS_REGION" --force
"""

import os
import sys
import argparse
import logging
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("s3-bucket")

def getenv_bool(name, default):
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("1", "true", "yes", "y"):
        return True
    if v in ("0", "false", "no", "n"):
        return False
    raise SystemExit(f"env {name} must be boolean-like, got '{v}'")

AWS_REGION = os.getenv("AWS_REGION", "").strip() or None
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET", "").strip() or None
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "data/raw/").strip()
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX", "data/chunked/").strip()
BACKUP_S3_BUCKET = os.getenv("BACKUP_S3_BUCKET", "").strip() or None
S3_BLOCK_PUBLIC_ACCESS = getenv_bool("S3_BLOCK_PUBLIC_ACCESS", True)
BACKUP_RETENTION_DAYS = int(os.getenv("BACKUP_RETENTION_DAYS") or 0)
BACKUP_S3_PATH_PREFIX = os.getenv("BACKUP_S3_PATH_PREFIX", "weaviate-backups/").rstrip("/") + "/"
DEFAULT_PREFIXES = [S3_RAW_PREFIX, S3_CHUNKED_PREFIX, "pulumi/"]

def require_env_and_creds():
    missing = []
    if not AWS_ACCESS_KEY_ID:
        missing.append("AWS_ACCESS_KEY_ID")
    if not AWS_SECRET_ACCESS_KEY:
        missing.append("AWS_SECRET_ACCESS_KEY")
    if not AWS_REGION:
        missing.append("AWS_REGION")
    if not S3_BUCKET:
        missing.append("S3_BUCKET")
    if not BACKUP_S3_BUCKET:
        missing.append("BACKUP_S3_BUCKET")
    if missing:
        log.error("Missing required environment variables: %s", ", ".join(missing))
        sys.exit(2)

def make_s3_client(region_name=None):
    kwargs = {}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
    kwargs["region_name"] = region_name or AWS_REGION
    return boto3.client("s3", **kwargs)

def make_s3_resource(region_name=None):
    kwargs = {}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
    kwargs["region_name"] = region_name or AWS_REGION
    return boto3.resource("s3", **kwargs)

def get_bucket_info(bucket_name):
    client = make_s3_client(region_name=AWS_REGION)
    try:
        client.head_bucket(Bucket=bucket_name)
        try:
            loc = client.get_bucket_location(Bucket=bucket_name).get("LocationConstraint")
            region = loc or "us-east-1"
        except Exception:
            region = AWS_REGION
        return True, True, region
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchBucket"):
            return False, False, None
        if code in ("403", "AccessDenied", "AllAccessDisabled"):
            try:
                loc = client.get_bucket_location(Bucket=bucket_name).get("LocationConstraint")
                region = loc or "us-east-1"
            except Exception:
                region = None
            return True, False, region
        return False, False, None

def create_bucket_if_missing(bucket_name, region):
    client = make_s3_client(region_name=region)
    exists, owned, existing_region = get_bucket_info(bucket_name)
    if exists:
        if not owned:
            log.error("Bucket '%s' already exists but is not owned/accessible by current credentials; aborting.", bucket_name)
            sys.exit(3)
        log.info("Bucket %s already exists and is owned by you (region=%s)", bucket_name, existing_region or region)
        return existing_region or region
    try:
        if (region or AWS_REGION) in (None, "us-east-1"):
            client.create_bucket(Bucket=bucket_name)
        else:
            client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": region or AWS_REGION})
        log.info("Created bucket %s in region %s without versioning", bucket_name, region or AWS_REGION)
        return region or AWS_REGION
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "BucketAlreadyOwnedByYou":
            log.info("Bucket %s already owned by you", bucket_name)
            loc = client.get_bucket_location(Bucket=bucket_name).get("LocationConstraint")
            return loc or AWS_REGION or "us-east-1"
        if code == "BucketAlreadyExists":
            log.error("Bucket name '%s' is taken globally by another account; choose a different bucket name.", bucket_name)
            sys.exit(4)
        log.exception("Failed to create bucket %s: %s", bucket_name, e)
        sys.exit(5)

def configure_bucket(bucket_name, region, set_lifecycle=False, lifecycle_days=0, prefixes=None):
    client = make_s3_client(region_name=region)
    try:
        pab = {
            "BlockPublicAcls": S3_BLOCK_PUBLIC_ACCESS,
            "IgnorePublicAcls": S3_BLOCK_PUBLIC_ACCESS,
            "BlockPublicPolicy": S3_BLOCK_PUBLIC_ACCESS,
            "RestrictPublicBuckets": S3_BLOCK_PUBLIC_ACCESS
        }
        client.put_public_access_block(Bucket=bucket_name, PublicAccessBlockConfiguration=pab)
        log.info("Configured public access block for bucket %s (blocked=%s)", bucket_name, S3_BLOCK_PUBLIC_ACCESS)
    except ClientError as e:
        log.error("Failed public access block on %s: %s", bucket_name, e)
        sys.exit(6)
    try:
        client.put_bucket_encryption(
            Bucket=bucket_name,
            ServerSideEncryptionConfiguration={"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]}
        )
        log.info("Enabled AES256 default encryption for bucket %s", bucket_name)
    except ClientError as e:
        log.warning("Bucket encryption failed for %s: %s", bucket_name, e)
    if set_lifecycle and lifecycle_days and lifecycle_days > 0:
        lifecycle_rules = [{
            "ID": "expire-after",
            "Status": "Enabled",
            "Filter": {"Prefix": ""},
            "Expiration": {"Days": lifecycle_days},
            "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7}
        }]
        try:
            client.put_bucket_lifecycle_configuration(Bucket=bucket_name, LifecycleConfiguration={"Rules": lifecycle_rules})
            log.info("Configured lifecycle on bucket %s: expire after %d days", bucket_name, lifecycle_days)
        except ClientError as e:
            log.warning("Lifecycle config failed for %s: %s", bucket_name, e)
    if prefixes:
        for prefix in prefixes:
            try:
                client.put_object(Bucket=bucket_name, Key=prefix)
                log.info("Created prefix %s in bucket %s", prefix, bucket_name)
            except ClientError as e:
                log.warning("Failed to create prefix '%s' in %s: %s", prefix, bucket_name, e)

def ensure_buckets_strict(data_bucket, backup_bucket, region):
    data_region = create_bucket_if_missing(data_bucket, region)
    configure_bucket(data_bucket, data_region, set_lifecycle=False, lifecycle_days=0, prefixes=DEFAULT_PREFIXES)
    backup_region = create_bucket_if_missing(backup_bucket, region)
    configure_bucket(backup_bucket, backup_region, set_lifecycle=(BACKUP_RETENTION_DAYS > 0), lifecycle_days=BACKUP_RETENTION_DAYS, prefixes=[BACKUP_S3_PATH_PREFIX])
    return data_bucket, backup_bucket

def purge_and_delete_bucket(bucket_name, region=None):
    s3res = make_s3_resource(region_name=region)
    bucket = s3res.Bucket(bucket_name)
    try:
        bucket.object_versions.delete()
    except ClientError as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        if code == "NoSuchBucket":
            log.warning("Bucket %s does not exist.", bucket_name)
            return
        log.exception("Failed deleting object versions from %s: %s", bucket_name, e)
        sys.exit(7)
    try:
        bucket.delete()
        log.info("Deleted bucket %s", bucket_name)
    except ClientError as e:
        log.exception("Failed to delete bucket %s: %s", bucket_name, e)
        sys.exit(8)

def parse_args():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--create", action="store_true")
    grp.add_argument("--delete", metavar="BUCKET")
    p.add_argument("--region", "-r")
    p.add_argument("--force", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    if args.delete:
        bucket_to_delete = args.delete
        if not args.force:
            confirm = input(f"ARE YOU SURE you want to DELETE bucket '{bucket_to_delete}' and ALL of its contents? Type 'yes' to confirm: ")
            if confirm.strip().lower() != "yes":
                log.info("Aborting deletion")
                return
        purge_and_delete_bucket(bucket_to_delete, args.region or AWS_REGION)
        return
    if args.create:
        require_env_and_creds()
        region = args.region or AWS_REGION
        if not region:
            log.error("AWS region not provided")
            sys.exit(2)
        try:
            data, backup = ensure_buckets_strict(S3_BUCKET, BACKUP_S3_BUCKET, region)
            print(data, backup)
        except SystemExit:
            raise
        except Exception as e:
            log.exception("Bucket creation/configuration failed: %s", e)
            sys.exit(9)

if __name__ == "__main__":
    main()
