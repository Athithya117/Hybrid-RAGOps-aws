import boto3

def generate_presigned_urls(bucket, key, expires=3600):
    s3 = boto3.client("s3")

    inline_url = s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": bucket,
            "Key": key,
            "ResponseContentDisposition": f'inline; filename="{key.split("/")[-1]}"',
            "ResponseContentType": "application/pdf"
        },
        ExpiresIn=expires
    )

    download_url = s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": bucket,
            "Key": key,
            "ResponseContentDisposition": f'attachment; filename="{key.split("/")[-1]}"',
            "ResponseContentType": "application/pdf"
        },
        ExpiresIn=expires
    )

    return inline_url, download_url


bucket = "e2e-rag-system"
key = "data/raw/2504.03656v1.pdf"

view_url, download_url = generate_presigned_urls(bucket, key)
print("View in browser:\n", view_url)
print("\nForce download:\n", download_url)
