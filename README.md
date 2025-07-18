


export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_REGION=ap-south-1
export S3_BUCKET_NAME=e2e-rag-system16
export PYTHONPATH=$(pwd)


aws s3 ls $S3_BUCKET_NAME/data/raw/
