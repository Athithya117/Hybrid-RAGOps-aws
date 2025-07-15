export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=


export AWS_REGION=ap-south-1
export S3_BUCKET_NAME=e2e-rag-system16
export SYNC_RAY_OBJECT_STORE_MEMORY=4294967296  # 4GB in bytes
export RAY_INCLUDE_DASHBOARD=false
export RAY_NUM_CPUS=8
export RAY_NUM_GPUS=0
export RAY_CONCURRENT_FILES=4
export PYTHONPATH=$(pwd)


aws s3 ls $S3_BUCKET_NAME/data/raw/


