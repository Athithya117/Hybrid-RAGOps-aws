push:
	git pull
	git add .
	git commit -m "new"
	git push origin main

s3:
	python3 utils/create_s3.py
	aws s3 ls "$S3_BUCKET_NAME/data/raw/" --recursive | head -n 25

install:
	sudo bash utils/bootstrap-amd.sh
