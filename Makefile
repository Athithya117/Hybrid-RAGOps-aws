.PHONY: s3 delete-s3 tree clean lc
s3:
	python3 utils/s3_bucket.py --create
	aws s3 ls "s3://$S3_BUCKET/" --recursive | head -n 100

delete-s3:
	python3 utils/s3_bucket.py --create
	aws s3 ls

tree:
	tree -a -I '.git|data|.venv|repos|production-stack|raw_data|.venv2|archive|tmp.md|docs|models|tmp|raw|chunked'

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + && find . -name "*.pyc" -delete
	clear




