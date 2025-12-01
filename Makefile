.PHONY: s3 delete-s3 tree clean lc push

push:
	git config --global http.postBuffer 524288000
	git config --global http.maxRequestBuffer 1000M
	git config --global http.lowSpeedLimit 0
	git config --global http.lowSpeedTime 999999
	git add .
	git commit -m "new"
	git push origin main --force

s3:
	python3 utils/s3_buckets.py --create
	aws s3 ls "s3://$S3_BUCKET/" --recursive | head -n 100

delete-s3:
	python3 utils/s3_buckets.py --create
	aws s3 ls

lc:
	bash utils/lc.sh

tree:
	tree -a -I '.git|.venv|repos|__pycache__|venv|commands.sh|production-stack|raw_data|.venv2|archive|tmp.md|docs|models|tmp|raw|chunked'

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + && find . -name "*.pyc" -delete
	clear

docker-login:
	echo "$$DOCKER_PASSWORD" | docker login -u "$$DOCKER_USERNAME" --password-stdin


