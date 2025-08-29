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

lc:
	bash utils/lc.sh

deploy-local-embedder:
	docker pull ghcr.io/huggingface/text-embeddings-inference:cpu-1.7
	kind load docker-image ghcr.io/huggingface/text-embeddings-inference:cpu-1.7 --name rag8s-local
	kubectl port-forward svc/gte-embedder 8080:80 -n inference &
	curl -X POST http://127.0.0.1:8080/embed -H 'Content-Type: application/json'
	
