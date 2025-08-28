s3:
	python3 utils/create_s3.py
	aws s3 ls "s3://$S3_BUCKET/" --recursive | head -n 100

tree:
	tree -a -I '.git|data|.venv|repos|production-stack|raw_data|.venv2|archive|tmp.md|docs|models|tmp|raw|chunked'

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + && find . -name "*.pyc" -delete
	clear

lc:
	bash utils/lc.sh

nuke-local-workspace:
	kind get clusters | xargs -n1 kind delete cluster --name || true
	sudo rm -f /usr/local/bin/kind || true
	docker system prune --volumes --force || true 
	sleep 3
	clear

