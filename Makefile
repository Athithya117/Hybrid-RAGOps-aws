pull:
	git add .
	git commit -m "new"
	git config pull.rebase false
	git pull

push:
	git add .
	git commit -m "new"
	git push origin main --force

s3:
	python3 utils/create_s3.py
	aws s3 ls "s3://$S3_BUCKET/" --recursive | head -n 100

tree:
	tree -a -I '.git|data|.venv|raw_data|.venv2|aws|docs|utils|models|tmp|raw|chunked'

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

