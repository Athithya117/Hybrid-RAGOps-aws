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

tree-o:
	tree -a -I '.git|.venv|aws|docs|tmp|aws-kustomization.yaml|raw|chunked'

backup:
	zip "$$(basename $$PWD)_$$(date +%Y%m%d_%H%M%S).zip" \
	$$(find . -type f -size -100M \
		! -path "*/.git/*" \
		! -path "*/.venv/*" \
		! -path "*/tmp/*" \
		! -path "models/*")



clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + && find . -name "*.pyc" -delete

