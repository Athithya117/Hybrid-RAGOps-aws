push:
	git add .
	git commit -m "new"
	git push origin main --force

s3:
	python3 utils/create_s3.py
	aws s3 ls "$S3_BUCKET_NAME/data/raw/" --recursive | head -n 25

install-o:
	bash utils/bootstrap-on-prem.sh

tree-o:
	tree -a -I '.git|.venv|aws|docs|tmp|utils|aws-kustomization.yaml|raw|chunked'

backup:
	zip "$$(basename $$PWD)_$$(date +%Y%m%d_%H%M%S).zip" \
	$$(find . -type f -size -100M \
		! -path "*/.git/*" \
		! -path "*/.venv/*" \
		! -path "*/tmp/*" \
		! -path "models/*")


