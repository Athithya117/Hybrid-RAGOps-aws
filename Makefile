.PHONY: s3 delete-s3 tree clean lc push

push:
	git add .
	git commit -m "new"
	git push origin main

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



SHELL := /bin/bash -lc
.PHONY: host-qdrant-locally host-qdrant-locally-foreground write-env stop-qdrant

QDRANT_PRIVATE_IP ?= 127.0.0.1
QDRANT_PORT ?= 6333
QDRANT_IMAGE ?= qdrant/qdrant:v1.15.4
QDRANT_HOST_DATA ?= /workspace/qdrant/data
QDRANT_HOST_BACKUPS ?= /workspace/qdrant/backups

host-qdrant-locally-foreground:
	sudo mkdir -p $(QDRANT_HOST_DATA) $(QDRANT_HOST_BACKUPS)/snapshots && \
	sudo chown -R 1000:1000 $(QDRANT_HOST_DATA) $(QDRANT_HOST_BACKUPS) && \
	docker rm -f qdrant_local 2>/dev/null || true && \
	docker run --rm --name qdrant_local -p $(QDRANT_PRIVATE_IP):$(QDRANT_PORT):$(QDRANT_PORT) -v $(QDRANT_HOST_DATA):/qdrant/storage -v $(QDRANT_HOST_BACKUPS):/qdrant/backups -e QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/backups/snapshots $(QDRANT_IMAGE)

host-qdrant-locally:
	sudo mkdir -p $(QDRANT_HOST_DATA) $(QDRANT_HOST_BACKUPS)/snapshots && \
	sudo chown -R 1000:1000 $(QDRANT_HOST_DATA) $(QDRANT_HOST_BACKUPS) && \
	docker rm -f qdrant_local 2>/dev/null || true && \
	docker run -d --name qdrant_local -p $(QDRANT_PRIVATE_IP):$(QDRANT_PORT):$(QDRANT_PORT) -v $(QDRANT_HOST_DATA):/qdrant/storage -v $(QDRANT_HOST_BACKUPS):/qdrant/backups -e QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/backups/snapshots $(QDRANT_IMAGE)
