







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
	tree -a -I '.git|.venv|.venv2|aws|docs|utils|models|tmp|raw|chunked'

.PHONY: backup

backup:
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	find . -type f -size -100M \
	  ! -path "*/.git/*" \
	  ! -path "*/.venv/*" \
	  ! -path "*/.venv2/*" \
	  ! -path "*/backups/*" \
	  ! -path "*/tmp/*" \
	  ! -path "models/*" \
	| zip "$$(basename $$PWD)_$$timestamp.zip" -@


.PHONY: local-llm create-local-llm-secret local-llm-helm-chart check-local-llm delete-local-llm

IMAGE_NAME=ctransformers-qwen3-cpu
FULL_IMAGE=ghcr.io/$(shell echo $(GITHUB_USER) | tr A-Z a-z)/$(IMAGE_NAME):$(TAG)

create-local-llm-secret:
	@echo "Creating GHCR image pull secret..."
	kubectl create secret docker-registry ghcr-secret \
		--docker-server=ghcr.io \
		--docker-username=$(GITHUB_USER) \
		--docker-password=$(GHCR_PAT) \
		--docker-email=$(EMAIL) \
		--dry-run=client -o yaml | kubectl apply -f -

local-llm-helm-chart:
	@echo "[1/3] Logging into GHCR..."
	echo "$(GHCR_PAT)" | docker login ghcr.io -u "$(GITHUB_USER)" --password-stdin

	@echo "[2/3] Building and pushing Docker image: $(FULL_IMAGE)..."
	DOCKER_BUILDKIT=1 \
	docker build \
		--secret id=hf_token,env=HF_TOKEN \
		--cache-to=type=inline \
		--cache-from=type=registry,ref=$(FULL_IMAGE) \
		-t $(FULL_IMAGE) \
		-f infra/ctransformers-chart/docker/Dockerfile \
		infra/ctransformers-chart/docker/

	docker push $(FULL_IMAGE)

	@echo "[3/3] Deploying Helm chart with image: $(FULL_IMAGE)..."
	cd infra/ctransformers-chart && \
	helm upgrade --install ctr-cpu . \
		--wait --timeout 300s \
		--set image.repository=ghcr.io/$(shell echo $(GITHUB_USER) | tr A-Z a-z)/$(IMAGE_NAME) \
		--set image.tag=$(TAG) \
		--set replicas.workerMax=$(LOCAL_QWEN3_MAX_WORKERS)

deploy-local-llm: create-local-llm-secret local-llm-helm-chart

check-local-llm:
	@echo "[*] Checking LLM deployment readiness..."
	@HEAD=$$(kubectl get pod -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}'); \
	echo "Ray head pod: $$HEAD"; \
	kubectl get pods -l ray.io/cluster=ctr-cpu-service; \
	kubectl get rayservice ctr-cpu-service -o yaml | yq '.status'; \
	echo "[*] Port-forwarding Ray Serve and waiting for /health..."; \
	kubectl port-forward pod/$$HEAD 8000:8000 8265:8265 & \
	PORT_PID=$$!; \
	for i in $$(seq 1 10); do \
		curl -sf http://localhost:8000/health >/dev/null && break; \
		echo "Waiting for Ray Serve /health ($$i)..."; \
		sleep 3; \
	done; \
	curl -s -X POST http://localhost:8000/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"messages":[{"role":"user","content":"What is Qwen3?"}]}' | jq; \
	kill $$PORT_PID

delete-local-llm:
	@echo "Cleaning up local LLM deployment..."
	kubectl delete rayservice ctr-cpu-service --ignore-not-found
	kubectl delete raycluster ctr-cpu-service-raycluster --ignore-not-found
	kubectl delete pod -l ray.io/node-type=worker --ignore-not-found
	kubectl delete pod -l ray.io/node-type=head --ignore-not-found
	helm uninstall ctr-cpu || true

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

