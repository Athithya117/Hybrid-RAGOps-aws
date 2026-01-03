create-sa:
	python3 infra/base_infra/storage_acc.py --create

delete-sa:
	python3 infra/base_infra/storage_acc.py --delete

pulumi-up:
	bash infra/pulumi_azure/run.sh --create || true

pulumi-destroy:
	bash infra/pulumi_azure/run.sh --delete || true

pulumi-preview:
	bash infra/pulumi_azure/run.sh --preview || true



set-aks-context:
	az aks get-credentials \
	  -g "$$AZURE_RESOURCE_GROUP_NAME" \
	  -n "$$(jq -r '.aks_cluster_name' infra/pulumi_azure/pulumi-outputs.json)" \
	  --overwrite-existing && kubectl get nodes

set-kind-context:
	kubectl config use-context kind-rag8s-local



index-image:
	bash apps/index/build_and_push_image.sh

frontend-image:
	bash apps/inference/frontend/build_and_push_frontend.sh

retrieval-image:
	bash apps/inference/retrieval/test_and_push_retriever.sh

sparse-image:
	bash apps/sparse/test_and_push_sparse.sh

dense-image:
	bash apps/dense/test_and_push_dense.sh

reranker-image:
	bash apps/reranker/test_and_push_reranker.sh


setup-flux:
	python3 infra/scripts/setup_fluxcd.py --auto-push

inspect-flux:
	tail -f infra/manifests/flux-system/setup_fluxcd.log

flux-status:
	flux check && flux get kustomizations -n flux-system



deploy-dense:
	python3 infra/generators/dense.py --apply

deploy-sparse:
	python3 infra/generators/sparse.py --apply

deploy-reranker:
	python3 infra/generators/reranker.py --apply

deploy-qdrant:
	bash infra/generators/qdrant_cluster.sh --apply

deploy-retriever:
	bash infra/generators/retriever.sh --apply

deploy-frontend:
	python3 infra/generators/frontend_auth.py --apply --confirm

deploy-cloudflared:
	python3 infra/generators/cloudflared.py --apply --replicas $${CLOUDFLARED_TUNNEL_REPLICAS} --namespace inference

deploy-clickhouse:
	python3 infra/generators/clickhouse.py --apply

deploy-vector:
	python3 infra/generators/vector_logger.py --apply

deploy-vm:
	bash infra/generators/monitoring.sh --apply

deploy-runbooks:
	bash infra/base_infra/az_runbooks.sh

deploy-alert-manager:
	python3 infra/generators/alerting.py --apply


deploy-models: deploy-dense deploy-sparse deploy-reranker
deploy-inference-svcs: deploy-retriever deploy-frontend
deploy-observability-stack:	deploy-prometheus deploy-clickhouse deploy-vector deploy-dashboards


run-indexing-cronjob-kind:
	@echo "[make fix-dns] invoking utils/fix_kind_cluster_dns.sh"
	@chmod +x utils/fix_kind_cluster_dns.sh || true
	@utils/fix_kind_cluster_dns.sh --timeout 60
	sleep 5
	python3 infra/generators/indexing_cronjob.py --delete
	python3 infra/generators/indexing_cronjob.py --apply
	python3 infra/runners/run_indexing_cronjob.py --wait-for-running --wait-running-timeout 120

fix-dns:
	@echo "[make fix-dns] invoking utils/fix_kind_cluster_dns.sh"
	@chmod +x utils/fix_kind_cluster_dns.sh || true
	@utils/fix_kind_cluster_dns.sh --timeout 60


PY ?= python3
CONTROL := infra/runners/backup_and_restore.sh

qdrant-backup:
	@bash $(CONTROL) backup

qdrant-restore:
	@bash $(CONTROL) restore

cloudflare-setup:
	bash infra/setup/cloudflared.sh 

cloudflare-logout:
	rm -rf ~/.cloudflared && rm -f ~/.config/rag/secrets.env && unset CLOUDFLARE_TUNNEL_TOKEN && unset CLOUDFLARE_TUNNEL_CREDENTIALS_B64 && unset CLOUDFLARE_TUNNEL_NAME

test-vector-connection:
	make fix-dns
	bash infra/tests/test_vector_clickhouse_connection.sh

test-retriever:
	make deploy-vm
	make deploy-retriever
	bash infra/tests/monitoring/test_retriever.sh || true

lc:
	bash utils/lc.sh

tree:
	tree -a -I '.git|.venv|repos|__pycache__|venv|commands.sh|raw_data|.venv-pulumi|.venv2|archive|tmp.md|docs|models|tmp|raw|chunked'

docker-login:
	echo "$$DOCKER_PASSWORD" | docker login -u "$$DOCKER_USERNAME" --password-stdin

push:
	git config --global http.postBuffer 524288000
	git config --global http.maxRequestBuffer 1000M
	git config --global http.lowSpeedLimit 0
	git config --global http.lowSpeedTime 999999
	git add .
	git commit -m "new"
	git push origin main --force

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.log" ! -path "./.git/*" -delete
	find . -type f -name "*.pulumi-logs" ! -path "./.git/*" -delete
	clear
