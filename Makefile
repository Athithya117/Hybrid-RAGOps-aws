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
	  --overwrite-existing && \
	echo "current context: " && \
	kubectl config current-context && \
	kubectl get nodes && \
	kubectl get pods -A

delete-aks-context:
	@CLUSTER_NAME="$$(jq -r '.aks_cluster_name' infra/pulumi_azure/pulumi-outputs.json)" ; \
	RG_NAME="$$AZURE_RESOURCE_GROUP_NAME" ; \
	kubectl config delete-context "$$CLUSTER_NAME" 2>/dev/null || true ; \
	kubectl config delete-cluster "$$CLUSTER_NAME" 2>/dev/null || true ; \
	kubectl config unset "users.clusterUser_$${RG_NAME}_$${CLUSTER_NAME}" 2>/dev/null || true ; \
	kubectl config get-contexts

set-kind-context:
	kubectl config use-context kind-rag8s-local


rollout-qdrant:
	python3 infra/generators/qdrant_cluster.py --rollout
rollout-qdrant-with-flux:
	python3 infra/generators/qdrant_cluster.py --rollout --flux


delete-qdrant:
	kubectl delete ns qdrant


dense-image:
	bash apps/dense/test_and_push_dense.sh

rollout-dense:
	python3 infra/generators/dense.py --rollout

delete-dense:
	python3 infra/generators/dense.py --delete


sparse-image:
	bash apps/sparse/test_and_push_sparse.sh

rollout-sparse:
	python3 infra/generators/sparse.py --rollout

delete-sparse:
	python3 infra/generators/sparse.py --delete


base-index-image:
	bash apps/index/build_and_push_base_image.sh

index-image:
	bash apps/index/build_and_push_image.sh

rollout-indexing-cronjob:
	python3 infra/generators/indexing_cronjob.py --rollout

delete-indexing-cronjob:
	kubectl delete ns indexing


frontend-image:
	bash apps/inference/frontend/build_and_push_frontend.sh

retrieval-image:
	bash apps/inference/retrieval/test_and_push_retriever.sh


rollout-reranker:
	python3 infra/generators/reranker.py --rollout

reranker-image:
	bash apps/reranker/test_and_push_reranker.sh

delete-reranker:
	python3 infra/generators/reranker.py --delete
	
setup-flux:
	curl -s https://fluxcd.io/install.sh | sudo FLUX_VERSION=2.7.5 bash || true
	python3 infra/setup/setup_fluxcd.py --auto-push



delete-flux:
	kubectl delete ns flux-system --grace-period=0 --force --wait=false || true
	kubectl get crd | grep fluxcd.io | awk '{print $$1}' | xargs -r kubectl delete crd --grace-period=0 --force || true
	kubectl delete crd gitrepositories.source.toolkit.fluxcd.io helmrepositories.source.toolkit.fluxcd.io --grace-period=0 --force || true
	kubectl get ns flux-system -o json 2>/dev/null | jq 'del(.spec.finalizers)' | kubectl replace --raw "/api/v1/namespaces/flux-system/finalize" -f - || true

inspect-flux:
	tail -f infra/manifests/flux-system/setup_fluxcd.log

flux-status:
	flux check && flux get kustomizations -n flux-system


rollout-retriever:
	python3 infra/generators/retriever.py --rollout

rollout-frontend:
	python3 infra/generators/frontend_auth.py --rollout --confirm

rollout-cloudflared-agents:
	python3 infra/generators/cloudflared.py --rollout --replicas $${CLOUDFLARED_TUNNEL_REPLICAS} --namespace inference

rollout-clickhouse:
	python3 infra/generators/clickhouse.py --rollout

rollout-vector:
	python3 infra/generators/vector_logger.py --rollout

rollout-vm:
	python3 infra/generators/monitoring.py --rollout

rollout-runbooks:
	bash infra/base_infra/az_runbooks.sh

rollout-alert-manager:
	python3 infra/generators/alerting.py --rollout

rollout-dashboards:
	python3 infra/generators/dashboards.py --rollout


delete-retriever:
	python3 infra/generators/retriever.py --delete || true

delete-frontend:
	python3 infra/generators/frontend_auth.py --delete --confirm || true

delete-cloudflared-agents:
	python3 infra/generators/cloudflared.py --delete --namespace inference || true

delete-clickhouse:
	python3 infra/generators/clickhouse.py --delete --confirm || true

delete-vector:
	python3 infra/generators/vector_logger.py --delete || true
 
delete-vm:
	python3 infra/generators/monitoring.py --delete || true

delete-runbooks:
	bash infra/base_infra/az_runbooks.sh --delete || true

delete-alert-manager:
	python3 infra/generators/alerting.py --delete --confirm || true

delete-dashboards:
	python3 infra/generators/dashboards.py --delete || true

rollout-models: rollout-dense rollout-sparse rollout-reranker
rollout-inference-svcs: rollout-retriever rollout-frontend
rollout-observability-stack: rollout-prometheus rollout-clickhouse rollout-vector rollout-dashboards


run-indexing-cronjob-kind:
	@echo "[make fix-kind-dns] invoking utils/fix_kind_cluster_dns.sh"
	@chmod +x utils/fix_kind_cluster_dns.sh || true
	@utils/fix_kind_cluster_dns.sh --timeout 60
	sleep 5
	python3 infra/generators/indexing_cronjob.py --rollout
	python3 infra/runners/run_indexing_cronjob_kind.py --wait-for-running --wait-running-timeout 120


fix-kind-dns:
	@echo "[make fix-kind-dns] invoking utils/fix_kind_cluster_dns.sh"
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
	rm -rf ~/.cloudflared && \
	rm -f ~/.config/rag/secrets.env && \
	unset CLOUDFLARE_TUNNEL_TOKEN && \
	unset CLOUDFLARE_TUNNEL_CREDENTIALS_B64 && \
	unset CLOUDFLARE_TUNNEL_NAME


test-vector-connection:
	make fix-dns
	bash infra/tests/test_vector_clickhouse_connection.sh

test-retriever:
	make rollout-vm
	make rollout-retriever
	bash infra/tests/monitoring/test_retriever.sh || true


lc:
	bash utils/lc.sh

tree:
	tree -a -I '.git|.venv|repos|__pycache__|venv|commands.sh|raw_data|.venv-pulumi|.venv2|archive|tmp.md|docs|models|tmp|raw|chunked'


docker-login:
	echo "$$DOCKER_PASSWORD" | docker login -u "$$DOCKER_USERNAME" --password-stdin

push:
	git add .
	git commit -m "new"
	git push origin main --force


clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.log" ! -path "./.git/*" -delete
	find . -type f -name "*.pulumi-logs" ! -path "./.git/*" -delete
	clear

fix-dns: fix-kind-dns

# ClickHouse convenience targets
CH_NS ?= observability

.PHONY: ch-shell ch-query

ch-shell:
	@POD="$$(kubectl -n $(CH_NS) get pods -l app=clickhouse -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || \
	kubectl -n $(CH_NS) get pods --no-headers 2>/dev/null | awk '/clickhouse|ch-single/ {print $$1; exit}')" ; \
	if [ -z "$$POD" ]; then \
		echo "[error] no ClickHouse pod found in namespace $(CH_NS)"; \
		kubectl -n $(CH_NS) get pods || true; \
		exit 1; \
	fi ; \
	echo "[info] connecting to ClickHouse pod $$POD in namespace $(CH_NS)"; \
	kubectl -n $(CH_NS) exec -it $$POD -- clickhouse-client

ch-query:
	@QUERY="$${QUERY}" ; \
	if [ -z "$$QUERY" ]; then \
		echo "Usage: make ch-query QUERY=\"SELECT ...\""; exit 2; \
	fi ; \
	POD="$$(kubectl -n $(CH_NS) get pods -l app=clickhouse -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || \
	kubectl -n $(CH_NS) get pods --no-headers 2>/dev/null | awk '/clickhouse|ch-single/ {print $$1; exit}')" ; \
	if [ -z "$$POD" ]; then \
		echo "[error] no ClickHouse pod found in namespace $(CH_NS)"; \
		kubectl -n $(CH_NS) get pods || true; \
		exit 1; \
	fi ; \
	echo "[info] running query on $$POD in namespace $(CH_NS)"; \
	echo "$$QUERY" | kubectl -n $(CH_NS) exec -i $$POD -- clickhouse-client --multiquery
