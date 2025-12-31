
make deploy-qdrant


python3 infra/generators/qdrant_cluster.py --delete && python3 infra/generators/qdrant_cluster.py --apply
python3 infra/generators/retriever.py --delete --confirm && python3 infra/generators/retriever.py --apply --confirm

python3 infra/generators/monitoring_alerts.py --delete --confirm && python3 infra/generators/monitoring_alerts.py --apply 



python3 infra/generators/clickhouse.py --delete --confirm || true && python3 infra/generators/clickhouse.py --apply && \
kubectl -n clickhouse logs -f -l app.kubernetes.io/name=clickhouse --all-containers=true



kubectl -n inference logs -f -l app.kubernetes.io/name=retrieval --all-containers=true


kubectl -n inference logs -f -l app.kubernetes.io/name=frontend-deployment --all-containers=true

kubectl -n inference logs -f -l app.kubernetes.io/name=cloudflared --all-containers=true

kubectl -n observability port-forward svc/kube-prom-stack-kube-prome-prometheus 9090:9090 &

kubectl describe servicemonitors qdrant-servicemonitor retrieval-servicemonitor -n observability

kubectl -n observability port-forward svc/kube-prom-stack-kube-prome-prometheus 9090:9090 &
