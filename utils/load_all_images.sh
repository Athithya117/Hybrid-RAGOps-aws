#!/usr/bin/env bash
# sleep 400
CLUSTER_NAME="${CLUSTER_NAME:-rag8s-local}"
echo "[INFO] Preloading images safely (post-bootstrap)..."
for node in $(kind get nodes --name "${CLUSTER_NAME}"); do
  echo "  → Loading into ${node}"
  docker exec "$node" bash -c 'ctr version >/dev/null 2>&1' || { echo "    [WARN] containerd not ready on $node — skipping"; continue; }
  for IMAGE in \
    docker.io/qdrant/qdrant:v1.16.0 \
    docker.io/athithya5354/dense:v1 \
    docker.io/athithya5354/sparse:v1 \
    docker.io/athithya5354/reranker:v1 \
    docker.io/athithya5354/retrieval:v23 \
    docker.io/athithya5354/frontend-and-auth:v12\
    docker.io/athithya5354/indexing_pipeline_cpu:v16
  do
    echo "    checking $IMAGE on $node..."
    docker exec "$node" sh -c "ctr -n k8s.io images ls | grep -q -- '$IMAGE'" >/dev/null 2>&1 || {
      echo "    pulling $IMAGE..."
      docker exec "$node" ctr -n k8s.io images pull "$IMAGE" || echo "    [WARN] failed pulling $IMAGE on $node"
    }
  done
done
