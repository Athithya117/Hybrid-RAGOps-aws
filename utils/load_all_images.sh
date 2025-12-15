#!/usr/bin/env bash
# sleep 400
CLUSTER_NAME="${CLUSTER_NAME:-rag8s-local}"
echo "[INFO] Preloading images safely (post-bootstrap)..."
for node in $(kind get nodes --name "${CLUSTER_NAME}"); do
  echo "  → Loading into ${node}"
  docker exec "$node" bash -c 'ctr version >/dev/null 2>&1' || { echo "    [WARN] containerd not ready on $node — skipping"; continue; }
  for IMAGE in \
    docker.io/qdrant/qdrant:v1.16.0 \
    docker.io/athithya5354/dense:amd64-arm64-v1 \
    docker.io/athithya5354/sparse:amd64-arm64-v2 \
    docker.io/athithya5354/reranker:amd64-arm64-v1 \
    docker.io/athithya5354/retrieval:amd64-arm64-v2 \
    docker.io/athithya5354/frontend-and-auth:v5 \
    docker.io/athithya5354/indexing_pipeline_cpu:v11@sha256:1dd7b8fe3bb4536ac1739819d3d831e6132d20a6553c1a373140c9f423394af7
  do
    echo "    checking $IMAGE on $node..."
    docker exec "$node" sh -c "ctr -n k8s.io images ls | grep -q -- '$IMAGE'" >/dev/null 2>&1 || {
      echo "    pulling $IMAGE..."
      docker exec "$node" ctr -n k8s.io images pull "$IMAGE" || echo "    [WARN] failed pulling $IMAGE on $node"
    }
  done
done

