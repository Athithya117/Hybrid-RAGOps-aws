#!/usr/bin/env bash
# infra/inference/dynamic-values/dynamic-kubeai-embedder-values.final.sh
# Ensures cache profile, (optionally) prunes other embedder-managed models,
# installs/updates kubeai Helm chart, applies the Model CR, waits, port-forwards and runs test.
set -euo pipefail

# -------------------------
# Configurable env vars (override before running)
# -------------------------
NAMESPACE=${NAMESPACE:-inference}
HELM_RELEASE=${HELM_RELEASE:-kubeai}
HELM_CHART=${HELM_CHART:-kubeai/kubeai}
HELM_REPO_NAME=${HELM_REPO_NAME:-kubeai}
HELM_REPO_URL=${HELM_REPO_URL:-https://charts.kubeai.org}

EMBEDDER=${EMBEDDER:-gte-modernbert-base-onnx-int8}
EMBEDDER_URL=${EMBEDDER_URL:-hf://RAG8s/gte-modernbert-base-onnx-int8}
EMBEDDER_RESOURCE_PROFILE=${EMBEDDER_RESOURCE_PROFILE:-"cpu"}   # a profile name or arbitrary string; match your resourceProfiles
EMBEDDER_MIN_REPLICAS=${EMBEDDER_MIN_REPLICAS:-1}
EMBEDDER_MAX_REPLICAS=${EMBEDDER_MAX_REPLICAS:-2}
ONNX_FILE_PATH=${ONNX_FILE_PATH:-model.onnx}

# If true, delete other models labelled kubeai-embedder-managed=true (except current name)
PRUNE=${PRUNE:-false}

# Cache/profile config
CACHE_PROFILE_NAME=${CACHE_PROFILE_NAME:-default}
# By default we create a minimal empty profile. Replace contents if you want shared storage etc.
# Keep indentation exactly as YAML requires below.
CACHE_PROFILE_SPEC="${CACHE_PROFILE_NAME}: {}"

# Timeouts and ports
WAIT_PODS_TIMEOUT=${WAIT_PODS_TIMEOUT:-300}
API_POLL_TIMEOUT=${API_POLL_TIMEOUT:-120}
PF_LOCAL_PORT=${PF_LOCAL_PORT:-8081}
PF_SVC_PORT=${PF_SVC_PORT:-80}

# -------------------------
# helpers
# -------------------------
log(){ printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"; }
die(){ echo "ERROR: $*" >&2; exit 1; }

# add helm repo if needed
if ! helm repo list | awk '{print $1}' | grep -qx "${HELM_REPO_NAME}"; then
  log "Adding helm repo ${HELM_REPO_NAME} -> ${HELM_REPO_URL}"
  helm repo add "${HELM_REPO_NAME}" "${HELM_REPO_URL}"
fi
helm repo update >/dev/null 2>&1 || true

# 1) Install/upgrade kubeai with cacheProfiles set
log "Installing/upgrading Helm release ${HELM_RELEASE} (namespace ${NAMESPACE}) with cacheProfiles.${CACHE_PROFILE_NAME}..."
cat <<EOF | helm upgrade --install "${HELM_RELEASE}" "${HELM_CHART}" --namespace "${NAMESPACE}" --create-namespace -f - --wait --timeout 5m
# minimal values override ensuring cache profile exists
cacheProfiles:
  ${CACHE_PROFILE_SPEC}
# helpful resourceProfiles defaults for small local clusters
resourceProfiles:
  cpu:
    imageName: cpu
    requests:
      cpu: "1"
      memory: 2Gi
EOF

# 2) Optionally prune other embedder-managed models
if [ "${PRUNE}" = "true" ] || [ "${PRUNE}" = "True" ]; then
  log "PRUNE=true: deleting other kubeai-embedder-managed models (except ${EMBEDDER})"
  mapfile -t _models < <(kubectl -n "${NAMESPACE}" get model -l kubeai-embedder-managed=true -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)
  for m in "${_models[@]:-}"; do
    [ -z "$m" ] && continue
    if [ "$m" != "${EMBEDDER}" ]; then
      log "Deleting model ${m}"
      kubectl -n "${NAMESPACE}" delete model "${m}" --ignore-not-found || true
    fi
  done
fi

# 3) Apply the Model CR
log "Applying Model ${EMBEDDER}..."
kubectl -n "${NAMESPACE}" apply -f - <<EOF
apiVersion: kubeai.org/v1
kind: Model
metadata:
  name: ${EMBEDDER}
  labels:
    kubeai-embedder-managed: "true"
spec:
  features: ["TextEmbedding"]
  owner: RAG8s
  url: "${EMBEDDER_URL}"
  engine: Infinity
  resourceProfile: "${EMBEDDER_RESOURCE_PROFILE}"
  cacheProfile: "${CACHE_PROFILE_NAME}"
  minReplicas: ${EMBEDDER_MIN_REPLICAS}
  maxReplicas: ${EMBEDDER_MAX_REPLICAS}
  args:
    - --file-name=${ONNX_FILE_PATH}
EOF

# ensure controller attempts to create replicas
kubectl -n "${NAMESPACE}" patch model "${EMBEDDER}" --type=merge -p "{\"spec\":{\"minReplicas\":${EMBEDDER_MIN_REPLICAS}}}" || true
kubectl -n "${NAMESPACE}" scale model "${EMBEDDER}" --replicas=${EMBEDDER_MIN_REPLICAS} || true

# 4) Wait for model pod to be Ready (with diagnostics on failure)
LABEL="kubeai.k8s.io/model=${EMBEDDER}"
log "Waiting up to ${WAIT_PODS_TIMEOUT}s for pod(s) with ${LABEL} to become Ready..."
if ! kubectl -n "${NAMESPACE}" wait --for=condition=Ready pod -l "${LABEL}" --timeout="${WAIT_PODS_TIMEOUT}s"; then
  echo "ERROR: model pods did not become Ready within ${WAIT_PODS_TIMEOUT}s" >&2
  kubectl -n "${NAMESPACE}" get pods -o wide || true
  kubectl -n "${NAMESPACE}" describe model "${EMBEDDER}" || true
  kubectl -n "${NAMESPACE}" logs -l kubeai.k8s.io/model="${EMBEDDER}" --tail=400 || true
  kubectl -n "${NAMESPACE}" logs deploy/"${HELM_RELEASE}" --tail=400 || true
  die "model not Ready; see diagnostics above"
fi
log "Model pods Ready."

# 5) Port-forward KubeAI service and wait for API
# kill any existing port-forward on that port
if command -v lsof >/dev/null 2>&1; then
  EXISTING=$(lsof -ti :"${PF_LOCAL_PORT}" || true)
  if [ -n "$EXISTING" ]; then
    log "Killing existing processes on port ${PF_LOCAL_PORT}"
    kill $EXISTING || true
  fi
fi

log "Starting port-forward svc/${HELM_RELEASE} ${PF_LOCAL_PORT}:${PF_SVC_PORT} ..."
kubectl -n "${NAMESPACE}" port-forward svc/"${HELM_RELEASE}" "${PF_LOCAL_PORT}:${PF_SVC_PORT}" >/dev/null 2>&1 &
PF_PID=$!
trap 'kill ${PF_PID} 2>/dev/null || true' EXIT
sleep 1

# poll API root
END=$((SECONDS + API_POLL_TIMEOUT))
API_OK=0
while [ $SECONDS -le $END ]; do
  if curl -sS --max-time 3 "http://localhost:${PF_LOCAL_PORT}/openai/v1/" >/dev/null 2>&1; then
    API_OK=1
    break
  fi
  sleep 2
done

if [ "${API_OK}" -ne 1 ]; then
  echo "ERROR: KubeAI API did not respond at http://localhost:${PF_LOCAL_PORT}/openai/v1" >&2
  kubectl -n "${NAMESPACE}" get pods -o wide || true
  kubectl -n "${NAMESPACE}" describe model "${EMBEDDER}" || true
  kubectl -n "${NAMESPACE}" logs -l kubeai.k8s.io/model="${EMBEDDER}" --tail=200 || true
  kill "${PF_PID}" || true
  die "API unavailable"
fi

# 6) Run embedder test if present
if [ -f infra/inference/tests/test_embedder.py ] && command -v python3 >/dev/null 2>&1; then
  log "Running infra/inference/tests/test_embedder.py smoke test..."
  # script expects args (model_name, text) or (text) depending on your repo; handle two common signatures
  if python3 infra/inference/tests/test_embedder.py --help >/dev/null 2>&1; then
    python3 infra/inference/tests/test_embedder.py "${EMBEDDER}" "Hello world" || {
      echo "Embedder test failed" >&2
      kill "${PF_PID}" || true
      die "embedder smoke test failed"
    }
  else
    # try older signature: python3 test_embedder.py "hello"
    python3 infra/inference/tests/test_embedder.py "Hello world" || {
      echo "Embedder test failed" >&2
      kill "${PF_PID}" || true
      die "embedder smoke test failed"
    }
  fi
  log "Embedder test succeeded."
else
  log "No embedder test found or python3 missing; skipping smoke test."
fi

# cleanup
kill "${PF_PID}" 2>/dev/null || true
trap - EXIT
log "Complete."
