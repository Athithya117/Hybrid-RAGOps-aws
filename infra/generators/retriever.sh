#!/usr/bin/env bash
set -euo pipefail
LOG(){ printf '%s %s\n' "$(date -Iseconds)" "$*"; }

: "${RENDER_DIR:=${PWD}/infra/manifests}"
mkdir -p "${RENDER_DIR}"

: "${RETRIEVAL_NAMESPACE:=inference}"
: "${RETRIEVAL_NAME:=retrieval}"
: "${RETRIEVAL_IMAGE:=docker.io/athithya5354/retrieval:v10}"
: "${RETRIEVAL_PORT:=8001}"
: "${RETRIEVER_REPLICAS:=1}"
: "${CONTRACT_LABELS:=app=retrieval,team=search}"
: "${CONTRACT_ANNOTATIONS:=monitoring.io/scrape=true,monitoring.io/port=${RETRIEVAL_PORT},monitoring.io/path=/metrics}"
: "${RETRIEVAL_RES_CPU:=200m}"
: "${RETRIEVAL_RES_MEM:=256Mi}"

: "${GROQ_API_KEY:=}"
: "${OPENAI_API_KEY:=}"
: "${LLM_API_KEY:=}"
: "${QDRANT_API_KEY:=}"
: "${AZURE_STORAGE_CONNECTION_STRING:=}"

: "${QDRANT_URL:=}"
: "${DENSE_URL:=}"
: "${SPARSE_URL:=}"
: "${RERANKER_URL:=}"
: "${COLLECTION_NAME:=}"
: "${HTTP_TIMEOUT:=10}"
: "${DENSE_DIM:=768}"
: "${MAX_CHUNKS_TO_LLM:=6}"
: "${ENV:=local}"

: "${LLM_MODEL:=llama-3.1-8b-instant}"
: "${LLM_MAX_TOKENS:=512}"
: "${LLM_TEMPERATURE:=0.2}"
: "${LLM_SYSTEM_PROMPT:=You are a clear concise assistant. Provide a short explanatory answer in 2-3 sentences. When you cite evidence, use only numeric tags like [1],[2]. Do NOT output filenames, URLs, raw page numbers.}"
: "${LLM_USER_PROMPT_TEMPLATE:=Summarize the following retrieved passages and answer the question in 2-3 sentences.\n\nQUESTION: {question}\n\nPASSAGES:\n{passages}\n\nAnswer:}"

: "${RERANKER_MODE:=AUTO}"
: "${RERANK_TOPK:=20}"
: "${RERANKER_TOP_K:=${RERANK_TOPK}}"
: "${RERANK_AUTO_THRESHOLD:=0.75}"
: "${RERANK_THRESHOLD:=30}"
: "${RERANK_MARGIN:=0.08}"
: "${RERANK_ALPHA:=0.6}"
: "${QUERY_TOPK_DENSE:=200}"
: "${QUERY_TOPK_SPARSE:=200}"
: "${RRF_TOP_N:=10}"

MANIFEST_NS="${RENDER_DIR}/10-retriever-namespace.yaml"
MANIFEST_DEP="${RENDER_DIR}/11-retriever-deploy.yaml"
MANIFEST_SVC="${RENDER_DIR}/12-retriever-svc.yaml"
MANIFEST_CM="${RENDER_DIR}/13-retriever-configmap.yaml"

check_kubectl(){ command -v kubectl >/dev/null 2>&1 || { LOG "kubectl required"; exit 1; } }

yaml_single_quote(){
  local v="$1"
  v="${v//\'/''}"
  printf "'%s'" "$v"
}

build_kv_yaml(){
  local csv="$1"; local indent="${2:-6}"
  local out=""
  IFS=',' read -ra pairs <<< "$csv"
  for pair in "${pairs[@]}"; do
    pair="${pair#"${pair%%[![:space:]]*}"}"
    pair="${pair%"${pair##*[![:space:]]}"}"
    [ -z "$pair" ] && continue
    local k="${pair%%=*}"
    local v="${pair#*=}"
    if [[ "$k" =~ [./\ ] || "$k" == *":"* ]]; then
      k=$(yaml_single_quote "$k")
    fi
    v=$(yaml_single_quote "$v")
    out="${out}\n$(printf '%*s' "${indent}" '')${k}: ${v}"
  done
  printf '%b' "${out}"
}

extract_monitoring_port(){
  local csv="$1"
  IFS=',' read -ra pairs <<< "$csv"
  for pair in "${pairs[@]}"; do
    pair="${pair#"${pair%%[![:space:]]*}"}"
    pair="${pair%"${pair##*[![:space:]]}"}"
    [ -z "$pair" ] && continue
    local k="${pair%%=*}"
    local v="${pair#*=}"
    if [[ "$k" == "monitoring.io/port" ]]; then
      printf '%s' "$v"
      return 0
    fi
  done
  return 1
}

ensure_config_and_secrets(){
  check_kubectl
  kubectl create namespace "${RETRIEVAL_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
  local port
  port=$(extract_monitoring_port "${CONTRACT_ANNOTATIONS}") || port="${RETRIEVAL_PORT}"
  if ! printf '%s\n' "${port}" | grep -qE '^[0-9]+$' ; then
    LOG "ERROR: monitoring.io/port must be numeric (found: ${port})"; exit 1
  fi

  cat > "${MANIFEST_CM}" <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: retrieval-config
  namespace: ${RETRIEVAL_NAMESPACE}
data:
  PORT: "${RETRIEVAL_PORT}"
  ENV: "${ENV}"
  QDRANT_URL: "${QDRANT_URL}"
  DENSE_URL: "${DENSE_URL}"
  SPARSE_URL: "${SPARSE_URL}"
  RERANKER_URL: "${RERANKER_URL}"
  COLLECTION_NAME: "${COLLECTION_NAME}"
  HTTP_TIMEOUT: "${HTTP_TIMEOUT}"
  DENSE_DIM: "${DENSE_DIM}"
  MAX_CHUNKS_TO_LLM: "${MAX_CHUNKS_TO_LLM}"
  LLM_MODEL: "${LLM_MODEL}"
  LLM_MAX_TOKENS: "${LLM_MAX_TOKENS}"
  LLM_TEMPERATURE: "${LLM_TEMPERATURE}"
  LLM_SYSTEM_PROMPT: |
$(printf '%s\n' "${LLM_SYSTEM_PROMPT}" | sed 's/^/    /')
  LLM_USER_PROMPT_TEMPLATE: |
$(printf '%s\n' "${LLM_USER_PROMPT_TEMPLATE}" | sed 's/^/    /')
  RERANKER_MODE: "${RERANKER_MODE}"
  RERANK_TOPK: "${RERANK_TOPK}"
  RERANKER_TOP_K: "${RERANKER_TOP_K}"
  RERANK_AUTO_THRESHOLD: "${RERANK_AUTO_THRESHOLD}"
  RERANK_THRESHOLD: "${RERANK_THRESHOLD}"
  RERANK_MARGIN: "${RERANK_MARGIN}"
  RERANK_ALPHA: "${RERANK_ALPHA}"
  QUERY_TOPK_DENSE: "${QUERY_TOPK_DENSE}"
  QUERY_TOPK_SPARSE: "${QUERY_TOPK_SPARSE}"
  RRF_TOP_N: "${RRF_TOP_N}"
EOF

  kubectl -n "${RETRIEVAL_NAMESPACE}" apply -f "${MANIFEST_CM}"
  kubectl -n "${RETRIEVAL_NAMESPACE}" create secret generic retrieval-secrets \
    --from-literal=GROQ_API_KEY="${GROQ_API_KEY}" \
    --from-literal=OPENAI_API_KEY="${OPENAI_API_KEY}" \
    --from-literal=LLM_API_KEY="${LLM_API_KEY}" \
    --from-literal=QDRANT_API_KEY="${QDRANT_API_KEY}" \
    --from-literal=AZURE_STORAGE_CONNECTION_STRING="${AZURE_STORAGE_CONNECTION_STRING}" \
    --dry-run=client -o yaml | kubectl apply -f -
  LOG "applied retrieval-config and retrieval-secrets in ${RETRIEVAL_NAMESPACE}"
}

render_manifests(){
  local labels_yaml selector_yaml podlabels_yaml annots_yaml
  labels_yaml=$(build_kv_yaml "${CONTRACT_LABELS}" 4)
  selector_yaml=$(build_kv_yaml "${CONTRACT_LABELS}" 6)
  podlabels_yaml=$(build_kv_yaml "${CONTRACT_LABELS}" 8)
  annots_yaml=$(build_kv_yaml "${CONTRACT_ANNOTATIONS}" 8)

cat >"${MANIFEST_NS}" <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: ${RETRIEVAL_NAMESPACE}
EOF

cat >"${MANIFEST_DEP}" <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${RETRIEVAL_NAME}
  namespace: ${RETRIEVAL_NAMESPACE}
  labels:
${labels_yaml}
spec:
  replicas: ${RETRIEVER_REPLICAS}
  selector:
    matchLabels:
${selector_yaml}
  template:
    metadata:
      labels:
${podlabels_yaml}
      annotations:
${annots_yaml}
    spec:
      containers:
        - name: ${RETRIEVAL_NAME}
          image: ${RETRIEVAL_IMAGE}
          envFrom:
            - configMapRef:
                name: retrieval-config
            - secretRef:
                name: retrieval-secrets
          env:
            - name: PORT
              value: "${RETRIEVAL_PORT}"
            - name: ENV
              value: "${ENV}"
            - name: GROQ_API_KEY
              valueFrom:
                secretKeyRef:
                  name: retrieval-secrets
                  key: GROQ_API_KEY
                  optional: true
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: retrieval-secrets
                  key: OPENAI_API_KEY
                  optional: true
            - name: LLM_API_KEY
              valueFrom:
                secretKeyRef:
                  name: retrieval-secrets
                  key: LLM_API_KEY
                  optional: true
            - name: LLM_MODEL
              valueFrom:
                configMapKeyRef:
                  name: retrieval-config
                  key: LLM_MODEL
                  optional: true
            - name: LLM_MAX_TOKENS
              valueFrom:
                configMapKeyRef:
                  name: retrieval-config
                  key: LLM_MAX_TOKENS
                  optional: true
            - name: LLM_TEMPERATURE
              valueFrom:
                configMapKeyRef:
                  name: retrieval-config
                  key: LLM_TEMPERATURE
                  optional: true
            - name: LLM_SYSTEM_PROMPT
              valueFrom:
                configMapKeyRef:
                  name: retrieval-config
                  key: LLM_SYSTEM_PROMPT
                  optional: true
            - name: LLM_USER_PROMPT_TEMPLATE
              valueFrom:
                configMapKeyRef:
                  name: retrieval-config
                  key: LLM_USER_PROMPT_TEMPLATE
                  optional: true
            - name: MAX_CHUNKS_TO_LLM
              valueFrom:
                configMapKeyRef:
                  name: retrieval-config
                  key: MAX_CHUNKS_TO_LLM
                  optional: true
            - name: RERANKER_MODE
              valueFrom:
                configMapKeyRef:
                  name: retrieval-config
                  key: RERANKER_MODE
                  optional: true
            - name: RERANK_TOPK
              valueFrom:
                configMapKeyRef:
                  name: retrieval-config
                  key: RERANK_TOPK
                  optional: true
            - name: RERANKER_TOP_K
              valueFrom:
                configMapKeyRef:
                  name: retrieval-config
                  key: RERANKER_TOP_K
                  optional: true
            - name: RERANK_AUTO_THRESHOLD
              valueFrom:
                configMapKeyRef:
                  name: retrieval-config
                  key: RERANK_AUTO_THRESHOLD
                  optional: true
            - name: RERANK_THRESHOLD
              valueFrom:
                configMapKeyRef:
                  name: retrieval-config
                  key: RERANK_THRESHOLD
                  optional: true
            - name: RERANK_MARGIN
              valueFrom:
                configMapKeyRef:
                  name: retrieval-config
                  key: RERANK_MARGIN
                  optional: true
            - name: RERANK_ALPHA
              valueFrom:
                configMapKeyRef:
                  name: retrieval-config
                  key: RERANK_ALPHA
                  optional: true
            - name: QUERY_TOPK_DENSE
              valueFrom:
                configMapKeyRef:
                  name: retrieval-config
                  key: QUERY_TOPK_DENSE
                  optional: true
            - name: QUERY_TOPK_SPARSE
              valueFrom:
                configMapKeyRef:
                  name: retrieval-config
                  key: QUERY_TOPK_SPARSE
                  optional: true
            - name: RRF_TOP_N
              valueFrom:
                configMapKeyRef:
                  name: retrieval-config
                  key: RRF_TOP_N
                  optional: true
          ports:
            - name: http
              containerPort: ${RETRIEVAL_PORT}
          readinessProbe:
            httpGet:
              path: /readyz
              port: ${RETRIEVAL_PORT}
            initialDelaySeconds: 5
            periodSeconds: 10
            failureThreshold: 6
          livenessProbe:
            httpGet:
              path: /healthz
              port: ${RETRIEVAL_PORT}
            initialDelaySeconds: 15
            periodSeconds: 20
            failureThreshold: 3
          resources:
            requests:
              cpu: ${RETRIEVAL_RES_CPU}
              memory: ${RETRIEVAL_RES_MEM}
            limits:
              cpu: ${RETRIEVAL_RES_CPU}
              memory: ${RETRIEVAL_RES_MEM}
EOF

cat >"${MANIFEST_SVC}" <<EOF
apiVersion: v1
kind: Service
metadata:
  name: ${RETRIEVAL_NAME}
  namespace: ${RETRIEVAL_NAMESPACE}
  labels:
${labels_yaml}
spec:
  selector:
${selector_yaml}
  ports:
    - name: http
      port: ${RETRIEVAL_PORT}
      targetPort: ${RETRIEVAL_PORT}
  type: ClusterIP
EOF

  if grep -q '\\\\' "${MANIFEST_DEP}" "${MANIFEST_SVC}" "${MANIFEST_CM}" 2>/dev/null ; then
    LOG "ERROR: generated manifest contains backslash escapes; aborting"
    exit 1
  fi
}

apply(){
  check_kubectl
  ensure_config_and_secrets
  render_manifests
  kubectl apply -f "${MANIFEST_NS}"
  kubectl apply -f "${MANIFEST_DEP}"
  kubectl apply -f "${MANIFEST_SVC}"
  LOG "retriever applied in ${RETRIEVAL_NAMESPACE}"
}

delete(){
  check_kubectl
  kubectl delete -f "${MANIFEST_SVC}" --ignore-not-found || true
  kubectl delete -f "${MANIFEST_DEP}" --ignore-not-found || true
  kubectl delete -f "${MANIFEST_NS}" --ignore-not-found || true
  kubectl -n "${RETRIEVAL_NAMESPACE}" delete configmap retrieval-config --ignore-not-found || true
  kubectl -n "${RETRIEVAL_NAMESPACE}" delete secret retrieval-secrets --ignore-not-found || true
  LOG "retriever deleted (best-effort)"
}

case "${1:-}" in
  --generate) ensure_config_and_secrets; render_manifests; LOG "rendered manifests to ${RENDER_DIR}" ;;
  --apply) apply ;;
  --delete) delete ;;
  *) LOG "usage: $0 --generate|--apply|--delete"; exit 1 ;;
esac
