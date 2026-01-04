#!/usr/bin/env bash
set -euo pipefail
LOG(){ printf '%s %s\n' "$(date -Iseconds)" "$*"; }
DBG(){ if [ "${VERBOSE:-0}" != "0" ]; then printf '%s %s\n' "$(date -Iseconds)" "$*"; fi; }

: "${RENDER_DIR:=${PWD}/infra/manifests/retriever}"
mkdir -p "${RENDER_DIR}"

: "${RETRIEVAL_NAMESPACE:=inference}"
: "${RETRIEVAL_NAME:=retrieval}"
: "${RETRIEVAL_IMAGE:=docker.io/athithya5354/retrieval:v11}"
: "${RETRIEVAL_HTTP_PORT:=8001}"
: "${RETRIEVAL_METRICS_PORT:=${RETRIEVAL_HTTP_PORT}}"
: "${RETRIEVER_REPLICAS:=1}"
: "${CONTRACT_LABELS:=app=retrieval,team=search}"
: "${CONTRACT_ANNOTATIONS:=""}"
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

: "${VM_NAMESPACE:=monitoring}"
: "${VMAGENT_SERVICE:=vmagent}"
: "${VMAGENT_PORT:=8429}"
: "${VICTORIA_SERVICE:=victoria-metrics}"
: "${VICTORIA_PORT:=8428}"

MANIFEST_NS="${RENDER_DIR}/10-retriever-namespace.yaml"
MANIFEST_DEP="${RENDER_DIR}/11-retriever-deploy.yaml"
MANIFEST_SVC="${RENDER_DIR}/12-retriever-svc.yaml"
MANIFEST_CM="${RENDER_DIR}/13-retriever-configmap.yaml"

TMP_FILES=()
cleanup(){ local rc=$?; for f in "${TMP_FILES[@]:-}"; do [ -f "$f" ] && rm -f "$f" || true; done; [ -n "${PF_PID:-}" ] && kill "${PF_PID}" >/dev/null 2>&1 || true; exit $rc; }
trap cleanup INT TERM EXIT

check_kubectl(){ command -v kubectl >/dev/null 2>&1 || { LOG "kubectl required"; exit 2; } }
check_jq(){ command -v jq >/dev/null 2>&1 || { LOG "jq required"; exit 2; } }

yaml_single_quote(){
  local v="$1"
  v="${v//\'/''}"
  printf "'%s'" "$v"
}

build_kv_yaml(){
  local csv="$1"; local indent="${2:-4}"
  local out=""
  IFS=',' read -ra pairs <<< "$csv"
  for pair in "${pairs[@]}"; do
    pair="${pair#"${pair%%[![:space:]]*}"}"
    pair="${pair%"${pair##*[![:space:]]}"}"
    [ -z "$pair" ] && continue
    local k="${pair%%=*}"
    local v="${pair#*=}"
    if [[ "$k" =~ [./\ ] || "$k" == *":"* ]]; then k=$(yaml_single_quote "$k"); fi
    v=$(yaml_single_quote "$v")
    out="${out}\n$(printf '%*s' "${indent}" '')${k}: ${v}"
  done
  printf '%b' "${out}"
}

validate_numeric_ports(){
  for v in RETRIEVAL_HTTP_PORT RETRIEVAL_METRICS_PORT VMAGENT_PORT VICTORIA_PORT; do
    val="$(eval "printf '%s' \"\$${v}\"")"
    if ! printf '%s' "${val}" | grep -qE '^[0-9]+$'; then
      LOG "ERROR: ${v} must be numeric (found: ${val})"
      exit 2
    fi
    if [ "${val}" -lt 1 ] || [ "${val}" -gt 65535 ]; then LOG "ERROR: ${v} out of TCP port range (found: ${val})"; exit 2; fi
  done
}

sanitize_multiline(){
  local in="$1"
  printf '%s' "${in}" | tr -d '\r' | sed 's/\\//g'
}

atomic_write(){
  local dest="$1"; local content="$2"
  mkdir -p "$(dirname "$dest")"
  local tmp
  tmp="$(mktemp "${dest}.tmp.XXXXXX")"
  TMP_FILES+=("$tmp")
  printf '%s' "$content" > "$tmp"
  mv "$tmp" "$dest"
  for i in "${!TMP_FILES[@]}"; do [ "${TMP_FILES[$i]}" = "$tmp" ] && unset 'TMP_FILES[i]' || true; done
  LOG "wrote ${dest}"
}

render_manifests(){
  validate_numeric_ports

  if [ -z "${CONTRACT_ANNOTATIONS:-}" ]; then
    CONTRACT_ANNOTATIONS="monitoring.io/scrape=true,monitoring.io/port=${RETRIEVAL_METRICS_PORT},monitoring.io/path=/metrics"
  fi

  local labels_yaml selector_yaml podlabels_yaml annots_yaml
  labels_yaml=$(build_kv_yaml "${CONTRACT_LABELS}" 4)
  selector_yaml=$(build_kv_yaml "${CONTRACT_LABELS}" 6)
  podlabels_yaml=$(build_kv_yaml "${CONTRACT_LABELS}" 8)
  annots_yaml=$(build_kv_yaml "${CONTRACT_ANNOTATIONS}" 8)

  local effective_metrics_port="${RETRIEVAL_METRICS_PORT}"
  local single_port="false"
  if [ "${RETRIEVAL_HTTP_PORT}" -eq "${effective_metrics_port}" ]; then single_port="true"; fi

  local ns_yaml
  ns_yaml=$(cat <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: ${RETRIEVAL_NAMESPACE}
EOF
)
  atomic_write "${MANIFEST_NS}" "${ns_yaml}"

  local sys_prompt_s
  sys_prompt_s="$(sanitize_multiline "${LLM_SYSTEM_PROMPT}")"
  local user_prompt_s
  user_prompt_s="$(sanitize_multiline "${LLM_USER_PROMPT_TEMPLATE}")"

  local cm_yaml
  cm_yaml=$(cat <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: retrieval-config
  namespace: ${RETRIEVAL_NAMESPACE}
data:
  PORT: "${RETRIEVAL_HTTP_PORT}"
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
  LLM_SYSTEM_PROMPT: |
$(printf '%s\n' "${sys_prompt_s}" | sed 's/^/    /')
  LLM_USER_PROMPT_TEMPLATE: |
$(printf '%s\n' "${user_prompt_s}" | sed 's/^/    /')
EOF
)
  atomic_write "${MANIFEST_CM}" "${cm_yaml}"

  local ports_yaml=""
  if [ "${single_port}" = "true" ]; then
    ports_yaml=$(cat <<EOF
          ports:
            - name: http
              containerPort: ${RETRIEVAL_HTTP_PORT}
EOF
)
  else
    ports_yaml=$(cat <<EOF
          ports:
            - name: http
              containerPort: ${RETRIEVAL_HTTP_PORT}
            - name: metrics
              containerPort: ${effective_metrics_port}
EOF
)
  fi

  local dep_yaml
  dep_yaml=$(cat <<EOF
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
              value: "${RETRIEVAL_HTTP_PORT}"
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
${ports_yaml}
          readinessProbe:
            httpGet:
              path: /readyz
              port: ${RETRIEVAL_HTTP_PORT}
            initialDelaySeconds: 5
            periodSeconds: 10
            failureThreshold: 6
          livenessProbe:
            httpGet:
              path: /healthz
              port: ${RETRIEVAL_HTTP_PORT}
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
)
  atomic_write "${MANIFEST_DEP}" "${dep_yaml}"

  local svc_ports_yaml=""
  if [ "${single_port}" = "true" ]; then
    svc_ports_yaml=$(cat <<EOF
  ports:
    - name: http
      port: ${RETRIEVAL_HTTP_PORT}
      targetPort: ${RETRIEVAL_HTTP_PORT}
EOF
)
  else
    svc_ports_yaml=$(cat <<EOF
  ports:
    - name: http
      port: ${RETRIEVAL_HTTP_PORT}
      targetPort: ${RETRIEVAL_HTTP_PORT}
    - name: metrics
      port: ${effective_metrics_port}
      targetPort: ${effective_metrics_port}
EOF
)
  fi

  local svc_yaml
  svc_yaml=$(cat <<EOF
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
${svc_ports_yaml}
  type: ClusterIP
EOF
)
  atomic_write "${MANIFEST_SVC}" "${svc_yaml}"

  if grep -q "\\\\" "${MANIFEST_NS}" "${MANIFEST_DEP}" "${MANIFEST_SVC}" "${MANIFEST_CM}" 2>/dev/null ; then
    LOG "ERROR: generated manifest contains backslash escapes; aborting"
    exit 1
  fi

  LOG "rendered manifests: ${MANIFEST_NS}, ${MANIFEST_DEP}, ${MANIFEST_SVC}, ${MANIFEST_CM}"
}

ensure_config_and_secrets(){
  check_kubectl
  kubectl create namespace "${RETRIEVAL_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1
  kubectl -n "${RETRIEVAL_NAMESPACE}" apply -f "${MANIFEST_CM}" >/dev/null 2>&1 || true
  kubectl -n "${RETRIEVAL_NAMESPACE}" create secret generic retrieval-secrets \
    --from-literal=GROQ_API_KEY="${GROQ_API_KEY}" \
    --from-literal=OPENAI_API_KEY="${OPENAI_API_KEY}" \
    --from-literal=LLM_API_KEY="${LLM_API_KEY}" \
    --from-literal=QDRANT_API_KEY="${QDRANT_API_KEY}" \
    --from-literal=AZURE_STORAGE_CONNECTION_STRING="${AZURE_STORAGE_CONNECTION_STRING}" \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1
  LOG "applied configmap and secrets in ${RETRIEVAL_NAMESPACE}"
}

patch_deployment_metrics_port_if_missing(){
  check_kubectl
  check_jq
  local dep_json
  dep_json="$(kubectl -n "${RETRIEVAL_NAMESPACE}" get deployment "${RETRIEVAL_NAME}" -o json 2>/dev/null || true)"
  if [ -z "${dep_json}" ]; then LOG "deployment not found; skipping metrics port patch"; return 0; fi
  local container_name
  container_name="$(echo "${dep_json}" | jq -r '.spec.template.spec.containers[0].name // empty')"
  if [ -z "${container_name}" ]; then LOG "no container name discovered; skipping patch"; return 0; fi
  local exists
  exists="$(echo "${dep_json}" | jq --arg port "${RETRIEVAL_METRICS_PORT}" '[.spec.template.spec.containers[]?.ports[]? | select((.containerPort|tostring) == $port)] | length' 2>/dev/null || echo 0)"
  if [ "${exists:-0}" -gt 0 ]; then LOG "metrics port ${RETRIEVAL_METRICS_PORT} already declared"; return 0; fi
  local patch
  patch="$(jq -n --arg name "${container_name}" --arg pname "metrics" --argjson port "${RETRIEVAL_METRICS_PORT}" '{ "spec": { "template": { "spec": { "containers": [ { "name": $name, "ports": [ { "name": $pname, "containerPort": $port, "protocol": "TCP" } ] } ] } } } }')"
  if kubectl -n "${RETRIEVAL_NAMESPACE}" patch deployment "${RETRIEVAL_NAME}" --type=merge -p "${patch}" >/dev/null 2>&1; then
    LOG "patched deployment to add metrics port ${RETRIEVAL_METRICS_PORT} to container ${container_name}"
    kubectl -n "${RETRIEVAL_NAMESPACE}" rollout restart deployment "${RETRIEVAL_NAME}" >/dev/null 2>&1 || true
    return 0
  else
    LOG "warning: metrics port patch failed"
    return 1
  fi
}

apply(){
  check_kubectl
  check_jq
  validate_numeric_ports
  ensure_config_and_secrets
  render_manifests
  kubectl apply -f "${MANIFEST_NS}"
  kubectl apply -f "${MANIFEST_CM}"
  kubectl apply -f "${MANIFEST_DEP}"
  kubectl apply -f "${MANIFEST_SVC}"
  LOG "applied retriever manifests into ${RETRIEVAL_NAMESPACE}"
  patch_deployment_metrics_port_if_missing || true
  LOG "retriever apply complete in ${RETRIEVAL_NAMESPACE}"
}

delete(){
  check_kubectl
  kubectl -n "${RETRIEVAL_NAMESPACE}" delete -f "${MANIFEST_SVC}" --ignore-not-found || true
  kubectl -n "${RETRIEVAL_NAMESPACE}" delete -f "${MANIFEST_DEP}" --ignore-not-found || true
  kubectl -n "${RETRIEVAL_NAMESPACE}" delete -f "${MANIFEST_CM}" --ignore-not-found || true
  kubectl -n "${RETRIEVAL_NAMESPACE}" delete namespace "${RETRIEVAL_NAMESPACE}" --ignore-not-found || true
  kubectl -n "${RETRIEVAL_NAMESPACE}" delete secret retrieval-secrets --ignore-not-found || true
  LOG "retriever deleted (best-effort)"
}

usage(){ printf '%s\n' "usage: $0 --generate|--apply|--delete [--verbose]"; exit 1; }

if [ "$#" -eq 0 ]; then usage; fi
cmd=""; force=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --generate) cmd="generate"; shift;;
    --apply) cmd="apply"; shift;;
    --delete) cmd="delete"; shift;;
    --verbose) VERBOSE=1; shift;;
    --force) force="--force"; shift;;
    *) usage;;
  esac
done

if [ "$cmd" = "generate" ]; then render_manifests && LOG "rendered manifests to ${RENDER_DIR}"; exit 0; fi
if [ "$cmd" = "apply" ]; then apply; exit $?; fi
if [ "$cmd" = "delete" ]; then delete; exit 0; fi
usage
