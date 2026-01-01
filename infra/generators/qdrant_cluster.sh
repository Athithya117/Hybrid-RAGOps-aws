#!/usr/bin/env bash
set -euo pipefail
LOG(){ printf '%s %s\n' "$(date -Iseconds)" "$*"; }
DBG(){ if [ "${VERBOSE:-0}" != "0" ]; then printf '%s %s\n' "$(date -Iseconds)" "$*"; fi; }

MANIFESTS_DIR=${MANIFESTS_DIR:-infra/manifests/qdrant}
VALUES_FILE="${MANIFESTS_DIR}/values.yaml"
SERVICE_PATCH_FILE="${MANIFESTS_DIR}/service-patch.yaml"
SAMPLES_DIR="${MANIFESTS_DIR}/_samples"
LAST_SUMMARY="${MANIFESTS_DIR}/last_deploy_summary.json"
INPUTS_HASH_PATH="${MANIFESTS_DIR}/.inputs_hash"

QDRANT_RELEASE=${QDRANT_RELEASE:-qdrant}
QDRANT_NAMESPACE=${QDRANT_NAMESPACE:-qdrant}
QDRANT_IMAGE=${QDRANT_IMAGE:-qdrant/qdrant:v1.16.0}
CHART_VERSION=${CHART_VERSION:-1.16.0}
QDRANT_REPLICAS=${QDRANT_REPLICAS:-1}
QDRANT_CPU=${QDRANT_CPU:-1}
QDRANT_MEMORY=${QDRANT_MEMORY:-2Gi}
QDRANT_METRICS_PORT=${QDRANT_METRICS_PORT:-6333}
QDRANT_METRICS_PORT_NAME=${QDRANT_METRICS_PORT_NAME:-metrics}
SECRET_BACKUP_NAME=${SECRET_BACKUP_NAME:-}
SECRET_SERVICE_NAME=${SECRET_SERVICE_NAME:-qdrant-service-creds}
QDRANT__SERVICE__API_KEY=${QDRANT__SERVICE__API_KEY:-}
FAIL_ON_MISCONFIG=${FAIL_ON_MISCONFIG:-false}
SERVICE_VALIDATION_WAIT=${SERVICE_VALIDATION_WAIT:-120}
VENDOR_CHART_DIR=${VENDOR_CHART_DIR:-infra/archive/qdrant-helm-chart/qdrant}
HELM_REPO_NAME=${HELM_REPO_NAME:-qdrant}
HELM_PRIMARY_REPO=${HELM_PRIMARY_REPO:-https://qdrant.github.io/qdrant-helm}
HELM_FALLBACK_REPO=${HELM_FALLBACK_REPO:-https://qdrant.to/helm}
ENV=${ENV:-STAGING}
APPLY_STAGING_SECRETS=${APPLY_STAGING_SECRETS:-true}
AZURE_STORAGE_CONNECTION_STRING=${AZURE_STORAGE_CONNECTION_STRING:-}
AZURE_STORAGE_ACCOUNT_NAME=${AZURE_STORAGE_ACCOUNT_NAME:-}
AZURE_STORAGE_ACCOUNT_KEY=${AZURE_STORAGE_ACCOUNT_KEY:-}

TMP_FILES=()
cleanup(){ local rc=$?; for f in "${TMP_FILES[@]:-}"; do [ -f "$f" ] && rm -f "$f" || true; done; [ -n "${PF_PID:-}" ] && kill "${PF_PID}" >/dev/null 2>&1 || true; exit $rc; }
trap cleanup INT TERM EXIT

require_cmds(){ local miss=0; for c in kubectl helm python3 sha256sum awk grep sed mktemp jq; do if ! command -v "$c" >/dev/null 2>&1; then LOG "ERROR: required command not found: $c"; miss=1; fi; done; [ "$miss" -eq 0 ] || exit 2; }

ensure_dir(){ mkdir -p "${MANIFESTS_DIR}" "${SAMPLES_DIR}"; }

atomic_write(){
  local dest="$1"; local content="$2"
  mkdir -p "$(dirname "$dest")"
  local tmp
  tmp="$(mktemp "${dest}.tmp.XXXXXX")"
  TMP_FILES+=("$tmp")
  printf '%s' "$content" > "$tmp"
  mv "$tmp" "$dest"
  for i in "${!TMP_FILES[@]}"; do [ "${TMP_FILES[$i]}" = "$tmp" ] && unset 'TMP_FILES[i]' || true; done
}

sha_inputs(){
  printf '%s\n' \
    "${QDRANT_RELEASE}" "${QDRANT_NAMESPACE}" "${QDRANT_IMAGE}" "${CHART_VERSION}" \
    "${QDRANT_REPLICAS}" "${QDRANT_CPU}" "${QDRANT_MEMORY}" "${QDRANT_METRICS_PORT}" \
    "${SECRET_BACKUP_NAME}" "${SECRET_SERVICE_NAME}" "${QDRANT__SERVICE__API_KEY}" \
  | sha256sum | awk '{print $1}'
}

validate_numeric_port(){
  if ! printf '%s' "${QDRANT_METRICS_PORT}" | grep -qE '^[0-9]+$'; then
    LOG "ERROR: QDRANT_METRICS_PORT must be numeric (found: ${QDRANT_METRICS_PORT})"
    exit 2
  fi
}

render_values_yaml(){
  python3 - <<'PY' > "${VALUES_FILE}.tmp"
import os, sys, yaml
def getenv(k,d=None):
    return os.environ.get(k,d)
def geti(k,d):
    v=os.environ.get(k)
    try:
        return int(v) if v is not None else d
    except:
        return d
vals={}
vals["replicaCount"]=geti("QDRANT_REPLICAS",1)
repo_tag=getenv("QDRANT_IMAGE","qdrant/qdrant:v1.16.0")
if ":" in repo_tag:
    repo, tag = repo_tag.split(":",1)
else:
    repo, tag = repo_tag, "latest"
vals["image"]={"repository": repo, "tag": tag, "pullPolicy":"IfNotPresent"}
vals["service"]={"type":"ClusterIP","labels":{"app.kubernetes.io/name":getenv("QDRANT_RELEASE","qdrant"),"app.kubernetes.io/component":"qdrant"}}
metrics_port = int(getenv("QDRANT_METRICS_PORT","6333"))
metrics_name = getenv("QDRANT_METRICS_PORT_NAME","metrics")
vals["podAnnotations"]={
  "monitoring.io/scrape":"true",
  "monitoring.io/port": str(metrics_port),
  "monitoring.io/path": "/metrics"
}
vals["ports"]=[{"name": metrics_name, "containerPort": metrics_port, "protocol": "TCP"}]
vals["p2p"]={"port":6335}
replicas=geti("QDRANT_REPLICAS",1)
peers=["http://%s-%d.%s-headless:6335"%(getenv("QDRANT_RELEASE","qdrant"), i, getenv("QDRANT_RELEASE","qdrant")) for i in range(replicas)]
vals["cluster"]={"enabled": True, "peers": peers}
vals["snapshots"]={"enabled": False, "s3":{"bucket":"","endpoint":"","region":"","prefix":""}}
extra_env=[]
if getenv("SECRET_BACKUP_NAME"):
    for key in ("AZURE_STORAGE_CONNECTION_STRING","AZURE_STORAGE_ACCOUNT_NAME","AZURE_STORAGE_ACCOUNT_KEY"):
        extra_env.append({"name": key, "valueFrom": {"secretKeyRef": {"name": getenv("SECRET_BACKUP_NAME"), "key": key}}})
if getenv("QDRANT__SERVICE__API_KEY") and getenv("SECRET_SERVICE_NAME"):
    extra_env.append({"name":"QDRANT__SERVICE__API_KEY","valueFrom":{"secretKeyRef":{"name":getenv("SECRET_SERVICE_NAME"),"key":"QDRANT__SERVICE__API_KEY"}}})
vals["extraEnv"]=extra_env if extra_env else []
vals["resources"]={"requests":{"cpu":getenv("QDRANT_CPU","1"),"memory":getenv("QDRANT_MEMORY","2Gi")},"limits":{"cpu":getenv("QDRANT_CPU","1"),"memory":getenv("QDRANT_MEMORY","2Gi")}}
vals["tolerations"]=[]
vals["persistence"]={"enabled":False}
vals["config"]={"params":{"shard_number":geti("QDRANT_SHARD_NUMBER",1),"replication_factor":geti("QDRANT_REPLICATION_FACTOR",geti("QDRANT_REPLICAS",1)),"write_consistency_factor":geti("QDRANT_WRITE_CONSISTENCY_FACTOR",1)},"log_level":getenv("QDRANT_LOG_LEVEL","INFO"),"storage":{"storage_path":getenv("QDRANT__STORAGE__STORAGE_PATH","/qdrant/storage"),"snapshots_path":getenv("QDRANT__STORAGE__SNAPSHOTS_PATH","/qdrant/snapshots")}}
yaml.safe_dump(vals, sys.stdout, sort_keys=False)
PY
  mv "${VALUES_FILE}.tmp" "${VALUES_FILE}"
  LOG "Rendered ${VALUES_FILE}"
}

render_service_patch(){
  cat > "${SERVICE_PATCH_FILE}.tmp" <<EOF
apiVersion: v1
kind: Service
metadata:
  name: ${QDRANT_RELEASE}
  namespace: ${QDRANT_NAMESPACE}
  labels:
    app.kubernetes.io/name: ${QDRANT_RELEASE}
    app.kubernetes.io/component: qdrant
spec:
  selector:
    app.kubernetes.io/name: ${QDRANT_RELEASE}
  ports:
    - name: ${QDRANT_METRICS_PORT_NAME}
      port: ${QDRANT_METRICS_PORT}
      targetPort: ${QDRANT_METRICS_PORT}
      protocol: TCP
  type: ClusterIP
EOF
  mv "${SERVICE_PATCH_FILE}.tmp" "${SERVICE_PATCH_FILE}"
  LOG "Rendered ${SERVICE_PATCH_FILE}"
}

create_azure_backup_secret(){
  local secret_name="${SECRET_BACKUP_NAME:-}"
  if [ -z "${secret_name}" ]; then
    LOG "no SECRET_BACKUP_NAME configured; skipping backup secret creation"; return 0
  fi
  if ! command -v kubectl >/dev/null 2>&1; then
    LOG "kubectl not found; cannot create secret"; return 2
  fi
  kubectl create namespace "${QDRANT_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1
  if [ -n "${AZURE_STORAGE_CONNECTION_STRING:-}" ]; then
    kubectl -n "${QDRANT_NAMESPACE}" create secret generic "${secret_name}" \
      --from-literal=AZURE_STORAGE_CONNECTION_STRING="${AZURE_STORAGE_CONNECTION_STRING:-}" \
      --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1
    LOG "created/updated secret ${secret_name} (connection string)"; return 0
  fi
  if [ -n "${AZURE_STORAGE_ACCOUNT_NAME:-}" ] && [ -n "${AZURE_STORAGE_ACCOUNT_KEY:-}" ]; then
    kubectl -n "${QDRANT_NAMESPACE}" create secret generic "${secret_name}" \
      --from-literal=AZURE_STORAGE_ACCOUNT_NAME="${AZURE_STORAGE_ACCOUNT_NAME:-}" \
      --from-literal=AZURE_STORAGE_ACCOUNT_KEY="${AZURE_STORAGE_ACCOUNT_KEY:-}" \
      --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1
    LOG "created/updated secret ${secret_name} (account+key)"; return 0
  fi
  LOG "ERROR: neither AZURE_STORAGE_CONNECTION_STRING nor AZURE_STORAGE_ACCOUNT_NAME+AZURE_STORAGE_ACCOUNT_KEY present; cannot create backup secret"
  return 2
}

generate_manifests(){
  ensure_dir
  validate_numeric_port
  local inputs_hash
  inputs_hash="$(sha_inputs)"
  if [ -f "${INPUTS_HASH_PATH}" ] && [ "$(cat "${INPUTS_HASH_PATH}")" = "${inputs_hash}" ] && [ "${1:-}" != "--force" ]; then
    LOG "No non-secret changes detected; generation skipped."; return
  fi
  render_values_yaml
  if [ -n "${SECRET_BACKUP_NAME:-}" ]; then
    cat > "${SAMPLES_DIR}/secret-sample.placeholder.yaml" <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: ${SECRET_BACKUP_NAME}
  namespace: ${QDRANT_NAMESPACE}
type: Opaque
stringData:
  AZURE_STORAGE_CONNECTION_STRING: REPLACE_ME
  AZURE_STORAGE_ACCOUNT_NAME: REPLACE_ME
  AZURE_STORAGE_ACCOUNT_KEY: REPLACE_ME
EOF
  fi
  render_service_patch
  printf '%s' "${inputs_hash}" > "${INPUTS_HASH_PATH}"
  LOG "Wrote manifests to ${MANIFESTS_DIR}"
}

detect_storageclass(){
  if ! command -v kubectl >/dev/null 2>&1; then
    return
  fi
  kubectl get storageclass -o jsonpath='{.items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")].metadata.name}' 2>/dev/null || true
}

ensure_namespace(){
  if ! command -v kubectl >/dev/null 2>&1; then
    LOG "kubectl not found"; return 1
  fi
  kubectl create namespace "${QDRANT_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1
  return 0
}

helm_repo_add_if_missing(){
  local helm
  helm="$(which helm 2>/dev/null || true)"
  [ -n "${helm}" ] || { return 1; }
  helm repo add --force-update "${HELM_REPO_NAME}" "${HELM_PRIMARY_REPO}" >/dev/null 2>&1 || true
  helm repo update >/dev/null 2>&1 || true
  return 0
}

helm_upgrade_install(){
  if ! command -v helm >/dev/null 2>&1; then
    LOG "helm not found"; return 2
  fi
  ensure_namespace
  local tries=0 max=3 backoff=2
  while [ $tries -lt $max ]; do
    if [ -d "${VENDOR_CHART_DIR}" ] && [ -f "${VENDOR_CHART_DIR}/Chart.yaml" ]; then
      DBG "Using vendor chart ${VENDOR_CHART_DIR}"
      if helm upgrade --install "${QDRANT_RELEASE}" "${VENDOR_CHART_DIR}" --namespace "${QDRANT_NAMESPACE}" --create-namespace -f "${VALUES_FILE}" --wait --timeout 10m; then
        return 0
      fi
    else
      DBG "Attempting helm repo install (try $((tries+1))/${max})"
      helm_repo_add_if_missing || true
      if helm upgrade --install "${QDRANT_RELEASE}" "${HELM_REPO_NAME}/qdrant" --version "${CHART_VERSION}" --namespace "${QDRANT_NAMESPACE}" -f "${VALUES_FILE}" --wait --timeout 10m; then
        return 0
      fi
      if helm upgrade --install "${QDRANT_RELEASE}" "qdrant/qdrant" --version "${CHART_VERSION}" --repo "${HELM_FALLBACK_REPO}" --namespace "${QDRANT_NAMESPACE}" -f "${VALUES_FILE}" --wait --timeout 10m; then
        return 0
      fi
    fi
    tries=$((tries+1))
    sleep $((backoff * tries))
  done
  return 1
}

patch_statefulset_ports_if_missing(){
  if ! command -v kubectl >/dev/null 2>&1; then
    LOG "kubectl not found; cannot patch StatefulSet"
    return 1
  fi
  local ss_json
  ss_json="$(kubectl -n "${QDRANT_NAMESPACE}" get statefulset "${QDRANT_RELEASE}" -o json 2>/dev/null || true)"
  if [ -z "${ss_json}" ]; then
    LOG "StatefulSet ${QDRANT_RELEASE} not present; skipping port patch"
    return 0
  fi
  local container_name
  container_name="$(echo "${ss_json}" | jq -r '.spec.template.spec.containers[0].name // empty')"
  if [ -z "${container_name}" ]; then
    LOG "no container name discovered in StatefulSet; skipping patch"
    return 0
  fi
  local port_present
  port_present="$(echo "${ss_json}" | jq --arg port "${QDRANT_METRICS_PORT}" '[.spec.template.spec.containers[]?.ports[]? | select((.containerPort|tostring) == $port)] | length' 2>/dev/null || echo 0)"
  if [ "${port_present:-0}" -gt 0 ]; then
    LOG "StatefulSet already declares metrics port ${QDRANT_METRICS_PORT}; no patch required"
    return 0
  fi
  local patch
  patch="$(jq -n --arg name "${container_name}" --arg pname "${QDRANT_METRICS_PORT_NAME}" --argjson port "${QDRANT_METRICS_PORT}" '{ "spec": { "template": { "spec": { "containers": [ { "name": $name, "ports": [ { "name": $pname, "containerPort": $port, "protocol": "TCP" } ] } ] } } } }')"
  if kubectl -n "${QDRANT_NAMESPACE}" patch statefulset "${QDRANT_RELEASE}" --type=merge -p "${patch}" >/dev/null 2>&1; then
    LOG "patched StatefulSet ${QDRANT_RELEASE} to add metrics port ${QDRANT_METRICS_PORT} to container ${container_name}"
    return 0
  else
    LOG "warning: attempt to patch StatefulSet ports failed; continuing with validation"
    return 1
  fi
}

validate_service_post_install(){
  if ! command -v kubectl >/dev/null 2>&1; then
    LOG "kubectl not found; skipping validation"; return 0
  fi

  local selector="app.kubernetes.io/name=${QDRANT_RELEASE}"
  local wait_timeout="${SERVICE_VALIDATION_WAIT}"
  LOG "waiting up to ${wait_timeout}s for pods matching '${selector}' to appear/ready"
  if kubectl -n "${QDRANT_NAMESPACE}" wait --for=condition=Ready pod -l "${selector}" --timeout="${wait_timeout}s" >/dev/null 2>&1; then
    DBG "kubectl wait succeeded"
  else
    DBG "kubectl wait timed out or not supported; will poll get pods"
  fi

  local tmpjson
  tmpjson="$(mktemp /tmp/qdrant-pods.XXXXXX.json)"
  TMP_FILES+=("$tmpjson")
  local end=$((SECONDS + wait_timeout))
  local items_count=0
  while [ "${SECONDS}" -lt "${end}" ]; do
    kubectl -n "${QDRANT_NAMESPACE}" get pods -l "${selector}" -o json > "${tmpjson}" 2>/dev/null || true
    if [ -s "${tmpjson}" ]; then
      items_count=$(jq '.items | length' "${tmpjson}" 2>/dev/null || echo 0)
      if [ "${items_count:-0}" -gt 0 ]; then
        DBG "found ${items_count} pods"; break
      fi
    fi
    sleep 2
  done

  if [ -z "${tmpjson}" ] || [ ! -s "${tmpjson}" ]; then
    LOG "no pods JSON obtained after ${wait_timeout}s"; return 1
  fi
  if [ "${items_count:-0}" -eq 0 ]; then
    LOG "no pods found after ${wait_timeout}s"; return 1
  fi

  local tmp_errors
  tmp_errors="$(mktemp /tmp/qdrant-annot-errors.XXXXXX.txt)"; TMP_FILES+=("$tmp_errors")
  jq -r --arg expected_port "${QDRANT_METRICS_PORT}" '.items[] | .metadata.name as $n | .metadata.annotations as $ann | [$n, ($ann["monitoring.io/scrape"] // ""), ($ann["monitoring.io/port"] // ""), ($ann["monitoring.io/path"] // "")] | @tsv' "${tmpjson}" | while IFS=$'\t' read -r name scrape port path; do
    if [ "${scrape,,}" != "true" ]; then
      printf '%s\n' "ERR ${name} missing monitoring.io/scrape=true" >> "${tmp_errors}"
      continue
    fi
    if ! printf '%s' "${port}" | grep -qE '^[0-9]+$'; then
      printf '%s\n' "ERR ${name} monitoring.io/port must be numeric (found: ${port})" >> "${tmp_errors}"
      continue
    fi
    if [ "${port}" != "${QDRANT_METRICS_PORT}" ]; then
      printf '%s\n' "ERR ${name} monitoring.io/port mismatch expected ${QDRANT_METRICS_PORT} found ${port}" >> "${tmp_errors}"
    fi
    if [ "${path}" != "/metrics" ]; then
      printf '%s\n' "ERR ${name} monitoring.io/path must be /metrics found ${path}" >> "${tmp_errors}"
    fi
  done

  if [ -s "${tmp_errors}" ]; then
    LOG "annotation validation errors:"
    sed -n '1,200p' "${tmp_errors}" || true
    return 2
  fi

  LOG "pod annotations contract satisfied for all pods"

  LOG "checking StatefulSet container ports presence"
  local ss_json
  ss_json="$(kubectl -n "${QDRANT_NAMESPACE}" get statefulset "${QDRANT_RELEASE}" -o json 2>/dev/null || true)"
  if [ -z "${ss_json}" ]; then
    LOG "StatefulSet ${QDRANT_RELEASE} not present; skipping container-port info check"
    return 0
  fi
  local port_declared
  port_declared="$(echo "${ss_json}" | jq --arg port "${QDRANT_METRICS_PORT}" '[.spec.template.spec.containers[]?.ports[]? | select((.containerPort|tostring) == $port)] | length' 2>/dev/null || echo 0)"
  if [ "${port_declared:-0}" -gt 0 ]; then
    LOG "StatefulSet container spec declares metrics port ${QDRANT_METRICS_PORT}"
  else
    LOG "StatefulSet container spec does NOT declare metrics port ${QDRANT_METRICS_PORT}; will attempt a safe patch to add it"
    patch_statefulset_ports_if_missing || LOG "patch attempt returned non-zero or failed"
  fi

  return 0
}

apply_to_cluster(){
  require_cmds
  if [ "${ENV}" = "STAGING" ] && [ "${APPLY_STAGING_SECRETS}" = "true" ]; then
    if [ -z "${AZURE_STORAGE_CONNECTION_STRING:-}" ] && { [ -z "${AZURE_STORAGE_ACCOUNT_NAME:-}" ] || [ -z "${AZURE_STORAGE_ACCOUNT_KEY:-}" ]; }; then
      LOG "ENV=STAGING requires either AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME+AZURE_STORAGE_ACCOUNT_KEY when APPLY_STAGING_SECRETS=true"
      exit 2
    fi
  fi
  generate_manifests
  ensure_namespace
  if [ "${ENV}" = "STAGING" ] && [ "${APPLY_STAGING_SECRETS:-true}" = "true" ] && [ -n "${SECRET_BACKUP_NAME:-}" ]; then
    create_azure_backup_secret || { LOG "failed to create backup secret"; exit 2; }
  fi
  if [ -n "${QDRANT__SERVICE__API_KEY:-}" ] && [ -n "${SECRET_SERVICE_NAME:-}" ]; then
    kubectl -n "${QDRANT_NAMESPACE}" create secret generic "${SECRET_SERVICE_NAME}" --from-literal=QDRANT__SERVICE__API_KEY="${QDRANT__SERVICE__API_KEY:-}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1
    LOG "created/updated secret ${SECRET_SERVICE_NAME}"
  fi
  if ! helm_upgrade_install; then
    LOG "helm install failed"; exit 2
  fi
  LOG "helm install/upgrade succeeded"

  if ! kubectl apply -f "${SERVICE_PATCH_FILE}" >/dev/null 2>&1; then
    LOG "service patch apply returned non-zero"
  fi

  sleep 2

  if ! validate_service_post_install; then
    LOG "post-install validation errors"
    if [ "${FAIL_ON_MISCONFIG}" = "true" ]; then
      exit 2
    else
      LOG "continuing despite validation errors (FAIL_ON_MISCONFIG=false)"
    fi
  fi

  local summary
  summary=$(cat <<JSON
{
  "release":"${QDRANT_RELEASE}",
  "namespace":"${QDRANT_NAMESPACE}",
  "replicas":${QDRANT_REPLICAS},
  "values_file":"${VALUES_FILE}",
  "chart_version":"${CHART_VERSION}",
  "image":"${QDRANT_IMAGE}",
  "generated_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "secrets_created":$( [ -n "${SECRET_BACKUP_NAME:-}" ] || [ -n "${QDRANT__SERVICE__API_KEY:-}" ] && echo true || echo false )
}
JSON
)
  atomic_write "${LAST_SUMMARY}" "${summary}"
  LOG "Wrote deploy summary -> ${LAST_SUMMARY}"
}

delete_from_cluster(){
  if command -v kubectl >/dev/null 2>&1; then
    kubectl -n "${QDRANT_NAMESPACE}" delete deployment "${QDRANT_RELEASE}" --ignore-not-found || true
    kubectl -n "${QDRANT_NAMESPACE}" delete statefulset "${QDRANT_RELEASE}" --ignore-not-found || true
    kubectl -n "${QDRANT_NAMESPACE}" delete service "${QDRANT_RELEASE}" --ignore-not-found || true
    if [ -n "${SECRET_BACKUP_NAME:-}" ]; then
      kubectl -n "${QDRANT_NAMESPACE}" delete secret "${SECRET_BACKUP_NAME}" --ignore-not-found || true
    fi
    if [ -n "${SECRET_SERVICE_NAME:-}" ]; then
      kubectl -n "${QDRANT_NAMESPACE}" delete secret "${SECRET_SERVICE_NAME}" --ignore-not-found || true
    fi
  fi
  rm -rf "${MANIFESTS_DIR}" || true
  LOG "deleted manifests and cluster objects (best-effort)"
}

usage(){ printf '%s\n' "usage: $0 --generate|--apply|--delete [--force]"; exit 1; }

if [ "$#" -eq 0 ]; then usage; fi
cmd=""; force=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --generate) cmd="generate"; shift;;
    --apply) cmd="apply"; shift;;
    --delete) cmd="delete"; shift;;
    --force) force="--force"; shift;;
    --verbose) VERBOSE=1; shift;;
    *) usage;;
  esac
done

if [ "$cmd" = "generate" ]; then generate_manifests "${force}"; exit 0; fi
if [ "$cmd" = "apply" ]; then apply_to_cluster; exit $?; fi
if [ "$cmd" = "delete" ]; then delete_from_cluster; exit 0; fi
usage
