#!/usr/bin/env bash
# bootstrap_ops.sh — Minimal Ops workstation bootstrap (includes Pulumi)
set -euo pipefail
IFS=$'\n\t'

# ---- Version pins (edit if you need other versions) ----
ARGOCD_VERSION="v3.0.12"
KUBECTL_VERSION="v1.33.3"
EKSCTL_VERSION="v0.212.0"
HELM_VERSION="v3.17.2"
PULUMI_VERSION="3.196.0"

# ---- Helpers ----
TS() { date --iso-8601=seconds; }
log(){ printf '%s %s\n' "$(TS)" "$*"; }
download_with_retries(){
  local url="$1"; local out="$2"; local tries="${3:-5}"
  local i=0
  while :; do
    i=$((i+1))
    log "Downloading ${url} (attempt ${i}/${tries})"
    if curl -fsSL --retry 3 -o "${out}" "${url}"; then
      return 0
    fi
    if [ "${i}" -ge "${tries}" ]; then
      log "FAILED to download ${url}"
      return 1
    fi
    sleep $((i*i))
  done
}

install_if_missing(){ # install_if_missing <cmd> <check_version_substring> <installer_fn>
  local cmd="$1"; local want="$2"; local installer_fn="$3"
  if command -v "${cmd}" >/dev/null 2>&1; then
    if [ -n "${want}" ]; then
      ver="$(${cmd} --version 2>/dev/null || true)"
      if [[ "${ver}" == *"${want#v}"* || "${ver}" == *"${want}"* ]]; then
        log "${cmd} already installed and matches ${want}"
        return 0
      fi
    else
      log "${cmd} already installed"
      return 0
    fi
  fi
  ${installer_fn}
}

# ---- Sudo check ----
if ! sudo -n true 2>/dev/null; then
  log "sudo required: asking for credentials..."
  sudo -v
fi

# ---- Apt base packages ----
log "apt update + base packages"
sudo apt-get update -yq
sudo apt-get install -yq --no-install-recommends \
  ca-certificates curl vim make wget git jq unzip python3 python3-venv python3-pip make build-essential \
  lsb-release gnupg apt-transport-https dpkg-dev || true

# ---- yq (mikefarah) ----
install_yq(){
  local dest="/usr/local/bin/yq"
  tmp="$(mktemp)"
  download_with_retries "https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64" "${tmp}" || { log "yq download failed"; return 1; }
  sudo install -m 0755 "${tmp}" "${dest}"
  rm -f "${tmp}"
  log "yq installed to ${dest}"
}
install_if_missing yq "" install_yq

# ---- jq already ensured by apt (above) ----

# ---- AWS CLI v2 ----
install_aws(){
  tmpzip="$(mktemp)"; tmpd="$(mktemp -d)"
  download_with_retries "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" "${tmpzip}" || { log "aws cli download failed"; return 1; }
  unzip -q "${tmpzip}" -d "${tmpd}"
  sudo "${tmpd}/aws/install" --update || sudo "${tmpd}/aws/install" || true
  rm -rf "${tmpzip}" "${tmpd}"
  log "aws cli installed"
}
install_if_missing aws "" install_aws

# ---- kubectl ----
install_kubectl(){
  tmp="$(mktemp)"
  download_with_retries "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" "${tmp}" || { log "kubectl download failed"; return 1; }
  sudo install -m 0555 "${tmp}" /usr/local/bin/kubectl
  rm -f "${tmp}"
  log "kubectl ${KUBECTL_VERSION} installed"
}
install_if_missing kubectl "${KUBECTL_VERSION}" install_kubectl

# ---- eksctl ----
install_eksctl(){
  tmpgz="$(mktemp)"; tmpd="$(mktemp -d)"
  download_with_retries "https://github.com/eksctl-io/eksctl/releases/download/${EKSCTL_VERSION}/eksctl_Linux_amd64.tar.gz" "${tmpgz}" || { log "eksctl download failed"; return 1; }
  tar -xzf "${tmpgz}" -C "${tmpd}"
  sudo install -m 0755 "${tmpd}/eksctl" /usr/local/bin/eksctl
  rm -rf "${tmpgz}" "${tmpd}"
  log "eksctl ${EKSCTL_VERSION} installed"
}
install_if_missing eksctl "${EKSCTL_VERSION}" install_eksctl

# ---- argocd CLI ----
install_argocd(){
  tmp="$(mktemp)"
  download_with_retries "https://github.com/argoproj/argo-cd/releases/download/${ARGOCD_VERSION}/argocd-linux-amd64" "${tmp}" || { log "argocd download failed"; return 1; }
  sudo install -m 0555 "${tmp}" /usr/local/bin/argocd
  rm -f "${tmp}"
  log "argocd ${ARGOCD_VERSION} installed"
}
install_if_missing argocd "${ARGOCD_VERSION}" install_argocd

# ---- helm ----
install_helm(){
  tmp="$(mktemp)"; tgtd="$(mktemp -d)"
  download_with_retries "https://get.helm.sh/helm-${HELM_VERSION}-linux-amd64.tar.gz" "${tmp}" || { log "helm download failed"; return 1; }
  tar -xzf "${tmp}" -C "${tgtd}"
  sudo install -m 0555 "${tgtd}/linux-amd64/helm" /usr/local/bin/helm
  rm -rf "${tmp}" "${tgtd}"
  log "helm ${HELM_VERSION} installed"
}
install_if_missing helm "${HELM_VERSION}" install_helm

# ---- pulumi ----
install_pulumi(){
  tmpf="$(mktemp)"; tmpd="$(mktemp -d)"; attempts=0; max=5
  url="https://get.pulumi.com/releases/sdk/pulumi-v${PULUMI_VERSION}-linux-x64.tar.gz"
  while ! download_with_retries "${url}" "${tmpf}" 3; do
    attempts=$((attempts+1))
    if [ "${attempts}" -ge "${max}" ]; then
      log "pulumi download failed after ${attempts} attempts; trying official installer"
      break
    fi
    sleep $((attempts*2))
  done
  if [ -f "${tmpf}" ]; then
    tar -xzf "${tmpf}" -C "${tmpd}" || { log "pulumi extract failed"; rm -f "${tmpf}"; rm -rf "${tmpd}"; return 1; }
    find "${tmpd}" -type f -perm /111 -exec sudo install -m 0755 {} /usr/local/bin/ \; || true
    rm -f "${tmpf}"; rm -rf "${tmpd}"
    log "pulumi binaries installed"
  else
    # fallback to official install script (safe for workstation)
    if curl -fsSL https://get.pulumi.com | sudo -E sh -s -- --version "${PULUMI_VERSION}"; then
      log "pulumi installed via official installer"
    else
      log "pulumi install failed (both archive and installer)"
      return 1
    fi
  fi
}
install_if_missing pulumi "${PULUMI_VERSION}" install_pulumi

log "Bootstrap completed. Open a new shell or source your rc file to pick up any env changes."
