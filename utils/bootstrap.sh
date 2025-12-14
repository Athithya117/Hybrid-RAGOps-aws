#!/usr/bin/env bash
# bootstrap_ops.sh — Minimal Azure Ops workstation bootstrap (AKS + Pulumi)
set -euo pipefail
IFS=$'\n\t'

# ---- Version pins (edit intentionally) ----
AZ_CLI_VERSION="2.61.0-1~jammy"
HELM_VERSION="v3.17.2"
PULUMI_VERSION="3.206.0"

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

install_if_missing(){ # <cmd> <version_substring> <installer_fn>
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
  log "sudo required"
  sudo -v
fi

# ---- Base packages ----
log "Installing base packages"
sudo apt-get update -yq
sudo apt-get install -yq --no-install-recommends \
  ca-certificates curl vim make wget git jq unzip \
  python3 python3-venv python3-pip build-essential \
  lsb-release gnupg apt-transport-https dpkg-dev

# ---- yq ----
install_yq(){
  local tmp
  tmp="$(mktemp)"
  download_with_retries \
    "https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64" \
    "${tmp}"
  sudo install -m 0755 "${tmp}" /usr/local/bin/yq
  rm -f "${tmp}"
}
install_if_missing yq "" install_yq

# ---- Azure CLI (PINNED, apt, clean) ----
install_azure_cli(){
  log "Installing Azure CLI ${AZ_CLI_VERSION}"

  sudo rm -f /etc/apt/sources.list.d/azure-cli.*
  sudo rm -f /etc/apt/keyrings/microsoft.gpg
  sudo rm -rf ~/.azure || true

  sudo mkdir -p /etc/apt/keyrings
  curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
    | gpg --dearmor \
    | sudo tee /etc/apt/keyrings/microsoft.gpg > /dev/null

  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/microsoft.gpg] \
https://packages.microsoft.com/repos/azure-cli/ $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/azure-cli.list

  sudo apt-get update -yq
  sudo apt-get install -y \
    azure-cli="${AZ_CLI_VERSION}" \
    --allow-downgrades \
    --allow-change-held-packages
}
install_if_missing az "2.61.0" install_azure_cli

# ---- AKS kubectl (Azure-managed) ----
install_kubectl_aks(){
  log "Installing kubectl via Azure CLI (AKS-supported)"
  az aks install-cli
}
install_if_missing kubectl "" install_kubectl_aks

# ---- Helm ----
install_helm(){
  local tmp tgtd
  tmp="$(mktemp)"
  tgtd="$(mktemp -d)"
  download_with_retries \
    "https://get.helm.sh/helm-${HELM_VERSION}-linux-amd64.tar.gz" \
    "${tmp}"
  tar -xzf "${tmp}" -C "${tgtd}"
  sudo install -m 0555 "${tgtd}/linux-amd64/helm" /usr/local/bin/helm
  rm -rf "${tmp}" "${tgtd}"
}
install_if_missing helm "${HELM_VERSION}" install_helm

# ---- Pulumi ----
install_pulumi(){
  local tmp tmpd
  tmp="$(mktemp)"
  tmpd="$(mktemp -d)"
  download_with_retries \
    "https://get.pulumi.com/releases/sdk/pulumi-v${PULUMI_VERSION}-linux-x64.tar.gz" \
    "${tmp}"
  tar -xzf "${tmp}" -C "${tmpd}"
  sudo install -m 0755 "${tmpd}/pulumi/pulumi" /usr/local/bin/pulumi
  sudo install -m 0755 "${tmpd}/pulumi/pulumi-language-python" /usr/local/bin/
  rm -rf "${tmp}" "${tmpd}"
}
install_if_missing pulumi "${PULUMI_VERSION}" install_pulumi

pip install --no-cache-dir \
  azure-core==1.30.2 \
  azure-identity==1.16.0 \
  azure-mgmt-core==1.4.0 \
  azure-mgmt-storage==21.2.1

log "Bootstrap completed (Azure-only). Open a new shell if needed."
