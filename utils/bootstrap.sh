#!/usr/bin/env bash
# utils/bootstrap.sh — idempotent workstation bootstrap (Azure CLI, kubectl, Helm, Pulumi pinned)
# - Safe to re-run
# - No silent redirections: all command output is shown
# - PYTHON_PKGS declared as an array to avoid IFS/word-splitting bugs
set -euo pipefail
IFS=$'\n\t'   # keep newline+tab in IFS for safe iteration, but arrays are used for space-splitting

# ---- Version pins (edit to upgrade deliberately) ----
AZ_CLI_VERSION="2.61.0-1~jammy"
HELM_VERSION="v3.17.2"
PULUMI_VERSION="3.212.0"

# ---- Python packages declared as an array (safe with any IFS) ----
PYTHON_PKGS=(
  "azure-core==1.30.2"
  "azure-identity==1.16.0"
  "azure-mgmt-storage==21.2.1"
  "azure-storage-blob==12.27.1"
  "jinja2==3.1.6"
  "ruamel.yaml==0.18.16"
)

# ---- Helpers ----
TS(){ date --iso-8601=seconds; }
log(){ printf '%s %s\n' "$(TS)" "$*"; }
die(){ printf '%s ERROR: %s\n' "$(TS)" "$*" >&2; exit 1; }

download_with_retries(){
  local url="$1"; local out="$2"; local tries="${3:-5}"
  local i=0
  while :; do
    i=$((i+1))
    log "Downloading ${url} (attempt ${i}/${tries})"
    if curl -fsSL --retry 3 --retry-delay 2 -o "${out}" "${url}"; then
      return 0
    fi
    log "Warning: download attempt ${i} failed for ${url}"
    if [ "${i}" -ge "${tries}" ]; then
      log "FAILED to download ${url} after ${tries} attempts"
      return 1
    fi
    sleep $((i*i))
  done
}

install_if_missing(){
  local cmd="$1"; local want_substr="$2"; local installer_fn="$3"
  if command -v "${cmd}" >/dev/null 2>&1; then
    if [ -n "${want_substr}" ]; then
      local ver
      ver="$("${cmd}" --version 2>/dev/null || true)"
      if printf '%s' "${ver}" | grep -q "${want_substr#v}" || printf '%s' "${ver}" | grep -q "${want_substr}"; then
        log "${cmd} already installed and matches ${want_substr}"
        return 0
      fi
    else
      log "${cmd} already installed"
      return 0
    fi
  fi
  "${installer_fn}"
}

# ---- Require sudo ----
if ! sudo -n true 2>/dev/null; then
  log "sudo is required for some package operations; prompting for password"
  sudo -v
fi

# ---- Base packages ----
log "Installing base OS packages (apt)"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  ca-certificates curl wget git jq unzip gnupg lsb-release \
  python3 python3-venv python3-pip python3-dev build-essential apt-transport-https \
  libffi-dev libssl-dev

# Ensure user-local bin and pulumi bin are on PATH for this run
export PATH="${HOME}/.local/bin:${HOME}/.pulumi/bin:${PATH}"

# ---- yq ----
install_yq(){
  local tmp; tmp="$(mktemp)"
  download_with_retries "https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64" "${tmp}"
  sudo install -m 0755 "${tmp}" /usr/local/bin/yq
  rm -f "${tmp}"
  log "yq installed at /usr/local/bin/yq"
}
install_if_missing yq "" install_yq

# ---- Azure CLI (apt pinned) ----
install_azure_cli(){
  log "Installing Azure CLI ${AZ_CLI_VERSION}"
  sudo rm -f /etc/apt/sources.list.d/azure-cli.* || true
  sudo mkdir -p /etc/apt/keyrings
  curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | sudo tee /etc/apt/keyrings/microsoft.gpg >/dev/null
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/azure-cli/ $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/azure-cli.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y --allow-downgrades --allow-change-held-packages azure-cli="${AZ_CLI_VERSION}"
  log "az installed"
}
install_if_missing az "2.61.0" install_azure_cli

# ---- kubectl ----
install_kubectl_aks(){
  log "Installing kubectl via Azure CLI (az aks install-cli)"
  az aks install-cli
  log "kubectl installed/updated"
}
install_if_missing kubectl "" install_kubectl_aks

# ---- Helm ----
install_helm(){
  log "Installing Helm ${HELM_VERSION}"
  local tmp tgtd
  tmp="$(mktemp)"
  tgtd="$(mktemp -d)"
  download_with_retries "https://get.helm.sh/helm-${HELM_VERSION}-linux-amd64.tar.gz" "${tmp}"
  tar -xzf "${tmp}" -C "${tgtd}"
  sudo install -m 0555 "${tgtd}/linux-amd64/helm" /usr/local/bin/helm
  rm -rf "${tmp}" "${tgtd}"
  log "helm installed at /usr/local/bin/helm"
}
install_if_missing helm "${HELM_VERSION}" install_helm

# ---- Pulumi ----
install_pulumi(){
  log "Installing Pulumi ${PULUMI_VERSION} to ${HOME}/.pulumi/bin and /usr/local/bin"
  local tmp tmpd relurl
  tmp="$(mktemp)"
  tmpd="$(mktemp -d)"
  relurl="https://get.pulumi.com/releases/sdk/pulumi-v${PULUMI_VERSION}-linux-x64.tar.gz"
  download_with_retries "${relurl}" "${tmp}"
  tar -xzf "${tmp}" -C "${tmpd}"
  if [ -d "${tmpd}/pulumi" ]; then
    mkdir -p "${HOME}/.pulumi/bin"
    sudo install -m 0755 "${tmpd}/pulumi/pulumi" /usr/local/bin/pulumi || true
    sudo install -m 0755 "${tmpd}/pulumi/pulumi-language-python" /usr/local/bin/pulumi-language-python || true
    [ -f "${tmpd}/pulumi/pulumi-language-python-exec" ] && sudo install -m 0755 "${tmpd}/pulumi/pulumi-language-python-exec" /usr/local/bin/pulumi-language-python-exec || true
    install -m 0755 "${tmpd}/pulumi/pulumi" "${HOME}/.pulumi/bin/pulumi" || true
    [ -f "${tmpd}/pulumi/pulumi-language-python" ] && install -m 0755 "${tmpd}/pulumi/pulumi-language-python" "${HOME}/.pulumi/bin/pulumi-language-python" || true
    [ -f "${tmpd}/pulumi/pulumi-language-python-exec" ] && install -m 0755 "${tmpd}/pulumi/pulumi-language-python-exec" "${HOME}/.pulumi/bin/pulumi-language-python-exec" || true
    export PATH="${HOME}/.pulumi/bin:${PATH}"
    log "Pulumi ${PULUMI_VERSION} installed"
    if ! command -v pulumi-language-python-exec >/dev/null 2>&1; then
      log "pulumi-language-python-exec not found; using get.pulumi.com fallback"
      curl -fsSL https://get.pulumi.com | sh
      export PATH="${HOME}/.pulumi/bin:${PATH}"
    fi
    if ! command -v pulumi-language-python-exec >/dev/null 2>&1; then
      die "pulumi-language-python-exec still missing after install; aborting"
    fi
    rm -rf "${tmp}" "${tmpd}"
  else
    rm -rf "${tmp}" "${tmpd}"
    die "Pulumi archive layout unexpected"
  fi
}
install_if_missing pulumi "${PULUMI_VERSION}" install_pulumi

# ---- Python packages (system pip) — show all output directly ----
log "Installing selected Python packages with system pip (outputs shown on stdout/stderr)."

# Upgrade pip/setuptools/wheel first (no redirection)
log "Upgrading pip, setuptools, wheel (may require sudo for system-managed environments)."
set +o errexit
python3 -m pip install --upgrade pip setuptools wheel --no-cache-dir
UPGRADE_EXIT=$?
set -o errexit
if [ "${UPGRADE_EXIT}" -ne 0 ]; then
  log "pip upgrade failed with exit ${UPGRADE_EXIT}. Trying sudo elevation with --break-system-packages if available."
  set +o errexit
  sudo -H python3 -m pip install --upgrade pip setuptools wheel --no-cache-dir --break-system-packages || true
  set -o errexit
  log "After sudo attempt, pip version:"
  python3 -m pip --version || true
fi

# Install packages — pass array so each package is a distinct argv
set +o errexit
python3 -m pip install --upgrade --no-cache-dir "${PYTHON_PKGS[@]}"
PKG_EXIT=$?
set -o errexit

if [ "${PKG_EXIT}" -ne 0 ]; then
  log "Primary pip install failed (exit ${PKG_EXIT}). Retrying with sudo elevation."
  set +o errexit
  # Use --break-system-packages if pip indicates system protection (pip >= 23 typically)
  pip_major=0
  pip_ver="$(python3 -m pip --version 2>/dev/null || true)"
  if [ -n "${pip_ver}" ]; then
    pip_major="$(printf '%s' "${pip_ver}" | awk '{print $2}' | cut -d. -f1 2>/dev/null || echo 0)"
  fi

  if [ "${pip_major}" -ge 23 ]; then
    sudo -H python3 -m pip install --upgrade --no-cache-dir --break-system-packages "${PYTHON_PKGS[@]}"
    PKG_EXIT=$?
  else
    sudo -H python3 -m pip install --upgrade --no-cache-dir "${PYTHON_PKGS[@]}"
    PKG_EXIT=$?
  fi
  set -o errexit
fi

if [ "${PKG_EXIT}" -ne 0 ]; then
  log "System pip installs failed after sudo attempt (exit ${PKG_EXIT}). Showing diagnostics and performing virtualenv fallback."
  log "python3 --version: $(python3 --version 2>&1 || true)"
  log "which python3: $(command -v python3 || true)"
  log "python3 -m pip --version: $(python3 -m pip --version 2>&1 || true)"
  log "pip list (system):"
  python3 -m pip list || true

  log "Creating a virtualenv at ${HOME}/.bootstrap-venv and installing packages there (outputs shown)."
  python3 -m venv "${HOME}/.bootstrap-venv"
  "${HOME}/.bootstrap-venv/bin/python" -m pip install --upgrade pip setuptools wheel
  "${HOME}/.bootstrap-venv/bin/python" -m pip install --no-cache-dir "${PYTHON_PKGS[@]}"
  log "Virtualenv fallback completed; to use those packages: source ${HOME}/.bootstrap-venv/bin/activate"
  die "pip install of python pkgs failed system-wide. Virtualenv created. Inspect above pip errors to diagnose; common causes: distro package protection or missing build deps."
fi

log "Python packages installed system-wide successfully."

# ---- Final checks ----
if ! command -v pulumi >/dev/null 2>&1; then
  die "pulumi binary not found after install"
fi
if ! command -v pulumi-language-python-exec >/dev/null 2>&1; then
  die "pulumi-language-python-exec missing after install; see above output"
fi
if ! command -v az >/dev/null 2>&1; then
  die "az CLI not found after install"
fi
if ! command -v helm >/dev/null 2>&1; then
  die "helm not found after install"
fi

# Ensure pulumi bin directory is present in common shell rc files (idempotent)
add_path_to_rc(){
  local rcfile="$1"
  local line='export PATH="$HOME/.pulumi/bin:$HOME/.local/bin:$PATH"'
  if [ -f "${rcfile}" ]; then
    if ! grep -Fq "${line}" "${rcfile}"; then
      printf "\n# Added by bootstrap.sh - ensure pulumi & user-local bin in PATH\n%s\n" "${line}" | tee -a "${rcfile}" >/dev/null
      log "Appended PATH export to ${rcfile}"
    fi
  else
    printf "%s\n" "${line}" > "${rcfile}"
    log "Created ${rcfile} with PATH export"
  fi
}
add_path_to_rc "${HOME}/.profile"
add_path_to_rc "${HOME}/.bashrc"

log "Bootstrap completed. Pulumi v: $(pulumi version 2>/dev/null || true) pulumi-language-python-exec: $(command -v pulumi-language-python-exec || true)"
log "If you use a login shell, ensure PATH contains ${HOME}/.pulumi/bin."
exit 0
