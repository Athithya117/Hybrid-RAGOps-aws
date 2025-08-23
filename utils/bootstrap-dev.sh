#!/usr/bin/env bash



IFS=$'\n\t'

# --- Predefined env (user requested)
export DEBIAN_FRONTEND=noninteractive
export TZ=Etc/UTC
export MODEL_HOME="/opt/models"
export HF_HOME="$MODEL_HOME/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export RAPIDOCR_MODEL_DIR="$MODEL_HOME/rapidocr"

# Tool versions (pinned)
ARGOCD_VERSION="v3.0.12"
KUBECTL_VERSION="v1.33.3"
EKSCTL_VERSION="v0.212.0"
HELM_VERSION="v3.17.2"
PULUMI_VERSION="3.186.0"
NODE_VERSION="22.x"         # Node setup script will install Node.js v22 LTS (22.15.x)
SOPS_VERSION="v3.10.2"
K3D_TAG="v5.7.5"            # default k3d tag; installer supports TAG override

# Helpers
log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"; }
require_sudo() {
  if ! sudo -n true 2>/dev/null; then
    log "sudo access required: you'll be prompted for password..."
  fi
}

echo "Installing latest Tesseract 5.x from alex-p PPA..."
sudo apt-get update -y
if ! grep -q "^deb .\+ppa.launchpadcontent.net/alex-p/tesseract-ocr5" /etc/apt/sources.list*; then
   add-apt-repository -y ppa:alex-p/tesseract-ocr5
fi
sudo apt-get update -y
sudo apt-get install -y yamllint tesseract-ocr libtesseract-dev libleptonica-dev
sudo snap install yq

sudo add-apt-repository ppa:libreoffice/ppa -y && \
sudo apt-get update && \
sudo apt-get install -y libreoffice \
  ttf-mscorefonts-installer fonts-dejavu fonts-liberation && \


sudo curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
sudo apt-get update
sudo apt-get install -y git-lfs
sudo git lfs install

# Basic system prep
require_sudo
log "[1/14] apt update/upgrade + essential packages"
sudo apt-get update -yq
sudo apt-get upgrade -yq
sudo apt-get install -yq --no-install-recommends \
  ca-certificates curl wget git sudo tree jq unzip make \
  python3.10-venv python3-pip build-essential \
  gnupg lsb-release software-properties-common zip \
  unzip apt-transport-https

# Atomic download+install helper for single-file binaries
dl_and_install_single_binary() {
  local url="$1" binname="$2" mode="${3:-0555}"
  local tmp="/tmp/${binname}.$$"
  log "Downloading ${binname} from ${url}"
  curl -fSL --retry 3 --retry-delay 2 -o "${tmp}" "${url}"
  sudo install -m "${mode}" "${tmp}" "/usr/local/bin/${binname}"
  rm -f "${tmp}"
  log "Installed /usr/local/bin/${binname}"
}

# ----- Tool installations -----

# AWS CLI v2 (official installer)
install_aws_cli() {
  if ! command -v aws >/dev/null 2>&1; then
    log "Installing AWS CLI v2"
    local tmpzip="/tmp/awscliv2.$$ .zip"
    curl -fSL --retry 3 -o /tmp/awscliv2.zip "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"
    unzip -q /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install --update
    rm -rf /tmp/aws /tmp/awscliv2.zip
  else
    log "aws CLI present (skipping)"
  fi
}

# kubectl (pinned)
install_kubectl() {
  if ! command -v kubectl >/dev/null 2>&1 || [[ "$(kubectl version --client --short 2>/dev/null || true)" != *"${KUBECTL_VERSION#v}"* ]]; then
    log "Installing kubectl ${KUBECTL_VERSION}"
    curl -fSL --retry 3 -o /tmp/kubectl "${KUBECTL_DOWNLOAD:-https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl}"
    sudo install -m 0555 /tmp/kubectl /usr/local/bin/kubectl
    rm -f /tmp/kubectl
  else
    log "kubectl already matches ${KUBECTL_VERSION} (skipping)"
  fi
}

# eksctl (pinned)
install_eksctl() {
  if ! command -v eksctl >/dev/null 2>&1 || [[ "$(eksctl version 2>/dev/null || true)" != *"${EKSCTL_VERSION#v}"* ]]; then
    log "Installing eksctl ${EKSCTL_VERSION}"
    local platform="Linux_amd64"
    # GitHub releases provide compressed tarball; use official pattern:
    curl -fSL --retry 3 -o /tmp/eksctl.tar.gz "https://github.com/eksctl-io/eksctl/releases/download/${EKSCTL_VERSION}/eksctl_${platform}.tar.gz"
    tar -xzf /tmp/eksctl.tar.gz -C /tmp
    sudo install -m 0755 /tmp/eksctl /usr/local/bin/eksctl
    rm -f /tmp/eksctl /tmp/eksctl.tar.gz
  else
    log "eksctl already matches ${EKSCTL_VERSION} (skipping)"
  fi
}

# Argo CD CLI (pinned)
install_argocd() {
  if ! command -v argocd >/dev/null 2>&1 || [[ "$(argocd version --client 2>/dev/null || true)" != *"${ARGOCD_VERSION#v}"* ]]; then
    log "Installing argocd ${ARGOCD_VERSION}"
    local url="https://github.com/argoproj/argo-cd/releases/download/${ARGOCD_VERSION}/argocd-linux-amd64"
    dl_and_install_single_binary "${url}" "argocd" 0555
  else
    log "argocd already matches ${ARGOCD_VERSION} (skipping)"
  fi
}

# Helm (pinned)
install_helm() {
  if ! command -v helm >/dev/null 2>&1 || [[ "$(helm version --short 2>/dev/null || true)" != *"${HELM_VERSION}"* ]]; then
    log "Installing helm ${HELM_VERSION}"
    curl -fSL --retry 3 -o /tmp/helm.tgz "https://get.helm.sh/helm-${HELM_VERSION}-linux-amd64.tar.gz"
    tar -xzf /tmp/helm.tgz -C /tmp
    sudo install -m 0555 /tmp/linux-amd64/helm /usr/local/bin/helm
    rm -rf /tmp/helm.tgz /tmp/linux-amd64
  else
    log "helm already matches ${HELM_VERSION} (skipping)"
  fi
}

# Pulumi (pinned)
install_pulumi() {
  if ! command -v pulumi >/dev/null 2>&1 || [[ "$(pulumi version 2>/dev/null || true)" != *"${PULUMI_VERSION}"* ]]; then
    log "Installing pulumi ${PULUMI_VERSION}"
    curl -fSL --retry 3 -o /tmp/pulumi.tgz "https://get.pulumi.com/releases/sdk/pulumi-v${PULUMI_VERSION}-linux-x64.tar.gz"
    sudo tar -xzf /tmp/pulumi.tgz -C /usr/local/bin --strip-components=1
    rm -f /tmp/pulumi.tgz
  else
    log "pulumi already matches ${PULUMI_VERSION} (skipping)"
  fi
}


# k3d (official installer; supports TAG override)
install_k3d() {
  if ! command -v k3d >/dev/null 2>&1 ; then
    log "Installing k3d (installer will pick TAG=${K3D_TAG})"
    # default installer supports environment TAG override for pinned release
    curl -sSL "https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh" | TAG="${K3D_TAG}" bash
  else
    log "k3d present (skipping)"
  fi
}

# Tesseract 5.x via PPA
install_tesseract() {
  log "Installing Tesseract 5.x (alex-p PPA)"
  if ! grep -q "^deb .\+ppa.launchpadcontent.net/alex-p/tesseract-ocr5" /etc/apt/sources.list* 2>/dev/null; then
    sudo add-apt-repository -y ppa:alex-p/tesseract-ocr5
  fi
  sudo apt-get update -y
  sudo apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev
}

# Misc directories & perms
setup_dirs() {
  log "Creating model and workspace directories"
  sudo mkdir -p "${MODEL_HOME}/hf/hub" "${MODEL_HOME}/hf/assets" /workspace/backups/dbs/arrangodb /workspace/data
  sudo chmod -R 0775 "${MODEL_HOME}" /workspace || true
  sudo chown -R "$(id -u):$(id -g)" "${MODEL_HOME}" /workspace || true
}

# Add env to user's bashrc if missing
append_if_missing() {
  local line="$1" target="$2"
  if ! grep -Fxq "${line}" "${target}" 2>/dev/null; then
    printf '%s\n' "${line}" >> "${target}"
  fi
}

# Run installers
install_aws_cli
install_kubectl
install_eksctl
install_argocd
install_helm
install_pulumi
install_k3d
install_tesseract

# Persist environment variables to ~/.bashrc
append_if_missing "export MODEL_HOME=\"${MODEL_HOME}\"" ~/.bashrc
append_if_missing "export HF_HOME=\"${HF_HOME}\"" ~/.bashrc
append_if_missing "export HF_HUB_CACHE=\"${HF_HUB_CACHE}\"" ~/.bashrc
append_if_missing "export HF_ASSETS_CACHE=\"${HF_ASSETS_CACHE}\"" ~/.bashrc
append_if_missing "export RAPIDOCR_MODEL_DIR=\"${RAPIDOCR_MODEL_DIR}\"" ~/.bashrc
append_if_missing 'export DEBIAN_FRONTEND=noninteractive' ~/.bashrc

# Ensure pythonpath convenience
append_if_missing 'export PYTHONPATH=$(pwd)' ~/.bashrc

# Final verification summary
log "Verification (versions):"
for cmd in aws kubectl eksctl argocd helm pulumi k3d; do
  if command -v "${cmd}" >/dev/null 2>&1; then
    printf '  %-8s -> %s\n' "${cmd}" "$(${cmd} --version 2>/dev/null | head -n1 || echo 'version unknown')"
  else
    printf '  %-8s -> not installed\n' "${cmd}"
  fi
done

log "Bootstrap completed. Restart your shell or run: source ~/.bashrc"
