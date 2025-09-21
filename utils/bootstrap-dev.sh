#!/usr/bin/env bash
IFS=$'\n\t'
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export TZ=Etc/UTC
export MODEL_HOME="/opt/models"
export HF_HOME="$MODEL_HOME/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export RAPIDOCR_MODEL_DIR="$MODEL_HOME/rapidocr"

ARGOCD_VERSION="v3.0.12"
KUBECTL_VERSION="v1.33.3"
EKSCTL_VERSION="v0.212.0"
HELM_VERSION="v3.17.2"
PULUMI_VERSION="3.196.0"
SOPS_VERSION="v3.10.2"
K3D_TAG="v5.7.5"

log(){ printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"; }
require_sudo(){ if ! sudo -n true 2>/dev/null; then log "sudo access required: you'll be prompted for password..."; fi }
append_if_missing(){ local line="$1" target="$2"; if ! grep -Fxq "$line" "$target" 2>/dev/null; then printf '%s\n' "$line" >> "$target"; fi }

dl_and_install_single_binary(){ local url="$1" binname="$2" mode="${3:-0555}"; local tmp="/tmp/${binname}.$$"; log "Downloading ${binname}"; curl -fSL --retry 3 -o "${tmp}" "${url}"; sudo install -m "${mode}" "${tmp}" "/usr/local/bin/${binname}"; rm -f "${tmp}"; log "Installed /usr/local/bin/${binname}"; }

install_aws_cli(){
  if ! command -v aws >/dev/null 2>&1; then
    log "Installing aws CLI v2"
    curl -fSL --retry 3 -o /tmp/awscliv2.zip "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"
    unzip -q /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install --update
    rm -rf /tmp/aws /tmp/awscliv2.zip
  else
    log "aws present (skipping)"
  fi
}

install_kubectl(){
  if ! command -v kubectl >/dev/null 2>&1 || [[ "$(kubectl version --client --short 2>/dev/null || true)" != *"${KUBECTL_VERSION#v}"* ]]; then
    log "Installing kubectl ${KUBECTL_VERSION}"
    curl -fSL --retry 3 -o /tmp/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl"
    sudo install -m 0555 /tmp/kubectl /usr/local/bin/kubectl
    rm -f /tmp/kubectl
  else
    log "kubectl matches (skipping)"
  fi
}

install_eksctl(){
  if ! command -v eksctl >/dev/null 2>&1 || [[ "$(eksctl version 2>/dev/null || true)" != *"${EKSCTL_VERSION#v}"* ]]; then
    log "Installing eksctl ${EKSCTL_VERSION}"
    curl -fSL --retry 3 -o /tmp/eksctl.tar.gz "https://github.com/eksctl-io/eksctl/releases/download/${EKSCTL_VERSION}/eksctl_Linux_amd64.tar.gz"
    tar -xzf /tmp/eksctl.tar.gz -C /tmp
    sudo install -m 0755 /tmp/eksctl /usr/local/bin/eksctl
    rm -f /tmp/eksctl /tmp/eksctl.tar.gz
  else
    log "eksctl matches (skipping)"
  fi
}

install_argocd(){
  if ! command -v argocd >/dev/null 2>&1 || [[ "$(argocd version --client 2>/dev/null || true)" != *"${ARGOCD_VERSION#v}"* ]]; then
    log "Installing argocd ${ARGOCD_VERSION}"
    dl_and_install_single_binary "https://github.com/argoproj/argo-cd/releases/download/${ARGOCD_VERSION}/argocd-linux-amd64" "argocd" 0555
  else
    log "argocd matches (skipping)"
  fi
}

install_helm(){
  if ! command -v helm >/dev/null 2>&1 || [[ "$(helm version --short 2>/dev/null || true)" != *"${HELM_VERSION}"* ]]; then
    log "Installing helm ${HELM_VERSION}"
    curl -fSL --retry 3 -o /tmp/helm.tgz "https://get.helm.sh/helm-${HELM_VERSION}-linux-amd64.tar.gz"
    tar -xzf /tmp/helm.tgz -C /tmp
    sudo install -m 0555 /tmp/linux-amd64/helm /usr/local/bin/helm
    rm -rf /tmp/helm.tgz /tmp/linux-amd64
  else
    log "helm matches (skipping)"
  fi
}

install_pulumi(){
  if ! command -v pulumi >/dev/null 2>&1 || [[ "$(pulumi version 2>/dev/null || true)" != *"${PULUMI_VERSION}"* ]]; then
    log "Installing pulumi ${PULUMI_VERSION}"
    curl -fSL --retry 3 -o /tmp/pulumi.tgz "https://get.pulumi.com/releases/sdk/pulumi-v${PULUMI_VERSION}-linux-x64.tar.gz"
    sudo tar -xzf /tmp/pulumi.tgz -C /usr/local/bin --strip-components=1
    rm -f /tmp/pulumi.tgz
  else
    log "pulumi matches (skipping)"
  fi
}

install_k3d(){
  if ! command -v k3d >/dev/null 2>&1 ; then
    log "Installing k3d (TAG=${K3D_TAG})"
    curl -sSL "https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh" | TAG="${K3D_TAG}" bash
  else
    log "k3d present (skipping)"
  fi
}

install_tesseract(){
  if ! grep -q "^deb .*ppa.launchpadcontent.net/alex-p/tesseract-ocr5" /etc/apt/sources.list* 2>/dev/null; then
    sudo add-apt-repository -y ppa:alex-p/tesseract-ocr5
  fi
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends tesseract-ocr libtesseract-dev libleptonica-dev
}

setup_dirs(){
  sudo mkdir -p "${MODEL_HOME}/hf/hub" "${MODEL_HOME}/hf/assets" "${RAPIDOCR_MODEL_DIR}" /workspace/backups/dbs/arrangodb /workspace/data
  sudo chmod -R 0775 "${MODEL_HOME}" /workspace || true
  sudo chown -R "$(id -u):$(id -g)" "${MODEL_HOME}" /workspace || true
}

require_sudo
docker run --privileged --rm tonistiigi/binfmt --install all >/dev/null 2>&1 || true

log "Updating apt and installing base packages"
sudo apt-get update -yq
sudo apt-get upgrade -yq || true
sudo apt-get install -yq --no-install-recommends \
  ca-certificates curl wget git gh sudo tree jq unzip vim make python3.10-venv python3-pip \
  build-essential gnupg lsb-release software-properties-common zip apt-transport-https fonts-dejavu fonts-liberation dos2unix yamllint unzip

log "Installing yq"
sudo wget -q https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/local/bin/yq
sudo chmod +x /usr/local/bin/yq

log "Installing git-lfs"
sudo curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get update -yq
sudo apt-get install -y git-lfs
sudo git lfs install || true

install_aws_cli
install_kubectl
install_eksctl
install_argocd
install_helm
install_pulumi
install_k3d
install_tesseract

setup_dirs

append_if_missing "export MODEL_HOME=\"${MODEL_HOME}\"" ~/.bashrc
append_if_missing "export HF_HOME=\"${HF_HOME}\"" ~/.bashrc
append_if_missing "export HF_HUB_CACHE=\"${HF_HUB_CACHE}\"" ~/.bashrc
append_if_missing "export HF_ASSETS_CACHE=\"${HF_ASSETS_CACHE}\"" ~/.bashrc
append_if_missing "export RAPIDOCR_MODEL_DIR=\"${RAPIDOCR_MODEL_DIR}\"" ~/.bashrc
append_if_missing 'export DEBIAN_FRONTEND=noninteractive' ~/.bashrc
append_if_missing 'export PYTHONPATH=$(pwd)' ~/.bashrc

log "Verification (versions):"
for cmd in aws kubectl eksctl argocd helm pulumi k3d git python3 pip3; do
  if command -v "${cmd}" >/dev/null 2>&1; then
    printf '  %-8s -> %s\n' "${cmd}" "$(${cmd} --version 2>/dev/null | head -n1 || echo 'version unknown')"
  else
    printf '  %-8s -> not installed\n' "${cmd}"
  fi
done

log "Installing Python packages"
pip3 install --upgrade pip || true
pip3 install huggingface_hub==0.34.4 || true
pip3 install pulumi==${PULUMI_VERSION} pulumi-aws==7.7.0 || true

if [ -f utils/archive/download_hf.py ]; then
  sudo python3 utils/archive/download_hf.py || true
fi

cd /usr/local/bin
if curl -I -s https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz >/dev/null 2>&1; then
  log "Downloading ffmpeg static (best-effort)"
  sudo curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o /tmp/ffmpeg.tar.xz || true
  if [ -f /tmp/ffmpeg.tar.xz ]; then
    tar -xf /tmp/ffmpeg.tar.xz -C /tmp || true
    sudo cp /tmp/ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ 2>/dev/null || true
    sudo cp /tmp/ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ 2>/dev/null || true
    rm -rf /tmp/ffmpeg* || true
  fi
fi
cd - >/dev/null 2>&1 || true

docker pull athithya324/embedder-cpu-inference:linux-amd64-arm64 || true

pulumi plugin install resource aws v7.7.0 2>/dev/null || true

clear
echo "Bootstrap completed. Open a new terminal or run: source ~/.bashrc"
