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
PULUMI_VERSION="3.186.0"

log(){ printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"; }
require_sudo(){ if ! sudo -n true 2>/dev/null; then log "sudo access required: you'll be prompted for password..."; fi }

# --- Small helpers ---
append_if_missing(){ local line="$1" target="$2"; if ! grep -Fxq "$line" "$target" 2>/dev/null; then printf '%s\n' "$line" >> "$target"; fi }

setup_dirs(){
  log "Creating model and workspace directories"
  sudo mkdir -p "${MODEL_HOME}/hf/hub" "${MODEL_HOME}/hf/assets" /workspace/data
  sudo chmod -R 0775 "${MODEL_HOME}" /workspace || true
  sudo chown -R "$(id -u):$(id -g)" "${MODEL_HOME}" /workspace || true
}

install_aws_cli(){
  if ! command -v aws >/dev/null 2>&1; then
    log "Installing AWS CLI v2"
    curl -fSL --retry 3 -o /tmp/awscliv2.zip "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"
    unzip -q /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install --update
    rm -rf /tmp/aws /tmp/awscliv2.zip
  else
    log "aws CLI present (skipping)"
  fi
}

install_pulumi(){
  if ! command -v pulumi >/dev/null 2>&1 || [[ "$(pulumi version 2>/dev/null || true)" != *"${PULUMI_VERSION}"* ]]; then
    log "Installing pulumi ${PULUMI_VERSION}"
    curl -fSL --retry 3 -o /tmp/pulumi.tgz "https://get.pulumi.com/releases/sdk/pulumi-v${PULUMI_VERSION}-linux-x64.tar.gz"
    sudo tar -xzf /tmp/pulumi.tgz -C /usr/local/bin --strip-components=1
    rm -f /tmp/pulumi.tgz
  else
    log "pulumi already matches ${PULUMI_VERSION} (skipping)"
  fi
}

install_tesseract(){
  log "Installing Tesseract 5.x (alex-p PPA)"
  sudo apt-get update -y
  if ! grep -q "^deb .*ppa.launchpadcontent.net/alex-p/tesseract-ocr5" /etc/apt/sources.list* 2>/dev/null; then
    sudo add-apt-repository -y ppa:alex-p/tesseract-ocr5
  fi
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends tesseract-ocr libtesseract-dev libleptonica-dev
}

# --- Main ---
require_sudo
log "Starting simplified bootstrap"

# Optional: enable binfmt for multiarch docker builds (no-op if already present)
docker run --privileged --rm tonistiigi/binfmt --install all >/dev/null 2>&1 || true

log "Update & install base packages"
sudo apt-get update -yq
sudo apt-get upgrade -yq || true
sudo apt-get install -yq --no-install-recommends \
  ca-certificates curl wget git sudo tree jq unzip make python3.10-venv python3-pip \
  build-essential gnupg lsb-release software-properties-common zip apt-transport-https fonts-dejavu fonts-liberation \
  unzip

# Git LFS
log "Installing git-lfs"
sudo curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get update -yq
sudo apt-get install -y git-lfs
sudo git lfs install || true

install_aws_cli
install_pulumi
install_tesseract

setup_dirs

# Persist key env vars to ~/.bashrc
append_if_missing "export MODEL_HOME=\"${MODEL_HOME}\"" ~/.bashrc
append_if_missing "export HF_HOME=\"${HF_HOME}\"" ~/.bashrc
append_if_missing "export HF_HUB_CACHE=\"${HF_HUB_CACHE}\"" ~/.bashrc
append_if_missing "export HF_ASSETS_CACHE=\"${HF_ASSETS_CACHE}\"" ~/.bashrc
append_if_missing "export RAPIDOCR_MODEL_DIR=\"${RAPIDOCR_MODEL_DIR}\"" ~/.bashrc
append_if_missing 'export DEBIAN_FRONTEND=noninteractive' ~/.bashrc
append_if_missing 'export PYTHONPATH=$(pwd)' ~/.bashrc

# Verify basic tools
log "Verification (versions):"
for cmd in aws pulumi git python3 pip3; do
  if command -v "${cmd}" >/dev/null 2>&1; then
    printf '  %-8s -> %s\n' "${cmd}" "$( ${cmd} --version 2>/dev/null | head -n1 || echo 'version unknown')"
  else
    printf '  %-8s -> not installed\n' "${cmd}"
  fi
done

# Optional: install ffmpeg static if available
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

# Python packages and optional HF download hook
pip3 install --upgrade pip || true
pip3 install huggingface_hub==0.34.4 || true
if [ -f utils/archive/download_hf.py ]; then
  sudo python3 utils/archive/download_hf.py || true
fi

# LibreOffice (PPA) - optional but useful for converting xls/xlsx
if ! command -v libreoffice >/dev/null 2>&1; then
  log "Installing LibreOffice (headless)"
  sudo add-apt-repository ppa:libreoffice/ppa -y || true
  sudo apt-get update -yq
  sudo apt-get install -y libreoffice-core libreoffice-writer libreoffice-calc --no-install-recommends || true
fi

# Pull a helpful docker image if available (non-fatal)
docker pull athithya324/embedder-cpu-inference:linux-amd64-arm64 >/dev/null 2>&1 || true

log "Bootstrap completed. Reload shell: source ~/.bashrc"
