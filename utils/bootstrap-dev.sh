#!/usr/bin/env bash
IFS=$'\n\t'

docker run --privileged --rm tonistiigi/binfmt --install all

export DEBIAN_FRONTEND=noninteractive
export TZ=Etc/UTC
export MODEL_HOME="/opt/models"
export HF_HOME="$MODEL_HOME/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export RAPIDOCR_MODEL_DIR="$MODEL_HOME/rapidocr"

PULUMI_VERSION="3.186.0"

log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"; }
require_sudo() {
  if ! sudo -n true 2>/dev/null; then
    log "sudo access required: you'll be prompted for password..."
  fi
}

sudo wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/local/bin/yq
sudo chmod +x /usr/local/bin/yq
yq --version


echo "Installing latest Tesseract 5.x from alex-p PPA..."
sudo apt-get update -y
if ! grep -q "^deb .\+ppa.launchpadcontent.net/alex-p/tesseract-ocr5" /etc/apt/sources.list*; then
   add-apt-repository -y ppa:alex-p/tesseract-ocr5
fi
sudo apt-get update -y
sudo apt-get install -y dos2unix yamllint tesseract-ocr libtesseract-dev libleptonica-dev

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

# Tesseract 5.x via PPA
install_tesseract() {
  log "Installing Tesseract 5.x (alex-p PPA)"
  if ! grep -q "^deb .\+ppa.launchpadcontent.net/alex-p/tesseract-ocr5" /etc/apt/sources.list* 2>/dev/null; then
    sudo add-apt-repository -y ppa:alex-p/tesseract-ocr5
  fi
  sudo apt-get update -y
  sudo apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev
}

setup_dirs() {
  log "Creating model and workspace directories"
  sudo mkdir -p "${MODEL_HOME}/hf/hub" "${MODEL_HOME}/hf/assets" /workspace/backups/dbs/arrangodb /workspace/data
  sudo chmod -R 0775 "${MODEL_HOME}" /workspace || true
  sudo chown -R "$(id -u):$(id -g)" "${MODEL_HOME}" /workspace || true
}

append_if_missing() {
  local line="$1" target="$2"
  if ! grep -Fxq "${line}" "${target}" 2>/dev/null; then
    printf '%s\n' "${line}" >> "${target}"
  fi
}

install_aws_cli
install_pulumi
install_tesseract

append_if_missing "export MODEL_HOME=\"${MODEL_HOME}\"" ~/.bashrc
append_if_missing "export HF_HOME=\"${HF_HOME}\"" ~/.bashrc
append_if_missing "export HF_HUB_CACHE=\"${HF_HUB_CACHE}\"" ~/.bashrc
append_if_missing "export HF_ASSETS_CACHE=\"${HF_ASSETS_CACHE}\"" ~/.bashrc
append_if_missing "export RAPIDOCR_MODEL_DIR=\"${RAPIDOCR_MODEL_DIR}\"" ~/.bashrc
append_if_missing 'export DEBIAN_FRONTEND=noninteractive' ~/.bashrc

append_if_missing 'export PYTHONPATH=$(pwd)' ~/.bashrc

log "Verification (versions):"
for cmd in aws pulumi; do
  if command -v "${cmd}" >/dev/null 2>&1; then
    printf '  %-8s -> %s\n' "${cmd}" "$(${cmd} --version 2>/dev/null | head -n1 || echo 'version unknown')"
  else
    printf '  %-8s -> not installed\n' "${cmd}"
  fi
done

# sudo apt purge ttf-mscorefonts-installer -y

pip install huggingface_hub==0.34.4
sudo python3 utils/archive/download_hf.py

docker pull athithya324/embedder-cpu-inference:linux-amd64-arm64

log "Bootstrap completed. Restart your shell or run: source ~/.bashrc"
