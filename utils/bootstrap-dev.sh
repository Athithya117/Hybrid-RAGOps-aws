#!/usr/bin/env bash
IFS=$'\n\t'
set -euo pipefail

docker run --privileged --rm tonistiigi/binfmt --install all || true

export DEBIAN_FRONTEND=noninteractive
export TZ=Etc/UTC
export MODEL_HOME="/opt/models"
export HF_HOME="$MODEL_HOME/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export RAPIDOCR_MODEL_DIR="$MODEL_HOME/rapidocr"
PULUMI_VERSION="3.186.0"
SOFFICE_WRAPPER_PATH="/usr/local/bin/soffice"
DOCKER_IMAGE="lscr.io/linuxserver/libreoffice:latest"
APT_PKGS="libreoffice-core libreoffice-writer libreoffice-common libreoffice-java-common libreoffice-avmedia-backend-gstreamer libreoffice-headless fonts-dejavu fonts-liberation"
DNF_PKGS="libreoffice-headless libreoffice-writer libreoffice-core libreoffice-common"
APK_PKGS="libreoffice libreoffice-writer libreoffice-common"

log(){ printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"; }
require_sudo(){ if ! sudo -n true 2>/dev/null; then log "sudo access required: you'll be prompted for password..."; fi; }

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

install_tesseract() {
  log "Installing Tesseract 5.x (alex-p PPA)"
  sudo apt-get update -y
  if ! grep -q "^deb .\+ppa.launchpadcontent.net/alex-p/tesseract-ocr5" /etc/apt/sources.list* 2>/dev/null; then
    sudo add-apt-repository -y ppa:alex-p/tesseract-ocr5
  fi
  sudo apt-get update -y
  sudo apt-get install -y dos2unix yamllint tesseract-ocr libtesseract-dev libleptonica-dev
}

found_soffice(){
  if command -v soffice >/dev/null 2>&1; then
    command -v soffice
    return 0
  fi
  if command -v libreoffice >/dev/null 2>&1; then
    LO=$(command -v libreoffice)
    tmpf=$(mktemp)
    cat > "${tmpf}" <<EOF
#!/usr/bin/env bash
exec "${LO}" "\$@"
EOF
    sudo install -m 0755 "${tmpf}" "${SOFFICE_WRAPPER_PATH}"
    rm -f "${tmpf}"
    echo "${SOFFICE_WRAPPER_PATH}"
    return 0
  fi
  return 1
}

install_via_apt(){
  if ! command -v apt-get >/dev/null 2>&1; then
    return 1
  fi
  log "Attempting apt install of LibreOffice (headless)"
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends ${APT_PKGS}
  return 0
}

install_via_dnf(){
  if ! command -v dnf >/dev/null 2>&1; then
    return 1
  fi
  log "Attempting dnf install of LibreOffice (headless)"
  sudo dnf install -y ${DNF_PKGS}
  return 0
}

install_via_apk(){
  if ! command -v apk >/dev/null 2>&1; then
    return 1
  fi
  log "Attempting apk install of LibreOffice"
  sudo apk add --no-cache ${APK_PKGS}
  return 0
}

install_via_snap(){
  if ! command -v snap >/dev/null 2>&1; then
    return 1
  fi
  log "Attempting snap install of LibreOffice"
  sudo snap install libreoffice --classic
  return 0
}

create_docker_wrapper(){
  if ! command -v docker >/dev/null 2>&1; then
    log "Docker not available; cannot create docker-backed soffice wrapper."
    return 1
  fi
  log "Pulling Docker image ${DOCKER_IMAGE}"
  sudo docker pull "${DOCKER_IMAGE}"
  tmpf=$(mktemp)
  cat > "${tmpf}" <<'EOF'
#!/usr/bin/env bash
set -e
UID=$(id -u)
GID=$(id -g)
docker run --rm -u "${UID}:${GID}" -v "$(pwd)":/workspace -w /workspace lscr.io/linuxserver/libreoffice:latest soffice "$@"
EOF
  sudo install -m 0755 "${tmpf}" "${SOFFICE_WRAPPER_PATH}"
  rm -f "${tmpf}"
  log "Created ${SOFFICE_WRAPPER_PATH} (calls LibreOffice in Docker)."
  return 0
}

setup_dirs(){
  log "Creating model and workspace directories"
  sudo mkdir -p "${MODEL_HOME}/hf/hub" "${MODEL_HOME}/hf/assets" /workspace/backups/dbs/arrangodb /workspace/data
  sudo chmod -R 0775 "${MODEL_HOME}" /workspace || true
  sudo chown -R "$(id -u):$(id -g)" "${MODEL_HOME}" /workspace || true
}

append_if_missing(){
  local line="$1" target="$2"
  if ! grep -Fxq "${line}" "${target}" 2>/dev/null; then
    printf '%s\n' "${line}" >> "${target}"
  fi
}

main_soffice_install(){
  require_sudo
  if soffice_path=$(found_soffice 2>/dev/null || true); then
    log "soffice present at: ${soffice_path}"
    "${soffice_path}" --version 2>/dev/null || true
    return 0
  fi
  DISTRO=""
  if [ -f /etc/os-release ]; then
    DISTRO=$(grep -E '^ID=' /etc/os-release | head -n1 | cut -d'=' -f2 | tr -d '"' || true)
  fi
  log "No local soffice binary found. Detected distro: ${DISTRO:-unknown}"
  if install_via_apt; then
    if found_soffice >/dev/null 2>&1; then
      log "soffice installed via apt"
      soffice --version 2>/dev/null || true
      return 0
    fi
  fi
  if install_via_dnf; then
    if found_soffice >/dev/null 2>&1; then
      log "soffice installed via dnf"
      soffice --version 2>/dev/null || true
      return 0
    fi
  fi
  if install_via_apk; then
    if found_soffice >/dev/null 2>&1; then
      log "soffice installed via apk"
      soffice --version 2>/dev/null || true
      return 0
    fi
  fi
  if install_via_snap; then
    if found_soffice >/dev/null 2>&1; then
      log "soffice installed via snap"
      soffice --version 2>/dev/null || true
      return 0
    fi
  fi
  if create_docker_wrapper; then
    log "Docker-backed soffice wrapper created at ${SOFFICE_WRAPPER_PATH}"
    return 0
  fi
  log "Failed to install or provide soffice. Please install LibreOffice on this host or ensure docker is available."
  return 1
}

require_sudo
log "Installing latest Tesseract 5.x from alex-p PPA..."
sudo apt-get update -y
if ! grep -q "^deb .\+ppa.launchpadcontent.net/alex-p/tesseract-ocr5" /etc/apt/sources.list* 2>/dev/null; then
  sudo add-apt-repository -y ppa:alex-p/tesseract-ocr5
fi
sudo apt-get update -y
sudo apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev

sudo add-apt-repository ppa:libreoffice/ppa -y || true
sudo apt-get update -y || true
main_soffice_install || true

sudo curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
sudo apt-get update
sudo apt-get install -y git-lfs
sudo git lfs install

log "[1/14] apt update/upgrade + essential packages"
sudo apt-get update -yq
sudo apt-get upgrade -yq
sudo apt-get install -yq --no-install-recommends ca-certificates curl wget git sudo tree jq unzip make python3.10-venv python3-pip build-essential gnupg lsb-release software-properties-common zip unzip apt-transport-https

install_aws_cli
install_pulumi
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
for cmd in aws pulumi; do
  if command -v "${cmd}" >/dev/null 2>&1; then
    printf '  %-8s -> %s\n' "${cmd}" "$( ${cmd} --version 2>/dev/null | head -n1 || echo 'version unknown')"
  else
    printf '  %-8s -> not installed\n' "${cmd}"
  fi
done

cd /usr/local/bin
sudo curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o ffmpeg.tar.xz || true
if [ -f ffmpeg.tar.xz ]; then
  tar -xvf ffmpeg.tar.xz || true
  cd ffmpeg-*-amd64-static || true
  sudo cp ffmpeg ffprobe /usr/local/bin/ || true
  ffmpeg -version || true
  cd /usr/local/bin || true
fi

pip install --upgrade pip
pip install huggingface_hub==0.34.4 || true
if [ -f utils/archive/download_hf.py ]; then
  sudo python3 utils/archive/download_hf.py || true
fi

docker pull athithya324/embedder-cpu-inference:linux-amd64-arm64 || true

log "Bootstrap completed. Restart your shell or run: source ~/.bashrc"
