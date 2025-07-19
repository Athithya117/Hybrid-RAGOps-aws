#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

export DEBIAN_FRONTEND=noninteractive

# Versions
KUBECTL_VERSION="v1.29.0"
EKSCTL_VERSION="v0.174.0"
FLUX_VERSION="v2.2.3"
HELM_VERSION="v3.14.3"
NODE_VERSION="20.x"
PULUMI_VERSION="3.114.0"
FASTTEXT_MODEL_URL="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_MODEL_FILE="lid.176.bin"

preconfigure() {
  echo "[*] Preconfiguring system..."
  sudo apt-get update -yq
  sudo apt-get upgrade -yq
  sudo apt-get install -yq tree debconf-utils gh git zip unzip make python3.10-venv python3-pip
  for q in \
    "needrestart needrestart/restart boolean true" \
    "needrestart needrestart/restart-without-asking boolean true" \
    "needrestart needrestart/restart string a"; do
    echo "$q" | sudo debconf-set-selections
  done
  # only if needrestart exists
  if [ -f /etc/needrestart/needrestart.conf ] && \
     grep -q '^#\$nrconf{restart} = ' /etc/needrestart/needrestart.conf; then
    sudo sed -i "s/#\$nrconf{restart} = .*/\$nrconf{restart} = 'a';/" /etc/needrestart/needrestart.conf
  fi
}

install_prereqs() {
  echo "[*] Installing prerequisites..."
  sudo apt-get update -yq
  sudo apt-get upgrade -yq
  sudo apt-get install -yq --no-install-recommends \
    build-essential curl unzip wget gnupg lsb-release software-properties-common jq \
    make libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
}

install_docker() {
  echo "[*] Installing Docker..."

  export DEBIAN_FRONTEND=noninteractive

  DOCKER_VERSION="5:24.0.7-1~ubuntu.22.04~jammy"
  DOCKER_CLI_VERSION="5:24.0.7-1~ubuntu.22.04~jammy"
  CONTAINERD_VERSION="1.6.25-1"

  if dpkg -l | grep -q "docker-ce.*${DOCKER_VERSION}"; then
    echo " - Docker ${DOCKER_VERSION} already installed"
  else
    # Remove old Docker versions (ignore errors)
    sudo apt-get remove -y docker docker-engine docker.io containerd runc || true

    sudo apt-get update -y
    sudo apt-get install -y ca-certificates curl gnupg lsb-release

    sudo install -m 0755 -d /etc/apt/keyrings

    # Overwrite GPG file non-interactively
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.gpg > /dev/null

    # Add Docker repo
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update -y

    # Allow downgrades and install specific versions
    sudo apt-get install -y \
      docker-ce="${DOCKER_VERSION}" \
      docker-ce-cli="${DOCKER_CLI_VERSION}" \
      containerd.io="${CONTAINERD_VERSION}" \
      docker-buildx-plugin docker-compose-plugin \
      --allow-downgrades

    # Add user to docker group (idempotent)
    sudo groupadd docker 2>/dev/null || true
    sudo usermod -aG docker "$USER"

    echo " - Docker installed"
  fi

  # Append docker group logic to ~/.bashrc (if not already added)
  if ! grep -q 'DOCKER_GROUP_APPLIED' ~/.bashrc; then
    cat >> ~/.bashrc <<'EOF'

# >>> docker group init >>>
if [ -n "$BASH_VERSION" ] && [ -z "$DOCKER_GROUP_APPLIED" ]; then
  if id -nG | grep -qw docker; then
    export DOCKER_GROUP_APPLIED=1
    newgrp docker <<EOS
    echo "[*] Docker group applied"
EOS
  fi
fi
# <<< docker group init <<<
EOF
    echo " - Added docker group activation block to ~/.bashrc"
  fi
}


install_aws_cli() {
  echo "[*] Installing AWS CLI v2..."
  if ! command -v aws &>/dev/null; then
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install --update
    rm -rf /tmp/aws /tmp/awscliv2.zip
  else
    echo " - AWS CLI already installed"
  fi
}

install_kubectl() {
  echo "[*] Installing kubectl..."
  local cur=""
  if command -v kubectl &>/dev/null; then
    cur=$(kubectl version --client -o json | jq -r '.clientVersion.gitVersion' 2>/dev/null || echo "")
  fi
  if [[ "$cur" != "$KUBECTL_VERSION" ]]; then
    echo " - Installing kubectl ${KUBECTL_VERSION} (was: ${cur:-none})"
    curl -fsSL "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" -o kubectl
    sudo install -m0755 kubectl /usr/local/bin/kubectl
    rm -f kubectl
  else
    echo " - kubectl ${KUBECTL_VERSION} already installed"
  fi
}

install_eksctl() {
  echo "[*] Installing eksctl..."
  if ! command -v eksctl &>/dev/null || \
     [[ "$(eksctl version | grep -oP '[0-9]+\.[0-9]+\.[0-9]+')" != "${EKSCTL_VERSION#v}" ]]; then
    curl -fsSL "https://github.com/weaveworks/eksctl/releases/download/${EKSCTL_VERSION}/eksctl_Linux_amd64.tar.gz" -o /tmp/eksctl.tar.gz
    tar -xzf /tmp/eksctl.tar.gz -C /tmp
    sudo mv /tmp/eksctl /usr/local/bin/
    rm -f /tmp/eksctl.tar.gz
  else
    echo " - eksctl ${EKSCTL_VERSION} already installed"
  fi
}

install_flux() {
  echo "[*] Installing flux..."
  if ! command -v flux &>/dev/null || \
     [[ "$(flux --version | grep -oP '\d+\.\d+\.\d+')" != "${FLUX_VERSION#v}" ]]; then
    local version="${FLUX_VERSION#v}"
    local archive="flux_${version}_linux_amd64.tar.gz"
    curl -fsSL "https://github.com/fluxcd/flux2/releases/download/${FLUX_VERSION}/${archive}" -o "$archive"
    tar -xzf "$archive"
    sudo install -m0755 flux /usr/local/bin/flux
    rm -f flux "$archive"
  else
    echo " - flux ${FLUX_VERSION} already installed"
  fi
}

install_helm() {
  echo "[*] Installing helm ${HELM_VERSION}..."
  local cur=""
  if command -v helm &>/dev/null; then
    cur=$(helm version --short 2>/dev/null | grep -oP '\d+\.\d+\.\d+')
  fi
  if [[ "$cur" != "$HELM_VERSION" ]]; then
    echo " - Installing helm ${HELM_VERSION} (was: ${cur:-none})"
    curl -fsSL "https://get.helm.sh/helm-${HELM_VERSION}-linux-amd64.tar.gz" -o /tmp/helm.tgz
    tar -xzf /tmp/helm.tgz -C /tmp
    sudo mv /tmp/linux-amd64/helm /usr/local/bin/helm
    sudo chmod +x /usr/local/bin/helm
    rm -rf /tmp/helm.tgz /tmp/linux-amd64
  else
    echo " - helm ${HELM_VERSION} already installed"
  fi
}

install_pulumi() {
  echo "[*] Installing Pulumi ${PULUMI_VERSION}..."
  local path="/usr/local/lib/pulumi-${PULUMI_VERSION}"
  local link="/usr/local/bin/pulumi"
  local cur=""
  if command -v pulumi &>/dev/null; then
    cur=$(pulumi version 2>/dev/null || echo "")
  fi
  if [[ "$cur" == "$PULUMI_VERSION" && \
        "$(readlink -f "$link")" == "${path}/pulumi" ]]; then
    echo " - Pulumi ${PULUMI_VERSION} already installed"
    return
  fi
  echo " - Installing Pulumi ${PULUMI_VERSION}"
  curl -fsSL "https://get.pulumi.com/releases/sdk/pulumi-v${PULUMI_VERSION}-linux-x64.tar.gz" -o /tmp/pulumi.tgz
  tar -xzf /tmp/pulumi.tgz -C /tmp
  sudo rm -rf "$path"
  sudo mv /tmp/pulumi "$path"
  sudo ln -sf "${path}/pulumi" "$link"
  rm -f /tmp/pulumi.tgz
}

install_node_vite() {
  echo "[*] Installing Node.js & Vite..."
  if ! command -v node &>/dev/null || ! command -v vite &>/dev/null; then
    curl -fsSL "https://deb.nodesource.com/setup_${NODE_VERSION}" | sudo -E bash -
    sudo apt-get install -yq nodejs
    sudo npm install -g vite --no-progress
  else
    echo " - Node.js & Vite already installed"
  fi
}

install_fasttext() {
  echo "[*] Downloading FastText model..."
  if [ ! -f "$FASTTEXT_MODEL_FILE" ]; then
    wget -nc "$FASTTEXT_MODEL_URL"
  else
    echo " - FastText model already present"
  fi
}

# Main
preconfigure
install_prereqs
install_docker
install_aws_cli
install_kubectl
install_eksctl
install_flux
install_helm
install_pulumi
install_node_vite
install_fasttext

pip install --upgrade pip wheel cache
curl -LO https://github.com/getsops/sops/releases/download/v3.8.1/sops-v3.8.1.linux.amd64
chmod +x sops-v3.8.1.linux.amd64
sudo mv sops-v3.8.1.linux.amd64 /usr/local/bin/sops
sops --version


clear
