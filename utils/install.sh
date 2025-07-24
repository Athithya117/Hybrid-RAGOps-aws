#!/usr/bin/env bash


# Provide fallback for USER variable to avoid unbound errors
: "${USER:=root}"

echo "[*] Running setup.sh as user: $USER"

export DEBIAN_FRONTEND=noninteractive
export TZ=Etc/UTC

apt-get update -yq
apt-get upgrade -yq

# Install base dependencies
apt-get install -yq --no-install-recommends \
  curl git sudo tree jq wget unzip make python3.10-venv python3-pip \
  ca-certificates gnupg lsb-release software-properties-common build-essential \
  unzip zip

# Docker install function
install_docker() {
  DOCKER_VERSION="5:24.0.7-1~ubuntu.22.04~jammy"
  DOCKER_CLI_VERSION="5:24.0.7-1~ubuntu.22.04~jammy"
  CONTAINERD_VERSION="1.6.25-1"

  if dpkg -l | grep -q "docker-ce.*${DOCKER_VERSION}"; then
    echo " - Docker ${DOCKER_VERSION} already installed"
  else
    apt-get remove -y docker docker-engine docker.io containerd runc || true
    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
      > /etc/apt/sources.list.d/docker.list
    apt-get update -y
    apt-get install -y --allow-downgrades \
      docker-ce="${DOCKER_VERSION}" \
      docker-ce-cli="${DOCKER_CLI_VERSION}" \
      containerd.io="${CONTAINERD_VERSION}" \
      docker-buildx-plugin docker-compose-plugin
  fi
}

install_aws_cli() {
  if ! command -v aws &>/dev/null; then
    curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp
    /tmp/aws/install --update
    rm -rf /tmp/aws /tmp/awscliv2.zip
  else
    echo " - AWS CLI already installed"
  fi
}

install_kubectl() {
  KUBECTL_VERSION="v1.29.0"
  local cur=""
  if command -v kubectl &>/dev/null; then
    cur=$(kubectl version --client -o json | jq -r '.clientVersion.gitVersion' 2>/dev/null || echo "")
  fi
  if [[ "$cur" != "$KUBECTL_VERSION" ]]; then
    echo " - Installing kubectl ${KUBECTL_VERSION} (was: ${cur:-none})"
    curl -fsSL https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl -o /usr/local/bin/kubectl
    chmod +x /usr/local/bin/kubectl
  else
    echo " - kubectl ${KUBECTL_VERSION} already installed"
  fi
}

install_eksctl() {
  EKSCTL_VERSION="v0.174.0"
  if ! command -v eksctl &>/dev/null || [[ "$(eksctl version | grep -oP '[0-9]+\.[0-9]+\.[0-9]+')" != "${EKSCTL_VERSION#v}" ]]; then
    curl -fsSL https://github.com/weaveworks/eksctl/releases/download/${EKSCTL_VERSION}/eksctl_Linux_amd64.tar.gz -o /tmp/eksctl.tar.gz
    tar -xzf /tmp/eksctl.tar.gz -C /tmp
    mv /tmp/eksctl /usr/local/bin/
    chmod +x /usr/local/bin/eksctl
    rm -f /tmp/eksctl.tar.gz
  else
    echo " - eksctl ${EKSCTL_VERSION} already installed"
  fi
}

install_flux() {
  FLUX_VERSION="v2.2.3"
  if ! command -v flux &>/dev/null || [[ "$(flux --version | grep -oP '\d+\.\d+\.\d+')" != "${FLUX_VERSION#v}" ]]; then
    local version="${FLUX_VERSION#v}"
    local archive="flux_${version}_linux_amd64.tar.gz"
    curl -fsSL https://github.com/fluxcd/flux2/releases/download/${FLUX_VERSION}/${archive} -o "$archive"
    tar -xzf "$archive"
    mv flux /usr/local/bin/flux
    chmod +x /usr/local/bin/flux
    rm -f "$archive"
  else
    echo " - flux ${FLUX_VERSION} already installed"
  fi
}

install_helm() {
  HELM_VERSION="v3.14.3"
  local cur=""
  if command -v helm &>/dev/null; then
    cur=$(helm version --short 2>/dev/null | grep -oP '\d+\.\d+\.\d+')
  fi
  if [[ "$cur" != "${HELM_VERSION#v}" ]]; then
    echo " - Installing helm ${HELM_VERSION} (was: ${cur:-none})"
    curl -fsSL https://get.helm.sh/helm-${HELM_VERSION}-linux-amd64.tar.gz -o /tmp/helm.tgz
    tar -xzf /tmp/helm.tgz -C /tmp
    mv /tmp/linux-amd64/helm /usr/local/bin/helm
    chmod +x /usr/local/bin/helm
    rm -rf /tmp/helm.tgz /tmp/linux-amd64
  else
    echo " - helm ${HELM_VERSION} already installed"
  fi
}

install_pulumi() {
  PULUMI_VERSION="3.114.0"
  local path="/usr/local/lib/pulumi-${PULUMI_VERSION}"
  local link="/usr/local/bin/pulumi"
  local cur=""
  if command -v pulumi &>/dev/null; then
    cur=$(pulumi version 2>/dev/null || echo "")
  fi
  if [[ "$cur" == "$PULUMI_VERSION" && "$(readlink -f "$link")" == "${path}/pulumi" ]]; then
    echo " - Pulumi ${PULUMI_VERSION} already installed"
    return
  fi
  curl -fsSL https://get.pulumi.com/releases/sdk/pulumi-v${PULUMI_VERSION}-linux-x64.tar.gz -o /tmp/pulumi.tgz
  tar -xzf /tmp/pulumi.tgz -C /tmp
  rm -rf "$path"
  mv /tmp/pulumi "$path"
  ln -sf "${path}/pulumi" "$link"
  rm -f /tmp/pulumi.tgz
}

install_node_vite() {
  NODE_VERSION="20.x"
  if ! command -v node &>/dev/null || ! command -v vite &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION} | bash -
    apt-get install -y nodejs
    npm install -g vite --no-progress
  else
    echo " - Node.js & Vite already installed"
  fi
}

# Upgrade pip and install python packages
pip3 install --upgrade pip wheel cache

# Install sops
if ! command -v sops &>/dev/null; then
  SOPS_VERSION="3.8.1"
  curl -LO https://github.com/getsops/sops/releases/download/v${SOPS_VERSION}/sops-v${SOPS_VERSION}.linux.amd64
  chmod +x sops-v${SOPS_VERSION}.linux.amd64
  mv sops-v${SOPS_VERSION}.linux.amd64 /usr/local/bin/sops
fi

# Create backup folders
mkdir -p /workspace/backups/dbs/qdrant
mkdir -p /workspace/backups/dbs/arrangodb

# Run installs
install_docker
install_aws_cli
install_kubectl
install_eksctl
install_flux
install_helm
install_pulumi
install_node_vite

echo "[*] Setup complete"
