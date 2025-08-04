#!/usr/bin/env bash

: "${USER:=root}"
echo "[*] Running setup.sh as user: $USER"

export DEBIAN_FRONTEND=noninteractive
export TZ=Etc/UTC


apt-get update -yq
apt-get upgrade -yq

apt-get install -yq --no-install-recommends \
  curl git sudo tree jq wget unzip make \
  python3.10-venv python3-pip \
  ca-certificates gnupg lsb-release \
  software-properties-common build-essential \
  zip

pip3 install --upgrade pip wheel cachecontrol

install_aws_cli() {
  if ! command -v aws &>/dev/null; then
    curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp
    /tmp/aws/install --update
    rm -rf /tmp/aws /tmp/awscliv2.zip
  fi
}

install_kubectl() {
  KUBECTL_VERSION="v1.29.0"
  if [[ "$(kubectl version --client -o json 2>/dev/null | jq -r .clientVersion.gitVersion)" != "$KUBECTL_VERSION" ]]; then
    curl -fsSL https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl -o /usr/local/bin/kubectl
    chmod +x /usr/local/bin/kubectl
  fi
}

install_eksctl() {
  EKSCTL_VERSION="v0.174.0"
  curl -fsSL https://github.com/weaveworks/eksctl/releases/download/${EKSCTL_VERSION}/eksctl_Linux_amd64.tar.gz -o /tmp/eksctl.tar.gz
  tar -xzf /tmp/eksctl.tar.gz -C /tmp
  mv /tmp/eksctl /usr/local/bin/
  chmod +x /usr/local/bin/eksctl
  rm -f /tmp/eksctl.tar.gz
}

install_argocd() {
  ARGOCD_VERSION="v2.11.3"
  curl -sSL -o /tmp/argocd https://github.com/argoproj/argo-cd/releases/download/${ARGOCD_VERSION}/argocd-linux-amd64
  chmod +x /tmp/argocd
  mv /tmp/argocd /usr/local/bin/argocd
}

install_helm() {
  HELM_VERSION="v3.14.3"
  curl -fsSL https://get.helm.sh/helm-${HELM_VERSION}-linux-amd64.tar.gz -o /tmp/helm.tgz
  tar -xzf /tmp/helm.tgz -C /tmp
  mv /tmp/linux-amd64/helm /usr/local/bin/
  chmod +x /usr/local/bin/helm
  rm -rf /tmp/helm.tgz /tmp/linux-amd64
}

install_pulumi() {
  PULUMI_VERSION="3.114.0"
  curl -fsSL https://get.pulumi.com/releases/sdk/pulumi-v${PULUMI_VERSION}-linux-x64.tar.gz -o /tmp/pulumi.tgz
  tar -xzf /tmp/pulumi.tgz -C /usr/local/bin --strip-components=1
  chmod +x /usr/local/bin/pulumi
  rm -f /tmp/pulumi.tgz
}

install_node_vite() {
  NODE_VERSION="20.x"
  curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION} | bash -
  apt-get install -y nodejs
  npm install -g vite --no-progress
}

install_sops() {
  SOPS_VERSION="3.8.1"
  curl -LO https://github.com/getsops/sops/releases/download/v${SOPS_VERSION}/sops-v${SOPS_VERSION}.linux.amd64
  chmod +x sops-v${SOPS_VERSION}.linux.amd64
  mv sops-v${SOPS_VERSION}.linux.amd64 /usr/local/bin/sops
}

install_k3d() {
  echo "[*] Installing k3d..."
  if ! command -v k3d &>/dev/null; then
    curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
  fi
}



# Run all install functions
install_aws_cli
install_kubectl
install_eksctl
install_argocd
install_helm
install_pulumi
install_node_vite
install_sops
install_k3d

pip install optimum[exporters] onnxruntime





mkdir -p /workspace/backups/dbs/qdrant /workspace/backups/dbs/arrangodb /workspace/data/



echo "Installing latest Tesseract 5.x from alex-p PPA..."
apt-get update -y
if ! grep -q "^deb .\+ppa.launchpadcontent.net/alex-p/tesseract-ocr5" /etc/apt/sources.list*; then
   add-apt-repository -y ppa:alex-p/tesseract-ocr5
fi
apt-get update -y
apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev


add-apt-repository ppa:libreoffice/ppa -y && \
apt-get update && \
apt-get install -y libreoffice \
  ttf-mscorefonts-installer fonts-dejavu fonts-liberation && \
fc-cache -f -v

clear

echo "Bootstrap completed"


