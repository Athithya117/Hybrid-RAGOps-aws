#!/usr/bin/env bash

IFS=$'\n\t'

export DEBIAN_FRONTEND=noninteractive
export APT_LISTCHANGES_FRONTEND=none
export PYTHONPATH="$(pwd)"
export PATH="$HOME/.local/bin:$PATH"

KUBECTL_VERSION="v1.29.0"
EKSCTL_VERSION="v0.174.0"
FLUX_VERSION="v2.2.3"
HELM_VERSION="v3.14.3"
NODE_VERSION="20.x"
PYTHON_VERSION="3.11.8"

preconfigure() {
  echo "[*] Preconfiguring system to avoid interactive prompts..."
  sudo apt-get update -yq
  sudo apt-get install -yq tree debconf-utils
  for q in \
    "needrestart needrestart/restart boolean true" \
    "needrestart needrestart/restart-without-asking boolean true" \
    "needrestart needrestart/restart string a"; do
    echo "$q" | sudo debconf-set-selections
  done
  sudo sed -i 's/#\$nrconf{restart} = .*/\$nrconf{restart} = '\''a'\'';/' /etc/needrestart/needrestart.conf || true
}

install_prereqs() {
  echo "[*] Installing base packages & pyenv build deps..."
  sudo apt-get update -yq
  sudo apt-get upgrade -yq
  sudo apt-get install -yq --no-install-recommends \
    build-essential curl unzip wget gnupg lsb-release software-properties-common git \
    python3.10 python3.10-venv python3.10-dev python3-pip jq \
    make libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
}

install_pyenv() {
  echo "[*] Installing pyenv..."
  [ -d "$HOME/.pyenv" ] || git clone https://github.com/pyenv/pyenv.git ~/.pyenv
  if ! grep -q 'export PYENV_ROOT' ~/.bashrc; then
    cat >> ~/.bashrc <<'EOF'

# pyenv setup
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv >/dev/null 2>&1; then
  eval "$(pyenv init --path)"
  eval "$(pyenv init -)"
fi
EOF
  fi
  export PYENV_ROOT="$HOME/.pyenv"
  export PATH="$PYENV_ROOT/bin:$PATH"
  eval "$(pyenv init --path)"
  eval "$(pyenv init -)"
}

install_aws_cli() {
  echo "[*] Installing AWS CLI v2..."
  if ! command -v aws &>/dev/null; then
    rm -rf /tmp/aws /tmp/awscliv2.zip
    curl -fSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install --update
    rm -rf /tmp/aws /tmp/awscliv2.zip
  fi
}

install_kubectl() {
  echo "[*] Installing kubectl ${KUBECTL_VERSION}..."
  current="$(kubectl version --client --output=json 2>/dev/null | jq -r .clientVersion.gitVersion || echo)"
  if [[ "$current" != "$KUBECTL_VERSION" ]]; then
    curl -fSL "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" -o kubectl
    sudo install -m0755 kubectl /usr/local/bin/kubectl
    rm -f kubectl
  fi
}

install_eksctl() {
  echo "[*] Installing eksctl ${EKSCTL_VERSION}..."
  current="$(eksctl version 2>/dev/null || echo)"
  if [[ "$current" != "$EKSCTL_VERSION" ]]; then
    curl -fSL "https://github.com/weaveworks/eksctl/releases/download/${EKSCTL_VERSION}/eksctl_Linux_amd64.tar.gz" -o /tmp/eksctl.tar.gz
    tar -xzf /tmp/eksctl.tar.gz -C /tmp
    sudo mv /tmp/eksctl /usr/local/bin/eksctl
    rm -f /tmp/eksctl.tar.gz
  fi
}

install_flux() {
  echo "[*] Installing flux ${FLUX_VERSION}..."
  version="${FLUX_VERSION#v}"
  archive="flux_${version}_linux_amd64.tar.gz"
  curl -fSL "https://github.com/fluxcd/flux2/releases/download/${FLUX_VERSION}/${archive}" -o "$archive"
  tar -xzf "$archive"
  sudo install -m0755 flux /usr/local/bin/flux
  rm -f flux "$archive"
}

install_helm() {
  echo "[*] Installing helm ${HELM_VERSION}..."
  archive="helm-${HELM_VERSION}-linux-amd64.tar.gz"
  current="$(helm version --short --client 2>/dev/null || echo)"
  if [[ "$current" != "${HELM_VERSION}" ]]; then
    curl -fSL "https://get.helm.sh/${archive}" -o "$archive"
    tar -xzf "$archive"
    sudo mv linux-amd64/helm /usr/local/bin/helm
    rm -rf "$archive" linux-amd64
  fi
}

install_pulumi() {
  local PULUMI_VERSION="${1:-3.114.0}"  # Default to 3.114.0 if not passed
  echo "[*] Installing Pulumi ${PULUMI_VERSION}..."

  # Skip if Pulumi is already installed and matches version
  if command -v pulumi &>/dev/null && [[ "$(pulumi version)" == "${PULUMI_VERSION}" ]]; then
    echo "[*] Pulumi ${PULUMI_VERSION} already installed."
    return 0
  fi

  local os="linux"
  local arch="x64"
  local tarball="pulumi-v${PULUMI_VERSION}-${os}-${arch}.tar.gz"

  curl -fSL "https://get.pulumi.com/releases/sdk/${tarball}" -o "/tmp/${tarball}"
  tar -xzf "/tmp/${tarball}" -C /tmp

  sudo mv "/tmp/pulumi" /usr/local/lib/pulumi-${PULUMI_VERSION}
  sudo ln -sf "/usr/local/lib/pulumi-${PULUMI_VERSION}/pulumi" /usr/local/bin/pulumi

  rm -f "/tmp/${tarball}"

  echo "[*] Pulumi ${PULUMI_VERSION} installed successfully."
}


install_node_vite() {
  echo "[*] Installing Node.js ${NODE_VERSION} & Vite..."
  curl -fSL "https://deb.nodesource.com/setup_${NODE_VERSION}" | sudo -E bash -
  sudo apt-get install -yq nodejs
  sudo npm install -g vite --no-progress
}

install_python() {
  echo "[*] Installing Python ${PYTHON_VERSION} via pyenv..."
  if ! pyenv versions --bare | grep -qx "${PYTHON_VERSION}"; then
    pyenv install "${PYTHON_VERSION}"
  fi
  if [[ "$(pyenv global)" != "${PYTHON_VERSION}" ]]; then
    pyenv global "${PYTHON_VERSION}"
  fi
}

preconfigure
install_prereqs
install_pyenv
install_aws_cli
install_kubectl
install_eksctl
install_node_vite
install_flux
install_helm
install_pulumi
install_python
mkdir -p tmp
mkdir -p efs && cd efs && wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin && cd -

clear


echo "→ Open a new terminal or run 'source ~/.bashrc' to use ${PYTHON_VERSION} by default."


