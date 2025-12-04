#!/usr/bin/env bash
# bootstrap script if building indexer container manually

set -euo pipefail
IFS=$'\n\t'

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
WAIT_PULUMI_RETRIES=6
WAIT_PULUMI_DELAY=2
TS() { date --iso-8601=seconds; }
printf '%s %s\n' "$(TS)" "bootstrap start"

if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
  TARGET_RC="$(getent passwd "${SUDO_USER}" | cut -d: -f6)/.bashrc"
else
  TARGET_RC="${HOME:-$(getent passwd $(id -u) | cut -d: -f6)}/.bashrc"
fi
mkdir -p "$(dirname "$TARGET_RC")"
append_line_if_missing=false

if ! sudo -n true 2>/dev/null; then
  printf '%s %s\n' "$(TS)" "sudo access required: prompt may appear"
  sudo -v
fi

mkdir -p "${MODEL_HOME}/hf/hub" "${MODEL_HOME}/hf/assets" "${RAPIDOCR_MODEL_DIR}" /workspace/data
sudo chmod -R 0775 "${MODEL_HOME}" /workspace || true
sudo chown -R "$(id -u):$(id -g)" "${MODEL_HOME}" /workspace || true

printf '%s %s\n' "$(TS)" "apt update + upgrade + base packages"
sudo apt-get update -yq
sudo apt-get upgrade -yq || true
sudo apt-get install -yq --no-install-recommends ca-certificates curl wget git gh sudo tree jq unzip vim make python3.10-venv python3-pip build-essential gnupg lsb-release software-properties-common zip apt-transport-https fonts-dejavu fonts-liberation dos2unix yamllint || true

if [ ! -x /usr/local/bin/yq ]; then
  sudo wget -q -O /usr/local/bin/yq "https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64"
  sudo chmod +x /usr/local/bin/yq
else
  printf '%s %s\n' "$(TS)" "yq present (skipping)"
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  sudo curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
  sudo apt-get update -yq
  sudo apt-get install -y git-lfs || true
  sudo git lfs install || true
else
  printf '%s %s\n' "$(TS)" "git-lfs present (skipping)"
fi

printf '%s %s\n' "$(TS)" "Installing CLI tools"

if ! command -v aws >/dev/null 2>&1; then
  printf '%s %s\n' "$(TS)" "Installing aws CLI v2"
  TMPZIP="$(mktemp)" TMPDIR="$(mktemp -d)"
  curl -fSL --retry 3 -o "${TMPZIP}" "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"
  unzip -q "${TMPZIP}" -d "${TMPDIR}"
  sudo "${TMPDIR}/aws/install" --update || true
  rm -rf "${TMPDIR}" "${TMPZIP}"
else
  printf '%s %s\n' "$(TS)" "aws present (skipping)"
fi

if command -v kubectl >/dev/null 2>&1 && [[ "$(kubectl version --client --short 2>/dev/null || true)" == *"${KUBECTL_VERSION#v}"* ]]; then
  printf '%s %s\n' "$(TS)" "kubectl matches (skipping)"
else
  printf '%s %s\n' "$(TS)" "Installing kubectl ${KUBECTL_VERSION}"
  TMP="$(mktemp)"
  curl -fSL --retry 3 -o "${TMP}" "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl"
  sudo install -m 0555 "${TMP}" /usr/local/bin/kubectl
  rm -f "${TMP}"
fi

if command -v eksctl >/dev/null 2>&1 && [[ "$(eksctl version 2>/dev/null || true)" == *"${EKSCTL_VERSION#v}"* ]]; then
  printf '%s %s\n' "$(TS)" "eksctl matches (skipping)"
else
  printf '%s %s\n' "$(TS)" "Installing eksctl ${EKSCTL_VERSION}"
  TMPGZ="$(mktemp)" TMPD="$(mktemp -d)"
  curl -fSL --retry 3 -o "${TMPGZ}" "https://github.com/eksctl-io/eksctl/releases/download/${EKSCTL_VERSION}/eksctl_Linux_amd64.tar.gz"
  tar -xzf "${TMPGZ}" -C "${TMPD}"
  sudo install -m 0755 "${TMPD}/eksctl" /usr/local/bin/eksctl
  rm -rf "${TMPGZ}" "${TMPD}"
fi

if command -v argocd >/dev/null 2>&1 && [[ "$(argocd version --client 2>/dev/null || true)" == *"${ARGOCD_VERSION#v}"* ]]; then
  printf '%s %s\n' "$(TS)" "argocd matches (skipping)"
else
  printf '%s %s\n' "$(TS)" "Installing argocd ${ARGOCD_VERSION}"
  TMPBIN="$(mktemp)"
  curl -fSL --retry 3 -o "${TMPBIN}" "https://github.com/argoproj/argo-cd/releases/download/${ARGOCD_VERSION}/argocd-linux-amd64"
  sudo install -m 0555 "${TMPBIN}" /usr/local/bin/argocd
  rm -f "${TMPBIN}"
fi

if command -v helm >/dev/null 2>&1 && [[ "$(helm version --short 2>/dev/null || true)" == *"${HELM_VERSION}"* ]]; then
  printf '%s %s\n' "$(TS)" "helm matches (skipping)"
else
  printf '%s %s\n' "$(TS)" "Installing helm ${HELM_VERSION}"
  TMP="$(mktemp)" TGTDIR="$(mktemp -d)"
  curl -fSL --retry 3 -o "${TMP}" "https://get.helm.sh/helm-${HELM_VERSION}-linux-amd64.tar.gz"
  tar -xzf "${TMP}" -C "${TGTDIR}"
  sudo install -m 0555 "${TGTDIR}/linux-amd64/helm" /usr/local/bin/helm
  rm -rf "${TMP}" "${TGTDIR}"
fi

if command -v pulumi >/dev/null 2>&1 && [[ "$(pulumi version 2>/dev/null || true)" == *"${PULUMI_VERSION}"* ]]; then
  printf '%s %s\n' "$(TS)" "pulumi matches (skipping)"
else
  printf '%s %s\n' "$(TS)" "Installing pulumi ${PULUMI_VERSION}"
  URL="https://get.pulumi.com/releases/sdk/pulumi-v${PULUMI_VERSION}-linux-x64.tar.gz"
  TMPFILE="$(mktemp)" TMPD="$(mktemp -d)" attempt=0
  while true; do
    attempt=$((attempt+1))
    if curl -fSL --retry 3 --retry-delay 2 --retry-max-time 120 -C - -o "${TMPFILE}" "${URL}"; then break; fi
    if [ "$attempt" -ge "$WAIT_PULUMI_RETRIES" ]; then
      printf '%s %s\n' "$(TS)" "Failed to download pulumi archive after ${attempt} attempts"
      rm -f "${TMPFILE}"
      break
    fi
    sleep $((WAIT_PULUMI_DELAY * attempt))
  done
  if [ -f "${TMPFILE}" ]; then
    tar -xzf "${TMPFILE}" -C "${TMPD}" || { printf '%s %s\n' "$(TS)" "Failed to extract pulumi archive"; rm -rf "${TMPFILE}" "${TMPD}"; }
    mapfile -t BINS < <(find "${TMPD}" -type f -perm /111 -print 2>/dev/null || true)
    if [ "${#BINS[@]}" -eq 0 ]; then
      printf '%s %s\n' "$(TS)" "Pulumi archive layout unexpected, falling back to official installer"
      rm -rf "${TMPFILE}" "${TMPD}"
      if curl -fsSL https://get.pulumi.com | sh -s -- --version "${PULUMI_VERSION}"; then
        printf '%s %s\n' "$(TS)" "Pulumi installed via official installer"
      else
        printf '%s %s\n' "$(TS)" "Pulumi official installer failed"
      fi
    else
      for b in "${BINS[@]}"; do sudo install -m 0755 "${b}" /usr/local/bin/ || true; done
      rm -rf "${TMPFILE}" "${TMPD}"
      if command -v pulumi >/dev/null 2>&1; then
        INSTALLED="$(pulumi version 2>/dev/null || true)"
        if [[ "${INSTALLED}" == *"${PULUMI_VERSION}"* ]]; then
          printf '%s %s\n' "$(TS)" "pulumi ${PULUMI_VERSION} installed"
        else
          printf '%s %s\n' "$(TS)" "pulumi installed but version mismatch: found='${INSTALLED}' expected='${PULUMI_VERSION}'"
        fi
      else
        printf '%s %s\n' "$(TS)" "pulumi binary not found after install"
      fi
    fi
  fi
fi

if ! grep -Rqs "alex-p/tesseract-ocr5" /etc/apt 2>/dev/null; then
  sudo add-apt-repository -y ppa:alex-p/tesseract-ocr5 || true
fi
sudo apt-get update -y
sudo apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev || true

python3 -m pip install --upgrade pip
python3 -m pip install huggingface_hub==0.34.4 || true
python3 -m pip install pulumi==${PULUMI_VERSION} pulumi-aws==7.7.0 pyyaml==6.0.3|| true

if [ -f utils/archive/download_hf/download_faster_whisper.py ]; then
  sudo python3 utils/archive/download_hf/download_faster_whisper.py || true
fi

sudo mkdir -p /models/spacy
sudo chown -R "$(id -u):$(id -g)" /models || true
python3 -m spacy download en_core_web_sm --target /models/spacy || true

printf '%s %s\n' "$(TS)" "Downloading RapidOCR models"
sudo mkdir -p /opt/models/rapidocr
sudo chown -R "$(id -u):$(id -g)" /opt/models || true
cd /opt/models/rapidocr || true
if [ ! -f ch_PP-OCRv4_det_infer.onnx ]; then
  curl -fSL --retry 5 -C - -o ch_PP-OCRv4_det_infer.onnx "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx" || true
else printf '%s %s\n' "$(TS)" "exists: ch_PP-OCRv4_det_infer.onnx"; fi
if [ ! -f ch_PP-OCRv4_rec_infer.onnx ]; then
  curl -fSL --retry 5 -C - -o ch_PP-OCRv4_rec_infer.onnx "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx" || true
else printf '%s %s\n' "$(TS)" "exists: ch_PP-OCRv4_rec_infer.onnx"; fi
if [ ! -f ch_ppocr_mobile_v2.0_cls_infer.onnx ]; then
  curl -fSL --retry 5 -C - -o ch_ppocr_mobile_v2.0_cls_infer.onnx "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_ppocr_mobile_v2.0_cls_infer.onnx" || true
else printf '%s %s\n' "$(TS)" "exists: ch_ppocr_mobile_v2.0_cls_infer.onnx"; fi
cd - >/dev/null 2>&1 || true

CURRENT_PWD="$(pwd -P)"
grep -Fxq "export MODEL_HOME=\"${MODEL_HOME}\"" "${TARGET_RC}" 2>/dev/null || printf 'export MODEL_HOME="%s"\n' "${MODEL_HOME}" >> "${TARGET_RC}"
grep -Fxq "export HF_HOME=\"${HF_HOME}\"" "${TARGET_RC}" 2>/dev/null || printf 'export HF_HOME="%s"\n' "${HF_HOME}" >> "${TARGET_RC}"
grep -Fxq "export HF_HUB_CACHE=\"${HF_HUB_CACHE}\"" "${TARGET_RC}" 2>/dev/null || printf 'export HF_HUB_CACHE="%s"\n' "${HF_HUB_CACHE}" >> "${TARGET_RC}"
grep -Fxq "export HF_ASSETS_CACHE=\"${HF_ASSETS_CACHE}\"" "${TARGET_RC}" 2>/dev/null || printf 'export HF_ASSETS_CACHE="%s"\n' "${HF_ASSETS_CACHE}" >> "${TARGET_RC}"
grep -Fxq "export RAPIDOCR_MODEL_DIR=\"${RAPIDOCR_MODEL_DIR}\"" "${TARGET_RC}" 2>/dev/null || printf 'export RAPIDOCR_MODEL_DIR="%s"\n' "${RAPIDOCR_MODEL_DIR}" >> "${TARGET_RC}"
grep -Fxq 'export DEBIAN_FRONTEND=noninteractive' "${TARGET_RC}" 2>/dev/null || printf 'export DEBIAN_FRONTEND=noninteractive\n' >> "${TARGET_RC}"
grep -Fxq "export PYTHONPATH=\"${CURRENT_PWD}\"" "${TARGET_RC}" 2>/dev/null || printf 'export PYTHONPATH="%s"\n' "${CURRENT_PWD}" >> "${TARGET_RC}"

printf '%s %s\n' "$(TS)" "Verification (versions):"
for cmd in aws kubectl eksctl argocd helm pulumi k3d git python3 pip3; do
  if command -v "${cmd}" >/dev/null 2>&1; then
    ver="$(${cmd} --version 2>/dev/null | head -n1 || echo 'version unknown')"
    printf '  %-8s -> %s\n' "${cmd}" "${ver}"
  else
    printf '  %-8s -> not installed\n' "${cmd}"
  fi
done


mkdir -p samples && cd samples && \
wget -O sample.doc   "https://filesamples.com/samples/document/doc/sample1.doc" && \
wget -O sample.xls   "https://filesamples.com/samples/document/xls/sample1.xls" && \
wget -O sample.xlsx  "https://filesamples.com/samples/document/xlsx/sample1.xlsx" && \
wget -O sample.mp3   "https://download.samplelib.com/mp3/sample-3s.mp3"


clear
printf '%s %s\n' "$(TS)" "Bootstrap completed. Open a new terminal or run: source \"${TARGET_RC}\""