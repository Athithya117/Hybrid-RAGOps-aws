#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

export DEBIAN_FRONTEND=noninteractive
export WORKSPACE_MODELS=${WORKSPACE_MODELS:-/workspace/models}
export HF_TOKEN=${HF_TOKEN:-}
export CUDA_CONTAINER_IMAGE=${CUDA_CONTAINER_IMAGE:-nvidia/cuda:12.2.0-runtime-ubuntu22.04}
export PIP_NO_WARN_SCRIPT_LOCATION=1

LOG() { echo "[$(date -Is)] $*"; }

LOG "Starting GPU-ready setup script"
mkdir -p "$WORKSPACE_MODELS"

LOG "Updating apt and installing base packages"
apt-get update -yq
apt-get install -yq --no-install-recommends \
  python3 python3-venv python3-pip python3-distutils \
  git curl ca-certificates gnupg lsb-release software-properties-common jq \
  linux-headers-$(uname -r) || true

LOG "Upgrading pip and installing python deps"
python3 -m pip install --no-warn-script-location --upgrade pip==25.2.0
python3 -m pip install --no-cache-dir huggingface-hub==0.34.3 tqdm==4.67.1 requests==2.32.4

LOG "Writing model download helper to /opt/download_models.py"
cat >/opt/download_models.py <<'PY'
import os
import sys
import logging
import shutil
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except Exception as e:
    raise ImportError(
        "huggingface_hub is required. Install with: pip install huggingface_hub"
    ) from e

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("download_hf")

WORKSPACE_MODELS = Path(os.getenv("WORKSPACE_MODELS", "/workspace/models"))
FORCE = os.getenv("FORCE_DOWNLOAD", "0").lower() in ("1", "true", "yes")

MODELS = [
    {
        "repo_id": "RAG8s/gte-modernbert-base-onnx-int8",
        "name": "gte-modernbert-base-onnx-int8",
        "base": "onnx",
        "items": [
            "model.onnx",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    }
]

def download_one(repo_id: str, remote: str, target: Path) -> bool:
    if target.exists() and not FORCE:
        logger.info("SKIP exists %s", target)
        return True

    tmp_dir = Path("/tmp") / "hf_download"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        got = hf_hub_download(
            repo_id=repo_id,
            filename=remote,
            local_dir=str(tmp_dir),
            local_dir_use_symlinks=False,
            force_download=FORCE,
        )
        got_path = Path(got)
        if got_path.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                try:
                    target.unlink()
                except Exception:
                    pass
            shutil.move(str(got_path), str(target))
            try:
                os.chmod(str(target), 0o444)
            except Exception:
                pass
            logger.info("Downloaded %s -> %s", remote, target)
            return True
    except Exception as e:
        logger.warning("Failed to download %s:%s (%s)", repo_id, remote, e)
    return False


def ensure_model(model: dict) -> bool:
    repo_id = model["repo_id"]
    name = model["name"]
    base = model.get("base", "llm")

    model_root = WORKSPACE_MODELS / base / name
    ok = True
    for item in model.get("items", []):
        remote_rel = str(item)
        target = model_root / Path(remote_rel)
        required = not remote_rel.endswith("special_tokens_map.json")
        success = download_one(repo_id, remote_rel, target)
        if not success and required:
            ok = False
            logger.error("Missing required %s:%s", name, remote_rel)
    return ok


def main() -> None:
    all_ok = True
    for m in MODELS:
        if not ensure_model(m):
            all_ok = False

    if not all_ok:
        logger.error("Some required files failed to download")
        sys.exit(2)

    logger.info("All model artifacts are present under %s", WORKSPACE_MODELS)


if __name__ == "__main__":
    main()
PY

LOG "Making download script executable and running it"
/usr/bin/chmod 755 /opt/download_models.py
python3 /opt/download_models.py || {
  LOG "Model download failed (non-fatal). Continuing - check logs for details."
}

LOG "Setting permissions on $WORKSPACE_MODELS"
chown -R 1000:1000 "$WORKSPACE_MODELS" || true
chmod -R 755 "$WORKSPACE_MODELS" || true
chmod -R a+rX "$WORKSPACE_MODELS" || true


LOG "Installing Docker (official repository)"
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /tmp/docker_gpg.key
mkdir -p /etc/apt/keyrings
gpg --batch --yes --dearmor -o /etc/apt/keyrings/docker.gpg /tmp/docker_gpg.key
chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update -yq

# Try to pick specific versions available; fallback to latest if not found
DOCKER_VERSION=$(apt-cache madison docker-ce | awk '{print $3; exit}' || true)
CLI_VERSION=$(apt-cache madison docker-ce-cli | awk '{print $3; exit}' || echo "$DOCKER_VERSION")
CONTAINERD_VERSION=$(apt-cache madison containerd.io | awk '{print $3; exit}' || true)

if [ -n "$DOCKER_VERSION" ] && [ -n "$CLI_VERSION" ] && [ -n "$CONTAINERD_VERSION" ]; then
  LOG "Installing specific docker versions: $DOCKER_VERSION $CLI_VERSION $CONTAINERD_VERSION"
  apt-get install -yq --allow-downgrades docker-ce="$DOCKER_VERSION" docker-ce-cli="$CLI_VERSION" containerd.io="$CONTAINERD_VERSION"
else
  LOG "Installing docker packages (latest available)"
  apt-get install -yq docker-ce docker-ce-cli containerd.io
fi

apt-mark hold docker-ce docker-ce-cli containerd.io || true
systemctl enable --now docker || true
usermod -aG docker "${SUDO_USER:-ubuntu}" || true

# --------------------------------------------------------------------------------
# NVIDIA driver (only if nvidia-smi not present)
# --------------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  LOG "nvidia-smi found: NVIDIA driver already installed"
else
  LOG "nvidia-smi not found - attempting to install NVIDIA driver (ubuntu-drivers autoinstall)"
  apt-get update -yq
  apt-get install -yq ubuntu-drivers-common || true

  # Try automatic recommended driver install
  if ubuntu-drivers devices >/dev/null 2>&1; then
    ubuntu-drivers autoinstall || LOG "ubuntu-drivers autoinstall failed (continuing to next step)"
  fi

  # re-check
  if command -v nvidia-smi >/dev/null 2>&1; then
    LOG "NVIDIA driver installed successfully via ubuntu-drivers"
  else
    LOG "nvidia-smi still missing after autoinstall. Attempting to install kernel headers and retry."
    apt-get install -yq --no-install-recommends linux-headers-$(uname -r) || true
    apt-get -f install -y || true
    # try again
    ubuntu-drivers autoinstall || LOG "Second ubuntu-drivers attempt failed"
  fi

  # Final check
  if command -v nvidia-smi >/dev/null 2>&1; then
    LOG "NVIDIA driver available now"
  else
    LOG "WARNING: nvidia-smi still not available. Driver install may require a reboot or a DLAMI. Continuing, but GPU containers will fail until host driver is present."
  fi
fi

# --------------------------------------------------------------------------------
# NVIDIA Container Toolkit (nvidia-docker)
# --------------------------------------------------------------------------------
LOG "Installing NVIDIA Container Toolkit"
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
# fallback mapping for ubuntu 22.04
if [ "$distribution" = "ubuntu22.04" ] || [ "$distribution" = "ubuntu20.04" ] || [ "$distribution" = "ubuntu18.04" ]; then
  :
else
  # if IDVERSION not matched, set to ubuntu22.04 (safe default for 22.04)
  distribution="ubuntu22.04"
fi

curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-docker-archive-keyring.gpg
curl -sSL "https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list" \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-docker-archive-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-docker.list

apt-get update -yq
apt-get install -yq nvidia-container-toolkit || {
  LOG "nvidia-container-toolkit install failed; attempting to install nvidia-container-runtime"
  apt-get install -yq nvidia-container-runtime || true
}

systemctl restart docker || true

# --------------------------------------------------------------------------------
# Quick verification: host & docker GPU test
# --------------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  LOG "Host nvidia-smi output:"
  nvidia-smi || true
else
  LOG "Host nvidia-smi not found - GPU driver not available on host"
fi

# Pull the CUDA runtime container and test GPU from inside container
LOG "Pulling CUDA container image: $CUDA_CONTAINER_IMAGE"
docker pull "$CUDA_CONTAINER_IMAGE" || { LOG "Failed to pull $CUDA_CONTAINER_IMAGE"; exit 1; }

LOG "Testing GPU from within container"
if docker run --rm --gpus all "$CUDA_CONTAINER_IMAGE" nvidia-smi; then
  LOG "Containerized nvidia-smi succeeded"
else
  LOG "Containerized nvidia-smi failed - the host driver or toolkit may be missing or incompatible"
  # keep going (non-zero would interrupt cloud-init); user can inspect logs.
fi

# --------------------------------------------------------------------------------
# Final cleanup and housekeeping
# --------------------------------------------------------------------------------
LOG "Cleaning apt caches and temporary files"
apt-get autoremove -yq || true
apt-get clean -yq || true
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* || true

if command -v cloud-init >/dev/null 2>&1; then
  LOG "Cleaning cloud-init state"
  cloud-init clean -s -l || true
fi

LOG "Done. GPU-EC2 setup script finished."
exit 0


