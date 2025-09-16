#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

WORKSPACE_MODELS="${WORKSPACE_MODELS:-/workspace/models}"
HF_TOKEN="${HF_TOKEN:-}"

# ----- HARD-CODED PINNED VERSIONS (edit if apt can't find them) -----
DOCKER_CE_VERSION="5:24.0.2~3-0~ubuntu-jammy"
DOCKER_CLI_VERSION="5:24.0.2~3-0~ubuntu-jammy"
CONTAINERD_VERSION="1.7.3-1"
# ------------------------------------------------------------------

apt-get update -yq
apt-get install -yq --no-install-recommends \
  python3 python3-venv python3-pip git curl ca-certificates gnupg lsb-release \
  software-properties-common jq

python3 -m pip install --no-warn-script-location --upgrade pip==25.2.0
python3 -m pip install --no-cache-dir huggingface-hub==0.34.3 tqdm==4.67.1 requests==2.32.4

mkdir -p "$WORKSPACE_MODELS"

cat >/opt/download_models.py <<'PY'
import os, sys, logging, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("download_hf")
WORKSPACE_MODELS = Path(os.getenv("WORKSPACE_MODELS", "/workspace/models"))
FORCE = os.getenv("FORCE_DOWNLOAD", "0").lower() in ("1", "true", "yes")

MODELS = [
    {"repo_id":"RAG8s/gte-modernbert-base-onnx-int8","name":"gte-modernbert-base-onnx-int8","base":"onnx","items":["model.onnx","config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json"]},
    {"repo_id":"Systran/faster-whisper-base","name":"faster-whisper-base","base":"faster_whisper","items":["model.bin","config.json","tokenizer.json","vocabulary.txt","README.md"]},
]

def download_one(repo_id, remote, target):
    if target.exists() and not FORCE:
        logger.info("SKIP exists %s", target); return True
    tmp_dir = Path("/tmp") / "hf_download"; tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        got = hf_hub_download(repo_id=repo_id, filename=remote, local_dir=str(tmp_dir), local_dir_use_symlinks=False, force_download=FORCE)
        got_path = Path(got)
        if got_path.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                try: target.unlink()
                except Exception: pass
            shutil.move(str(got_path), str(target))
            try: os.chmod(str(target), 0o444)
            except Exception: pass
            logger.info("Downloaded %s -> %s", remote, target); return True
    except Exception as e:
        logger.warning("Failed to download %s:%s (%s)", repo_id, remote, e)
    return False

def ensure_model(m):
    repo_id = m["repo_id"]; name = m["name"]; base = m.get("base", "llm")
    model_root = WORKSPACE_MODELS / base / name
    ok = True
    for item in m.get("items", []):
        target = model_root / Path(item)
        required = not item.endswith("special_tokens_map.json")
        success = download_one(repo_id, item, target)
        if not success and required:
            ok = False; logger.error("Missing required %s:%s", name, item)
    return ok

def main():
    all_ok = True
    for m in MODELS:
        if not ensure_model(m): all_ok = False
    if not all_ok:
        logger.error("Some required files failed to download"); sys.exit(2)
    logger.info("All model artifacts are present under %s", WORKSPACE_MODELS)

if __name__ == "__main__":
    main()
PY

chmod 755 /opt/download_models.py
python3 /opt/download_models.py

chown -R 1000:1000 "$WORKSPACE_MODELS" || true
chmod -R 755 "$WORKSPACE_MODELS" || true
chmod -R a+rX "$WORKSPACE_MODELS" || true

apt-get remove -yq docker docker-engine docker.io containerd runc || true

curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /tmp/docker_gpg.key
mkdir -p /etc/apt/keyrings
gpg --batch --yes --dearmor -o /etc/apt/keyrings/docker.gpg /tmp/docker_gpg.key
chmod a+r /etc/apt/keyrings/docker.gpg

ARCH="$(dpkg --print-architecture || true)"
DISTRO="$(lsb_release -cs || echo ubuntu)"
echo "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${DISTRO} stable" >/etc/apt/sources.list.d/docker.list
apt-get update -yq

set +e
apt-get install -yq --allow-downgrades "docker-ce=${DOCKER_CE_VERSION}" "docker-ce-cli=${DOCKER_CLI_VERSION}" "containerd.io=${CONTAINERD_VERSION}"
rc=$?
set -e
if [ $rc -ne 0 ]; then
  cat >&2 <<'ERR'

ERROR: Installing the pinned Docker packages failed.
Possible causes:
 - The hard-coded package strings do not exist in the Docker apt repo for your distro/arch.

To list available versions run:
  apt-cache madison docker-ce | awk '{print $3}' | uniq | sed -n '1,20p'
  apt-cache madison docker-ce-cli | awk '{print $3}' | uniq | sed -n '1,20p'
  apt-cache madison containerd.io | awk '{print $3}' | uniq | sed -n '1,20p'

Edit this script and replace the three DOCKER_* variables with one of the strings listed above, then re-run.

ERR
  exit 1
fi

apt-mark hold docker-ce docker-ce-cli containerd.io || true
systemctl enable --now docker || true
usermod -aG docker "${SUDO_USER:-ubuntu}" || true

apt-get autoremove -yq || true
apt-get clean -yq || true
rm -rf /var/lib/apt/lists/* || true
if command -v cloud-init >/dev/null 2>&1; then cloud-init clean -s -l || true; fi
truncate -s 0 /var/log/*log || true
rm -rf /tmp/hf_download || true

echo "docker --version:"
docker --version || true
echo "Pinned packages installed:"
echo "  docker-ce: ${DOCKER_CE_VERSION}"
echo "  docker-ce-cli: ${DOCKER_CLI_VERSION}"
echo "  containerd.io: ${CONTAINERD_VERSION}"

docker pull athithya324/frontend-streamlit:v1 || true
# docker pull --platform=linux/amd64 athithya324/embedder-cpu-inference:linux-amd64-arm64

docker pull --platform=linux/arm64 athithya324/embedder-cpu-inference:linux-amd64-arm64

exit 0

