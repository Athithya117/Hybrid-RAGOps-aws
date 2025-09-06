#!/usr/bin/env bash
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive
export WORKSPACE_MODELS=${WORKSPACE_MODELS:-/workspace/models}
export HF_TOKEN=${HF_TOKEN:-}

apt-get update -yq
apt-get install -yq --no-install-recommends python3 python3-venv python3-pip git curl ca-certificates gnupg lsb-release software-properties-common jq
python3 -m pip install --no-warn-script-location --upgrade pip==25.2.0
python3 -m pip install --no-cache-dir huggingface-hub==0.34.3 tqdm==4.67.1 requests==2.32.4
mkdir -p "$WORKSPACE_MODELS"
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
chmod 755 /opt/download_models.py
python3 /opt/download_models.py
chown -R 1000:1000 "$WORKSPACE_MODELS"
chmod -R 755 "$WORKSPACE_MODELS"
chmod -R a+rX "$WORKSPACE_MODELS"
apt-get remove -yq docker docker-engine docker.io containerd runc || true
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /tmp/docker_gpg.key
mkdir -p /etc/apt/keyrings
gpg --batch --yes --dearmor -o /etc/apt/keyrings/docker.gpg /tmp/docker_gpg.key
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" >/etc/apt/sources.list.d/docker.list
apt-get update -yq
DOCKER_VERSION=$(apt-cache madison docker-ce | awk '{print $3; exit}' || true)
CLI_VERSION=$(apt-cache madison docker-ce-cli | awk '{print $3; exit}' || echo "$DOCKER_VERSION")
CONTAINERD_VERSION=$(apt-cache madison containerd.io | awk '{print $3; exit}' || true)
if [ -n "$DOCKER_VERSION" ] && [ -n "$CLI_VERSION" ] && [ -n "$CONTAINERD_VERSION" ]; then apt-get install -yq --allow-downgrades docker-ce="$DOCKER_VERSION" docker-ce-cli="$CLI_VERSION" containerd.io="$CONTAINERD_VERSION"; else apt-get install -yq docker-ce docker-ce-cli containerd.io; fi
apt-mark hold docker-ce docker-ce-cli containerd.io || true
systemctl enable --now docker || true
usermod -aG docker "${SUDO_USER:-ubuntu}" || true
apt-get autoremove -yq
apt-get clean -yq
rm -rf /var/lib/apt/lists/*
if command -v cloud-init >/dev/null 2>&1; then cloud-init clean -s -l || true; fi
truncate -s 0 /var/log/*log || true
rm -rf /tmp/hf_download || true

docker pull athithya324/frontend-streamlit:v1
docker pull athithya324/embedder-cpu-inference:v1

exit 0

