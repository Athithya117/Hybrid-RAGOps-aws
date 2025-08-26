#!/usr/bin/env bash
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive
export WORKSPACE_MODELS=${WORKSPACE_MODELS:-/workspace/models}
export HF_TOKEN=${HF_TOKEN:-}
if [ "$(uname -m)" != "aarch64" ]; then echo "expected aarch64/Graviton base AMI"; exit 2; fi
apt-get update -yq
apt-get install -yq --no-install-recommends python3 python3-venv python3-pip git curl ca-certificates gnupg lsb-release software-properties-common jq
python3 -m pip install --no-warn-script-location --upgrade pip==25.2.0
python3 -m pip install --no-cache-dir huggingface-hub==0.34.3 tqdm==4.67.1 requests==2.32.4
mkdir -p "$WORKSPACE_MODELS"
cat >/opt/download_models.py <<'PY'
import os,shutil,time,logging
from pathlib import Path
from huggingface_hub import hf_hub_download
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
logger=logging.getLogger("download_hf")
WORKSPACE_MODELS=os.getenv("WORKSPACE_MODELS","/workspace/models")
FORCE=os.getenv("FORCE_DOWNLOAD","0").lower() in ("1","true","yes")
TOKEN=os.getenv("HF_TOKEN") or None
MODELS=[
 {"repo_id":"RAG8s/gte-modernbert-base-onnx-int8","name":"gte-modernbert-base-onnx-int8","items":[("onnx/model_int8.onnx","onnx/model_int8.onnx"),("model_int8.onnx","onnx/model_int8.onnx"),("config.json","config.json"),("tokenizer.json","tokenizer.json"),("tokenizer_config.json","tokenizer_config.json"),("special_tokens_map.json","special_tokens_map.json")]},
 {"repo_id":"RAG8s/gte-reranker-modernbert-base-onnx-int8","name":"gte-reranker-modernbert-base-onnx-int8","items":[("onnx/model_int8.onnx","onnx/model_int8.onnx"),("model_int8.onnx","onnx/model_int8.onnx"),("config.json","config.json"),("tokenizer.json","tokenizer.json"),("tokenizer_config.json","tokenizer_config.json"),("special_tokens_map.json","special_tokens_map.json")]},
 {"repo_id":"Qwen/Qwen3-4B-AWQ","name":"Qwen3-4B-AWQ","items":[("config.json","config.json"),("generation_config.json","generation_config.json"),("tokenizer.json","tokenizer.json"),("tokenizer_config.json","tokenizer_config.json"),("special_tokens_map.json","special_tokens_map.json"),("pytorch_model.bin","pytorch_model.bin")]}
]
def download_one(repo_id,name,remote_candidates,target_rel):
    target=Path(WORKSPACE_MODELS)/"onnx"/name/Path(target_rel)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and not FORCE: logger.info("SKIP exists %s",target); return True
    last_exc=None
    tmp_dir=Path("/tmp")/"hf_download"/name
    tmp_dir.mkdir(parents=True,exist_ok=True)
    for remote in remote_candidates:
        for attempt in range(3):
            try:
                logger.info("Attempt %s for %s -> %s",attempt+1,remote,target)
                got=hf_hub_download(repo_id=repo_id,filename=remote,local_dir=str(tmp_dir),local_dir_use_symlinks=False,force_download=FORCE,token=TOKEN)
                got_path=Path(got)
                if got_path.exists():
                    if got_path.resolve() != target.resolve():
                        shutil.move(str(got_path),str(target))
                    os.chmod(str(target),0o444)
                    logger.info("Downloaded %s to %s",remote,target)
                    return True
            except Exception as e:
                last_exc=e
                time.sleep(2*(attempt+1))
    logger.warning("All candidates failed for %s:%s last=%s",repo_id,target_rel,last_exc)
    return False
def ensure_model(model):
    repo_id=model["repo_id"]
    name=model["name"]
    ok=True
    for remote_rel,target_rel in model["items"]:
        required=not remote_rel.endswith("special_tokens_map.json")
        candidates=[remote_rel]
        if "/" in remote_rel: candidates.append(remote_rel.split("/",1)[1])
        if remote_rel.startswith("onnx/"): candidates.append(remote_rel.split("onnx/",1)[1])
        success=download_one(repo_id,name,candidates,target_rel)
        if not success and required:
            ok=False
            logger.error("Required file missing for %s: %s",name,target_rel)
    return ok
def main():
    all_ok=True
    for m in MODELS:
        if not ensure_model(m): all_ok=False
    if not all_ok: logger.error("One or more required files failed to download"); raise SystemExit(2)
    logger.info("All model artifacts present under %s/onnx",WORKSPACE_MODELS)
if __name__=="__main__": main()
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
exit 0
