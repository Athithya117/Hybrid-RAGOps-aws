#!/usr/bin/env bash
# Ubuntu 22.04 custom AMI for launching graviton based ec2s

sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip git curl ca-certificates
python3 -m pip install --upgrade pip==25.2
pip install huggingface-hub==0.34.3 tqdm==4.67.1 requests==2.32.4

cat >/opt/download_models.py <<'EOF'
import os,logging,shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
logger=logging.getLogger("download_hf")
WORKSPACE_MODELS=os.getenv("WORKSPACE_MODELS","/workspace/models")
FORCE=os.getenv("FORCE_DOWNLOAD","0").lower() in ("1","true","yes")
MODELS=[
 {"repo_id":"RAG8s/gte-modernbert-base-onnx-int8","name":"gte-modernbert-base-onnx-int8","items":[("onnx/model_int8.onnx","onnx/model_int8.onnx"),("model_int8.onnx","onnx/model_int8.onnx"),("config.json","config.json"),("tokenizer.json","tokenizer.json"),("tokenizer_config.json","tokenizer_config.json"),("special_tokens_map.json","special_tokens_map.json")]},
 {"repo_id":"RAG8s/gte-reranker-modernbert-base-onnx-int8","name":"gte-reranker-modernbert-base-onnx-int8","items":[("onnx/model_int8.onnx","onnx/model_int8.onnx"),("model_int8.onnx","onnx/model_int8.onnx"),("config.json","config.json"),("tokenizer.json","tokenizer.json"),("tokenizer_config.json","tokenizer_config.json"),("special_tokens_map.json","special_tokens_map.json")]}
]
def download_one(repo_id,name,remote_candidates,target_rel):
    target=Path(WORKSPACE_MODELS)/"onnx"/name/Path(target_rel)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and not FORCE:
        logger.info("SKIP exists %s",target)
        return True
    last_exc=None
    tmp_dir=Path("/tmp")/"hf_download"/name
    tmp_dir.mkdir(parents=True,exist_ok=True)
    for remote in remote_candidates:
        try:
            logger.info("Attempting %s -> %s",remote,str(target))
            got=hf_hub_download(repo_id=repo_id,filename=remote,local_dir=str(tmp_dir),local_dir_use_symlinks=False,force_download=FORCE)
            got_path=Path(got)
            if got_path.exists():
                if got_path.resolve() != target.resolve():
                    shutil.move(str(got_path),str(target))
                os.chmod(str(target),0o444)
                logger.info("Downloaded %s to %s",remote,str(target))
                return True
        except Exception as e:
            last_exc=e
            logger.debug("failed candidate %s: %s",remote,e)
    logger.warning("All candidates failed for %s:%s last=%s",repo_id,target_rel,last_exc)
    return False
def ensure_model(model):
    repo_id=model["repo_id"]
    name=model["name"]
    ok=True
    for remote_rel,target_rel in model["items"]:
        required=not remote_rel.endswith("special_tokens_map.json")
        candidates=[remote_rel]
        if "/" in remote_rel:
            candidates.append(remote_rel.split("/",1)[1])
        if remote_rel.startswith("onnx/"):
            candidates.append(remote_rel.split("onnx/",1)[1])
        success=download_one(repo_id,name,candidates,target_rel)
        if not success and required:
            ok=False
            logger.error("Required file missing for %s: %s",name,target_rel)
    return ok
def main():
    all_ok=True
    for m in MODELS:
        if not ensure_model(m):
            all_ok=False
    if not all_ok:
        logger.error("One or more required files failed to download")
        raise SystemExit(2)
    logger.info("All model artifacts present under %s/onnx",WORKSPACE_MODELS)
if __name__=="__main__":
    main()
EOF

sudo chmod 755 /opt/download_models.py

export WORKSPACE_MODELS=/workspace/models
mkdir -p $WORKSPACE_MODELS
python3 /opt/download_models.py

sudo chown -R 1000:1000 /workspace/models
sudo chmod -R 755 /workspace/models
sudo chmod -R a+rX /workspace/models


export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get remove -y docker docker-engine docker.io containerd runc || true
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release software-properties-common
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /tmp/docker_gpg.key
sudo gpg --batch --yes --dearmor -o /etc/apt/keyrings/docker.gpg /tmp/docker_gpg.key
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
DOCKER_VERSION=$(apt-cache madison docker-ce 2>/dev/null | awk '{print $3; exit}' || true)
CLI_VERSION=$(apt-cache madison docker-ce-cli 2>/dev/null | awk '{print $3; exit}' || echo "$DOCKER_VERSION")
CONTAINERD_VERSION=$(apt-cache madison containerd.io 2>/dev/null | awk '{print $3; exit}' || true)
if [ -n "$DOCKER_VERSION" ] && [ -n "$CLI_VERSION" ] && [ -n "$CONTAINERD_VERSION" ]; then sudo apt-get install -y --allow-downgrades docker-ce="$DOCKER_VERSION" docker-ce-cli="$CLI_VERSION" containerd.io="$CONTAINERD_VERSION"; else sudo apt-get install -y docker-ce docker-ce-cli containerd.io; fi
sudo apt-mark hold docker-ce docker-ce-cli containerd.io || true
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER" || true
sudo docker --version
