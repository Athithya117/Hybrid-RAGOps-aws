#!/usr/bin/env bash
# run as root in ubuntu-22.04 arm ec2 instance
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run as root"; exit 1; }
export DEBIAN_FRONTEND=noninteractive
WORKSPACE_MODELS="${WORKSPACE_MODELS:-/workspace/models}"
HF_TOKEN="${HF_TOKEN:-}"
apt-get update -yq
apt-get install -yq --no-install-recommends ca-certificates curl gnupg lsb-release python3 python3-venv python3-pip git jq
python3 -m pip install --no-warn-script-location --upgrade pip==25.2.0
python3 -m pip install --no-cache-dir huggingface-hub==0.34.3 tqdm==4.67.1 requests==2.32.4
mkdir -p "$WORKSPACE_MODELS"
cat >/opt/download_models.py <<'PY'
import os,sys,shutil,logging
from pathlib import Path
from huggingface_hub import hf_hub_download
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
WORKSPACE_MODELS=Path(os.getenv("WORKSPACE_MODELS","/workspace/models"))
FORCE=os.getenv("FORCE_DOWNLOAD","0").lower() in ("1","true","yes")
MODELS=[
 {"repo_id":"RAG8s/gte-modernbert-base-onnx-int8","name":"gte-modernbert-base-onnx-int8","base":"onnx","items":["model.onnx","config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json"]},
 {"repo_id":"Systran/faster-whisper-base","name":"faster-whisper-base","base":"faster_whisper","items":["model.bin","config.json","tokenizer.json","vocabulary.txt","README.md"]},
]
def dl(repo,fn,target):
 if target.exists() and not FORCE: return True
 tmp=Path("/tmp/hf_download"); tmp.mkdir(parents=True,exist_ok=True)
 try:
  got=hf_hub_download(repo_id=repo,filename=fn,local_dir=str(tmp),local_dir_use_symlinks=False,force_download=FORCE)
  p=Path(got)
  if p.exists():
   target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists(): target.unlink()
   shutil.move(str(p),str(target))
   try: os.chmod(str(target),0o444)
   except: pass
   logging.getLogger("download_hf").info("downloaded %s -> %s",fn,target)
   return True
 except Exception as e:
  logging.getLogger("download_hf").warning("fail %s:%s %s",repo,fn,e)
 return False
def ensure(m):
 ok=True
 root=WORKSPACE_MODELS/ m.get("base","llm")/ m["name"]
 for it in m.get("items",[]):
  tgt=root/it
  req= not it.endswith("special_tokens_map.json")
  if not dl(m["repo_id"],it,tgt) and req: ok=False
 return ok
def main():
 all_ok=True
 for m in MODELS:
  if not ensure(m): all_ok=False
 if not all_ok:
  logging.error("missing"); sys.exit(2)
 logging.info("models under %s",WORKSPACE_MODELS)
if __name__=="__main__":
 main()
PY
chmod 755 /opt/download_models.py
python3 /opt/download_models.py || true
chown -R 1000:1000 "$WORKSPACE_MODELS" || true
chmod -R a+rX "$WORKSPACE_MODELS" || true
apt-get remove -yq docker docker-engine docker.io containerd runc || true
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg || true
ARCH="$(dpkg --print-architecture || true)"
DISTRO="$(lsb_release -cs || echo jammy)"
echo "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${DISTRO} stable" >/etc/apt/sources.list.d/docker.list
apt-get update -yq
set +e
apt-get install -yq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
rc=$?
set -e
if [ $rc -ne 0 ]; then
 apt-get update -yq
 apt-get install -yq docker.io || curl -fsSL https://get.docker.com | sh
fi
systemctl enable --now docker || true
usermod -aG docker "${SUDO_USER:-ubuntu}" || true
apt-get autoremove -yq || true
apt-get clean -yq || true
rm -rf /var/lib/apt/lists/* || true
if command -v cloud-init >/dev/null 2>&1; then cloud-init clean -s -l || true; fi
truncate -s 0 /var/log/*log || true
rm -rf /tmp/hf_download || true
docker --version || true
IM1="athithya324/frontend-streamlit:v1"
IM2="athithya324/embedder-cpu-inference:linux-amd64-arm64"
docker pull --platform=linux/arm64 "$IM1" || docker pull "$IM1" || true
arch1="$(docker image inspect --format '{{.Architecture}}' "$IM1" 2>/dev/null || true)"
if [ "$arch1" != "arm64" ]; then echo "WARNING: $IM1 is $arch1 (no arm64 manifest). It may not run on this host without emulation."; fi
docker pull --platform=linux/arm64 "$IM2" || docker pull "$IM2" || true
arch2="$(docker image inspect --format '{{.Architecture}}' "$IM2" 2>/dev/null || true)"
if [ "$arch2" != "arm64" ]; then echo "WARNING: $IM2 is $arch2 (no arm64 manifest)."; fi
exit 0
