
sudo dpkg --configure -a
sudo apt update
sudo DEBIAN_FRONTEND=noninteractive apt -y install git-lfs
sudo git lfs install

git remote set-url origin https://RAG8s:hf_iqoNpjlyjPBeJ@huggingface.co/RAG8s/gte-modernbert-base-onnx-int8
git clone https://huggingface.co/RAG8s/gte-modernbert-base-onnx-int8
cd gte-modernbert-base-onnx-int8
tree

git rm onnx/model_int8.onnx
git add onnx/model.onnx
git commit -m "Remove old int8 model file, keep canonical model.onnx"
git push

git remote set-url origin https://RAG8s:hfJ@huggingface.co/RAG8s/gte-modernbert-reranker-base-onnx-int8
git clone https://huggingface.co/RAG8s/gte-reranker-modernbert-base-onnx-int8
cd gte-reranker-modernbert-base-onnx-int8
tree
git rm onnx/model_int8.onnx
git add onnx/model.onnx
git commit -m "Remove old int8 model file, keep canonical model.onnx"
git push


pip install --upgrade pip

# Core: Hugging Face Transformers & Optimum
pip install "transformers>=4.41.0" "optimum[onnxruntime]>=1.18.0"

# ONNX Runtime for inference (CPU version is enough unless you want GPU CUDA)
pip install onnxruntime

# Optional but recommended: accelerate (better device placement)
pip install accelerate


# sanity: ensure in repo root
cd ~/RAG8s/gte-modernbert-base-onnx-int8

# ensure git-lfs present
git lfs install --local || true

# create a local backup of current HEAD (safe)
git branch backup-before-reset-$(date +%Y%m%d%H%M%S)

# fetch all remote refs and LFS objects
git fetch --all --tags --prune
git lfs fetch --all

# verify commit exists anywhere (remote or local)
if git cat-file -e s535a43^{commit}; then
  echo "commit exists locally"
else
  echo "commit not found locally; attempting to fetch from origin"
  git fetch origin s535a43 || true
fi

# hard reset to the desired commit (danger: rewrites working tree)
git reset --hard s535a43

# ensure LFS files for that commit are checked out
git lfs checkout

# show the status and that model.onnx exists now
ls -l model.onnx || ls -l onnx/model.onnx || true
git show --name-only --pretty="" s535a43 | sed -n '1,200p'

# force-push the branch (overwrite remote main/master). Replace 'main' with your remote branch if different.
REMOTE_BRANCH=${REMOTE_BRANCH:-main}
git push origin HEAD:${REMOTE_BRANCH} --force

# push LFS objects (best-effort)
git lfs push --all origin
