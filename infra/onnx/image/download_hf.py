import os,logging,shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
logger=logging.getLogger("download_hf")
WORKSPACE_MODELS=os.getenv("WORKSPACE_MODELS","/workspace/models")
FORCE=os.getenv("FORCE_DOWNLOAD","0").lower() in ("1","true","yes")
MODELS=[{"repo_id":"RAG8s/gte-modernbert-base-onnx-int8","name":"gte-modernbert-base-onnx-int8","items":[("onnx/model_int8.onnx","onnx/model_int8.onnx"),("model_int8.onnx","onnx/model_int8.onnx"),("config.json","config.json"),("tokenizer.json","tokenizer.json"),("tokenizer_config.json","tokenizer_config.json"),("special_tokens_map.json","special_tokens_map.json")]},{"repo_id":"RAG8s/gte-reranker-modernbert-base-onnx-int8","name":"gte-reranker-modernbert-base-onnx-int8","items":[("onnx/model_int8.onnx","onnx/model_int8.onnx"),("model_int8.onnx","onnx/model_int8.onnx"),("config.json","config.json"),("tokenizer.json","tokenizer.json"),("tokenizer_config.json","tokenizer_config.json"),("special_tokens_map.json","special_tokens_map.json")]}]
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
