import os,logging,shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
logger=logging.getLogger("download_hf")
WORKSPACE_MODELS=os.getenv("WORKSPACE_MODELS","/workspace/models")
FORCE=os.getenv("FORCE_DOWNLOAD","0").lower() in ("1","true","yes")
MODELS=[
 {"repo_id":"RAG8s/gte-modernbert-base-onnx-int8","name":"gte-modernbert-base-onnx-int8","base":"onnx","items":[("onnx/model_int8.onnx","onnx/model_int8.onnx"),("model_int8.onnx","onnx/model_int8.onnx"),("config.json","config.json"),("tokenizer.json","tokenizer.json"),("tokenizer_config.json","tokenizer_config.json"),("special_tokens_map.json","special_tokens_map.json")]},
 {"repo_id":"RAG8s/gte-reranker-modernbert-base-onnx-int8","name":"gte-reranker-modernbert-base-onnx-int8","base":"onnx","items":[("onnx/model_int8.onnx","onnx/model_int8.onnx"),("model_int8.onnx","onnx/model_int8.onnx"),("config.json","config.json"),("tokenizer.json","tokenizer.json"),("tokenizer_config.json","tokenizer_config.json"),("special_tokens_map.json","special_tokens_map.json")]},
 {"repo_id":"Qwen/Qwen3-4B-AWQ","name":"qwen3-4b-awq","base":"llm","items":[("model.safetensors","model.safetensors"),("config.json","config.json"),("tokenizer.json","tokenizer.json"),("tokenizer_config.json","tokenizer_config.json"),("vocab.json","vocab.json"),("merges.txt","merges.txt"),("generation_config.json","generation_config.json")]}
]
def download_one(repo_id,name,remote_candidates,target_rel,base):
    target=Path(WORKSPACE_MODELS)/base/name/Path(target_rel)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and not FORCE:
        logger.info("SKIP exists %s",target)
        return True
    last_exc=None
    tmp_dir=Path("/tmp")/"hf_download"/base/name
    tmp_dir.mkdir(parents=True,exist_ok=True)
    for remote in remote_candidates:
        try:
            logger.info("Attempting %s -> %s",remote,str(target))
            got=hf_hub_download(repo_id=repo_id,filename=remote,local_dir=str(tmp_dir),local_dir_use_symlinks=False,force_download=FORCE)
            got_path=Path(got)
            if got_path.exists():
                if got_path.resolve()!=target.resolve():
                    shutil.move(str(got_path),str(target))
                os.chmod(str(target),0o444)
                logger.info("Downloaded %s to %s",remote,str(target))
                return True
        except Exception as e:
            last_exc=e
            logger.debug("failed candidate %s: %s",remote,e)
    logger.error("All candidates failed for %s:%s last=%s",repo_id,target_rel,last_exc)
    return False
def ensure_model(model):
    repo_id=model["repo_id"]
    name=model["name"]
    base=model["base"]
    ok=True
    for remote_rel,target_rel in model["items"]:
        required=not remote_rel.endswith("special_tokens_map.json")
        candidates=[remote_rel]
        if "/" in remote_rel:
            candidates.append(remote_rel.split("/",1)[1])
        if remote_rel.startswith("onnx/"):
            candidates.append(remote_rel.split("onnx/",1)[1])
        success=download_one(repo_id,name,candidates,target_rel,base)
        if not success and required:
            ok=False
            logger.error("Required file missing for %s: %s",name,target_rel)
    return ok
def remove_stray_onnx_duplicates(models):
    onnx_root=Path(WORKSPACE_MODELS)/"onnx"
    if not onnx_root.exists() or not onnx_root.is_dir():
        return
    for m in models:
        if m.get("base")=="onnx":
            continue
        llm_dir=Path(WORKSPACE_MODELS)/m.get("base","llm")/m["name"]
        if not llm_dir.exists():
            continue
        repo_last=m["repo_id"].split("/",1)[1] if "/" in m["repo_id"] else m["repo_id"]
        candidates=set([m["name"].lower(),repo_last.lower()])
        try:
            for child in onnx_root.iterdir():
                if not child.is_dir():
                    continue
                if child.name.lower() in candidates:
                    try:
                        shutil.rmtree(child)
                        logger.info("Removed stray onnx duplicate %s",str(child))
                    except Exception as e:
                        logger.warning("Failed to remove %s: %s",str(child),e)
        except Exception as e:
            logger.warning("Scanning onnx root failed: %s",e)
def main():
    all_ok=True
    for m in MODELS:
        if not ensure_model(m):
            all_ok=False
    if not all_ok:
        logger.error("One or more required files failed to download")
        raise SystemExit(2)
    remove_stray_onnx_duplicates(MODELS)
    logger.info("All model artifacts present under %s",WORKSPACE_MODELS)
if __name__=="__main__":
    main()
