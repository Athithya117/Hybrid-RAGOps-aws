"""
Pre-conversions pipeline:
 - Groups all blobs under STORAGE_RAW_PREFIX into canonical subfolders (idempotent).
 - Then runs conversions for docs -> pdfs, sheets -> csvs, audio -> wav.
 - Uses Azure Blob SDK streaming, deterministic metadata, and safe rename logic.
Environment:
  - AZURE_CONTAINER (required)
  - Preferred: AZURE_STORAGE_CONNECTION_STRING (or AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY / AZURE_SAS_TOKEN)
  - STORAGE_RAW_PREFIX (default: data/raw/)
Flags (env):
  - OVERWRITE_ALL_AUDIO_FILES, OVERWRITE_OTHER_TO_PDF, OVERWRITE_SPREADSHEETS_WITH_CSV
  - AZURE_USE_MANAGED_IDENTITY / USE_MANAGED_IDENTITY + UAI_RAG_RW_CLIENT_ID
"""
from __future__ import annotations
import os,sys,time,json,base64,hashlib,subprocess,shutil,resource,tempfile,threading
from pathlib import Path
from typing import Dict,Optional,Tuple,Any,Iterable
TS=lambda:time.strftime("%Y-%m-%dT%H:%M:%SZ")
def log(level:str,event:str,msg:str,**k:Any)->None:
    o={"ts":TS(),"level":level,"event":event,"msg":msg}
    if k:o.update(k)
    print(json.dumps(o,default=str),flush=True)
USE_MANAGED_IDENTITY=os.getenv("AZURE_USE_MANAGED_IDENTITY",os.getenv("USE_MANAGED_IDENTITY","")).strip().lower() in ("1","true","yes")
UAI_RAG_RW_CLIENT_ID=os.getenv("UAI_RAG_RW_CLIENT_ID","").strip()
STORAGE_RAW_PREFIX=(os.getenv("STORAGE_RAW_PREFIX",os.getenv("AZURE_RAW_PREFIX",os.getenv("S3_RAW_PREFIX","data/raw/"))).lstrip("/").rstrip("/"))+"/"
AZ_CONN=os.getenv("AZURE_STORAGE_CONNECTION_STRING","").strip()
AZ_ACCOUNT=os.getenv("AZURE_STORAGE_ACCOUNT_NAME","").strip()
AZ_KEY=os.getenv("AZURE_STORAGE_ACCOUNT_KEY","").strip()
AZ_SAS=os.getenv("AZURE_SAS_TOKEN","").strip()
AZ_ENDPOINT_SUFFIX=os.getenv("AZURE_ENDPOINT_SUFFIX","core.windows.net").strip()
AZURE_STRICT_VALIDATE=os.getenv("AZURE_STRICT_VALIDATE","true").strip().lower() in ("1","true","yes")
CONTAINER=os.getenv("AZURE_CONTAINER") or os.getenv("STORAGE_CONTAINER") or os.getenv("AZ_CONTAINER")
if not CONTAINER:
    log("ERROR","missing_env","AZURE_CONTAINER (or STORAGE_CONTAINER/AZ_CONTAINER) must be set")
    raise SystemExit(2)
TMP_ROOT=os.getenv("TMP_ROOT",os.getenv("TMP_DIR","/tmp/preconv"))
Path(TMP_ROOT).mkdir(parents=True,exist_ok=True)
OVERWRITE_ALL_AUDIO_FILES=os.getenv("OVERWRITE_ALL_AUDIO_FILES","true").strip().lower() in ("1","true","yes")
OVERWRITE_OTHER_TO_PDF=os.getenv("OVERWRITE_OTHER_TO_PDF","true").strip().lower() in ("1","true","yes")
OVERWRITE_SPREADSHEETS_WITH_CSV=os.getenv("OVERWRITE_SPREADSHEETS_WITH_CSV","true").strip().lower() in ("1","true","yes")
DESIRED_NOFILE=int(os.getenv("DESIRED_NOFILE","262144"))
FDS_PER_CONVERT=int(os.getenv("FDS_PER_CONVERT","60"))
SYSTEM_RESERVED_FDS=int(os.getenv("SYSTEM_RESERVED_FDS","200"))
ENV_MAX_PARALLEL=int(os.getenv("MAX_PARALLEL_CONVERTS",os.getenv("MAX_PARALLEL_CONVERTS_DEFAULT","2")))
try:
    from azure.storage.blob import BlobServiceClient,ContentSettings
except Exception as e:
    log("ERROR","import_failure","missing azure-storage-blob (pip install azure-storage-blob)",error=str(e))
    raise SystemExit(2)
if USE_MANAGED_IDENTITY:
    try:
        from azure.identity import DefaultAzureCredential,ManagedIdentityCredential
    except Exception as e:
        log("ERROR","import_failure","missing azure-identity (pip install azure-identity)",error=str(e))
        raise SystemExit(2)
def ensure_nofile_limit(desired:int)->int:
    try:
        soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft=min(desired,hard) if hard!=resource.RLIM_INFINITY else desired
        if new_soft>soft:
            resource.setrlimit(resource.RLIMIT_NOFILE,(new_soft,hard))
            log("INFO","nofile.raised","Raised RLIMIT_NOFILE",old_soft=soft,new_soft=new_soft,hard=hard)
            return new_soft
        log("DEBUG","nofile.unchanged","RLIMIT_NOFILE unchanged",soft=soft,hard=hard)
        return soft
    except Exception as e:
        try:
            soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
        except Exception:
            soft=1024
        log("WARN","nofile_raise_failed","Failed to raise RLIMIT_NOFILE",error=str(e))
        return soft
def current_fd_count()->int:
    try:
        return len([p for p in Path("/proc/self/fd").iterdir()])
    except Exception:
        return 0
def compute_max_workers(soft_limit:int,fds_per_worker:int,reserve:int,env_max:int)->int:
    usable=max(0,soft_limit-reserve)
    calc=max(1,usable//max(1,fds_per_worker))
    return max(1,min(calc,env_max))
def b64(v:str)->str:
    return base64.b64encode(v.encode("utf-8")).decode("ascii") if v is not None else ""
def compute_hashes(path:str,chunk_size:int=8*1024*1024)->Tuple[str,str]:
    md5=hashlib.md5();sha=hashlib.sha256()
    with open(path,"rb") as fh:
        while True:
            chunk=fh.read(chunk_size)
            if not chunk:break
            md5.update(chunk);sha.update(chunk)
    return md5.hexdigest(),sha.hexdigest()
def prepare_metadata(d:Dict[str,Optional[str]])->Dict[str,str]:
    out={}
    for k,v in (d or {}).items():
        kk=str(k).replace(" ","_").lower()
        out[kk]=b64(str(v)) if v is not None else ""
    return out
def build_blob_service_client()->"BlobServiceClient":
    if USE_MANAGED_IDENTITY:
        if not AZ_ACCOUNT:
            log("ERROR","env_missing","AZURE_STORAGE_ACCOUNT_NAME required for managed identity mode")
            raise SystemExit(2)
        account_url=f"https://{AZ_ACCOUNT}.{AZ_ENDPOINT_SUFFIX}"
        try:
            cred=ManagedIdentityCredential(client_id=UAI_RAG_RW_CLIENT_ID) if UAI_RAG_RW_CLIENT_ID else DefaultAzureCredential()
            client=BlobServiceClient(account_url=account_url,credential=cred)
            log("INFO","client_init","Initialized BlobServiceClient (managed identity)",account=AZ_ACCOUNT)
            return client
        except Exception as e:
            log("ERROR","mi_client_failed","Failed to init BlobServiceClient (MI)",error=str(e))
            raise SystemExit(2)
    if AZ_CONN:
        try:
            client=BlobServiceClient.from_connection_string(AZ_CONN)
            log("INFO","client_init","Initialized BlobServiceClient (connection string)")
            return client
        except Exception as e:
            log("ERROR","connstr_init_failed","Failed to init client from connection string",error=str(e))
            raise SystemExit(2)
    if AZ_ACCOUNT and AZ_KEY:
        try:
            account_url=f"https://{AZ_ACCOUNT}.{AZ_ENDPOINT_SUFFIX}"
            client=BlobServiceClient(account_url=account_url,credential=AZ_KEY)
            log("INFO","client_init","Initialized BlobServiceClient (account+key)",account=AZ_ACCOUNT)
            return client
        except Exception as e:
            log("ERROR","acctkey_init_failed","Failed to init client from account+key",error=str(e))
            raise SystemExit(2)
    if AZ_ACCOUNT and AZ_SAS:
        try:
            account_url=f"https://{AZ_ACCOUNT}.{AZ_ENDPOINT_SUFFIX}"
            token=AZ_SAS if AZ_SAS.startswith("?") else ("?"+AZ_SAS)
            client=BlobServiceClient(account_url=account_url+token)
            log("INFO","client_init","Initialized BlobServiceClient (account+SAS)",account=AZ_ACCOUNT)
            return client
        except Exception as e:
            log("ERROR","sas_init_failed","Failed to init client from SAS token",error=str(e))
            raise SystemExit(2)
    log("ERROR","auth_missing","Provide AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME+KEY or AZURE_SAS_TOKEN")
    raise SystemExit(2)
BLOB_SERVICE=build_blob_service_client()
try:
    CONTAINER_CLIENT=BLOB_SERVICE.get_container_client(CONTAINER)
except Exception as e:
    log("ERROR","container_client_failed","Unable to get container client for container",error=str(e))
    raise SystemExit(2)
if AZURE_STRICT_VALIDATE:
    try:
        CONTAINER_CLIENT.get_container_properties()
        log("INFO","container_validation","container accessible",container=CONTAINER)
    except Exception as e:
        log("ERROR","container_validation_failed","Failed to validate container",error=str(e))
        raise SystemExit(2)
def download_blob_to_file(blob_name:str,dst_path:str)->bool:
    try:
        bc=CONTAINER_CLIENT.get_blob_client(blob_name)
        downloader=bc.download_blob()
        Path(dst_path).parent.mkdir(parents=True,exist_ok=True)
        with open(dst_path,"wb") as fh:
            for chunk in downloader.chunks():
                fh.write(chunk)
        return True
    except Exception as e:
        log("WARN","download_failed","download failed",blob=blob_name,error=str(e))
        return False
def upload_file_to_blob(blob_name:str,src_path:str,metadata:Dict[str,str],content_type:str="application/octet-stream",overwrite:bool=True)->Dict[str,Any]:
    try:
        bc=CONTAINER_CLIENT.get_blob_client(blob_name)
        cs=ContentSettings(content_type=content_type)
        with open(src_path,"rb") as fh:
            bc.upload_blob(fh,overwrite=overwrite,content_settings=cs,metadata=metadata)
        props=bc.get_blob_properties()
        remote_meta=props.metadata or {}
        verified=all((k in remote_meta and remote_meta[k]==metadata[k]) for k in metadata)
        return {"action":"uploaded","verified":bool(verified),"etag":props.etag}
    except Exception as e:
        log("ERROR","az_upload_failed","upload failed",target=blob_name,error=str(e))
        return {"action":"failed","error":str(e)}
def get_blob_props(blob_name:str)->Dict[str,Any]:
    try:
        bc=CONTAINER_CLIENT.get_blob_client(blob_name)
        p=bc.get_blob_properties()
        return {"exists":True,"etag":p.etag,"size":int(getattr(p,"size",0) or 0),"metadata":p.metadata or {}}
    except Exception:
        return {"exists":False}
def make_unique_target(base_target:str)->str:
    if not get_blob_props(base_target).get("exists"):
        return base_target
    stem=str(Path(base_target).with_suffix("")).rstrip(".")
    suffix=Path(base_target).suffix
    i=1
    while True:
        candidate=f"{stem}-{i}{suffix}"
        if not get_blob_props(candidate).get("exists"):
            return candidate
        i+=1
def safe_move_blob(src:str,dst:str)->Tuple[bool,str]:
    src_props=get_blob_props(src)
    dst_props=get_blob_props(dst)
    if dst_props.get("exists"):
        try:
            src_size=src_props.get("size",0);dst_size=dst_props.get("size",0)
            if src_size and dst_size and src_size==dst_size:
                CONTAINER_CLIENT.delete_blob(src)
                log("INFO","group_dedup","deleted source as identical target exists",src=src,dst=dst)
                return True,dst
            dst=make_unique_target(dst)
        except Exception:
            dst=make_unique_target(dst)
    tmp=str(Path(TMP_ROOT)/"move"/Path(src).name)
    try:
        ok=download_blob_to_file(src,tmp)
        if not ok:
            return False,dst
        md5,sha=compute_hashes(tmp)
        meta=prepare_metadata({"sha256":sha,"original_name":Path(src).name})
        up=upload_file_to_blob(dst,tmp,meta,content_type="application/octet-stream",overwrite=True)
        if up.get("action")=="uploaded":
            try:CONTAINER_CLIENT.delete_blob(src)
            except Exception:pass
            return True,dst
        else:
            return False,dst
    except Exception as e:
        log("WARN","group_move_failed","exception during move",src=src,dst=dst,error=str(e))
        return False,dst
    finally:
        try:
            if os.path.exists(tmp):os.remove(tmp)
        except Exception:pass
def ext_to_subdir(name:str)->str:
    ext=Path(name).suffix.lstrip(".").lower()
    mapping={e:"audio" for e in ("mp3","m4a","aac","wav","flac","ogg","opus","webm","amr","wma","aiff","aif")}
    mapping.update({e:"images" for e in ("jpg","jpeg","png","webp","tif","tiff","bmp","gif")})
    mapping.update({e:"docs" for e in ("doc","docx")})
    mapping.update({e:"ppts" for e in ("ppt","pptx")})
    mapping.update({e:"sheets" for e in ("xls","xlsx","ods","xlsm","xlsb")})
    mapping.update({"pdf":"pdfs","csv":"csvs","md":"mds","txt":"txts","html":"htmls","htm":"htmls"})
    return mapping.get(ext,"others")
def list_blobs(prefix:str)->Iterable[str]:
    try:
        for b in CONTAINER_CLIENT.list_blobs(name_starts_with=prefix):
            yield b.name
    except Exception as e:
        log("ERROR","list_failed","listing blobs failed",prefix=prefix,error=str(e))
        raise
def run_soffice_convert(src:str,outdir:str,convert_to:str)->Tuple[bool,str]:
    env=os.environ.copy();env["SAL_USE_VCLPLUGIN"]=env.get("SAL_USE_VCLPLUGIN","gen");env["HOME"]=env.get("HOME","/tmp")
    cmd=["soffice","--headless","--invisible","--nologo","--nodefault","--nofirststartwizard","--nolockcheck","--convert-to",convert_to,"--outdir",outdir,src]
    try:
        res=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,timeout=120,close_fds=True,cwd=outdir)
        stderr=(res.stderr or b"").decode("utf-8",errors="ignore").strip()
        if res.returncode==0:return True,""
        return False,("\n".join(stderr.splitlines()[:6]) if stderr else f"soffice rc={res.returncode}")
    except Exception as e:
        return False,str(e)
def run_ffmpeg_convert(src:str,dst:str,extra_args:Optional[list]=None)->Tuple[bool,str]:
    args=["ffmpeg","-y","-hide_banner","-loglevel","error","-i",src,"-ar","16000","-ac","1","-sample_fmt","s16",dst]
    if extra_args:args=args[:-1]+extra_args+[args[-1]]
    try:
        res=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120,close_fds=True,cwd=Path(dst).parent)
        if res.returncode==0:return True,""
        out=(res.stderr or b"").decode("utf-8",errors="ignore").splitlines()[:6]
        return False,("\n".join(out) if out else "ffmpeg error")
    except Exception as e:
        return False,str(e)
def _flush_tmp_cleanup(path:str)->None:
    try:
        if os.path.exists(path):
            if os.path.isfile(path):os.remove(path)
            else:shutil.rmtree(path,ignore_errors=True)
    except Exception:
        pass
def process_doc(blob_name:str)->None:
    name=Path(blob_name).name;ext=Path(name).suffix.lstrip(".").lower()
    if ext not in ("doc","docx"):return
    with tempfile.TemporaryDirectory(prefix="preconv-",dir=TMP_ROOT) as tmpdir:
        src_local=str(Path(tmpdir)/"src"/name);outdir=str(Path(tmpdir)/"out");Path(src_local).parent.mkdir(parents=True,exist_ok=True);Path(outdir).mkdir(parents=True,exist_ok=True)
        if not download_blob_to_file(blob_name,src_local):
            log("WARN","doc_download_failed","download failed",blob=blob_name);return
        ok,err=run_soffice_convert(src_local,outdir,"pdf:writer_pdf_Export")
        if not ok:
            qname=f"{STORAGE_RAW_PREFIX}quarantine/{name}.corrupt"
            metadata=prepare_metadata({"quarantined_from":blob_name,"error":err,"original_name":name})
            upload_file_to_blob(qname,src_local,metadata,content_type="application/octet-stream",overwrite=True)
            log("WARN","doc_convert_failed","soffice failed",blob=blob_name,error=err)
            return
        pdf_path=None
        for p in Path(outdir).glob(f"{Path(name).stem}*.pdf"):
            pdf_path=str(p);break
        if not pdf_path:
            log("WARN","doc_no_pdf","conversion produced no pdf",blob=blob_name);return
        tgt_blob=f"{STORAGE_RAW_PREFIX}pdfs/{Path(name).stem}.pdf"
        md5,sha=compute_hashes(src_local);src_props=get_blob_props(blob_name);src_etag=src_props.get("etag","")
        meta=prepare_metadata({"sha256":sha,"original_name":name,"original_ext":ext,"converted_from":blob_name,"converted_etag":src_etag})
        up=upload_file_to_blob(tgt_blob,pdf_path,meta,content_type="application/pdf",overwrite=True)
        if up.get("action")=="uploaded":
            log("INFO","doc_uploaded","Uploaded pdf",target=tgt_blob,result=up)
            if OVERWRITE_OTHER_TO_PDF:
                try:CONTAINER_CLIENT.delete_blob(blob_name);log("INFO","doc_deleted_old","Deleted original doc",name=blob_name)
                except Exception as e:log("WARN","delete_old_failed","failed delete original doc",name=blob_name,error=str(e))
        else:
            log("ERROR","doc_upload_failed","upload failed",target=tgt_blob,result=up)
def process_sheet(blob_name:str)->None:
    name=Path(blob_name).name;ext=Path(name).suffix.lstrip(".").lower()
    if ext not in ("xls","xlsx","ods","xlsm","xlsb"):return
    with tempfile.TemporaryDirectory(prefix="preconv-",dir=TMP_ROOT) as tmpdir:
        src_local=str(Path(tmpdir)/"src"/name);outdir=str(Path(tmpdir)/"out");Path(src_local).parent.mkdir(parents=True,exist_ok=True);Path(outdir).mkdir(parents=True,exist_ok=True)
        if not download_blob_to_file(blob_name,src_local):
            log("WARN","sheet_download_failed","download failed",blob=blob_name);return
        ok,err=run_soffice_convert(src_local,outdir,"csv")
        if not ok:
            log("WARN","sheet_convert_failed","soffice csv conversion failed",blob=blob_name,error=err);return
        created=list(Path(outdir).glob("*.csv"))
        if not created:
            log("WARN","sheet_no_csv","no csvs produced",blob=blob_name);return
        for f in created:
            dest_blob=f"{STORAGE_RAW_PREFIX}csvs/{Path(name).stem}/{f.name}"
            md5,sha=compute_hashes(src_local);src_props=get_blob_props(blob_name);src_etag=src_props.get("etag","")
            meta=prepare_metadata({"sha256":sha,"original_name":name,"original_ext":ext,"converted_from":blob_name,"converted_etag":src_etag})
            up=upload_file_to_blob(dest_blob,str(f),meta,content_type="text/csv",overwrite=True)
            if up.get("action")=="uploaded":log("INFO","sheet_uploaded","Uploaded csv",target=dest_blob,result=up)
        if OVERWRITE_SPREADSHEETS_WITH_CSV:
            try:CONTAINER_CLIENT.delete_blob(blob_name);log("INFO","sheet_deleted_old","Deleted original sheet",name=blob_name)
            except Exception as e:log("WARN","delete_old_failed","failed deleting original sheet",name=blob_name,error=str(e))
def process_audio(blob_name:str)->None:
    name=Path(blob_name).name;ext=Path(name).suffix.lstrip(".").lower()
    if ext not in ("mp3","m4a","aac","wav","flac","ogg","opus","webm","amr","wma","aiff","aif"):return
    with tempfile.TemporaryDirectory(prefix="preconv-",dir=TMP_ROOT) as tmpdir:
        src_local=str(Path(tmpdir)/"src"/name);out_local=str(Path(tmpdir)/"out"/(Path(name).stem+".wav"));Path(src_local).parent.mkdir(parents=True,exist_ok=True);Path(out_local).parent.mkdir(parents=True,exist_ok=True)
        if not download_blob_to_file(blob_name,src_local):
            log("WARN","audio_download_failed","download failed",blob=blob_name);return
        ok,err=run_ffmpeg_convert(src_local,out_local)
        if not ok:
            log("ERROR","audio_convert_failed","ffmpeg failed",blob=blob_name,error=err);return
        tgt_blob=f"{STORAGE_RAW_PREFIX}audio/{Path(name).stem}.wav"
        md5,sha=compute_hashes(src_local);src_props=get_blob_props(blob_name);src_etag=src_props.get("etag","")
        meta=prepare_metadata({"sha256":sha,"original_name":name,"original_ext":ext,"converted_from":blob_name,"converted_etag":src_etag})
        up=upload_file_to_blob(tgt_blob,out_local,meta,content_type="audio/wav",overwrite=True)
        if up.get("action")=="uploaded":
            log("INFO","audio_uploaded","Uploaded audio",target=tgt_blob,result=up)
            if OVERWRITE_ALL_AUDIO_FILES and tgt_blob!=blob_name:
                try:CONTAINER_CLIENT.delete_blob(blob_name);log("INFO","audio_deleted_old","Deleted original audio",name=blob_name)
                except Exception as e:log("WARN","delete_old_failed","failed deleting original audio",name=blob_name,error=str(e))
        else:
            log("ERROR","audio_upload_failed","upload failed",target=tgt_blob,result=up)
def group_all(prefix:str)->None:
    log("INFO","group_start","Grouping start",prefix=prefix)
    blobs=list(list_blobs(prefix))
    for full in blobs:
        try:
            if full.endswith("/") or full.endswith(".manifest.json"):continue
            if not full.startswith(prefix):continue
            rel=full[len(prefix):]
            if not rel or rel.strip()=="":continue
            parts=rel.split("/",1);first=parts[0]
            basename=Path(rel).name
            correct_subdir=ext_to_subdir(basename)
            if first in ("audio","images","pdfs","docs","ppts","sheets","csvs","mds","txts","htmls","chunked","quarantine","others"):
                if first==correct_subdir:continue
                target=f"{prefix}{correct_subdir}/{basename}"
                ok,final=safe_move_blob(full,target)
                if ok:log("INFO","group_moved","moved to correct dir",src=full,dst=final)
                else:log("WARN","group_move_failed","failed to move to correct dir",src=full,dst=target)
                continue
            target=f"{prefix}{correct_subdir}/{basename}"
            ok,final=safe_move_blob(full,target)
            if ok:log("INFO","group_moved","grouped file",src=full,dst=final)
            else:log("WARN","group_move_failed","failed to group file",src=full,dst=target)
        except Exception as e:
            log("ERROR","group_iteration_failed","error grouping blob",blob=full,error=str(e))
    log("INFO","group_done","Grouping completed",prefix=prefix)
def worker_wrapper(blob:str)->None:
    try:
        if blob.endswith("/") or blob.endswith(".manifest.json"):return
        rel=blob[len(STORAGE_RAW_PREFIX):] if blob.startswith(STORAGE_RAW_PREFIX) else blob
        first=rel.split("/",1)[0] if "/" in rel else rel
        if first in ("csvs","pdfs","chunked","quarantine"):return
        ext=Path(blob).suffix.lstrip(".").lower()
        if ext in ("doc","docx"):process_doc(blob)
        elif ext in ("xls","xlsx","ods","xlsm","xlsb"):process_sheet(blob)
        elif ext in ("mp3","m4a","wav","flac","ogg","opus","webm","amr","wma","aiff","aif"):process_audio(blob)
    except Exception as e:
        log("ERROR","processing_failed",f"processing {blob} failed",error=str(e))
def main()->None:
    soft=ensure_nofile_limit(DESIRED_NOFILE)
    log("INFO","nofile.limits","Current RLIMIT_NOFILE",soft=soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)[1])
    try:
        group_all(STORAGE_RAW_PREFIX)
    except Exception as e:
        log("ERROR","grouping_failed","grouping phase failed",error=str(e));raise SystemExit(2)
    try:
        blobs=list(list_blobs(STORAGE_RAW_PREFIX))
    except Exception as e:
        log("ERROR","list_after_group_failed","failed listing blobs after grouping",error=str(e));raise SystemExit(2)
    workers=compute_max_workers(soft,FDS_PER_CONVERT,SYSTEM_RESERVED_FDS,ENV_MAX_PARALLEL)
    log("INFO","concurrency","Computed worker concurrency",workers=workers,fds_per_worker=FDS_PER_CONVERT,reserved=SYSTEM_RESERVED_FDS,env_max=ENV_MAX_PARALLEL)
    from concurrent.futures import ThreadPoolExecutor,as_completed
    total=len(blobs);processed=0;skipped=0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures={ex.submit(worker_wrapper,blob):blob for blob in blobs}
        for fut in as_completed(futures):
            blob=futures.get(fut)
            try:
                fut.result()
                processed+=1
            except Exception as e:
                log("WARN","fut.failed","future failed for blob",blob=blob,error=str(e))
    try:
        BLOB_SERVICE.close()
    except Exception:
        pass
    log("INFO","finished","pre_conversions completed",processed=processed,skipped=skipped)
if __name__=="__main__":
    main()
