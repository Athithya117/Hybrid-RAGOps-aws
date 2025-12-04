#!/usr/bin/env python3
from __future__ import annotations
import os,sys,time,json
from typing import Optional
env=os.environ
BACKUP_S3_BUCKET=env.get("BACKUP_S3_BUCKET","")
BACKUP_S3_PATH=env.get("BACKUP_S3_PATH","backups")
AWS_REGION=env.get("AWS_REGION","")
AWS_ACCESS_KEY_ID=env.get("AWS_ACCESS_KEY_ID","")
AWS_SECRET_ACCESS_KEY=env.get("AWS_SECRET_ACCESS_KEY","")
WEAVIATE_URL=env.get("WEAVIATE_URL","http://127.0.0.1:8080")
WEAVIATE_API_KEY=env.get("WEAVIATE_API_KEY","")
BACKUP_WAIT_TIMEOUT=int(env.get("BACKUP_WAIT_TIMEOUT","1800"))
POLL_INTERVAL=float(env.get("BACKUP_POLL_INTERVAL","5.0"))
def err(msg:str,code:int=1):
    print(msg,file=sys.stderr);sys.exit(code)
def load_manifest_from_s3(bucket:str,path:str,region:Optional[str]=None):
    try:
        import boto3
    except Exception as e:
        err(f"boto3 required: {e}")
    s3=boto3.client("s3",region_name=region or None)
    key=path.rstrip("/")+"/latest/latest_backup.json"
    try:
        obj=s3.get_object(Bucket=bucket,Key=key)
        data=obj["Body"].read()
        return json.loads(data)
    except Exception as e:
        print(f"No manifest found at s3://{bucket}/{key} ({e})")
        return None
def do_restore(manifest:dict):
    try:
        import weaviate
    except Exception as e:
        err(f"weaviate-client required: {e}")
    backup_id=manifest.get("backup_id")
    if not backup_id:
        err("manifest missing backup_id",2)
    headers={}
    if WEAVIATE_API_KEY:
        headers["X-Api-Key"]=WEAVIATE_API_KEY
    client=weaviate.Client(url=WEAVIATE_URL,additional_headers=headers)
    if not client.is_ready():
        err("weaviate not ready",3)
    try:
        status=client.backup.get_restore_status(backup_id=backup_id,backend="s3")
        st=getattr(getattr(status,"status",None),"value",None)
    except Exception:
        st=None
    if st and st not in ("SUCCESS","FAILED","ERROR"):
        print(f"restore already in progress (status={st}) - exiting")
        return 0
    try:
        client.backup.restore(backup_id=backup_id,backend="s3",wait_for_completion=False)
    except Exception as e:
        err(f"restore API call failed: {e}",4)
    start=time.time()
    while time.time()-start<BACKUP_WAIT_TIMEOUT:
        try:
            status=client.backup.get_restore_status(backup_id=backup_id,backend="s3")
            st=getattr(getattr(status,"status",None),"value",None)
        except Exception:
            st=None
        print("restore status:",st)
        if st=="SUCCESS":
            print("restore completed successfully")
            return 0
        if st in ("FAILED","ERROR"):
            print("restore failed:",status)
            return 3
        time.sleep(POLL_INTERVAL)
    err("restore timed out",5)
def main():
    if not BACKUP_S3_BUCKET or not BACKUP_S3_PATH:
        err("BACKUP_S3_BUCKET and BACKUP_S3_PATH must be set",2)
    manifest=load_manifest_from_s3(BACKUP_S3_BUCKET,BACKUP_S3_PATH,region=AWS_REGION or None)
    if not manifest:
        print("no manifest -> nothing to restore");return
    return_code=do_restore(manifest)
    sys.exit(return_code if isinstance(return_code,int) else 0)
if __name__=="__main__":
    main()
