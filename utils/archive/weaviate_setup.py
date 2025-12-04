#!/usr/bin/env python3
from __future__ import annotations
import os,subprocess,sys,time,json,urllib.request,urllib.error,shutil,signal
from pathlib import Path
from typing import Any,Dict,Optional,Tuple,List
import argparse
parser=argparse.ArgumentParser(prog="weaviate_setup")
parser.add_argument("--delete",action="store_true",help="delete weaviate resources managed by this script")
parser.add_argument("--delete-namespace",action="store_true",help="also delete namespace (requires --delete)")
args=parser.parse_args()
ENV=os.environ
WEAVIATE_REPLICAS=int(ENV.get("WEAVIATE_REPLICAS","1"))
WEAVIATE_PDB_REPLICAS=int(ENV.get("WEAVIATE_PDB_REPLICAS","1"))
WEAVIATE_IMAGE=ENV.get("WEAVIATE_IMAGE","cr.weaviate.io/semitechnologies/weaviate:1.34.0")
WEAVIATE_RELEASE=ENV.get("WEAVIATE_RELEASE","my-weaviate")
WEAVIATE_NAMESPACE=ENV.get("WEAVIATE_NAMESPACE","weaviate")
HTTP_PORT=int(ENV.get("HTTP_PORT","8080"))
LOCAL_HTTP_PORT=int(ENV.get("LOCAL_HTTP_PORT",str(HTTP_PORT)))
PERSISTENCE_SIZE=ENV.get("PERSISTENCE_SIZE","10Gi")
PERSISTENCE_STORAGE_CLASS=ENV.get("PERSISTENCE_STORAGE_CLASS","")
WEAVIATE_ROOT_KEY=ENV.get("WEAVIATE_ROOT_KEY","")
BACKUP_S3_BUCKET=ENV.get("BACKUP_S3_BUCKET","")
AWS_ACCESS_KEY_ID=ENV.get("AWS_ACCESS_KEY_ID","")
AWS_SECRET_ACCESS_KEY=ENV.get("AWS_SECRET_ACCESS_KEY","")
BACKUP_S3_PATH=ENV.get("BACKUP_S3_PATH","backups")
BACKUP_S3_ENDPOINT_RAW=ENV.get("BACKUP_S3_ENDPOINT","")
BACKUP_S3_USE_SSL=ENV.get("BACKUP_S3_USE_SSL","true").lower() in ("1","true","yes")
AWS_REGION=ENV.get("AWS_REGION","ap-south-1")
MANIFEST_DIR=Path(__file__).resolve().parent/"manifests"/"local"
MANIFEST_DIR.mkdir(parents=True,exist_ok=True)
def run(cmd:List[str],input_data:Optional[str]=None,capture:bool=False,check:bool=True)->Optional[str]:
    try:
        p=subprocess.run(cmd,input=input_data,text=True,capture_output=capture,check=check)
        return p.stdout if capture else None
    except subprocess.CalledProcessError as e:
        if capture:
            return (e.stdout or "")+(e.stderr or "")
        raise
def check_bin(name:str)->None:
    if shutil.which(name) is None:
        print(f"missing binary: {name}",file=sys.stderr); sys.exit(1)
check_bin("kubectl")
def kubectl_create_namespace(ns:str)->None:
    run(["kubectl","create","ns",ns],check=False)
def kubectl_apply_json(obj:Dict[str,Any],delete_first:bool=False)->None:
    kind=obj.get("kind","").lower(); name=obj.get("metadata",{}).get("name"); namespace=obj.get("metadata",{}).get("namespace")
    if delete_first and kind and name and namespace:
        run(["kubectl","delete",kind,name,"-n",namespace,"--ignore-not-found"])
    j=json.dumps(obj)
    run(["kubectl","apply","-f","-"],input_data=j)
def create_or_update_secret(name:str,literals:Dict[str,str],namespace:str)->None:
    parts=["kubectl","create","secret","generic",name]
    for k,v in literals.items():
        parts.append(f"--from-literal={k}={v}")
    parts+=["-n",namespace,"--dry-run=client","-o","yaml"]
    y=run(parts,capture=True)
    run(["kubectl","apply","-f","-"],input_data=y)
def wait_for_pods_ready(label_sel:str,namespace:str,timeout:int=900)->bool:
    try:
        run(["kubectl","wait","--for=condition=Ready","pod","-l",label_sel,"-n",namespace,f"--timeout={timeout}s"])
        return True
    except Exception:
        return False
def gather_diagnostics(namespace:str)->None:
    pods=run(["kubectl","get","pods","-n",namespace,"-o","name"],capture=True) or ""
    for pn in [p.strip() for p in pods.splitlines() if p.strip()]:
        d=run(["kubectl","describe",pn,"-n",namespace],capture=True) or ""
        print(d)
        lg=run(["kubectl","logs",pn.replace("pod/",""),"-n",namespace],capture=True) or ""
        print(lg)
def compute_endpoint_host(raw:Optional[str],region:Optional[str])->str:
    if raw:
        r=raw.strip()
        if r.startswith("http://"): r=r[7:]
        elif r.startswith("https://"): r=r[8:]
        return r.rstrip("/")
    if region:
        return f"s3.{region}.amazonaws.com"
    return ""
def delete_all()->None:
    objs=[("configmap","weaviate-config"),("secret","weaviate-root-key"),("secret","weaviate-backup-s3"),("service","weaviate-headless"),("service","weaviate"),("statefulset","weaviate"),("poddisruptionbudget","weaviate-pdb")]
    for kind,name in objs:
        run(["kubectl","delete",kind,name,"-n",WEAVIATE_NAMESPACE,"--ignore-not-found"],check=False)
    if args.delete_namespace:
        run(["kubectl","delete","ns",WEAVIATE_NAMESPACE,"--ignore-not-found"],check=False)
    sys.exit(0)
if args.delete:
    delete_all()
kubectl_create_namespace(WEAVIATE_NAMESPACE)
conf_yaml="authentication:\n  anonymous_access:\n    enabled: true\nauthorization:\n  rbac:\n    enabled: false\nquery_defaults:\n  limit: 100\ndebug: false\n"
configmap={"apiVersion":"v1","kind":"ConfigMap","metadata":{"name":"weaviate-config","namespace":WEAVIATE_NAMESPACE,"labels":{"app":"weaviate"}},"data":{"conf.yaml":conf_yaml}}
kubectl_apply_json(configmap)
create_or_update_secret("weaviate-root-key",{"WEAVIATE_ROOT_KEY":WEAVIATE_ROOT_KEY},WEAVIATE_NAMESPACE)
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    create_or_update_secret("weaviate-backup-s3",{"AWS_ACCESS_KEY_ID":AWS_ACCESS_KEY_ID,"AWS_SECRET_ACCESS_KEY":AWS_SECRET_ACCESS_KEY,"AWS_REGION":AWS_REGION},WEAVIATE_NAMESPACE)
headless_svc={"apiVersion":"v1","kind":"Service","metadata":{"name":"weaviate-headless","namespace":WEAVIATE_NAMESPACE,"labels":{"app":"weaviate"}},"spec":{"type":"ClusterIP","clusterIP":None,"selector":{"app":"weaviate"},"ports":[{"name":"gossip","port":7000,"protocol":"TCP","targetPort":7000},{"name":"data","port":7001,"protocol":"TCP","targetPort":7001},{"name":"metrics","port":2112,"protocol":"TCP","targetPort":2112}],"publishNotReadyAddresses":True}}
kubectl_apply_json(headless_svc,delete_first=True)
main_svc={"apiVersion":"v1","kind":"Service","metadata":{"name":"weaviate","namespace":WEAVIATE_NAMESPACE,"labels":{"app":"weaviate"},"annotations":{"prometheus.io/scrape":"true","prometheus.io/port":"2112","prometheus.io/path":"/metrics"}},"spec":{"type":"ClusterIP","selector":{"app":"weaviate"},"ports":[{"name":"http","port":80,"protocol":"TCP","targetPort":HTTP_PORT},{"name":"grpc","port":50051,"protocol":"TCP","targetPort":50051},{"name":"metrics","port":2112,"protocol":"TCP","targetPort":2112}]}}
kubectl_apply_json(main_svc)
endpoint_host=compute_endpoint_host(BACKUP_S3_ENDPOINT_RAW,AWS_REGION)
env_vars=[{"name":"ENABLE_MODULES","value":ENV.get("ENABLE_MODULES","backup-s3")},{"name":"BACKUP_S3_BUCKET","value":BACKUP_S3_BUCKET},{"name":"BACKUP_S3_PATH","value":BACKUP_S3_PATH},{"name":"BACKUP_S3_ENDPOINT","value":endpoint_host},{"name":"BACKUP_S3_USE_SSL","value":"true" if BACKUP_S3_USE_SSL else "false"},{"name":"AWS_REGION","value":AWS_REGION},{"name":"WEAVIATE_ROOT_KEY","valueFrom":{"secretKeyRef":{"name":"weaviate-root-key","key":"WEAVIATE_ROOT_KEY"}}}]
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    env_vars.append({"name":"AWS_ACCESS_KEY_ID","valueFrom":{"secretKeyRef":{"name":"weaviate-backup-s3","key":"AWS_ACCESS_KEY_ID"}}})
    env_vars.append({"name":"AWS_SECRET_ACCESS_KEY","valueFrom":{"secretKeyRef":{"name":"weaviate-backup-s3","key":"AWS_SECRET_ACCESS_KEY"}}})
data_host_path=ENV.get("WEAVIATE_NVME_MOUNT_PATH","/var/lib/weaviate")
sts={"apiVersion":"apps/v1","kind":"StatefulSet","metadata":{"name":"weaviate","namespace":WEAVIATE_NAMESPACE,"labels":{"app":"weaviate"}},"spec":{"serviceName":"weaviate-headless","replicas":WEAVIATE_REPLICAS,"selector":{"matchLabels":{"app":"weaviate"}},"template":{"metadata":{"labels":{"app":"weaviate"}},"spec":{"containers":[{"name":"weaviate","image":WEAVIATE_IMAGE,"imagePullPolicy":"IfNotPresent","command":["/bin/weaviate"],"args":["--host","0.0.0.0","--port",str(HTTP_PORT),"--scheme","http","--config-file","/weaviate-config/conf.yaml","--read-timeout=60s","--write-timeout=60s"],"ports":[{"containerPort":HTTP_PORT,"name":"http"},{"containerPort":50051,"name":"grpc"},{"containerPort":2112,"name":"metrics"}],"env":env_vars,"volumeMounts":[{"name":"weaviate-config","mountPath":"/weaviate-config","readOnly":True},{"name":"weaviate-data","mountPath":"/var/lib/weaviate"}],"livenessProbe":{"httpGet":{"path":"/v1/.well-known/live","port":HTTP_PORT},"initialDelaySeconds":30,"periodSeconds":10,"timeoutSeconds":3,"failureThreshold":3},"readinessProbe":{"httpGet":{"path":"/v1/.well-known/ready","port":HTTP_PORT},"initialDelaySeconds":5,"periodSeconds":10,"timeoutSeconds":3,"failureThreshold":3}}],"volumes":[{"name":"weaviate-config","configMap":{"name":"weaviate-config"}}]}},"volumeClaimTemplates":[]}}
if WEAVIATE_REPLICAS<=1:
    sts["spec"]["template"]["spec"]["volumes"].append({"name":"weaviate-data","hostPath":{"path":data_host_path,"type":"DirectoryOrCreate"}})
else:
    pvc={"metadata":{"name":"weaviate-data"},"spec":{"accessModes":["ReadWriteOnce"],"resources":{"requests":{"storage":PERSISTENCE_SIZE}}}}
    if PERSISTENCE_STORAGE_CLASS:
        pvc["spec"]["storageClassName"]=PERSISTENCE_STORAGE_CLASS
    sts["spec"]["volumeClaimTemplates"].append(pvc)
kubectl_apply_json(sts,delete_first=True)
pdb={"apiVersion":"policy/v1","kind":"PodDisruptionBudget","metadata":{"name":"weaviate-pdb","namespace":WEAVIATE_NAMESPACE},"spec":{"minAvailable":max(1,WEAVIATE_PDB_REPLICAS),"selector":{"matchLabels":{"app":"weaviate"}}}}
kubectl_apply_json(pdb)
print("Applied manifests. Waiting for pods to become Ready (timeout 900s)...")
if not wait_for_pods_ready("app=weaviate",WEAVIATE_NAMESPACE,900):
    print("Pod readiness failed. Diagnostics:",file=sys.stderr)
    gather_diagnostics(WEAVIATE_NAMESPACE)
    sys.exit(2)
pod_name=(run(["kubectl","get","pod","-n",WEAVIATE_NAMESPACE,"-l","app=weaviate","-o","jsonpath={.items[0].metadata.name}"],capture=True) or "").strip()
if not pod_name:
    print("No pod found",file=sys.stderr); sys.exit(3)
def http_request(path:str,headers:Optional[Dict[str,str]]=None,timeout:int=5)->Tuple[int,str]:
    url=f"http://127.0.0.1:{LOCAL_HTTP_PORT}{path}"
    req=urllib.request.Request(url,headers=headers or {},method="GET")
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:
            return r.getcode(),r.read().decode("utf-8","ignore")
    except urllib.error.HTTPError as e:
        return e.code,e.read().decode("utf-8","ignore")
    except Exception as e:
        return 0,str(e)
pf=subprocess.Popen(["kubectl","port-forward",f"pod/{pod_name}",f"{LOCAL_HTTP_PORT}:{HTTP_PORT}","-n",WEAVIATE_NAMESPACE,"--address=127.0.0.1"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
time.sleep(1.0)
def wait_for_health(retries:int=90,delay:float=2.0)->None:
    for _ in range(retries):
        c,_=http_request("/v1/.well-known/ready")
        if c==200:
            return
        time.sleep(delay)
    raise RuntimeError("Health check failed")
try:
    wait_for_health(90,2.0)
except Exception:
    print("Local readiness failed",file=sys.stderr)
    pf.terminate()
    gather_diagnostics(WEAVIATE_NAMESPACE)
    sys.exit(4)
pf.terminate()
print("Weaviate deployed and ready:",pod_name)
print(f"Local API: http://127.0.0.1:{LOCAL_HTTP_PORT}")
