#!/usr/bin/env python3
from __future__ import annotations
import os,sys,json,time,tempfile,subprocess,hashlib,shutil,re
from pathlib import Path
try:
    import yaml
except Exception:
    print("ERROR: PyYAML required. Install with: pip install pyyaml", file=sys.stderr); sys.exit(2)

def LOG(*p): print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), *p, flush=True)
def DBG(*p):
    if os.environ.get("VERBOSE","0")!="0": LOG(*p)

ROOT=Path(__file__).resolve().parent.parent.parent
MANIFESTS_DIR=Path(os.environ.get("MANIFESTS_DIR","infra/manifests/qdrant"))
VALUES_FILE=MANIFESTS_DIR/"values.yaml"
SERVICE_PATCH_FILE=MANIFESTS_DIR/"service-patch.yaml"
NAMESPACE_FILE=MANIFESTS_DIR/"namespace.yaml"
LAST_SUMMARY=MANIFESTS_DIR/"last_deploy_summary.json"

QDRANT_RELEASE=os.environ.get("QDRANT_RELEASE","qdrant")
QDRANT_NAMESPACE=os.environ.get("QDRANT_NAMESPACE","qdrant")
QDRANT_IMAGE=os.environ.get("QDRANT_IMAGE","qdrant/qdrant:v1.16.0")
CHART_VERSION=os.environ.get("CHART_VERSION","1.16.0")
QDRANT_REPLICAS=int(os.environ.get("QDRANT_REPLICAS","1"))
QDRANT_METRICS_PORT=int(os.environ.get("QDRANT_METRICS_PORT","6333"))
QDRANT_METRICS_PORT_NAME=os.environ.get("QDRANT_METRICS_PORT_NAME","metrics")
QDRANT_ONDISK=os.environ.get("QDRANT_ONDISK","false").lower() in ("1","true","yes","y")
QDRANT_PERSISTENCE_ENABLED=os.environ.get("QDRANT_PERSISTENCE_ENABLED","false").lower() in ("1","true","yes","y")
QDRANT_PERSISTENCE_SIZE=os.environ.get("QDRANT_PERSISTENCE_SIZE","20Gi")
QDRANT_PERSISTENCE_STORAGE_CLASS=os.environ.get("QDRANT_PERSISTENCE_STORAGE_CLASS","")
SECRET_SERVICE_NAME=os.environ.get("SECRET_SERVICE_NAME","qdrant-service-creds")
QDRANT__SERVICE__API_KEY=os.environ.get("QDRANT__SERVICE__API_KEY", os.environ.get("QDRANT_API_KEY",""))
SERVICE_VALIDATION_WAIT=int(os.environ.get("SERVICE_VALIDATION_WAIT","120"))
VENDOR_CHART_DIR=os.environ.get("VENDOR_CHART_DIR","infra/archive/qdrant-helm-chart/qdrant")
HELM_FALLBACK_REPO=os.environ.get("HELM_FALLBACK_REPO","https://qdrant.to/helm")
TMP_FILES=[]

def atomic_write(path:Path,content:bytes):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name,dir=str(path.parent)); os.close(fd); TMP_FILES.append(tmp)
    with open(tmp,"wb") as f: f.write(content)
    os.replace(tmp,str(path))
    try: TMP_FILES.remove(tmp)
    except Exception: pass

def run(cmd,check=True,capture=False):
    DBG("run:", " ".join(cmd))
    try:
        if capture: return subprocess.check_output(cmd,stderr=subprocess.STDOUT,text=True)
        subprocess.check_call(cmd)
        return ""
    except subprocess.CalledProcessError as e:
        DBG("cmd failed:", e)
        if capture: return e.output
        if check: raise
        return ""
    except FileNotFoundError:
        if capture: return ""
        if check: raise
        return ""

def run_full(cmd):
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    return p.returncode,(p.stdout or "")

def render_values_dict() -> dict:
    cpu_req=os.environ.get("QDRANT_CPU_REQUEST") or os.environ.get("QDRANT_CPU") or "1"
    cpu_lim=os.environ.get("QDRANT_CPU_LIMIT") or os.environ.get("QDRANT_CPU") or cpu_req
    mem_req=os.environ.get("QDRANT_MEMORY_REQUEST") or os.environ.get("QDRANT_MEMORY") or "2Gi"
    mem_lim=os.environ.get("QDRANT_MEMORY_LIMIT") or os.environ.get("QDRANT_MEMORY") or mem_req
    repo_tag=QDRANT_IMAGE
    if ":" in repo_tag: repo,tag=repo_tag.split(":",1)
    else: repo,tag=repo_tag,"latest"
    peers=[f"http://{QDRANT_RELEASE}-{i}.{QDRANT_RELEASE}-headless:6335" for i in range(QDRANT_REPLICAS)]
    vals={
        "replicaCount":QDRANT_REPLICAS,
        "image":{"repository":repo,"tag":tag,"pullPolicy":"IfNotPresent"},
        "service":{"type":"ClusterIP","labels":{"app.kubernetes.io/name":QDRANT_RELEASE,"app.kubernetes.io/component":"qdrant"}},
        "podAnnotations":{"monitoring.io/scrape":"true","monitoring.io/port":str(QDRANT_METRICS_PORT),"monitoring.io/path":"/metrics"},
        "ports":[{"name":QDRANT_METRICS_PORT_NAME,"containerPort":QDRANT_METRICS_PORT,"protocol":"TCP"}],
        "p2p":{"port":6335},
        "cluster":{"enabled":True,"peers":peers},
        "snapshots":{"enabled":False,"s3":{"bucket":"","endpoint":"","region":"","prefix":""}},
        "extraEnv":[],"resources":{"requests":{"cpu":cpu_req,"memory":mem_req},"limits":{"cpu":cpu_lim,"memory":mem_lim}},
        "tolerations":[],"persistence":{"enabled":bool(QDRANT_PERSISTENCE_ENABLED),"size":QDRANT_PERSISTENCE_SIZE,"storageClass":QDRANT_PERSISTENCE_STORAGE_CLASS or ""},
        "config":{"on_disk_payload":QDRANT_ONDISK,"log_level":os.environ.get("QDRANT_LOG_LEVEL","INFO"),
                  "storage":{"storage_path":os.environ.get("QDRANT__STORAGE__STORAGE_PATH","/qdrant/storage"),
                            "snapshots_path":os.environ.get("QDRANT__STORAGE__SNAPSHOTS_PATH","/qdrant/snapshots")}},
        "hnsw_defaults":{"on_disk":QDRANT_ONDISK}
    }
    extra_env=[]
    if QDRANT__SERVICE__API_KEY and SECRET_SERVICE_NAME:
        extra_env.append({"name":"QDRANT__SERVICE__API_KEY","valueFrom":{"secretKeyRef":{"name":SECRET_SERVICE_NAME,"key":"QDRANT__SERVICE__API_KEY"}}})
    if extra_env: vals["extraEnv"]=extra_env
    checksum=hashlib.sha256(yaml.safe_dump(vals,sort_keys=False).encode("utf-8")).hexdigest()
    pod_ann=vals.get("podAnnotations",{}) or {}; pod_ann["qdrant/config-checksum"]=checksum; vals["podAnnotations"]=pod_ann
    return vals

def render_service_patch():
    svc={
        "apiVersion":"v1","kind":"Service",
        "metadata":{"name":QDRANT_RELEASE,"namespace":QDRANT_NAMESPACE,"labels":{"app.kubernetes.io/name":QDRANT_RELEASE,"app.kubernetes.io/component":"qdrant"}},
        "spec":{"selector":{"app.kubernetes.io/name":QDRANT_RELEASE},"ports":[{"name":QDRANT_METRICS_PORT_NAME,"port":QDRANT_METRICS_PORT,"targetPort":QDRANT_METRICS_PORT,"protocol":"TCP"}],"type":"ClusterIP"}
    }
    atomic_write(SERVICE_PATCH_FILE,yaml.safe_dump(svc,sort_keys=False).encode("utf-8")); LOG("Rendered",str(SERVICE_PATCH_FILE))

def render_values_file():
    vals=render_values_dict(); atomic_write(VALUES_FILE,yaml.safe_dump(vals,sort_keys=False).encode("utf-8")); LOG("Rendered",str(VALUES_FILE))

def ensure_namespace():
    try: run(["kubectl","create","namespace",QDRANT_NAMESPACE], check=False)
    except Exception: pass

def create_service_secret():
    if not QDRANT__SERVICE__API_KEY or not SECRET_SERVICE_NAME: return False
    # create secret using client-side dry-run then apply (safe for idempotence)
    run(["bash","-c",f"kubectl -n {QDRANT_NAMESPACE} create secret generic {SECRET_SERVICE_NAME} --from-literal=QDRANT__SERVICE__API_KEY='{QDRANT__SERVICE__API_KEY}' --dry-run=client -o yaml | kubectl apply -f -"], check=True)
    LOG("created/updated secret",SECRET_SERVICE_NAME); return True

def helm_install():
    ensure_namespace()
    if SECRET_SERVICE_NAME and QDRANT__SERVICE__API_KEY: create_service_secret()
    render_values_file(); render_service_patch()
    # vendor chart fallback path if present
    if Path(VENDOR_CHART_DIR).is_dir() and (Path(VENDOR_CHART_DIR)/"Chart.yaml").exists():
        rc,out=run_full(["helm","upgrade","--install",QDRANT_RELEASE,VENDOR_CHART_DIR,"--namespace",QDRANT_NAMESPACE,"--create-namespace","-f",str(VALUES_FILE),"--wait","--timeout","10m"])
        if rc==0: return True
        LOG("helm vendor install failed"); DBG(out); return False
    # Official repo add/update and install with pinned chart version
    run(["helm","repo","add","--force-update","qdrant","https://qdrant.github.io/qdrant-helm"], check=False)
    run(["helm","repo","update"], check=False)
    rc,out=run_full(["helm","upgrade","--install",QDRANT_RELEASE,"qdrant/qdrant","--version",CHART_VERSION,"--namespace",QDRANT_NAMESPACE,"-f",str(VALUES_FILE),"--wait","--timeout","10m"])
    if rc==0: return True
    # fallback repo if needed
    rc2,out2=run_full(["helm","upgrade","--install",QDRANT_RELEASE,"qdrant/qdrant","--version",CHART_VERSION,"--repo",HELM_FALLBACK_REPO,"--namespace",QDRANT_NAMESPACE,"-f",str(VALUES_FILE),"--wait","--timeout","10m"])
    if rc2==0: return True
    LOG("helm install failed; see helm output"); DBG(out); DBG(out2); return False

def validate_post_install():
    selector=f"app.kubernetes.io/name={QDRANT_RELEASE}"
    try: run(["kubectl","-n",QDRANT_NAMESPACE,"wait","--for=condition=Ready","pod","-l",selector,f"--timeout={SERVICE_VALIDATION_WAIT}s"], check=False)
    except Exception: pass
    end=time.time()+SERVICE_VALIDATION_WAIT
    pods=[]
    while time.time()<end:
        out=run(["kubectl","-n",QDRANT_NAMESPACE,"get","pods","-l",selector,"-o","json"], check=False, capture=True)
        try:
            pj=json.loads(out); items=pj.get("items",[])
            if items: pods=items; break
        except Exception: pass
        time.sleep(2)
    if not pods: LOG("no pods found after wait"); return False
    LOG("pods detected:", len(pods)); return True

def delete_qdrant():
    run(["kubectl","delete","ns",QDRANT_NAMESPACE,"--ignore-not-found"], check=False)
    if MANIFESTS_DIR.exists():
        try: shutil.rmtree(MANIFESTS_DIR)
        except Exception: pass
    LOG("deleted qdrant namespace and manifests (best-effort)")

def main():
    if len(sys.argv)==1: print("usage: qdrant_cluster.py --rollout|--delete"); sys.exit(1)
    cmd=None
    for a in sys.argv[1:]:
        if a=="--rollout": cmd="rollout"
        elif a=="--delete": cmd="delete"
        else: print("usage: qdrant_cluster.py --rollout|--delete"); sys.exit(1)
    if cmd=="rollout":
        LOG("rollout started")
        ok=helm_install()
        if not ok: LOG("helm install/upgrade failed"); sys.exit(2)
        LOG("helm install/upgrade succeeded")
        if not validate_post_install(): LOG("post-install validation issues (non-fatal)")
        summary={"release":QDRANT_RELEASE,"namespace":QDRANT_NAMESPACE,"replicas":QDRANT_REPLICAS,"chart_version":CHART_VERSION,"image":QDRANT_IMAGE,"generated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),"use_flux":False,"secrets_created":bool(QDRANT__SERVICE__API_KEY)}
        atomic_write(LAST_SUMMARY,json.dumps(summary,indent=2).encode("utf-8")); LOG("Wrote summary",str(LAST_SUMMARY))
    elif cmd=="delete":
        LOG("delete started")
        delete_qdrant()

if __name__=="__main__":
    main()
