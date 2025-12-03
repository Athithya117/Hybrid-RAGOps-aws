#!/usr/bin/env python3
import os,sys,json,hashlib,argparse,subprocess,datetime
from pathlib import Path
from jinja2 import Environment,BaseLoader
ROOT=Path(__file__).resolve().parents[2]
MANIFESTS_ROOT=ROOT/"infra"/"manifests"
OUT_DIR=MANIFESTS_ROOT/"argocd"
OUT_FILE=OUT_DIR/"applicationset.yaml"
MARKER=OUT_DIR/"applicationset.applied.json"
ROOT_MARKER=MANIFESTS_ROOT/"argocd-root"/"root_app.applied.json"
TEMPLATE="""apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: rag-applications
  namespace: argocd
spec:
  generators:
    - list:
        elements:
{% for e in elements %}
          - name: {{ e.name }}
            path: {{ e.path | tojson }}
            destNamespace: {{ e.destNamespace }}
{% endfor %}
  template:
    metadata:
      name: '{{name}}'
    spec:
      project: default
      source:
        repoURL: "{{ repo_url }}"
        targetRevision: "{{ target_revision }}"
        path: '{{path}}'
      destination:
        server: https://kubernetes.default.svc
        namespace: '{{destNamespace}}'
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
        syncOptions:
          - CreateNamespace=true
"""
def run(cmd):
    try:
        p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False)
        return p.returncode,p.stdout.strip(),p.stderr.strip()
    except FileNotFoundError:
        return 127,"","command not found: "+cmd[0]
def detect_repo():
    e=os.environ.get("REPO_URL")
    if e:
        return e
    rc,out,err=run(["git","remote","get-url","origin"])
    if rc==0 and out:
        return normalize(out)
    rc,out,err=run(["git","remote","-v"])
    if rc==0 and out:
        first=out.splitlines()[0].split()[1]
        return normalize(first)
    return "https://example.com/replace-with-repo.git"
def normalize(url):
    url=url.strip()
    if url.startswith("git@"):
        try:
            host,path=url.split(":",1)
            host=host.split("@",1)[1]
            return f"https://{host}/{path}"
        except:
            return url
    return url
def detect_branch():
    e=os.environ.get("TARGET_REVISION")
    if e:
        return e
    rc,out,err=run(["git","symbolic-ref","--quiet","refs/remotes/origin/HEAD"])
    if rc==0 and out and out.startswith("refs/remotes/origin/"):
        return out.split("refs/remotes/origin/")[1]
    rc,out,err=run(["git","rev-parse","--abbrev-ref","HEAD"])
    if rc==0 and out:
        return out.strip()
    return "main"
def enum_elements(manifests_root):
    skip={"argocd-root","argocd",".",".."}
    items=[]
    if not manifests_root.exists():
        return items
    for p in sorted([x for x in manifests_root.iterdir() if x.is_dir() and not x.name.startswith(".")]):
        name=p.name
        if name in skip: continue
        dest=map_dest_namespace(name)
        items.append({"name":name,"path":str((manifests_root/name).as_posix()),"destNamespace":dest})
    return items
def map_dest_namespace(name):
    m={"k8s-core":"kube-system","ingress":"ingress-nginx","karpenter":"karpenter","monitoring":"monitoring","qdrant":"qdrant","rag":"rag"}
    return m.get(name,name)
def render(ctx):
    env=Environment(loader=BaseLoader(),trim_blocks=True,lstrip_blocks=True)
    env.filters['tojson']=lambda v: json.dumps(v)
    return env.from_string(TEMPLATE).render(**ctx).strip()+"\n"
def atomic_write(path,content):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(content)
    tmp.replace(path)
def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
def detect_kubectl():
    rc,out,err=run(["kubectl","version","--client","-o","json"])
    if rc==0: return True
    rc,out,err=run(["kubectl","version","--client"])
    if rc==0: return True
    rc,out,err=run(["kubectl","version"])
    return rc==0
def kubectl_apply(path):
    rc,out,err=run(["kubectl","apply","-f",str(path)])
    if rc!=0: raise RuntimeError(err or out)
    return out
def write_marker(repo,rev,output):
    payload={"applied_at":datetime.datetime.utcnow().isoformat()+"Z","repo_url":repo,"revision":rev,"applied_by":os.environ.get("USER",os.environ.get("USERNAME","unknown")),"kubectl_output":output}
    atomic_write(MARKER,json.dumps(payload,indent=2))
    return payload
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--apply",action="store_true")
    p.add_argument("--force",action="store_true")
    args=p.parse_args()
    repo=detect_repo()
    rev=detect_branch()
    elements=enum_elements(MANIFESTS_ROOT)
    if not elements:
        print("No folders found under infra/manifests; nothing to generate")
        sys.exit(1)
    ctx={"elements":elements,"repo_url":repo,"target_revision":rev,"name":"{{name}}","path":"{{path}}","destNamespace":"{{destNamespace}}"}
    rendered=render(ctx)
    if args.dry_run:
        print("--- DRY RUN applicationset.yaml ---\n")
        print(rendered)
        sys.exit(0)
    prev=file_hash(OUT_FILE) if OUT_FILE.exists() else None
    new=hashlib.sha256(rendered.encode()).hexdigest()
    if prev==new and not args.force:
        print("No change for",OUT_FILE)
    else:
        atomic_write(OUT_FILE,rendered)
        print("Wrote",OUT_FILE)
    if args.apply:
        if ROOT_MARKER.exists():
            print("Root app marker present; prefer letting ArgoCD create ApplicationSet. Skipping manual apply.")
            return
        if MARKER.exists() and not args.force:
            print("Applicationset marker exists; skipping apply")
            return
        if not detect_kubectl():
            print("kubectl not found; cannot apply"); sys.exit(1)
        try:
            out=kubectl_apply(OUT_FILE)
            marker=write_marker(repo,rev,out)
            print("Applied applicationset; marker written at",MARKER)
            print(json.dumps(marker,indent=2))
        except Exception as e:
            print("Apply failed:",str(e)); sys.exit(1)
if __name__=="__main__":
    main()
