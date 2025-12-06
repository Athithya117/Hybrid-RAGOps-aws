import os,sys,subprocess,json,hashlib,time,re,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MANIFESTS_ROOT=Path(os.environ.get("MANIFEST_PATH","infra/manifests"))
FLUX_NS=os.environ.get("FLUX_NS","flux-system")
RECONCILE_INTERVAL_SECONDS=int(os.environ.get("RECONCILE_INTERVAL_SECONDS","60"))
GIT_PAT=os.environ.get("GIT_PAT","").strip()
GIT_TIMEOUT=8
WAIT_READY_SECONDS=int(os.environ.get("WAIT_READY_SECONDS",str(max(60,RECONCILE_INTERVAL_SECONDS*2))))
def run(cmd,input=None,timeout=None):
    try:
        p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,input=input,timeout=timeout)
        return p.returncode,(p.stdout or "").strip(),(p.stderr or "").strip()
    except FileNotFoundError:
        return 127,"","command not found: "+(cmd[0] if cmd else "")
    except subprocess.TimeoutExpired as e:
        return 124,getattr(e,"stdout","") or "","timeout"
def die(msg,extra=None):
    out={"error":msg}
    if extra: out["detail"]=extra
    print(json.dumps(out))
    sys.exit(1)
def info(k,v=None):
    if v is None:
        print(json.dumps({"info":k}))
    else:
        print(json.dumps({"info":k,"value":v}))
def check_bins():
    for b in ("kubectl","git","flux"):
        if shutil.which(b) is None:
            die("missing_binary:"+b)
def normalize(url):
    u=url.strip()
    if u.startswith("git@"):
        try:
            host,path=u.split(":",1)
            host=host.split("@",1)[1]
            return f"https://{host}/{path}"
        except:
            return u
    return u
def detect_repo():
    env=os.environ.get("REPO_URL","").strip()
    if env: return normalize(env)
    rc,out,err=run(["git","remote","get-url","origin"],timeout=GIT_TIMEOUT)
    if rc==0 and out: return normalize(out)
    rc,out,err=run(["git","remote","-v"],timeout=GIT_TIMEOUT)
    if rc==0 and out:
        first=out.splitlines()[0].split()[1]
        return normalize(first)
    die("no_repo_detected")
def detect_branch():
    e=os.environ.get("BRANCH") or os.environ.get("TARGET_REVISION")
    if e: return e
    rc,out,err=run(["git","symbolic-ref","--quiet","refs/remotes/origin/HEAD"],timeout=GIT_TIMEOUT)
    if rc==0 and out and out.startswith("refs/remotes/origin/"):
        return out.split("refs/remotes/origin/")[1]
    rc,out,err=run(["git","rev-parse","--abbrev-ref","HEAD"],timeout=GIT_TIMEOUT)
    if rc==0 and out: return out.strip()
    return "main"
def detect_git_user():
    rc,out,err=run(["git","config","user.name"],timeout=GIT_TIMEOUT)
    if rc==0 and out: return out.strip()
    rc,out,err=run(["git","config","user.email"],timeout=GIT_TIMEOUT)
    if rc==0 and out: return out.split("@",1)[0]
    return "git"
def sanitize(n):
    return re.sub(r'[^a-z0-9-]', '-', n.lower()).strip('-')[:63]
def short_hash(s):
    return hashlib.sha1(s.encode()).hexdigest()[:8]
def render_gitrepo(name,repo,branch,secret_name=None,interval_s=60):
    lines=["apiVersion: source.toolkit.fluxcd.io/v1","kind: GitRepository","metadata:","  name: "+name,"  namespace: "+FLUX_NS,"spec:","  interval: "+str(interval_s)+"s","  url: \""+repo+"\"","  ref:","    branch: \""+branch+"\""]
    if secret_name:
        lines += ["  secretRef:","    name: "+secret_name]
    return "\n".join(lines)
def render_kustomization(name,repo_name,path,interval_s=60,timeout_s=None):
    if timeout_s is None:
        timeout_s=max(30,interval_s*2)
    lines=["apiVersion: kustomize.toolkit.fluxcd.io/v1","kind: Kustomization","metadata:","  name: "+name,"  namespace: "+FLUX_NS,"spec:","  interval: "+str(interval_s)+"s","  retryInterval: "+str(interval_s)+"s","  timeout: "+str(timeout_s)+"s","  prune: true","  wait: true","  sourceRef:","    kind: GitRepository","    name: "+repo_name,"    namespace: "+FLUX_NS,"  path: \""+path+"\""]
    return "\n".join(lines)
def kubectl_apply(yaml):
    rc,out,err=run(["kubectl","apply","-f","-"],input=yaml,timeout=30)
    if rc!=0: die("kubectl_apply_failed",err or out)
    info("kubectl_apply_ok",yaml.splitlines()[0] if yaml else "")
def wait_ready(kind,name,timeout):
    start=time.time()
    while time.time()-start<timeout:
        rc,out,err=run(["kubectl","-n",FLUX_NS,"get",kind,name,"-o","json"],timeout=10)
        if rc==0 and out:
            try:
                obj=json.loads(out)
                conds=obj.get("status",{}).get("conditions",[])
                for c in conds:
                    if c.get("type")=="Ready" and c.get("status")=="True":
                        return True
            except:
                pass
        time.sleep(3)
    return False
def write_and_push(path,branch,user,pat,msg):
    rc,out,err=run(["git","add",str(path)],timeout=15)
    rc,out,err=run(["git","commit","-m",msg],timeout=15)
    if rc!=0:
        if "nothing to commit" in (out+err).lower():
            info("git_nothing_to_commit")
            return
        die("git_commit_failed",out+err)
    if not repo.startswith("https://"):
        die("git_push_auto_only_supports_https")
    rest=repo[len("https://"):]
    push_url=f"https://{user}:{pat}@{rest}"
    rc,out,err=run(["git","push",push_url,f"HEAD:{branch}"],timeout=60)
    if rc!=0: die("git_push_failed",err or out)
    info("git_push_ok")
def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--auto-push",action="store_true")
    args=parser.parse_args()
    check_bins()
    global repo
    repo=detect_repo()
    branch=detect_branch()
    git_user=detect_git_user()
    if MANIFESTS_ROOT.is_dir() is False: die("manifests_root_missing",str(MANIFESTS_ROOT))
    elements=[p.name for p in sorted(MANIFESTS_ROOT.iterdir()) if p.is_dir() and not p.name.startswith(".") and p.name!=FLUX_NS]
    if not elements: die("no_manifest_dirs_found")
    info("repo",repo)
    info("branch",branch)
    info("elements",elements)
    sanitized_repo_name=sanitize(re.sub(r"https?://","",repo).rstrip("/").replace("/","-"))
    repo_cr_name=sanitized_repo_name or "repo-"+short_hash(repo)
    secret_name=None
    info("bootstrapping_flux","start")
    rc,out,err=run(["flux","check","--pre"],timeout=60)
    if rc!=0:
        rc,out,err=run(["flux","install","--namespace",FLUX_NS],timeout=300)
        if rc!=0: die("flux_install_failed",err or out)
        info("flux_installed",FLUX_NS)
    else:
        info("flux_precheck_ok")
    if GIT_PAT:
        secret_name="repo-"+short_hash(repo)
        secret_obj={"apiVersion":"v1","kind":"Secret","metadata":{"name":secret_name,"namespace":FLUX_NS},"stringData":{"username":"git","password":GIT_PAT}}
        kubectl_apply(json.dumps(secret_obj))
        info("repo_secret_created",secret_name)
    gitrepo_yaml=render_gitrepo(repo_cr_name,repo,branch,secret_name,RECONCILE_INTERVAL_SECONDS)
    kubectl_apply(gitrepo_yaml)
    if not wait_ready("gitrepository",repo_cr_name,WAIT_READY_SECONDS): die("gitrepository_not_ready",repo_cr_name)
    info("gitrepository_ready",repo_cr_name)
    kustom_yamls=[]
    for d in elements:
        path=f"./{MANIFESTS_ROOT.as_posix()}/{d}"
        kn=sanitize(repo_cr_name+"-"+d)
        ky=render_kustomization(kn,repo_cr_name,path,RECONCILE_INTERVAL_SECONDS)
        kustom_yamls.append((kn,ky,d))
    combined="\n---\n".join(ky for (_,ky,_) in kustom_yamls)
    kubectl_apply(combined)
    failed=[]
    for kn,ky,d in kustom_yamls:
        if not wait_ready("kustomization",kn,WAIT_READY_SECONDS):
            failed.append(kn)
        else:
            info("kustomization_ready",kn)
    summary={"repo":repo,"branch":branch,"gitrepo":repo_cr_name,"kustomizations":[kn for (kn,_,_) in kustom_yamls],"failed":failed}
    print(json.dumps({"summary":summary},indent=2))
    if args.auto_push:
        if not GIT_PAT: die("GIT_PAT_required_for_auto_push")
        outdir=MANIFESTS_ROOT/FLUX_NS
        outdir.mkdir(parents=True,exist_ok=True)
        repo_file=outdir/"gitrepository.yaml"
        repo_file.write_text(gitrepo_yaml)
        for kn,ky,d in kustom_yamls:
            (outdir/f"{kn}.yaml").write_text(ky)
        write_and_push(outdir,branch,git_user,GIT_PAT,"gen: flux bootstrap + kustomizations")
if __name__=="__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        die("unhandled_exception",str(e))
