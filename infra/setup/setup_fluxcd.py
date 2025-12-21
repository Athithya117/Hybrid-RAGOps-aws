#!/usr/bin/env python3
import os,sys,subprocess,json,hashlib,time,re,shutil,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MANIFESTS_ROOT=Path(os.environ.get("MANIFEST_PATH","infra/manifests"))
FLUX_NS=os.environ.get("FLUX_NS","flux-system")
RECONCILE_INTERVAL_SECONDS=int(os.environ.get("RECONCILE_INTERVAL_SECONDS","60"))
GIT_PAT=os.environ.get("GIT_PAT","").strip()
GIT_TIMEOUT=8
WAIT_CONTROLLER_SECONDS=int(os.environ.get("WAIT_CONTROLLER_SECONDS","180"))
WAIT_READY_SECONDS=int(os.environ.get("WAIT_READY_SECONDS",str(max(60,RECONCILE_INTERVAL_SECONDS*2))))
# desired list (script will only use ones that actually exist)
DESIRED_DIRS=["dense","retrieval","frontend","ingress","monitoring","qdrant","reranker","sparse"]
LOG_DIR=MANIFESTS_ROOT/FLUX_NS
LOG_FILE=LOG_DIR/"setup_fluxcd.log"

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

def mask_token(tok):
    if not tok: return ""
    if len(tok) <= 8: return "***"
    return tok[:4]+"..."+tok[-4:]

def check_bins():
    for b in ("kubectl","git","flux"):
        if shutil.which(b) is None:
            die("missing_binary:"+b)

def normalize(url):
    u=(url or "").strip()
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

def ensure_dir(p):
    try:
        p.mkdir(parents=True,exist_ok=True)
    except Exception as e:
        die("create_log_dir_failed",str(e))

def ensure_namespace(ns):
    rc,out,err=run(["kubectl","get","ns",ns,"-o","name"],timeout=8)
    if rc==0 and out:
        info("namespace_exists",ns); return
    rc,out,err=run(["kubectl","create","ns",ns,"--dry-run=client","-o","yaml"],timeout=8)
    if rc!=0: die("namespace_create_dryrun_failed",err or out)
    rc,out,err=run(["kubectl","apply","-f","-"],input=out,timeout=8)
    if rc!=0: die("namespace_create_failed",err or out)
    info("namespace_created",ns)

def wait_deploy(name,timeout):
    start=time.time()
    while time.time()-start<timeout:
        rc,out,err=run(["kubectl","-n",FLUX_NS,"get","deploy",name,"-o","jsonpath={.status.availableReplicas}"],timeout=8)
        if rc==0 and out:
            try:
                if int(out)>0: return True
            except:
                pass
        time.sleep(2)
    return False

def controllers_ready(timeout):
    required=("source-controller","kustomize-controller")
    for d in required:
        if not wait_deploy(d,timeout):
            return False
    return True

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

def kubectl_apply(yaml,redact=False):
    rc,out,err=run(["kubectl","apply","-f","-"],input=yaml,timeout=30)
    if rc!=0: die("kubectl_apply_failed",err or out)
    if redact or "stringData" in yaml or "password" in yaml:
        info("kubectl_apply_ok","redacted")
    else:
        first = yaml.splitlines()[0] if yaml else ""
        info("kubectl_apply_ok",first)

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
        time.sleep(2)
    return False

def collect_debug(kind,name,controller):
    desc_rc,desc_out,desc_err=run(["kubectl","-n",FLUX_NS,"describe",kind,name],timeout=20)
    events_rc,events_out,events_err=run(["kubectl","-n",FLUX_NS,"get","events","--field-selector","involvedObject.name="+name,"-o","json"],timeout=20)
    logs_rc,logs_out,logs_err=run(["kubectl","-n",FLUX_NS,"logs","deployment/"+controller,"--tail=200"],timeout=30)
    return {"describe":desc_out or desc_err,"events":events_out or events_err,controller+"_logs":logs_out or logs_err}

# Robust write_and_push using a temporary clone (isolates from local workspace)
def write_and_push(path,branch,user,pat,msg):
    if not pat:
        die("GIT_PAT_required_for_auto_push")
    repo_url = repo
    if not repo_url.startswith("https://"):
        die("git_push_auto_only_supports_https")
    rest = repo_url[len("https://"):]
    auth_url = f"https://{user}:{pat}@{rest}"
    tmpdir = tempfile.mkdtemp(prefix="flux-push-")
    try:
        rc,out,err = run(["git","clone","--depth","1","--branch",branch,auth_url,tmpdir],timeout=120)
        if rc!=0:
            # fallback: clone default branch then create branch
            rc,out,err = run(["git","clone","--depth","1",auth_url,tmpdir],timeout=120)
            if rc!=0:
                shutil.rmtree(tmpdir,ignore_errors=True)
                die("git_clone_failed",err or out)
            rc,out,err = run(["git","-C",tmpdir,"checkout","-B",branch],timeout=30)
            if rc!=0:
                shutil.rmtree(tmpdir,ignore_errors=True)
                die("git_checkout_branch_failed",out+err)
        # compute destination path inside clone
        try:
            rel_target = Path(path).relative_to(ROOT)
        except Exception:
            rel_target = Path(path).name
        dest = Path(tmpdir)/rel_target
        if dest.exists():
            # remove previous generated folder so replaced files don't linger
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        dest.parent.mkdir(parents=True,exist_ok=True)
        # copy generated files
        if Path(path).is_dir():
            shutil.copytree(str(path),str(dest),dirs_exist_ok=True)
        else:
            shutil.copy2(str(path),str(dest))
        rc,out,err = run(["git","-C",tmpdir,"add",str(rel_target)],timeout=30)
        if rc!=0:
            shutil.rmtree(tmpdir,ignore_errors=True)
            die("git_add_failed",out+err)
        rc,out,err = run(["git","-C",tmpdir,"commit","-m",msg],timeout=30)
        if rc!=0:
            if "nothing to commit" in (out+err).lower():
                info("git_nothing_to_commit")
                shutil.rmtree(tmpdir,ignore_errors=True)
                return
            shutil.rmtree(tmpdir,ignore_errors=True)
            die("git_commit_failed",out+err)
        rc,out,err = run(["git","-C",tmpdir,"push","origin",f"HEAD:{branch}"],timeout=60)
        if rc!=0:
            shutil.rmtree(tmpdir,ignore_errors=True)
            die("git_push_failed",err or out)
        info("git_push_ok")
    finally:
        try:
            shutil.rmtree(tmpdir,ignore_errors=True)
        except Exception:
            pass

# daemonize child to keep watching kustomizations after parent exits
def daemonize_and_wait(kustom_names,wait_seconds,log_path):
    try:
        pid = os.fork()
        if pid > 0:
            return
    except AttributeError:
        pass
    os.setsid()
    try:
        pid2 = os.fork()
        if pid2 > 0:
            os._exit(0)
    except AttributeError:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    with open(log_path,"a",encoding="utf-8") as lf:
        def logj(k,v=None):
            entry={"ts":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
            entry["info"]=k
            if v is not None:
                entry["value"]=v
            lf.write(json.dumps(entry,ensure_ascii=False)+"\n")
            lf.flush()
        logj("daemon_start",{"kustomizations":kustom_names,"wait_seconds":wait_seconds})
        results=[]
        for kn in kustom_names:
            ok=wait_ready("kustomization",kn,wait_seconds)
            if ok:
                logj("kustomization_ready",kn)
                results.append({"name":kn,"ready":True})
            else:
                debug=collect_debug("kustomization",kn,"kustomize-controller")
                logj("kustomization_failed",{"name":kn,"debug":debug})
                results.append({"name":kn,"ready":False,"debug":debug})
        logj("daemon_done",{"results":results})
    os._exit(0)

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
    available_dirs=[p.name for p in sorted(MANIFESTS_ROOT.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    selected=[d for d in DESIRED_DIRS if (MANIFESTS_ROOT/d).is_dir()]
    if not selected:
        die("no_selected_manifest_dirs_found",{"desired":DESIRED_DIRS,"available":available_dirs})
    info("repo",repo); info("branch",branch); info("selected",selected)
    sanitized_base=sanitize(re.sub(r"https?://","",repo).rstrip("/").replace("/","-"))
    if sanitized_base:
        repo_cr_name = sanitized_base if sanitized_base.endswith("-git") else sanitized_base + "-git"
    else:
        repo_cr_name = "repo-" + short_hash(repo)
    ensure_dir(LOG_DIR)
    ensure_namespace(FLUX_NS)
    info("bootstrapping_flux","start")
    rc,out,err=run(["flux","check","--pre"],timeout=60)
    if rc!=0:
        rc,out,err=run(["flux","install","--namespace",FLUX_NS],timeout=300)
        if rc!=0: die("flux_install_failed",err or out)
        info("flux_installed",FLUX_NS)
    else:
        if not controllers_ready(5):
            rc,out,err=run(["flux","install","--namespace",FLUX_NS],timeout=300)
            if rc!=0: die("flux_install_failed",err or out)
            info("flux_installed",FLUX_NS)
    if not controllers_ready(WAIT_CONTROLLER_SECONDS):
        die("flux_controllers_not_ready")
    info("flux_controllers_ready")
    secret_name=None
    if GIT_PAT:
        secret_name="repo-"+short_hash(repo)
        secret_obj={"apiVersion":"v1","kind":"Secret","metadata":{"name":secret_name,"namespace":FLUX_NS},"stringData":{"username":"git","password":GIT_PAT}}
        kubectl_apply(json.dumps(secret_obj),redact=True)
        info("repo_secret_created",secret_name)
    gitrepo_yaml=render_gitrepo(repo_cr_name,repo,branch,secret_name,RECONCILE_INTERVAL_SECONDS)
    kubectl_apply(gitrepo_yaml)
    if not wait_ready("gitrepository",repo_cr_name,WAIT_READY_SECONDS):
        debug=collect_debug("gitrepository",repo_cr_name,"source-controller")
        die("gitrepository_not_ready",debug)
    info("gitrepository_ready",repo_cr_name)
    kustom_yamls=[]
    for d in selected:
        path=f"./{MANIFESTS_ROOT.as_posix()}/{d}"
        kn=sanitize(repo_cr_name+"-"+d)
        ky=render_kustomization(kn,repo_cr_name,path,RECONCILE_INTERVAL_SECONDS)
        kustom_yamls.append((kn,ky,d))
    combined="\n---\n".join(ky for (_,ky,_) in kustom_yamls)
    kubectl_apply(combined)
    if args.auto_push:
        # perform push from temp clone (isolated)
        outdir=MANIFESTS_ROOT/FLUX_NS
        outdir.mkdir(parents=True,exist_ok=True)
        repo_file=outdir/"gitrepository.yaml"
        repo_file.write_text(gitrepo_yaml)
        for kn,ky,d in kustom_yamls:
            (outdir/f"{kn}.yaml").write_text(ky)
        # commit & push using isolated clone
        write_and_push(outdir,branch,git_user,GIT_PAT,"gen: flux bootstrap + kustomizations")
    # print concise parent summary, then detach to background to wait
    kustom_names=[kn for (kn,_,_) in kustom_yamls]
    summary={"repo":repo,"branch":branch,"gitrepo":repo_cr_name,"kustomizations":kustom_names,"log_file":str(LOG_FILE)}
    if secret_name:
        summary["repo_secret"]=secret_name
        summary["repo_token_masked"]=mask_token(GIT_PAT)
    print(json.dumps({"summary":summary},indent=2))
    # detach and let child wait & record progress
    try:
        daemonize_and_wait(kustom_names,WAIT_READY_SECONDS,str(LOG_FILE))
    except Exception as e:
        die("daemonize_failed",str(e))

if __name__=="__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        die("unhandled_exception",str(e))
