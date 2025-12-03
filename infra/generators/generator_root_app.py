#!/usr/bin/env python3
import os,sys,subprocess,json,hashlib,argparse,datetime
from pathlib import Path
from jinja2 import Environment,BaseLoader

ROOT_DIR=Path(__file__).resolve().parents[2]
DEFAULT_MANIFESTS_PATH="infra/manifests/argocd"
OUT_DIR=ROOT_DIR / "infra" / "manifests" / "argocd-root"
OUT_FILE=OUT_DIR / "root-app.yaml"
MARKER_FILE=OUT_DIR / "root_app.applied.json"
TEMPLATE="""
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: {{ app_name }}
  namespace: {{ argocd_namespace }}
spec:
  project: {{ project }}
  source:
    repoURL: "{{ repo_url }}"
    targetRevision: "{{ target_revision }}"
    path: "{{ path }}"
  destination:
    server: "{{ destination_server }}"
    namespace: {{ destination_namespace }}
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
"""

def run_cmd(cmd):
    try:
        p=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"

def detect_repo_url():
    env_url=os.environ.get("REPO_URL")
    if env_url:
        return env_url
    rc,out,err=run_cmd(["git","remote","get-url","origin"])
    if rc==0 and out:
        return normalize_git_url(out)
    rc,out,err=run_cmd(["git","remote","-v"])
    if rc==0 and out:
        first=out.splitlines()[0].split()[1]
        return normalize_git_url(first)
    return None

def normalize_git_url(url):
    url=url.strip()
    if url.startswith("git@"):
        try:
            host,path=url.split(":",1)
            host=host.split("@",1)[1]
            return f"https://{host}/{path}"
        except Exception:
            return url
    return url

def detect_branch():
    env_branch=os.environ.get("TARGET_REVISION")
    if env_branch:
        return env_branch
    rc,out,err=run_cmd(["git","symbolic-ref","--quiet","refs/remotes/origin/HEAD"])
    if rc==0 and out and out.startswith("refs/remotes/origin/"):
        return out.split("refs/remotes/origin/")[1]
    rc,out,err=run_cmd(["git","rev-parse","--abbrev-ref","HEAD"])
    if rc==0 and out:
        return out.strip()
    return "main"

def render_yaml(ctx):
    env=Environment(loader=BaseLoader(),trim_blocks=True,lstrip_blocks=True)
    return env.from_string(TEMPLATE).render(**ctx).strip()+"\n"

def atomic_write(path:Path,content:str):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(content)
    tmp.replace(path)

def file_hash(path:Path):
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()

def detect_kubectl():
    rc,out,err=run_cmd(["kubectl","version","--client","-o","json"])
    if rc==0:
        return True
    rc,out,err=run_cmd(["kubectl","version","--client"])
    if rc==0:
        return True
    rc,out,err=run_cmd(["kubectl","version"])
    return rc==0

def kubectl_apply(path:Path):
    rc,out,err=run_cmd(["kubectl","apply","-f",str(path)])
    if rc!=0:
        raise RuntimeError(err or out)
    return out

def detect_argocd_cli():
    rc,out,err=run_cmd(["argocd","version"])
    return rc==0

def mask_token(t):
    if not t:
        return ""
    if len(t)<=8:
        return t[:2]+"..."+t[-2:]
    return t[:4]+"..."+t[-4:]

def attempt_argocd_repo_add(repo_url, git_user, git_pat, ssh_key_path=None):
    if ssh_key_path:
        cmd=["argocd","repo","add",repo_url,"--ssh-private-key-path",ssh_key_path]
    else:
        cmd=["argocd","repo","add",repo_url,"--username",git_user,"--password",git_pat]
    rc,out,err=run_cmd(cmd)
    return rc==0, out or err

def write_marker(repo_url,revision,applied_output):
    payload={
        "applied_at":datetime.datetime.utcnow().isoformat()+"Z",
        "repo_url":repo_url,
        "revision":revision,
        "applied_by":os.environ.get("USER",os.environ.get("USERNAME","unknown")),
        "kubectl_output":applied_output
    }
    atomic_write(MARKER_FILE,json.dumps(payload,indent=2))
    return payload

def print_repo_add_instructions(repo_url, git_user, git_pat, ssh_key_path):
    print("\n--- REPO ACCESS: manual step required ---")
    if ssh_key_path:
        print("You provided GIT_SSH_KEY_PATH. Run (on a machine with argocd CLI authenticated):")
        print(f"  argocd repo add {repo_url} --ssh-private-key-path {ssh_key_path}")
    elif git_pat:
        safe_mask=mask_token(git_pat)
        print("You provided GIT_PAT. Run (on a machine with argocd CLI authenticated):")
        print(f"  argocd repo add {repo_url} --username {git_user or '<your-git-username>'} --password <YOUR_PAT>")
        print("where <YOUR_PAT> is your Personal Access Token (not shown here). Example safe env usage:")
        print("  export GIT_PAT=ghp_xxx && argocd repo add {0} --username {1} --password \"$GIT_PAT\"".format(repo_url, git_user or "<user>"))
    else:
        print("No GIT_PAT or SSH key provided. For private repos add repo credentials to ArgoCD via `argocd repo add` or UI.")
    print("--- end instructions ---\n")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--force",action="store_true")
    p.add_argument("--apply",action="store_true")
    p.add_argument("--path",default=os.environ.get("ROOT_APP_PATH",DEFAULT_MANIFESTS_PATH))
    p.add_argument("--repo",default=os.environ.get("REPO_URL"))
    p.add_argument("--revision",default=os.environ.get("TARGET_REVISION"))
    p.add_argument("--app-name",default=os.environ.get("ROOT_APP_NAME","argocd-root"))
    p.add_argument("--argocd-namespace",default=os.environ.get("ARGOCD_NAMESPACE","argocd"))
    p.add_argument("--dest-namespace",default=os.environ.get("DEST_NAMESPACE","argocd"))
    args=p.parse_args()

    repo_url=args.repo or detect_repo_url() or "https://example.com/replace-with-repo.git"
    revision=args.revision or detect_branch() or "main"
    ctx={
        "app_name":args.app_name,
        "argocd_namespace":args.argocd_namespace,
        "project":"default",
        "repo_url":repo_url,
        "target_revision":revision,
        "path":args.path,
        "destination_server":"https://kubernetes.default.svc",
        "destination_namespace":args.dest_namespace
    }
    rendered=render_yaml(ctx)
    if args.dry_run:
        print("--- DRY RUN root-app.yaml ---\n")
        print(rendered)
        print("\n--- DRY RUN COMPLETE ---")
        sys.exit(0)

    prev=file_hash(OUT_FILE)
    new=hashlib.sha256(rendered.encode()).hexdigest()
    if prev==new and not args.force:
        print(f"No change for {OUT_FILE}; not rewriting.")
    else:
        atomic_write(OUT_FILE,rendered)
        print(f"Wrote {OUT_FILE}")

    if args.apply:
        if MARKER_FILE.exists() and not args.force:
            print(f"Marker {MARKER_FILE} exists; skipping apply. Use --force to reapply.")
            return
        git_pat=os.environ.get("GIT_PAT","")
        git_user=os.environ.get("GIT_USER","")
        ssh_key_path=os.environ.get("GIT_SSH_KEY_PATH","")
        if git_pat and not git_user:
            print("Warning: GIT_PAT provided but GIT_USER not set; try setting GIT_USER to your git username for argocd repo registration.")

        argocd_cli=detect_argocd_cli()
        if git_pat or ssh_key_path:
            if argocd_cli:
                ok,msg=attempt_argocd_repo_add(repo_url, git_user, git_pat, ssh_key_path if ssh_key_path else None)
                if ok:
                    print("argocd repo add succeeded (repo credential registered).")
                else:
                    print("argocd repo add attempted but failed:",msg)
                    print_repo_add_instructions(repo_url, git_user, git_pat, ssh_key_path)
            else:
                print("argocd CLI not present or not logged in. I will not attempt repo registration.")
                print_repo_add_instructions(repo_url, git_user, git_pat, ssh_key_path)

        if not detect_kubectl():
            print("kubectl not found or not functional; cannot apply. Aborting.")
            sys.exit(1)
        try:
            apply_out=kubectl_apply(OUT_FILE)
            marker=write_marker(repo_url,revision,apply_out)
            print("kubectl apply succeeded; marker written at:",MARKER_FILE)
            print(json.dumps(marker,indent=2))
        except Exception as e:
            print("kubectl apply failed:",str(e))
            sys.exit(1)
    else:
        print("Generate-only run complete. Use --apply to kubectl apply the root app.")

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print("Unhandled error:",str(e))
        sys.exit(1)
