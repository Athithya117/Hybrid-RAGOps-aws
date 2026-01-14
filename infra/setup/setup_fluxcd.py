#!/usr/bin/env python3
from __future__ import annotations
import os, sys, subprocess, json, hashlib, time, re, shutil, tempfile
from pathlib import Path

try:
    import yaml
except Exception:
    print(json.dumps({"error": "PyYAML required. Install with: pip install pyyaml"})); sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS_ROOT = Path(os.environ.get("MANIFEST_PATH", "infra/manifests"))
FLUX_NS = os.environ.get("FLUX_NS", "flux-system")
GIT_PAT = os.environ.get("GIT_PAT", "").strip()
GIT_TIMEOUT = 8
WAIT_CONTROLLER_SECONDS = int(os.environ.get("WAIT_CONTROLLER_SECONDS", "180"))
WAIT_CRDS_SECONDS = int(os.environ.get("WAIT_CRDS_SECONDS", "120"))

def run(cmd, input=None, timeout=None):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, input=input, timeout=timeout)
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except FileNotFoundError:
        return 127, "", "command not found: " + (cmd[0] if cmd else "")
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", "") or "", "timeout"

def die(msg, extra=None):
    out = {"error": msg}
    if extra: out["detail"] = extra
    print(json.dumps(out)); sys.exit(1)

def info(k, v=None):
    if v is None:
        print(json.dumps({"info": k}))
    else:
        print(json.dumps({"info": k, "value": v}))

def sanitize(n):
    return re.sub(r'[^a-z0-9-]', '-', n.lower()).strip('-')[:63]

def short_hash(s):
    return hashlib.sha1(s.encode()).hexdigest()[:8]

def detect_repo():
    env = os.environ.get("REPO_URL", "").strip()
    if env: return env
    rc, out, err = run(["git", "remote", "get-url", "origin"], timeout=GIT_TIMEOUT)
    if rc == 0 and out: return out
    rc, out, err = run(["git", "remote", "-v"], timeout=GIT_TIMEOUT)
    if rc == 0 and out:
        first = out.splitlines()[0].split()[1]; return first
    die("no_repo_detected")

def detect_branch():
    e = os.environ.get("BRANCH") or os.environ.get("TARGET_REVISION")
    if e: return e
    rc, out, err = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=GIT_TIMEOUT)
    if rc == 0 and out: return out.strip()
    return "main"

def detect_git_user():
    # Prefer git config user.name, fallback to user.email local-part, finally "git"
    rc, out, err = run(["git", "config", "--get", "user.name"])
    if rc == 0 and out: return out.strip()
    rc, out, err = run(["git", "config", "--get", "user.email"])
    if rc == 0 and out:
        return out.split("@", 1)[0]
    return "git"

def check_bins():
    for b in ("kubectl", "git", "flux"):
        if shutil.which(b) is None: die("missing_binary:" + b)

def ensure_namespace(ns):
    rc, out, err = run(["kubectl", "get", "ns", ns, "-o", "name"], timeout=8)
    if rc == 0 and out: info("namespace_exists", ns); return
    rc, out, err = run(["kubectl", "create", "ns", ns, "--dry-run=client", "-o", "yaml"], timeout=8)
    if rc != 0: die("namespace_create_dryrun_failed", err or out)
    rc2, out2, err2 = run(["kubectl", "apply", "-f", "-"], input=out, timeout=8)
    if rc2 != 0: die("namespace_create_failed", err2 or out2)
    info("namespace_created", ns)

# Wait for the Flux CRDs we need to appear in the API server.
def wait_for_crds(crd_names, timeout_seconds=WAIT_CRDS_SECONDS):
    deadline = time.time() + timeout_seconds
    missing = set(crd_names)
    while time.time() < deadline:
        for crd in list(missing):
            rc, out, err = run(["kubectl", "get", "crd", crd], timeout=8)
            if rc == 0:
                missing.discard(crd)
        if not missing:
            return True, []
        time.sleep(2)
    return False, sorted(list(missing))

def render_helmrelease(repo_cr_name, branch, component, target_namespace=None):
    # HelmRelease API (helm.toolkit.fluxcd.io/v2) fields follow Flux docs.
    if target_namespace is None: target_namespace = component
    replicas = int(os.environ.get(f"{component.upper()}_REPLICAS", os.environ.get("REPLICAS", "1")))
    # By default assume charts are stored as ./infra/archive/<component>-helm-chart/<component>
    chart_path = f"./infra/archive/{component}-helm-chart/{component}"
    vals = {"replicaCount": replicas}
    hr = {
        "apiVersion": "helm.toolkit.fluxcd.io/v2",
        "kind": "HelmRelease",
        "metadata": {"name": component, "namespace": FLUX_NS},
        "spec": {
            "interval": "5m",
            "install": {"createNamespace": True},
            "targetNamespace": target_namespace,
            "chart": {
                "spec": {
                    "chart": chart_path,
                    "sourceRef": {"kind": "GitRepository", "name": repo_cr_name, "namespace": FLUX_NS}
                }
            },
            "values": vals
        }
    }
    return yaml.safe_dump(hr, sort_keys=False)

def render_namespace_yaml(target_namespace):
    ns = {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": target_namespace}}
    return yaml.safe_dump(ns, sort_keys=False)

def render_service_patch(target_namespace, component):
    port = int(os.environ.get(f"{component.upper()}_METRICS_PORT", os.environ.get("METRICS_PORT", "6333")))
    svc = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": component, "namespace": target_namespace,
                     "labels": {"app.kubernetes.io/name": component, "app.kubernetes.io/component": component}},
        "spec": {"selector": {"app.kubernetes.io/name": component},
                 "ports": [{"name": "metrics", "port": port, "targetPort": port, "protocol": "TCP"}],
                 "type": "ClusterIP"}
    }
    return yaml.safe_dump(svc, sort_keys=False)

def render_gitrepo(name, repo, branch, secret_name=None, interval_s=60):
    # GitRepository apiVersion per Flux source API v1. (duration string accepted)
    obj = {"apiVersion": "source.toolkit.fluxcd.io/v1", "kind": "GitRepository",
           "metadata": {"name": name, "namespace": FLUX_NS},
           "spec": {"interval": f"{interval_s}s", "url": repo, "ref": {"branch": branch}}}
    if secret_name: obj["spec"]["secretRef"] = {"name": secret_name}
    return yaml.safe_dump(obj, sort_keys=False)

def render_kustomization(name, repo_name, path, interval_s=60):
    obj = {
        "apiVersion": "kustomize.toolkit.fluxcd.io/v1",
        "kind": "Kustomization",
        "metadata": {"name": name, "namespace": FLUX_NS},
        "spec": {
            "interval": f"{interval_s}s",
            "retryInterval": f"{interval_s}s",
            "timeout": "120s",
            "prune": True,
            "wait": True,
            "sourceRef": {"kind": "GitRepository", "name": repo_name, "namespace": FLUX_NS},
            "path": path
        }
    }
    return yaml.safe_dump(obj, sort_keys=False)

def kubectl_apply(yaml_str, redact=False, timeout=30):
    rc, out, err = run(["kubectl", "apply", "-f", "-"], input=yaml_str, timeout=timeout)
    if rc != 0:
        # return error tuple to caller so caller can decide whether to retry
        return False, (err or out)
    if redact or "stringData" in yaml_str or "password" in yaml_str:
        info("kubectl_apply_ok", "redacted")
    else:
        info("kubectl_apply_ok", yaml_str.splitlines()[0] if yaml_str else "")
    return True, ""

def write_and_push(repo, path, branch, user, pat, msg):
    if not pat: die("GIT_PAT_required_for_auto_push")
    repo_url = repo
    if not repo_url.startswith("https://"): die("git_push_auto_only_supports_https")
    rest = repo_url[len("https://"):]
    auth_url = f"https://{user}:{pat}@{rest}"
    tmpdir = tempfile.mkdtemp(prefix="flux-push-")
    try:
        rc, out, err = run(["git", "clone", "--depth", "1", "--branch", branch, auth_url, tmpdir], timeout=120)
        if rc != 0:
            rc, out, err = run(["git", "clone", "--depth", "1", auth_url, tmpdir], timeout=120)
            if rc != 0:
                shutil.rmtree(tmpdir, ignore_errors=True); die("git_clone_failed", err or out)
            rc, out, err = run(["git", "-C", tmpdir, "checkout", "-B", branch], timeout=30)
            if rc != 0:
                shutil.rmtree(tmpdir, ignore_errors=True); die("git_checkout_branch_failed", out + err)
        try:
            rel_target = Path(path).relative_to(ROOT)
        except Exception:
            rel_target = Path(path).name
        dest = Path(tmpdir) / rel_target
        if dest.exists():
            if dest.is_dir(): shutil.rmtree(dest)
            else: dest.unlink()
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(path).is_dir():
            shutil.copytree(str(path), str(dest), dirs_exist_ok=True)
        else:
            shutil.copy2(str(path), str(dest))
        rc, out, err = run(["git", "-C", tmpdir, "add", str(rel_target)], timeout=30)
        if rc != 0: shutil.rmtree(tmpdir, ignore_errors=True); die("git_add_failed", out + err)
        rc, out, err = run(["git", "-C", tmpdir, "commit", "-m", msg], timeout=30)
        if rc != 0:
            if "nothing to commit" in (out + err).lower():
                info("git_nothing_to_commit"); shutil.rmtree(tmpdir, ignore_errors=True); return
            shutil.rmtree(tmpdir, ignore_errors=True); die("git_commit_failed", out + err)
        rc, out, err = run(["git", "-C", tmpdir, "push", "origin", f"HEAD:{branch}"], timeout=60)
        if rc != 0: shutil.rmtree(tmpdir, ignore_errors=True); die("git_push_failed", err or out)
        info("git_push_ok")
    finally:
        try: shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception: pass

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Install Flux (if necessary) and generate per-component manifests/kustomizations under infra/manifests/<component>."
    )
    parser.add_argument("--auto-push", action="store_true", help="Push generated manifests to detected repo (requires GIT_PAT).")
    parser.add_argument("--qdrant", action="store_true", help="(backwards-compatible) generate qdrant manifests")
    args, remaining = parser.parse_known_args()

    # Build list of components:
    components = []
    if args.qdrant:
        components.append("qdrant")
    for tok in remaining:
        if tok.startswith("--"):
            name = tok.lstrip("-")
            if name: components.append(name)
        else:
            components.append(tok)
    if not components:
        components = ["qdrant"]

    check_bins()
    repo = detect_repo(); branch = detect_branch()
    info("repo", repo); info("branch", branch)
    ensure_namespace(FLUX_NS)

    rc, out, err = run(["flux", "check", "--pre"], timeout=60)
    if rc != 0:
        rc, out, err = run(["flux", "install", "--namespace", FLUX_NS], timeout=300)
        if rc != 0: die("flux_install_failed", err or out)
        info("flux_installed", FLUX_NS)

    # Wait until controllers show available replicas (best-effort loop)
    start = time.time()
    while (time.time() - start) < WAIT_CONTROLLER_SECONDS:
        rc, out, err = run(["kubectl", "-n", FLUX_NS, "get", "deploy", "source-controller", "-o", "jsonpath={.status.availableReplicas}"], timeout=8)
        if rc == 0 and out and out != "0": break
        time.sleep(2)
    info("flux_controllers_ready")

    # Ensure required Flux CRDs exist before we apply GitRepository / Kustomization
    required_crds = [
        "gitrepositories.source.toolkit.fluxcd.io",
        "kustomizations.kustomize.toolkit.fluxcd.io",
        "helmreleases.helm.toolkit.fluxcd.io",
    ]
    ok, missing = wait_for_crds(required_crds, timeout_seconds=WAIT_CRDS_SECONDS)
    if not ok:
        die("flux_crds_missing_after_install", missing)
    info("flux_crds_present")

    repo_cr_name = sanitize(re.sub(r'https?://', '', repo).rstrip('/').replace('/', '-'))
    if not repo_cr_name.endswith("-git"):
        repo_cr_name = repo_cr_name + "-git"

    secret_name = None
    if GIT_PAT:
        secret_name = "repo-" + short_hash(repo)
        secret_obj = {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": secret_name, "namespace": FLUX_NS},
                      "stringData": {"username": detect_git_user(), "password": GIT_PAT}}
        ok, detail = kubectl_apply(yaml.safe_dump(secret_obj), redact=True)
        if not ok: die("kubectl_apply_failed", detail)
        info("repo_secret_created", secret_name)

    gitrepo_yaml = render_gitrepo(repo_cr_name, repo, branch, secret_name, 60)
    ok, detail = kubectl_apply(gitrepo_yaml)
    if not ok:
        # If applying still fails (very unlikely after CRDs), show detail
        die("kubectl_apply_failed", detail)

    # Wait for GitRepository resource to be visible
    start = time.time()
    while (time.time() - start) < 120:
        rc, out, err = run(["kubectl", "-n", FLUX_NS, "get", "gitrepository", repo_cr_name, "-o", "json"], timeout=8)
        if rc == 0 and out: break
        time.sleep(1)
    info("gitrepository_ready", repo_cr_name)

    created_kustoms = []
    for comp in components:
        comp = comp.strip()
        if not comp: continue
        target_dir = MANIFESTS_ROOT / comp
        target_dir.mkdir(parents=True, exist_ok=True)

        hr = render_helmrelease(repo_cr_name, branch, comp, target_namespace=comp)
        ns = render_namespace_yaml(comp)
        sp = render_service_patch(comp, comp)

        (target_dir / "helmrelease.yaml").write_text(hr)
        (target_dir / "namespace.yaml").write_text(ns)
        (target_dir / "service-patch.yaml").write_text(sp)
        info("wrote_manifests", str(target_dir))

        kustom_name = sanitize(repo_cr_name + "-" + comp)
        kustom_yaml = render_kustomization(kustom_name, repo_cr_name, f"./infra/manifests/{comp}", 60)
        ok, detail = kubectl_apply(kustom_yaml)
        if not ok: die("kubectl_apply_failed", detail)
        info("kustomization_created", kustom_name)
        created_kustoms.append(kustom_name)

        if args.auto_push:
            write_and_push(repo, str(target_dir), branch, detect_git_user(), GIT_PAT, f"gen: flux {comp} manifests")

    summary = {"repo": repo, "branch": branch, "gitrepo": repo_cr_name, "kustomizations": created_kustoms}
    print(json.dumps({"summary": summary}, indent=2))

if __name__ == "__main__":
    main()
