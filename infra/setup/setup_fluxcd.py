from __future__ import annotations
import os, sys, subprocess, json, time, hashlib, re, shutil
from pathlib import Path

try:
    import yaml
except Exception:
    print(json.dumps({"error": "PyYAML required. Install with: pip install pyyaml"}))
    sys.exit(2)

ROOT = Path.cwd()
MANIFESTS_ROOT = ROOT / "infra" / "manifests"
ARCHIVE_DIR = ROOT / "infra" / "archive"
FLUX_NS = os.environ.get("FLUX_NS", "flux-system")
FLUX_VERSION = os.environ.get("FLUX_VERSION", "v2.7.5")
GIT_PAT = os.environ.get("GIT_PAT", "").strip()
WAIT_CONTROLLER_SECONDS = int(os.environ.get("WAIT_CONTROLLER_SECONDS", "180"))
WAIT_CRDS_SECONDS = int(os.environ.get("WAIT_CRDS_SECONDS", "180"))

def run(cmd, input=None, timeout=None, check=True):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, input=input, timeout=timeout)
        if check and p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout, stderr=p.stderr)
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0] if cmd else ''}"
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", "") or "", "timeout"

def info(k, v=None):
    if v is None:
        print(json.dumps({"info": k}))
    else:
        print(json.dumps({"info": k, "value": v}))

def die(msg, extra=None):
    out = {"error": msg}
    if extra: out["detail"] = extra
    print(json.dumps(out))
    sys.exit(1)

def sanitize(name: str) -> str:
    return re.sub(r'[^a-z0-9-]', '-', name.lower()).strip('-')[:63]

def check_bins(required=("kubectl", "git")):
    missing = []
    for b in required:
        if shutil.which(b) is None:
            missing.append(b)
    if missing:
        die("missing_binary:" + ",".join(missing))

def ensure_namespace(ns: str):
    rc, out, err = run(["kubectl", "get", "ns", ns, "-o", "name"], timeout=8, check=False)
    if rc == 0 and out:
        info("namespace_exists", ns); return
    rc, out, err = run(["kubectl", "create", "ns", ns, "--dry-run=client", "-o", "yaml"], timeout=8, check=False)
    if rc != 0:
        die("namespace_create_dryrun_failed", err or out)
    rc2, out2, err2 = run(["kubectl", "apply", "-f", "-"], input=out, timeout=8, check=False)
    if rc2 != 0:
        die("namespace_create_failed", err2 or out2)
    info("namespace_created", ns)

def install_flux(flux_ns=FLUX_NS, flux_version=FLUX_VERSION):
    # prefer flux CLI if present
    if shutil.which("flux"):
        info("install_method", "flux-cli")
        rc, out, err = run(["flux", "install", "--namespace", flux_ns], timeout=300, check=False)
        if rc == 0:
            info("flux_install_ok", flux_ns); return True
        info("flux_cli_failed", err or out)
    # fallback to applying install manifest
    info("install_method", "manifest-apply")
    manifest_url = f"https://github.com/fluxcd/flux2/releases/download/{flux_version}/install.yaml"
    rc, out, err = run(["kubectl", "apply", "-f", manifest_url], timeout=120, check=False)
    if rc != 0:
        die("flux_install_manifest_failed", err or out)
    info("flux_install_applied", flux_version)
    return True

def wait_for_crds(names, timeout_seconds=WAIT_CRDS_SECONDS):
    deadline = time.time() + timeout_seconds
    missing = set(names)
    while time.time() < deadline:
        for n in list(missing):
            rc, out, err = run(["kubectl", "get", "crd", n], timeout=8, check=False)
            if rc == 0:
                missing.discard(n)
        if not missing:
            return True, []
        time.sleep(2)
    return False, sorted(list(missing))

def wait_for_controllers(namespace=FLUX_NS, timeout_seconds=WAIT_CONTROLLER_SECONDS):
    controllers = ["source-controller", "kustomize-controller", "helm-controller", "notification-controller"]
    start = time.time()
    while (time.time() - start) < timeout_seconds:
        ready = True
        for c in controllers:
            rc, out, err = run(["kubectl", "-n", namespace, "get", "deploy", c, "-o", "jsonpath={.status.availableReplicas}"], timeout=8, check=False)
            if rc != 0 or not out or out == "0":
                ready = False
        if ready:
            info("flux_controllers_ready"); return True
        time.sleep(2)
    info("flux_controllers_unready"); return False

def kubectl_apply(yaml_str, redact=False, timeout=30):
    rc, out, err = run(["kubectl", "apply", "-f", "-"], input=yaml_str, timeout=timeout, check=False)
    if rc != 0:
        return False, (err or out)
    if redact or "stringData" in (yaml_str or "") or "password" in (yaml_str or ""):
        info("kubectl_apply_ok", "redacted")
    else:
        info("kubectl_apply_ok", yaml_str.splitlines()[0] if yaml_str else "")
    return True, ""

def detect_repo():
    env = os.environ.get("REPO_URL", "").strip()
    if env: return env
    rc, out, err = run(["git", "remote", "get-url", "origin"], timeout=8, check=False)
    if rc == 0 and out: return out.strip()
    rc, out, err = run(["git", "remote", "-v"], timeout=8, check=False)
    if rc == 0 and out:
        first = out.splitlines()[0].split()[1]; return first
    die("no_repo_detected")

def detect_branch():
    e = os.environ.get("BRANCH") or os.environ.get("TARGET_REVISION")
    if e: return e
    rc, out, err = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=8, check=False)
    if rc == 0 and out: return out.strip()
    return "main"

def create_repo_secret(secret_name):
    if not GIT_PAT:
        return None
    secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": secret_name, "namespace": FLUX_NS},
        "stringData": {"username": detect_git_user(), "password": GIT_PAT}
    }
    ok, detail = kubectl_apply(yaml.safe_dump(secret), redact=True)
    if not ok: die("kubectl_apply_failed", detail)
    info("repo_secret_created", secret_name)
    return secret_name

def detect_git_user():
    rc, out, err = run(["git", "config", "--get", "user.name"], timeout=4, check=False)
    if rc == 0 and out: return out.strip()
    rc, out, err = run(["git", "config", "--get", "user.email"], timeout=4, check=False)
    if rc == 0 and out: return out.split("@",1)[0]
    return "git"

def render_qdrant_if_possible():
    qdir = MANIFESTS_ROOT / "qdrant"
    chart_dir = ARCHIVE_DIR / "qdrant-helm-chart" / "qdrant"
    values = qdir / "values.yaml"
    outp = qdir / "rendered" / "qdrant.yaml"
    if chart_dir.exists() and values.exists():
        if shutil.which("helm") is None:
            die("helm_missing_for_qdrant_render")
        outp.parent.mkdir(parents=True, exist_ok=True)
        rc, out, err = run(["helm", "template", "qdrant", str(chart_dir), "-n", "qdrant", "-f", str(values), "--include-crds"], timeout=60, check=False)
        if rc != 0:
            die("helm_template_failed", err or out)
        outp.write_text(out)
        info("qdrant_rendered", str(outp))
    else:
        info("qdrant_render_skipped", f"chart_exists={chart_dir.exists()}, values_exists={values.exists()}")

def render_and_apply_gitrepo(repo, branch, secret_name=None, interval_s=60):
    name = sanitize(re.sub(r'https?://', '', repo).rstrip('/').replace('/', '-'))
    if not name.endswith("-git"): name = name + "-git"
    obj = {
        "apiVersion": "source.toolkit.fluxcd.io/v1",
        "kind": "GitRepository",
        "metadata": {"name": name, "namespace": FLUX_NS},
        "spec": {"interval": f"{interval_s}s", "url": repo, "ref": {"branch": branch}}
    }
    if secret_name:
        obj["spec"]["secretRef"] = {"name": secret_name}
    ok, detail = kubectl_apply(yaml.safe_dump(obj))
    if not ok: die("kubectl_apply_failed", detail)
    info("gitrepository_created", name)
    return name

def list_manifest_components():
    if not MANIFESTS_ROOT.exists():
        die("manifests_root_missing", str(MANIFESTS_ROOT))
    comps = []
    for p in sorted(MANIFESTS_ROOT.iterdir()):
        if p.is_dir() and p.name != "_samples":
            comps.append(p.name)
    return comps

def render_kustomization(repo_cr_name, comp_name, path, target_namespace=None, depends_on=None, interval_s=60, wait=True):
    if target_namespace is None: target_namespace = comp_name
    spec = {
        "interval": f"{interval_s}s",
        "timeout": "120s",
        "prune": True,
        "wait": wait,
        "sourceRef": {"kind": "GitRepository", "name": repo_cr_name, "namespace": FLUX_NS},
        "path": path,
        "targetNamespace": target_namespace
    }
    if depends_on:
        spec["dependsOn"] = depends_on
    obj = {"apiVersion": "kustomize.toolkit.fluxcd.io/v1", "kind": "Kustomization",
           "metadata": {"name": sanitize(repo_cr_name + "-" + comp_name), "namespace": FLUX_NS},
           "spec": spec}
    return yaml.safe_dump(obj, sort_keys=False)

def create_kustomizations_for_all(repo_cr_name, components):
    created = []
    # Ensure qdrant first
    ordered = []
    if "qdrant" in components:
        ordered.append("qdrant")
    for c in components:
        if c != "qdrant": ordered.append(c)
    q_kname = sanitize(repo_cr_name + "-qdrant")
    for comp in ordered:
        # skip empty or invalid
        if not comp: continue
        target_dir = MANIFESTS_ROOT / comp
        if not target_dir.exists():
            # create placeholder dir so Flux path exists
            target_dir.mkdir(parents=True, exist_ok=True)
            info("created_placeholder_manifest_dir", str(target_dir))
        depends = None
        if comp != "qdrant" and "qdrant" in components:
            depends = [{"name": q_kname, "namespace": FLUX_NS}]
        ktxt = render_kustomization(repo_cr_name, comp, f"./{str(MANIFESTS_ROOT.relative_to(ROOT))}/{comp}", target_namespace=(comp if comp!="qdrant" else "qdrant"), depends_on=depends)
        ok, detail = kubectl_apply(ktxt)
        if not ok:
            die("kubectl_apply_failed", detail)
        info("kustomization_created", sanitize(repo_cr_name + "-" + comp))
        created.append(sanitize(repo_cr_name + "-" + comp))
    return created

def git_auto_push(path_list, branch):
    # commit and push given list of paths; requires git credentials to be configured
    try:
        run(["git", "add"] + path_list)
        rc, out, err = run(["git", "commit", "-m", "gen: flux manifests (auto-push)"], check=False)
        if rc != 0 and "nothing to commit" in (out+err).lower():
            info("git_nothing_to_commit"); return
        elif rc != 0:
            die("git_commit_failed", out+err)
        rc, out, err = run(["git", "push", "origin", branch], check=False)
        if rc != 0:
            die("git_push_failed", err or out)
        info("git_push_ok")
    except Exception as e:
        die("git_auto_push_failed", str(e))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bootstrap Flux and create Kustomizations for all infra/manifests/* dirs (qdrant-first).")
    parser.add_argument("--auto-push", action="store_true", help="Commit & push generated files (requires git credentials).")
    args = parser.parse_args()

    check_bins(("kubectl", "git"))
    repo = detect_repo(); branch = detect_branch()
    info("repo", repo); info("branch", branch)
    ensure_namespace(FLUX_NS)

    install_flux(FLUX_NS, FLUX_VERSION)

    required_crds = [
        "gitrepositories.source.toolkit.fluxcd.io",
        "kustomizations.kustomize.toolkit.fluxcd.io",
        "helmreleases.helm.toolkit.fluxcd.io",
    ]
    ok, missing = wait_for_crds(required_crds, timeout_seconds=WAIT_CRDS_SECONDS)
    if not ok:
        die("flux_crds_missing_after_install", missing)
    info("flux_crds_present")

    wait_for_controllers(FLUX_NS, timeout_seconds=WAIT_CONTROLLER_SECONDS)

    # render qdrant if possible (safe: only creates rendered files under infra/manifests/qdrant)
    render_qdrant_if_possible()

    # If auto-push: stage any generated files under infra/manifests/*
    if args.auto_push:
        # we will add infra/manifests/* (safe)
        paths = ["infra/manifests"]
        git_auto_push(paths, branch)

    # create a GitRepository object
    secret_name = None
    if GIT_PAT:
        secret_name = "repo-" + hashlib.sha1(repo.encode()).hexdigest()[:8]
        create_repo_secret(secret_name)
    repo_cr_name = sanitize(re.sub(r'https?://', '', repo).rstrip('/').replace('/', '-'))
    if not repo_cr_name.endswith("-git"): repo_cr_name = repo_cr_name + "-git"
    render_and_apply_gitrepo(repo, branch, secret_name)

    # detect components and create kustomizations (qdrant-first)
    components = list_manifest_components()
    created = create_kustomizations_for_all(repo_cr_name, components)

    print(json.dumps({"summary": {"repo": repo, "branch": branch, "gitrepository": repo_cr_name, "kustomizations": created}}, indent=2))

if __name__ == "__main__":
    main()
