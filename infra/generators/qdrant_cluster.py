#!/usr/bin/env python3
from pathlib import Path
import os
import sys
import json
import yaml
import shutil
import subprocess
import hashlib
import uuid
import datetime
from typing import Tuple, List
def load_config():
    env = os.environ.get("ENV", "STAGING").upper()
    cfg = {}
    cfg["ENV"] = env
    cfg["MANIFESTS_DIR"] = Path(os.environ.get("MANIFESTS_DIR", "infra/manifests/qdrant"))
    cfg["ARCHIVE_DIR"] = Path(os.environ.get("ARCHIVE_DIR", "infra/archive/qdrant-helm-chart"))
    cfg["VENDOR_CHART_DIR"] = cfg["ARCHIVE_DIR"] / "qdrant"
    cfg["QDRANT_RELEASE"] = os.environ.get("QDRANT_RELEASE", "qdrant")
    cfg["QDRANT_NAMESPACE"] = os.environ.get("QDRANT_NAMESPACE", "qdrant")
    cfg["QDRANT_IMAGE"] = os.environ.get("QDRANT_IMAGE", "qdrant/qdrant:v1.16.0")
    cfg["CHART_VERSION"] = os.environ.get("CHART_VERSION", "1.16.0")
    cfg["QDRANT_REPLICAS"] = int(os.environ.get("QDRANT_REPLICAS", "2" if env == "STAGING" else "3"))
    cfg["QDRANT_CPU"] = os.environ.get("QDRANT_CPU", "1" if env == "STAGING" else "4")
    cfg["QDRANT_MEMORY"] = os.environ.get("QDRANT_MEMORY", "2Gi" if env == "STAGING" else "16Gi")
    cfg["QDRANT_STORAGE"] = os.environ.get("QDRANT_STORAGE", "emptyDir")
    cfg["QDRANT_NODE_SELECTOR"] = os.environ.get("QDRANT_NODE_SELECTOR", "")
    cfg["QDRANT_TAINT_KEY"] = os.environ.get("QDRANT_TAINT_KEY", "qdrant-dedicated")
    cfg["QDRANT_TAINT_EFFECT"] = os.environ.get("QDRANT_TAINT_EFFECT", "NoSchedule")
    cfg["BACKUP_AZURE_CONTAINER"] = os.environ.get("BACKUP_AZ_CONTAINER", "")
    cfg["BACKUP_AZURE_PREFIX"] = os.environ.get("BACKUP_AZURE_PREFIX", "qdrant/backups")
    cfg["AZURE_STORAGE_CONNECTION_STRING"] = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    cfg["AZURE_STORAGE_ACCOUNT_NAME"] = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME", "")
    cfg["AZURE_STORAGE_ACCOUNT_KEY"] = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY", "")
    cfg["QDRANT__SERVICE__API_KEY"] = os.environ.get("QDRANT__SERVICE__API_KEY", "mypassword")
    cfg["APPLY_STAGING_SECRETS"] = os.environ.get("APPLY_STAGING_SECRETS", "true").lower() in ("1", "true", "yes")
    cfg["TIMEOUT_SECONDS"] = int(os.environ.get("TIMEOUT_SECONDS", "600"))
    cfg["INPUTS_HASH_PATH"] = cfg["MANIFESTS_DIR"] / ".inputs_hash"
    cfg["UUID_SHORT"] = str(uuid.uuid4())[:8]
    cfg["HELM_PRIMARY_REPO"] = os.environ.get("HELM_PRIMARY_REPO", "https://qdrant.github.io/qdrant-helm")
    cfg["HELM_FALLBACK_REPO"] = os.environ.get("HELM_FALLBACK_REPO", "https://qdrant.to/helm")
    cfg["HELM_REPO_NAME"] = os.environ.get("HELM_REPO_NAME", "qdrant")
    cfg["QDRANT_SHARD_NUMBER"] = int(os.environ.get("QDRANT_SHARD_NUMBER", "1"))
    cfg["QDRANT_REPLICATION_FACTOR"] = int(os.environ.get("QDRANT_REPLICATION_FACTOR", os.environ.get("QDRANT_REPLICAS", "1")))
    cfg["QDRANT_WRITE_CONSISTENCY_FACTOR"] = int(os.environ.get("QDRANT_WRITE_CONSISTENCY_FACTOR", "1"))
    cfg["QDRANT_LOG_LEVEL"] = os.environ.get("QDRANT_LOG_LEVEL", "INFO")
    cfg["QDRANT__STORAGE__STORAGE_PATH"] = os.environ.get("QDRANT__STORAGE__STORAGE_PATH", "/qdrant/storage")
    cfg["QDRANT__STORAGE__SNAPSHOTS_PATH"] = os.environ.get("QDRANT__STORAGE__SNAPSHOTS_PATH", "/qdrant/snapshots")
    cfg["USE_LOCAL_NVME"] = os.environ.get("USE_LOCAL_NVME", "false").lower() in ("1", "true", "yes")
    cfg["QDRANT_LOCAL_PATH"] = os.environ.get("QDRANT_LOCAL_PATH", "/mnt/nvme/qdrant")
    cfg["TAINT_QDRANT_NODES"] = os.environ.get("TAINT_QDRANT_NODES", "false").lower() in ("1", "true", "yes")
    cfg["SECRET_BACKUP_NAME"] = os.environ.get("SECRET_BACKUP_NAME", "qdrant-backup-azure")
    cfg["SECRET_SERVICE_NAME"] = os.environ.get("SECRET_SERVICE_NAME", "qdrant-service-creds")
    return cfg
SENSITIVE_KEYS = {"AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_ACCOUNT_KEY", "QDRANT__SERVICE__API_KEY"}
def canonical_inputs_hash(cfg):
    serial = {}
    for k in sorted(cfg.keys()):
        if k == "INPUTS_HASH_PATH":
            continue
        if k in SENSITIVE_KEYS:
            serial[k] = bool(cfg.get(k))
        else:
            v = cfg.get(k)
            try:
                json.dumps(v)
                serial[k] = v
            except Exception:
                serial[k] = str(v)
    j = json.dumps(serial, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode("utf-8")).hexdigest()
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
def atomic_write(path: Path, content: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)
def run_cmd(cmd: List[str], capture=True, check=False, timeout=None, input_bytes=None):
    try:
        proc = subprocess.run(cmd, capture_output=capture, text=True, check=check, timeout=timeout, input=(input_bytes.decode() if isinstance(input_bytes, bytes) else input_bytes))
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout or "", e.stderr or ""
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or f"timeout after {timeout}s"
def detect_storageclass(kubectl_bin="kubectl"):
    if shutil.which(kubectl_bin) is None:
        return None
    cmd = [kubectl_bin, "get", "storageclass", "-o", "jsonpath={.items[?(@.metadata.annotations.storageclass\\.kubernetes\\.io/is-default-class==\"true\")].metadata.name}"]
    rc, out, err = run_cmd(cmd)
    out = out.strip()
    if out:
        return out
    cmd2 = [kubectl_bin, "get", "storageclass", "-o", "jsonpath={.items[0].metadata.name}"]
    rc2, out2, err2 = run_cmd(cmd2)
    return out2.strip() if out2.strip() else None
def ensure_namespace(cfg):
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return False, "kubectl-not-found"
    ns = cfg["QDRANT_NAMESPACE"]
    cmd1 = [kubectl, "create", "namespace", ns, "--dry-run=client", "-o", "yaml"]
    rc, out, err = run_cmd(cmd1, timeout=20)
    if rc != 0:
        rcg, outg, errg = run_cmd([kubectl, "get", "namespace", ns], timeout=20)
        if rcg == 0:
            return True, None
        return False, err or out or outg or errg
    rc2, out2, err2 = run_cmd([kubectl, "apply", "-f", "-"], input_bytes=out.encode("utf-8"), timeout=20)
    if rc2 != 0:
        return False, err2 or out2
    return True, None
def kubectl_create_secret_in_cluster(cfg, secret_name: str, env_keys: List[str]):
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return False, "kubectl-not-found"
    ok_ns, ns_err = ensure_namespace(cfg)
    if not ok_ns:
        return False, f"ensure-namespace-failed: {ns_err}"
    literals = []
    for k in env_keys:
        val = cfg.get(k, os.environ.get(k, ""))
        if val:
            literals += ["--from-literal", f"{k}={val}"]
    if not literals:
        return False, "no-secrets-present-for-this-secret"
    cmd = [kubectl, "create", "secret", "generic", secret_name, "-n", cfg["QDRANT_NAMESPACE"], "--dry-run=client", "-o", "yaml"] + literals
    rc1, out1, err1 = run_cmd(cmd, timeout=20)
    if rc1 != 0:
        return False, err1 or out1
    rc2, out2, err2 = run_cmd([kubectl, "apply", "-f", "-"], input_bytes=out1.encode("utf-8"), timeout=20)
    if rc2 != 0:
        return False, err2 or out2
    return True, None
def kubectl_resource_exists(kind: str, name: str, namespace: str) -> bool:
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return False
    rc, out, err = run_cmd([kubectl, "get", kind, name, "-n", namespace], capture=True)
    return rc == 0
def helm_repo_add_if_missing(cfg, verbose=False):
    helm = shutil.which("helm")
    if not helm:
        return False, "helm-not-found"
    cmd = [helm, "repo", "add", "--force-update", cfg["HELM_REPO_NAME"], cfg["HELM_PRIMARY_REPO"]]
    rc, out, err = run_cmd(cmd, timeout=30)
    if rc != 0:
        rc2, out2, err2 = run_cmd([helm, "repo", "add", cfg["HELM_REPO_NAME"], cfg["HELM_PRIMARY_REPO"]], timeout=30)
        if rc2 != 0:
            return False, err2 or err
    rc_upd, out_upd, err_upd = run_cmd([helm, "repo", "update"], timeout=30)
    return True, None
def vendor_chart_if_missing(cfg, verbose=False):
    vendor_dir = cfg["VENDOR_CHART_DIR"]
    helm = shutil.which("helm")
    if vendor_dir.exists() and (vendor_dir / "Chart.yaml").exists():
        return True, str(vendor_dir)
    if helm is None:
        return False, "helm-not-found"
    ok, err = helm_repo_add_if_missing(cfg, verbose=verbose)
    try:
        cmd = [helm, "pull", f"{cfg['HELM_REPO_NAME']}/qdrant", "--version", cfg["CHART_VERSION"], "--untar", "--untardir", str(cfg["ARCHIVE_DIR"])]
        rc, out, err = run_cmd(cmd, timeout=120)
        if rc == 0:
            if vendor_dir.exists():
                return True, str(vendor_dir)
            else:
                return False, "vendor-dir-missing-after-pull"
        else:
            return False, err or out
    except Exception as e:
        return False, str(e)
def helm_upgrade_install(cfg, values_file: Path, vendor_dir: Path, verbose=False):
    helm = shutil.which("helm")
    if not helm:
        return False, "helm-not-found", "", ""
    release = cfg["QDRANT_RELEASE"]
    ns = cfg["QDRANT_NAMESPACE"]
    timeout = "10m"
    if vendor_dir.exists():
        cmd = [helm, "upgrade", "--install", release, str(vendor_dir), "--namespace", ns, "--create-namespace", "-f", str(values_file), "--wait", "--timeout", timeout]
        rc, out, err = run_cmd(cmd, timeout=600)
        if rc == 0:
            return True, None, out, err
        vendor_err = err or out
    else:
        vendor_err = "vendor-not-present"
    cmd2 = [helm, "upgrade", "--install", release, f"{cfg['HELM_REPO_NAME']}/qdrant", "--version", cfg["CHART_VERSION"], "--namespace", ns, "-f", str(values_file), "--wait", "--timeout", timeout]
    rc2, out2, err2 = run_cmd(cmd2, timeout=600)
    if rc2 == 0:
        return True, None, out2, err2
    cmd3 = [helm, "upgrade", "--install", release, "qdrant/qdrant", "--version", cfg["CHART_VERSION"], "--repo", cfg["HELM_FALLBACK_REPO"], "--namespace", ns, "-f", str(values_file), "--wait", "--timeout", timeout]
    rc3, out3, err3 = run_cmd(cmd3, timeout=600)
    if rc3 == 0:
        return True, None, out3, err3
    cmd4 = [helm, "upgrade", "--install", release, "qdrant/qdrant", "--version", cfg["CHART_VERSION"], "--repo", cfg["HELM_PRIMARY_REPO"], "--namespace", ns, "-f", str(values_file), "--wait", "--timeout", timeout]
    rc4, out4, err4 = run_cmd(cmd4, timeout=600)
    if rc4 == 0:
        return True, None, out4, err4
    combined_err = "\n--- vendor attempt ---\n" + str(vendor_err) + "\n--- primary ---\n" + (err2 or out2) + "\n--- fallback ---\n" + (err3 or out3) + "\n--- retry primary ---\n" + (err4 or out4)
    return False, combined_err, out4, err4
def _kubectl_available():
    return shutil.which("kubectl") is not None
def _get_candidate_nodes(cfg):
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return []
    selector = cfg.get("QDRANT_NODE_SELECTOR", "").strip()
    if selector:
        cmd = [kubectl, "get", "nodes", "-l", selector, "-o", "jsonpath={.items[*].metadata.name}"]
        rc, out, err = run_cmd(cmd)
        names = out.strip().split() if out.strip() else []
        return names
    cmd = [kubectl, "get", "nodes", "-o", "json"]
    rc, out, err = run_cmd(cmd)
    if rc != 0 or not out:
        return []
    try:
        data = json.loads(out)
        names = []
        for it in data.get("items", []):
            labels = it.get("metadata", {}).get("labels", {})
            if "node-role.kubernetes.io/control-plane" in labels or "node-role.kubernetes.io/master" in labels:
                continue
            for cond in it.get("status", {}).get("conditions", []):
                if cond.get("type") == "Ready" and cond.get("status") == "True":
                    names.append(it.get("metadata", {}).get("name"))
                    break
        return names
    except Exception:
        return []
def _node_has_taint(node_name, taint_key, taint_effect):
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return False
    rc, out, err = run_cmd([kubectl, "describe", "node", node_name])
    if rc != 0:
        return False
    for line in out.splitlines():
        if line.strip().startswith("Taints:"):
            if ":" in line and "Taints:" in line and line.strip().endswith("<none>"):
                return False
        if taint_key in line and taint_effect in line:
            return True
    return False
def taint_nodes(cfg):
    if not cfg.get("TAINT_QDRANT_NODES"):
        return True, "TAINT_QDRANT_NODES=false; skipping taint"
    if not _kubectl_available():
        return False, "kubectl-not-found"
    nodes = _get_candidate_nodes(cfg)
    if not nodes:
        return False, "no-candidate-nodes-found"
    for n in nodes:
        if _node_has_taint(n, cfg["QDRANT_TAINT_KEY"], cfg["QDRANT_TAINT_EFFECT"]):
            continue
        cmd = ["kubectl", "taint", "nodes", n, f"{cfg['QDRANT_TAINT_KEY']}={cfg['QDRANT_TAINT_KEY']}:{cfg['QDRANT_TAINT_EFFECT']}"]
        rc, out, err = run_cmd(cmd, timeout=20)
        if rc != 0:
            return False, f"taint-failed: {n}: {err or out}"
    return True, f"tainted {len(nodes)} nodes"
def untaint_nodes(cfg):
    if not _kubectl_available():
        return False, "kubectl-not-found"
    nodes = _get_candidate_nodes(cfg)
    if not nodes:
        return True, "no-candidate-nodes-found"
    for n in nodes:
        if not _node_has_taint(n, cfg["QDRANT_TAINT_KEY"], cfg["QDRANT_TAINT_EFFECT"]):
            continue
        cmd = ["kubectl", "taint", "nodes", n, f"{cfg['QDRANT_TAINT_KEY']}:{cfg['QDRANT_TAINT_EFFECT']}-"]
        rc, out, err = run_cmd(cmd, timeout=20)
        if rc != 0:
            return False, f"untaint-failed: {n}: {err or out}"
    return True, f"untainted {len(nodes)} nodes"
def render_values_yaml(cfg, storage_class):
    repo, tag = cfg["QDRANT_IMAGE"].split(":", 1) if ":" in cfg["QDRANT_IMAGE"] else (cfg["QDRANT_IMAGE"], "latest")
    peers = []
    for i in range(0, cfg["QDRANT_REPLICAS"]):
        peers.append(f"http://{cfg['QDRANT_RELEASE']}-{i}.{cfg['QDRANT_RELEASE']}-headless:6335")
    values = {
        "replicaCount": cfg["QDRANT_REPLICAS"],
        "image": {"repository": repo, "tag": tag, "pullPolicy": "IfNotPresent"},
        "service": {"type": "ClusterIP"},
        "p2p": {"port": 6335},
        "cluster": {"enabled": True, "peers": peers},
        "snapshots": {"enabled": False, "s3": {"bucket": "", "endpoint": "", "region": "", "prefix": ""}},
        "extraEnv": [],
        "resources": {"requests": {"cpu": cfg["QDRANT_CPU"], "memory": cfg["QDRANT_MEMORY"]}, "limits": {"cpu": cfg["QDRANT_CPU"], "memory": cfg["QDRANT_MEMORY"]}},
    }
    backup_secret = cfg.get("SECRET_BACKUP_NAME")
    if backup_secret:
        for k in ("AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_ACCOUNT_KEY"):
            values["extraEnv"].append({"name": k, "valueFrom": {"secretKeyRef": {"name": backup_secret, "key": k}}})
    service_secret = cfg.get("SECRET_SERVICE_NAME")
    if service_secret:
        values["extraEnv"].append({"name": "QDRANT__SERVICE__API_KEY", "valueFrom": {"secretKeyRef": {"name": service_secret, "key": "QDRANT__SERVICE__API_KEY"}}})
    if cfg.get("TAINT_QDRANT_NODES"):
        values["tolerations"] = [{"key": cfg["QDRANT_TAINT_KEY"], "operator": "Exists", "effect": cfg["QDRANT_TAINT_EFFECT"]}]
    else:
        values["tolerations"] = []
    if cfg.get("USE_LOCAL_NVME"):
        values["persistence"] = {"enabled": False}
        host_path = cfg.get("QDRANT_LOCAL_PATH", "/mnt/nvme/qdrant")
        values["extraVolumes"] = [
            {"name": "qdrant-storage", "hostPath": {"path": host_path, "type": "DirectoryOrCreate"}},
            {"name": "qdrant-snapshots", "hostPath": {"path": f"{host_path}/snapshots", "type": "DirectoryOrCreate"}},
        ]
        values["extraVolumeMounts"] = [
            {"name": "qdrant-storage", "mountPath": cfg.get("QDRANT__STORAGE__STORAGE_PATH", "/qdrant/storage")},
            {"name": "qdrant-snapshots", "mountPath": cfg.get("QDRANT__STORAGE__SNAPSHOTS_PATH", "/qdrant/snapshots")},
        ]
        if cfg.get("QDRANT_NODE_SELECTOR"):
            try:
                k, v = cfg.get("QDRANT_NODE_SELECTOR").split("=", 1)
                values["nodeSelector"] = {k: v}
            except Exception:
                values["nodeSelector"] = {}
    else:
        if storage_class:
            values["persistence"] = {"enabled": True, "storageClass": storage_class, "size": "50Gi"}
        else:
            values["persistence"] = {"enabled": False}
            values["extraVolumes"] = [{"name": "qdrant-storage", "emptyDir": {}}]
            values["extraVolumeMounts"] = [{"name": "qdrant-storage", "mountPath": cfg.get("QDRANT__STORAGE__STORAGE_PATH", "/qdrant/storage")}]
    values["config"] = {
        "params": {
            "shard_number": int(cfg.get("QDRANT_SHARD_NUMBER", 1)),
            "replication_factor": int(cfg.get("QDRANT_REPLICATION_FACTOR", cfg.get("QDRANT_REPLICAS", 1))),
            "write_consistency_factor": int(cfg.get("QDRANT_WRITE_CONSISTENCY_FACTOR", 1)),
        },
        "log_level": cfg.get("QDRANT_LOG_LEVEL", "INFO"),
        "storage": {"storage_path": cfg.get("QDRANT__STORAGE__STORAGE_PATH", "/qdrant/storage"), "snapshots_path": cfg.get("QDRANT__STORAGE__SNAPSHOTS_PATH", "/qdrant/snapshots")},
    }
    return yaml.safe_dump(values, sort_keys=False)
def generate_manifests(cfg, dry_run=False, verbose=False):
    ensure_dir(cfg["MANIFESTS_DIR"])
    inputs_hash = canonical_inputs_hash(cfg)
    existing = None
    if cfg["INPUTS_HASH_PATH"].exists():
        existing = cfg["INPUTS_HASH_PATH"].read_text().strip()
    if existing == inputs_hash and not dry_run:
        print("No non-secret changes detected; generation skipped.")
        return
    storage_class = detect_storageclass() or None
    values_yaml = render_values_yaml(cfg, storage_class)
    atomic_write(cfg["MANIFESTS_DIR"] / "values.yaml", values_yaml)
    samples_dir = cfg["MANIFESTS_DIR"] / "_samples"
    ensure_dir(samples_dir)
    sample_secret_placeholder = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": cfg["SECRET_BACKUP_NAME"], "namespace": cfg["QDRANT_NAMESPACE"]},
        "type": "Opaque",
        "stringData": {"AZURE_STORAGE_CONNECTION_STRING": "REPLACE_ME", "AZURE_STORAGE_ACCOUNT_NAME": "REPLACE_ME", "AZURE_STORAGE_ACCOUNT_KEY": "REPLACE_ME"},
    }
    atomic_write(samples_dir / "secret-sample.placeholder.yaml", yaml.safe_dump(sample_secret_placeholder, sort_keys=False))
    cfg["INPUTS_HASH_PATH"].write_text(inputs_hash)
    print("Wrote manifests to", str(cfg["MANIFESTS_DIR"]))
    if verbose:
        print("--- values.yaml preview ---")
        for line in values_yaml.splitlines()[:200]:
            print(line)
    return
def apply_to_cluster(cfg, dry_run=False, verbose=False):
    kubectl = shutil.which("kubectl")
    helm = shutil.which("helm")
    if kubectl is None or helm is None:
        print("ERROR: kubectl and helm are required in PATH to apply to cluster.", file=sys.stderr)
        sys.exit(2)
    if cfg["ENV"] == "STAGING" and cfg["APPLY_STAGING_SECRETS"]:
        if not cfg.get("AZURE_STORAGE_CONNECTION_STRING") and not (cfg.get("AZURE_STORAGE_ACCOUNT_NAME") and cfg.get("AZURE_STORAGE_ACCOUNT_KEY")):
            print("ERROR: ENV=STAGING requires AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME+AZURE_STORAGE_ACCOUNT_KEY set when using --apply.", file=sys.stderr)
            sys.exit(2)
    generate_manifests(cfg, dry_run=dry_run, verbose=verbose)
    if dry_run:
        print("Dry-run mode: skipping kubectl/helm apply.")
        return
    ok_ns, ns_err = ensure_namespace(cfg)
    if not ok_ns:
        print("ERROR: failed to ensure namespace:", ns_err, file=sys.stderr)
        sys.exit(2)
    print("Namespace ensured:", cfg["QDRANT_NAMESPACE"])
    created_any_secret = False
    if cfg["ENV"] == "STAGING" and cfg["APPLY_STAGING_SECRETS"]:
        ok_s, err_s = kubectl_create_secret_in_cluster(cfg, cfg["SECRET_BACKUP_NAME"], ["AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_ACCOUNT_KEY"])
        if not ok_s:
            print("ERROR: failed to create/update backup secret in-cluster:", err_s, file=sys.stderr)
            sys.exit(2)
        print("Created/updated in-cluster secret:", cfg["SECRET_BACKUP_NAME"])
        created_any_secret = True
    else:
        if cfg["ENV"] == "PROD":
            print("ENV=PROD: skipping secret creation (expect managed identity).")
        else:
            print("Skipping in-cluster backup secret creation (APPLY_STAGING_SECRETS=false).")
    if cfg.get("QDRANT__SERVICE__API_KEY"):
        ok_srv, err_srv = kubectl_create_secret_in_cluster(cfg, cfg["SECRET_SERVICE_NAME"], ["QDRANT__SERVICE__API_KEY"])
        if not ok_srv:
            print("ERROR: failed to create/update service secret in-cluster:", err_srv, file=sys.stderr)
            sys.exit(2)
        print("Created/updated in-cluster secret:", cfg["SECRET_SERVICE_NAME"])
        created_any_secret = True
    if cfg.get("TAINT_QDRANT_NODES"):
        ok_t, msg_t = taint_nodes(cfg)
        if not ok_t:
            print("ERROR: failed to taint nodes:", msg_t, file=sys.stderr)
            sys.exit(2)
        print("Tainting result:", msg_t)
    else:
        ok_u, msg_u = untaint_nodes(cfg)
        if not ok_u and msg_u != "kubectl-not-found":
            print("Warning: failed to untaint nodes:", msg_u)
        else:
            if msg_u:
                print("Untainting result:", msg_u)
    v_ok, v_err = vendor_chart_if_missing(cfg, verbose=verbose)
    if not v_ok:
        print("Vendor chart not available locally; will attempt remote install. vendor error:", v_err)
    else:
        print("Vendor chart available at", cfg["VENDOR_CHART_DIR"])
    values_file = cfg["MANIFESTS_DIR"] / "values.yaml"
    ok, errtext, stdout_text, stderr_text = helm_upgrade_install(cfg, values_file, cfg["VENDOR_CHART_DIR"], verbose=verbose)
    if not ok:
        print("ERROR: helm upgrade/install failed. See summary below.", file=sys.stderr)
        if verbose:
            print("--- helm error (tail) ---")
            for line in (errtext or "").splitlines()[-200:]:
                print(line)
        print(errtext, file=sys.stderr)
        sys.exit(2)
    print("Helm install/upgrade succeeded for release:", cfg["QDRANT_RELEASE"])
    summary = {
        "release": cfg["QDRANT_RELEASE"],
        "namespace": cfg["QDRANT_NAMESPACE"],
        "replicas": cfg["QDRANT_REPLICAS"],
        "values_file": str(values_file),
        "chart_version": cfg["CHART_VERSION"],
        "image": cfg["QDRANT_IMAGE"],
        "vendor_chart_dir": str(cfg["VENDOR_CHART_DIR"]) if cfg["VENDOR_CHART_DIR"].exists() else None,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "secrets_created": created_any_secret,
    }
    atomic_write(cfg["MANIFESTS_DIR"] / "last_deploy_summary.json", json.dumps(summary, indent=2))
    print("Wrote deploy summary ->", str(cfg["MANIFESTS_DIR"] / "last_deploy_summary.json"))
    return
def delete_manifests(cfg):
    if cfg["MANIFESTS_DIR"].exists():
        for p in sorted(cfg["MANIFESTS_DIR"].glob("*")):
            try:
                p.unlink()
            except IsADirectoryError:
                shutil.rmtree(p)
        try:
            cfg["INPUTS_HASH_PATH"].unlink()
        except FileNotFoundError:
            pass
        print("Deleted manifests at", str(cfg["MANIFESTS_DIR"]))
    else:
        print("Manifests dir not present:", str(cfg["MANIFESTS_DIR"]))
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Generate/apply Qdrant Helm manifests (cluster-aware, Azure-native backup placeholders).")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--generate", action="store_true", help="Generate manifests to MANIFESTS_DIR.")
    grp.add_argument("--apply", action="store_true", help="Generate manifests and apply (create staging secret(s) + helm install).")
    grp.add_argument("--delete", action="store_true", help="Delete generated manifests and inputs hash.")
    p.add_argument("--dry-run", action="store_true", help="Render and validate but do not write or apply.")
    p.add_argument("--verbose", action="store_true", help="Print extra debug info (helm output tails on failure).")
    return p.parse_args()
def main():
    args = parse_args()
    cfg = load_config()
    if args.apply:
        if shutil.which("kubectl") is None or shutil.which("helm") is None:
            print("ERROR: To --apply you must have kubectl and helm installed and configured in PATH.", file=sys.stderr)
            sys.exit(2)
    if args.delete:
        delete_manifests(cfg)
        return
    if args.generate:
        generate_manifests(cfg, dry_run=args.dry_run, verbose=args.verbose)
        return
    if args.apply:
        apply_to_cluster(cfg, dry_run=args.dry_run, verbose=args.verbose)
        return
if __name__ == "__main__":
    main()
