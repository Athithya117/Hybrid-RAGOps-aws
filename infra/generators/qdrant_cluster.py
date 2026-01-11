#!/usr/bin/env python3
import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path

try:
    import yaml
except Exception:
    print("ERROR: PyYAML required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

def LOG(*parts):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), *parts, flush=True)

def DBG(*parts):
    if os.environ.get("VERBOSE", "0") != "0":
        LOG(*parts)

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFESTS_DIR = Path(os.environ.get("MANIFESTS_DIR", "infra/manifests/qdrant"))
VALUES_FILE = MANIFESTS_DIR / "values.yaml"
SERVICE_PATCH_FILE = MANIFESTS_DIR / "service-patch.yaml"
SAMPLES_DIR = MANIFESTS_DIR / "_samples"
LAST_SUMMARY = MANIFESTS_DIR / "last_deploy_summary.json"

ENV = os.environ.get("ENV", "PROD")
QDRANT_RELEASE = os.environ.get("QDRANT_RELEASE", "qdrant")
QDRANT_NAMESPACE = os.environ.get("QDRANT_NAMESPACE", "qdrant")
QDRANT_IMAGE = os.environ.get("QDRANT_IMAGE", "qdrant/qdrant:v1.16.0")
CHART_VERSION = os.environ.get("CHART_VERSION", "1.16.0")
QDRANT_REPLICAS = int(os.environ.get("QDRANT_REPLICAS", "1"))
QDRANT_METRICS_PORT = int(os.environ.get("QDRANT_METRICS_PORT", "6333"))
QDRANT_METRICS_PORT_NAME = os.environ.get("QDRANT_METRICS_PORT_NAME", "metrics")

QDRANT_ONDISK = os.environ.get("QDRANT_ONDISK", "false").lower() in ("1","true","yes","y")
QDRANT_PERSISTENCE_ENABLED = os.environ.get("QDRANT_PERSISTENCE_ENABLED", "false").lower() in ("1","true","yes","y")
QDRANT_PERSISTENCE_SIZE = os.environ.get("QDRANT_PERSISTENCE_SIZE", "20Gi")
QDRANT_PERSISTENCE_STORAGE_CLASS = os.environ.get("QDRANT_PERSISTENCE_STORAGE_CLASS", "")

SECRET_BACKUP_NAME = os.environ.get("SECRET_BACKUP_NAME", "")
SECRET_SERVICE_NAME = os.environ.get("SECRET_SERVICE_NAME", "qdrant-service-creds")
QDRANT__SERVICE__API_KEY = os.environ.get("QDRANT__SERVICE__API_KEY", os.environ.get("QDRANT_API_KEY", ""))

FAIL_ON_MISCONFIG = os.environ.get("FAIL_ON_MISCONFIG", "false").lower() in ("1","true","yes","y")
SERVICE_VALIDATION_WAIT = int(os.environ.get("SERVICE_VALIDATION_WAIT", "120"))
VENDOR_CHART_DIR = os.environ.get("VENDOR_CHART_DIR", "infra/archive/qdrant-helm-chart/qdrant")
HELM_REPO_NAME = os.environ.get("HELM_REPO_NAME", "qdrant")
HELM_PRIMARY_REPO = os.environ.get("HELM_PRIMARY_REPO", "https://qdrant.github.io/qdrant-helm")
HELM_FALLBACK_REPO = os.environ.get("HELM_FALLBACK_REPO", "https://qdrant.to/helm")
APPLY_STAGING_SECRETS = os.environ.get("APPLY_STAGING_SECRETS", "true").lower() in ("1","true","yes","y")

QDRANT_CPU_REQUEST = os.environ.get("QDRANT_CPU_REQUEST")
QDRANT_CPU_LIMIT = os.environ.get("QDRANT_CPU_LIMIT")
QDRANT_MEMORY_REQUEST = os.environ.get("QDRANT_MEMORY_REQUEST")
QDRANT_MEMORY_LIMIT = os.environ.get("QDRANT_MEMORY_LIMIT")
LEGACY_QDRANT_CPU = os.environ.get("QDRANT_CPU")
LEGACY_QDRANT_MEMORY = os.environ.get("QDRANT_MEMORY")

AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING","")
AZURE_STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME","")
AZURE_STORAGE_ACCOUNT_KEY = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY","")

TMP_FILES = []

def atomic_write(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    os.close(fd)
    TMP_FILES.append(tmp)
    with open(tmp, "wb") as f:
        f.write(content)
    os.replace(tmp, str(path))
    try:
        TMP_FILES.remove(tmp)
    except Exception:
        pass

def run(cmd, check=True, capture=False, text=True):
    DBG("run:", " ".join(cmd) if isinstance(cmd, (list,tuple)) else str(cmd))
    try:
        if capture:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=text)
            return out
        else:
            subprocess.check_call(cmd)
            return None
    except subprocess.CalledProcessError as e:
        LOG("Command failed:", e)
        if capture:
            return e.output
        if check:
            raise
        return None

def parse_cpu(v):
    if v is None:
        return None
    s = str(v).strip()
    if s.endswith("m"):
        try:
            return float(s[:-1]) / 1000.0
        except:
            return None
    try:
        return float(s)
    except:
        return None

def parse_mem_mib(v):
    if v is None:
        return None
    s = str(v).strip()
    sl = s.lower()
    if sl.endswith("gi"):
        try:
            num = float(s[:-2])
            return int(num * 1024)
        except:
            return None
    if sl.endswith("g"):
        try:
            num = float(s[:-1])
            return int(num * 1024)
        except:
            return None
    if sl.endswith("mi"):
        try:
            num = float(s[:-2])
            return int(num)
        except:
            return None
    if sl.endswith("m"):
        try:
            num = float(s[:-1])
            return int(num)
        except:
            return None
    try:
        return int(float(s))
    except:
        return None

def validate_replication_constraints():
    if QDRANT_REPLICAS < 1:
        LOG("ERROR: QDRANT_REPLICAS must be >= 1")
        sys.exit(2)

def validate_resource_contract():
    cpu_req = QDRANT_CPU_REQUEST or LEGACY_QDRANT_CPU or "1"
    cpu_lim = QDRANT_CPU_LIMIT or LEGACY_QDRANT_CPU or cpu_req
    mem_req = QDRANT_MEMORY_REQUEST or LEGACY_QDRANT_MEMORY or "2Gi"
    mem_lim = QDRANT_MEMORY_LIMIT or LEGACY_QDRANT_MEMORY or mem_req

    cpu_req_f = parse_cpu(cpu_req)
    cpu_lim_f = parse_cpu(cpu_lim)
    mem_req_mib = parse_mem_mib(mem_req)
    mem_lim_mib = parse_mem_mib(mem_lim)

    if cpu_req_f is None or cpu_lim_f is None:
        LOG("ERROR: invalid CPU request/limit:", cpu_req, cpu_lim)
        sys.exit(2)
    if cpu_req_f > cpu_lim_f:
        LOG("ERROR: CPU request cannot exceed CPU limit")
        sys.exit(2)
    if mem_req_mib is None or mem_lim_mib is None:
        LOG("ERROR: invalid Memory request/limit:", mem_req, mem_lim)
        sys.exit(2)
    if mem_req_mib > mem_lim_mib:
        LOG("ERROR: Memory request cannot exceed Memory limit")
        sys.exit(2)

def render_values_yaml():
    cpu_req = QDRANT_CPU_REQUEST or LEGACY_QDRANT_CPU or "1"
    cpu_lim = QDRANT_CPU_LIMIT or LEGACY_QDRANT_CPU or cpu_req
    mem_req = QDRANT_MEMORY_REQUEST or LEGACY_QDRANT_MEMORY or "2Gi"
    mem_lim = QDRANT_MEMORY_LIMIT or LEGACY_QDRANT_MEMORY or mem_req

    repo_tag = QDRANT_IMAGE
    if ":" in repo_tag:
        repo, tag = repo_tag.split(":",1)
    else:
        repo, tag = repo_tag, "latest"

    peers = [f"http://{QDRANT_RELEASE}-{i}.{QDRANT_RELEASE}-headless:6335" for i in range(QDRANT_REPLICAS)]

    vals = {
        "replicaCount": QDRANT_REPLICAS,
        "image": {"repository": repo, "tag": tag, "pullPolicy": "IfNotPresent"},
        "service": {"type": "ClusterIP", "labels": {"app.kubernetes.io/name": QDRANT_RELEASE, "app.kubernetes.io/component": "qdrant"}},
        "podAnnotations": {"monitoring.io/scrape": "true", "monitoring.io/port": str(QDRANT_METRICS_PORT), "monitoring.io/path": "/metrics"},
        "ports": [{"name": QDRANT_METRICS_PORT_NAME, "containerPort": QDRANT_METRICS_PORT, "protocol": "TCP"}],
        "p2p": {"port": 6335},
        "cluster": {"enabled": True, "peers": peers},
        "snapshots": {"enabled": False, "s3": {"bucket": "", "endpoint": "", "region": "", "prefix": ""}},
        "extraEnv": [],
        "resources": {"requests": {"cpu": cpu_req, "memory": mem_req}, "limits": {"cpu": cpu_lim, "memory": mem_lim}},
        "tolerations": [],
        "persistence": {"enabled": bool(QDRANT_PERSISTENCE_ENABLED), "size": QDRANT_PERSISTENCE_SIZE, "storageClass": QDRANT_PERSISTENCE_STORAGE_CLASS or ""},
        "config": {"on_disk_payload": QDRANT_ONDISK, "log_level": os.environ.get("QDRANT_LOG_LEVEL", "INFO"), "storage": {"storage_path": os.environ.get("QDRANT__STORAGE__STORAGE_PATH", "/qdrant/storage"), "snapshots_path": os.environ.get("QDRANT__STORAGE__SNAPSHOTS_PATH", "/qdrant/snapshots")}},
        "hnsw_defaults": {"on_disk": QDRANT_ONDISK}
    }

    extra_env = []
    if SECRET_BACKUP_NAME:
        for key in ("AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_ACCOUNT_KEY"):
            extra_env.append({"name": key, "valueFrom": {"secretKeyRef": {"name": SECRET_BACKUP_NAME, "key": key}}})
    if QDRANT__SERVICE__API_KEY and SECRET_SERVICE_NAME:
        extra_env.append({"name": "QDRANT__SERVICE__API_KEY", "valueFrom": {"secretKeyRef": {"name": SECRET_SERVICE_NAME, "key": "QDRANT__SERVICE__API_KEY"}}})
    if extra_env:
        vals["extraEnv"] = extra_env

    content = yaml.safe_dump(vals, sort_keys=False).encode("utf-8")
    atomic_write(VALUES_FILE, content)
    LOG("Rendered", str(VALUES_FILE))

def render_service_patch():
    svc = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": QDRANT_RELEASE, "namespace": QDRANT_NAMESPACE, "labels": {"app.kubernetes.io/name": QDRANT_RELEASE, "app.kubernetes.io/component": "qdrant"}},
        "spec": {"selector": {"app.kubernetes.io/name": QDRANT_RELEASE}, "ports": [{"name": QDRANT_METRICS_PORT_NAME, "port": QDRANT_METRICS_PORT, "targetPort": QDRANT_METRICS_PORT, "protocol": "TCP"}], "type": "ClusterIP"}
    }
    content = yaml.safe_dump(svc, sort_keys=False).encode("utf-8")
    atomic_write(SERVICE_PATCH_FILE, content)
    LOG("Rendered", str(SERVICE_PATCH_FILE))

def generate_samples():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    if SECRET_BACKUP_NAME:
        secret_sample = {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": SECRET_BACKUP_NAME, "namespace": QDRANT_NAMESPACE}, "type": "Opaque", "stringData": {"AZURE_STORAGE_CONNECTION_STRING": "REPLACE_ME", "AZURE_STORAGE_ACCOUNT_NAME": "REPLACE_ME", "AZURE_STORAGE_ACCOUNT_KEY": "REPLACE_ME"}}
        atomic_write(SAMPLES_DIR / "secret-sample.placeholder.yaml", yaml.safe_dump(secret_sample).encode("utf-8"))
    if SECRET_SERVICE_NAME:
        svc_secret = {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": SECRET_SERVICE_NAME, "namespace": QDRANT_NAMESPACE}, "type": "Opaque", "stringData": {"QDRANT__SERVICE__API_KEY": "REPLACE_ME"}}
        atomic_write(SAMPLES_DIR / "service-secret-sample.placeholder.yaml", yaml.safe_dump(svc_secret).encode("utf-8"))

def ensure_namespace():
    run(["kubectl", "create", "namespace", QDRANT_NAMESPACE, "--dry-run=client", "-o", "yaml"], check=False)
    run(["kubectl", "create", "namespace", QDRANT_NAMESPACE, "--dry-run=client", "-o", "yaml"], check=False)

def create_azure_backup_secret():
    if not SECRET_BACKUP_NAME:
        LOG("no SECRET_BACKUP_NAME configured; skipping backup secret creation")
        return True
    run(["kubectl", "create", "namespace", QDRANT_NAMESPACE, "--dry-run=client", "-o", "yaml"], check=False)
    if AZURE_STORAGE_CONNECTION_STRING:
        run(["bash", "-c", f"kubectl -n {QDRANT_NAMESPACE} create secret generic {SECRET_BACKUP_NAME} --from-literal=AZURE_STORAGE_CONNECTION_STRING='{AZURE_STORAGE_CONNECTION_STRING}' --dry-run=client -o yaml | kubectl apply -f -"], check=True)
        LOG("created/updated secret", SECRET_BACKUP_NAME)
        return True
    if AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY:
        run(["bash", "-c", f"kubectl -n {QDRANT_NAMESPACE} create secret generic {SECRET_BACKUP_NAME} --from-literal=AZURE_STORAGE_ACCOUNT_NAME='{AZURE_STORAGE_ACCOUNT_NAME}' --from-literal=AZURE_STORAGE_ACCOUNT_KEY='{AZURE_STORAGE_ACCOUNT_KEY}' --dry-run=client -o yaml | kubectl apply -f -"], check=True)
        LOG("created/updated secret", SECRET_BACKUP_NAME)
        return True
    LOG("ERROR: Azure storage credentials missing; cannot create backup secret")
    return False

def helm_repo_add_if_missing():
    try:
        run(["helm", "repo", "add", "--force-update", HELM_REPO_NAME, HELM_PRIMARY_REPO], check=False)
        run(["helm", "repo", "update"], check=False)
        return True
    except Exception:
        return False

def helm_upgrade_install():
    run(["kubectl", "create", "namespace", QDRANT_NAMESPACE, "--dry-run=client", "-o", "yaml"], check=False)
    tries = 0
    max_tries = 3
    while tries < max_tries:
        try:
            if Path(VENDOR_CHART_DIR).is_dir() and (Path(VENDOR_CHART_DIR)/"Chart.yaml").exists():
                DBG("Using vendor chart", VENDOR_CHART_DIR)
                run(["helm", "upgrade", "--install", QDRANT_RELEASE, VENDOR_CHART_DIR, "--namespace", QDRANT_NAMESPACE, "--create-namespace", "-f", str(VALUES_FILE), "--wait", "--timeout", "10m"], check=True)
                return True
            else:
                DBG("Attempting helm repo install try", tries+1)
                helm_repo_add_if_missing()
                run(["helm", "upgrade", "--install", QDRANT_RELEASE, f"{HELM_REPO_NAME}/qdrant", "--version", CHART_VERSION, "--namespace", QDRANT_NAMESPACE, "-f", str(VALUES_FILE), "--wait", "--timeout", "10m"], check=True)
                return True
        except Exception:
            try:
                run(["helm", "upgrade", "--install", QDRANT_RELEASE, "qdrant/qdrant", "--version", CHART_VERSION, "--repo", HELM_FALLBACK_REPO, "--namespace", QDRANT_NAMESPACE, "-f", str(VALUES_FILE), "--wait", "--timeout", "10m"], check=True)
                return True
            except Exception:
                pass
        tries += 1
        time.sleep(2 * tries)
    return False

def patch_statefulset_ports_if_missing():
    try:
        ss_json = run(["kubectl", "-n", QDRANT_NAMESPACE, "get", "statefulset", QDRANT_RELEASE, "-o", "json"], check=False, capture=True)
        if not ss_json:
            LOG("StatefulSet not present; skipping port patch")
            return False
        obj = json.loads(ss_json)
        containers = obj.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        if not containers:
            LOG("no containers discovered in StatefulSet; skipping patch")
            return False
        for c in containers:
            ports = c.get("ports", [])
            for p in ports:
                if str(p.get("containerPort","")) == str(QDRANT_METRICS_PORT):
                    LOG("StatefulSet already declares metrics port", QDRANT_METRICS_PORT)
                    return True
        container_name = containers[0].get("name")
        patch = {"spec": {"template": {"spec": {"containers": [{"name": container_name, "ports": [{"name": QDRANT_METRICS_PORT_NAME, "containerPort": QDRANT_METRICS_PORT, "protocol": "TCP"}]}]}}}}
        run(["kubectl", "-n", QDRANT_NAMESPACE, "patch", "statefulset", QDRANT_RELEASE, "--type=merge", "-p", json.dumps(patch)], check=False)
        LOG("patched StatefulSet to add metrics port")
        return True
    except Exception:
        return False

def validate_service_post_install():
    selector = f"app.kubernetes.io/name={QDRANT_RELEASE}"
    LOG(f"waiting up to {SERVICE_VALIDATION_WAIT}s for pods matching '{selector}'")
    try:
        run(["kubectl", "-n", QDRANT_NAMESPACE, "wait", "--for=condition=Ready", "pod", "-l", selector, f"--timeout={SERVICE_VALIDATION_WAIT}s"], check=False)
    except Exception:
        pass
    end = time.time() + SERVICE_VALIDATION_WAIT
    pods = []
    while time.time() < end:
        out = run(["kubectl", "-n", QDRANT_NAMESPACE, "get", "pods", "-l", selector, "-o", "json"], check=False, capture=True)
        try:
            pod_json = json.loads(out)
            items = pod_json.get("items", [])
            if items:
                pods = items
                break
        except Exception:
            pass
        time.sleep(2)
    if not pods:
        LOG("no pods found after wait")
        return False
    errors = []
    for p in pods:
        name = p.get("metadata", {}).get("name")
        ann = p.get("metadata", {}).get("annotations", {}) or {}
        scrape = ann.get("monitoring.io/scrape", "").lower()
        port = ann.get("monitoring.io/port", "")
        path = ann.get("monitoring.io/path", "")
        if scrape != "true":
            errors.append(f"{name}: missing monitoring.io/scrape=true")
        if not port.isdigit():
            errors.append(f"{name}: monitoring.io/port must be numeric (found: {port})")
        elif int(port) != QDRANT_METRICS_PORT:
            errors.append(f"{name}: monitoring.io/port mismatch expected {QDRANT_METRICS_PORT} found {port}")
        if path != "/metrics":
            errors.append(f"{name}: monitoring.io/path must be /metrics found {path}")
    if errors:
        LOG("annotation validation errors:")
        for e in errors[:200]:
            LOG(" ", e)
        return False
    ss_json = run(["kubectl", "-n", QDRANT_NAMESPACE, "get", "statefulset", QDRANT_RELEASE, "-o", "json"], check=False, capture=True)
    if not ss_json:
        LOG("StatefulSet not present; skipping container port check")
        return False
    try:
        obj = json.loads(ss_json)
        containers = obj.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        declared = False
        for c in containers:
            for p in c.get("ports", []):
                if str(p.get("containerPort","")) == str(QDRANT_METRICS_PORT):
                    declared = True
                    break
        if not declared:
            LOG("StatefulSet container spec does NOT declare metrics port; attempting patch")
            patch_statefulset_ports_if_missing()
    except Exception:
        LOG("Failed to parse StatefulSet JSON for port check")
    LOG("pod annotations contract satisfied")
    return True

def generate_manifests(force=False):
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    validate_replication_constraints()
    validate_resource_contract()
    render_values_yaml()
    render_service_patch()
    generate_samples()
    LOG("Wrote manifests to", str(MANIFESTS_DIR))

def apply_to_cluster():
    if ENV.upper() == "STAGING" and APPLY_STAGING_SECRETS and not (AZURE_STORAGE_CONNECTION_STRING or (AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY)):
        LOG("ENV=STAGING requires AZURE storage creds when APPLY_STAGING_SECRETS=true")
        sys.exit(2)
    validate_replication_constraints()
    validate_resource_contract()
    generate_manifests(force=True)
    # create the namespace for real (do not emit yaml)
    try:
        run(["kubectl", "create", "namespace", QDRANT_NAMESPACE], check=False)
    except Exception:
        pass
    if ENV.upper() == "STAGING" and APPLY_STAGING_SECRETS and SECRET_BACKUP_NAME:
        if not create_azure_backup_secret():
            LOG("failed to create backup secret")
            if FAIL_ON_MISCONFIG:
                sys.exit(2)
    if QDRANT__SERVICE__API_KEY and SECRET_SERVICE_NAME:
        run(["bash", "-c", f"kubectl -n {QDRANT_NAMESPACE} create secret generic {SECRET_SERVICE_NAME} --from-literal=QDRANT__SERVICE__API_KEY='{QDRANT__SERVICE__API_KEY}' --dry-run=client -o yaml | kubectl apply -f -"], check=True)
        LOG("created/updated secret", SECRET_SERVICE_NAME)
    if QDRANT_PERSISTENCE_ENABLED and not QDRANT_PERSISTENCE_STORAGE_CLASS:
        try:
            sc = run(["kubectl", "get", "storageclass", "-o", "jsonpath={.items[?(@.metadata.annotations.storageclass\\.kubernetes\\.io/is-default-class==\"true\")].metadata.name}"], check=False, capture=True)
            sc = sc.strip()
            if sc:
                LOG("No QDRANT_PERSISTENCE_STORAGE_CLASS specified; using cluster default:", sc)
                os.environ["QDRANT_PERSISTENCE_STORAGE_CLASS"] = sc
                render_values_yaml()
            else:
                LOG("ERROR: persistence enabled but no storageClass found; set QDRANT_PERSISTENCE_STORAGE_CLASS")
                if FAIL_ON_MISCONFIG:
                    sys.exit(2)
        except Exception:
            LOG("storageclass detection failed")
    if not helm_upgrade_install():
        LOG("helm install failed")
        sys.exit(2)
    LOG("helm install/upgrade succeeded")
    try:
        run(["kubectl", "apply", "-f", str(SERVICE_PATCH_FILE)], check=False)
    except Exception:
        LOG("service patch apply returned non-zero")
    time.sleep(2)
    if not validate_service_post_install():
        LOG("post-install validation errors")
        if FAIL_ON_MISCONFIG:
            sys.exit(2)
        else:
            LOG("continuing despite validation errors (FAIL_ON_MISCONFIG=false)")
    summary = {"release": QDRANT_RELEASE, "namespace": QDRANT_NAMESPACE, "replicas": QDRANT_REPLICAS, "values_file": str(VALUES_FILE), "chart_version": CHART_VERSION, "image": QDRANT_IMAGE, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "secrets_created": bool(SECRET_BACKUP_NAME or QDRANT__SERVICE__API_KEY)}
    atomic_write(LAST_SUMMARY, json.dumps(summary, indent=2).encode("utf-8"))
    LOG("Wrote deploy summary ->", str(LAST_SUMMARY))

def delete_from_cluster():
    try:
        run(["kubectl", "-n", QDRANT_NAMESPACE, "delete", "deployment", QDRANT_RELEASE, "--ignore-not-found"], check=False)
        run(["kubectl", "-n", QDRANT_NAMESPACE, "delete", "statefulset", QDRANT_RELEASE, "--ignore-not-found"], check=False)
        run(["kubectl", "-n", QDRANT_NAMESPACE, "delete", "service", QDRANT_RELEASE, "--ignore-not-found"], check=False)
        if SECRET_BACKUP_NAME:
            run(["kubectl", "-n", QDRANT_NAMESPACE, "delete", "secret", SECRET_BACKUP_NAME, "--ignore-not-found"], check=False)
        if SECRET_SERVICE_NAME:
            run(["kubectl", "-n", QDRANT_NAMESPACE, "delete", "secret", SECRET_SERVICE_NAME, "--ignore-not-found"], check=False)
    except Exception as e:
        LOG("delete attempts encountered errors:", e)
    try:
        if MANIFESTS_DIR.exists():
            shutil.rmtree(MANIFESTS_DIR)
            LOG("deleted manifests directory")
    except Exception as e:
        LOG("failed to delete manifests dir:", e)
    LOG("deleted cluster objects (best-effort)")

def usage():
    print("usage: qdrant_cluster.py --generate|--rollout|--delete [--force] [--verbose]")
    sys.exit(1)

def cleanup_and_exit(rc=0):
    for f in list(TMP_FILES):
        try:
            os.unlink(f)
        except Exception:
            pass
    sys.exit(rc)

def main():
    if len(sys.argv) == 1:
        usage()
    cmd = None
    force = False
    for arg in sys.argv[1:]:
        if arg == "--generate":
            cmd = "generate"
        elif arg == "--rollout":
            cmd = "rollout"
        elif arg == "--apply":
            # keep legacy support, treated as deprecated alias for rollout
            cmd = "apply"
        elif arg == "--delete":
            cmd = "delete"
        elif arg == "--force":
            force = True
        elif arg == "--verbose":
            os.environ["VERBOSE"] = "1"
        else:
            usage()
    try:
        if cmd == "generate":
            generate_manifests(force=force)
            cleanup_and_exit(0)
        elif cmd == "rollout":
            LOG("rollout started")
            apply_to_cluster()
            cleanup_and_exit(0)
        elif cmd == "apply":
            LOG("WARNING: --apply is deprecated; use --rollout (behavior unchanged)")
            apply_to_cluster()
            cleanup_and_exit(0)
        elif cmd == "delete":
            delete_from_cluster()
            cleanup_and_exit(0)
        else:
            usage()
    except KeyboardInterrupt:
        LOG("interrupted by user")
        cleanup_and_exit(1)
    except SystemExit as se:
        cleanup_and_exit(se.code if isinstance(se.code, int) else 1)
    except Exception as e:
        LOG("unexpected error:", e)
        cleanup_and_exit(2)

if __name__ == "__main__":
    main()
