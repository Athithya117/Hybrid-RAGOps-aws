#!/usr/bin/env python3
"""
generators/gen_cloudflared.py

Deterministic, secret-safe generator for Cloudflare Tunnel (cloudflared) Kubernetes manifests.

Behavior:
 - Writes only non-secret manifests to disk (Namespace, ServiceAccount, Deployment).
 - Secrets are NEVER written to disk. If CLOUDFLARE_TUNNEL_CREDENTIALS_B64 is provided and USE_AZURE_KEYVAULT is false,
   the generator will create/replace the k8s Secret in-cluster via kubectl (in-memory YAML applied via stdin).
 - If USE_AZURE_KEYVAULT=true, an ExternalSecret manifest is emitted (no secret values).
 - Commands: --generate (write manifests), --apply (create secret in-cluster if needed and kubectl apply), --delete.

Usage:
  python3 gen_cloudflared.py --generate
  python3 gen_cloudflared.py --apply --confirm
  python3 gen_cloudflared.py --delete --confirm
"""

from __future__ import annotations
import os
import sys
import base64
import json
import yaml
import hashlib
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, List
import datetime

# ---------------------------
# Helpers
# ---------------------------
def die(msg: str) -> None:
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(2)

def info(msg: str) -> None:
    print("INFO:", msg)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def atomic_write(path: Path, content: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
    try:
        path.chmod(mode)
    except Exception:
        pass

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def run_cmd_capture(cmd: List[str], input_bytes: Optional[bytes] = None, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=timeout)

def canonicalize(o: Any):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, dict):
        return {k: canonicalize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [canonicalize(v) for v in o]
    return o

def canonical_inputs_hash(cfg: Dict[str, Any]) -> str:
    serial = {}
    for k in sorted(cfg.keys()):
        if k in ("INPUTS_HASH_PATH", "SECRET_CREDENTIALS_DECODED"):
            continue
        serial[k] = canonicalize(cfg[k])
    j = json.dumps(serial, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode("utf-8")).hexdigest()

# ---------------------------
# Config loader
# ---------------------------
def load_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    cfg["MANIFESTS_DIR"] = Path(os.getenv("MANIFESTS_DIR", "infra/manifests/cloudflared"))
    cfg["INPUTS_HASH_PATH"] = cfg["MANIFESTS_DIR"] / ".inputs_hash"
    cfg["ENV"] = os.getenv("ENV", "STAGING").upper()
    cfg["USE_AZURE_KEYVAULT"] = os.getenv("USE_AZURE_KEYVAULT", "false").lower() in ("1", "true", "yes")
    cfg["ALLOW_MISSING_SECRETS"] = os.getenv("ALLOW_MISSING_SECRETS", "false").lower() in ("1", "true", "yes")

    # cloudflared inputs
    cfg["MODE"] = os.getenv("FRONTEND_HOST_MODE", "cloudflare_quick").lower()  # cloudflare_quick | cloudflare_named
    cfg["NAMESPACE"] = os.getenv("FRONTEND_NAMESPACE", "default")
    cfg["TUNNEL_NAME"] = os.getenv("CLOUDFLARE_TUNNEL_NAME", "rag-tunnel")
    cfg["REPLICAS"] = int(os.getenv("CLOUDFLARE_TUNNEL_REPLICAS", "2")) if cfg["ENV"] == "PROD" else int(os.getenv("CLOUDFLARE_TUNNEL_REPLICAS", "2"))
    cfg["IMAGE"] = os.getenv("CLOUDFLARE_IMAGE", "cloudflare/cloudflared:2025.10.0")
    cfg["SERVICE_ACCOUNT_NAME"] = os.getenv("CLOUDFLARE_SA", "cloudflared-sa")
    cfg["SERVICE_ACCOUNT_NAMESPACE"] = cfg["NAMESPACE"]

    # Cloudflare secrets & options
    cfg["CLOUDFLARE_TUNNEL_CREDENTIALS_B64"] = os.getenv("CLOUDFLARE_TUNNEL_CREDENTIALS_B64", "")
    cfg["CLOUDFLARE_API_TOKEN"] = os.getenv("CLOUDFLARE_API_TOKEN", "")
    cfg["CLOUDFLARE_ACCOUNT_ID"] = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
    cfg["FRONTEND_SERVICE_NAME"] = os.getenv("FRONTEND_SERVICE_NAME", "frontend")
    cfg["FRONTEND_SERVICE_PORT"] = int(os.getenv("FRONTEND_SERVICE_PORT", "80"))

    # files
    m = cfg["MANIFESTS_DIR"]
    cfg["FILES"] = {
        "namespace": m / "00-namespace.yaml",
        "sa": m / "01-serviceaccount.yaml",
        "deployment": m / "02-deployment.yaml",
        "externalsecret": m / "03-externalsecret.yaml",
        "inputs_hash": m / ".inputs_hash",
    }

    return cfg

# ---------------------------
# Renderers (non-secret manifests only)
# ---------------------------
def render_namespace(cfg: Dict[str, Any]) -> str:
    obj = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": cfg["NAMESPACE"], "labels": {"app.kubernetes.io/managed-by": "cloudflared-generator", "env": cfg["ENV"].lower()}},
    }
    return yaml.safe_dump(obj, sort_keys=False)

def render_serviceaccount(cfg: Dict[str, Any]) -> str:
    sa = {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {"name": cfg["SERVICE_ACCOUNT_NAME"], "namespace": cfg["NAMESPACE"]},
    }
    return yaml.safe_dump(sa, sort_keys=False)

def render_deployment(cfg: Dict[str, Any]) -> str:
    # Deployment references the secret by name but does not include secret contents
    secret_name = f"cloudflared-tunnel-creds"
    container = {
        "name": "cloudflared",
        "image": cfg["IMAGE"],
        "args": ["tunnel", "run", cfg["TUNNEL_NAME"]],
        "env": [
            # If API token exists, we still avoid embedding it in manifests; provide note
            {"name": "TUNNEL_TOKEN", "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "token"}}} if not cfg["CLOUDFLARE_TUNNEL_CREDENTIALS_B64"] else {"name": "TUNNEL_CREDENTIALS_JSON_PATH", "value": "/etc/cloudflared/credentials.json"},
        ],
        "volumeMounts": [
            {"name": "creds", "mountPath": "/etc/cloudflared", "readOnly": True},
        ],
        "ports": [{"containerPort": 7844, "name": "proxy"}, {"containerPort": 8080, "name": "local"}],
        "resources": {"requests": {"cpu": "100m", "memory": "128Mi"}, "limits": {"cpu": "500m", "memory": "512Mi"}},
        "livenessProbe": {"httpGet": {"path": "/", "port": 8080}, "initialDelaySeconds": 10, "periodSeconds": 10},
        "readinessProbe": {"httpGet": {"path": "/", "port": 8080}, "initialDelaySeconds": 5, "periodSeconds": 5},
    }

    pod_spec = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "cloudflared", "namespace": cfg["NAMESPACE"], "labels": {"app": "cloudflared"}},
        "spec": {
            "replicas": cfg["REPLICAS"],
            "selector": {"matchLabels": {"app": "cloudflared"}},
            "template": {
                "metadata": {"labels": {"app": "cloudflared"}},
                "spec": {
                    "serviceAccountName": cfg["SERVICE_ACCOUNT_NAME"],
                    "containers": [container],
                    "volumes": [
                        {
                            "name": "creds",
                            "secret": {
                                "secretName": secret_name,
                                # do not write secret contents to disk; this is only a reference
                            },
                        }
                    ],
                },
            },
        },
    }

    # If using CLOUDFLARE_TUNNEL_CREDENTIALS_B64, the deployment expects credentials.json in the secret (stringData -> credentials.json)
    return yaml.safe_dump(pod_spec, sort_keys=False)

def render_external_secret(cfg: Dict[str, Any]) -> Optional[str]:
    if not cfg["USE_AZURE_KEYVAULT"]:
        return None
    # ExternalSecret pointing to existing Azure Key Vault Secret containing cloudflared credentials
    key_name = os.getenv("CLOUDFLARE_KEYVAULT_SECRET_NAME", "cloudflared-credentials")
    if not key_name and not cfg["ALLOW_MISSING_SECRETS"]:
        die("USE_AZURE_KEYVAULT=true requires CLOUDFLARE_KEYVAULT_SECRET_NAME or set ALLOW_MISSING_SECRETS=true")
    es = {
        "apiVersion": "external-secrets.io/v1beta1",
        "kind": "ExternalSecret",
        "metadata": {"name": "cloudflared-externalsecret", "namespace": cfg["NAMESPACE"]},
        "spec": {
            "refreshInterval": "1h",
            "secretStoreRef": {"name": os.getenv("AZURE_KEYVAULT_SECRETSTORE", "azure-keyvault-store"), "kind": "ClusterSecretStore"},
            "target": {"name": "cloudflared-tunnel-creds", "creationPolicy": "Merge"},
            "data": [{"secretKey": "credentials.json", "remoteRef": {"key": key_name}}],
        },
    }
    return yaml.safe_dump(es, sort_keys=False)

# ---------------------------
# Secret creation (in-cluster) - never write secret files to disk
# ---------------------------
def kubectl_apply_stream(yaml_docs: str, dry_run: bool = False) -> Dict[str, Any]:
    kubectl = which("kubectl")
    if not kubectl:
        return {"applied": False, "error": "kubectl-not-found"}
    cmd = [kubectl, "apply", "-f", "-"]
    if dry_run:
        cmd = [kubectl, "apply", "--dry-run=client", "-f", "-"]
    try:
        proc = subprocess.run(cmd, input=yaml_docs.encode("utf-8"), capture_output=True, check=True, timeout=120)
        out = proc.stdout.decode() if proc.stdout else ""
        return {"applied": True, "stdout": out}
    except subprocess.CalledProcessError as e:
        return {"applied": False, "stderr": e.stderr.decode() if e.stderr else str(e)}
    except Exception as e:
        return {"applied": False, "stderr": str(e)}

def create_or_update_secret_in_cluster(cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """
    Create/update k8s secret from base64-encoded tunnel credentials (CLOUDFLARE_TUNNEL_CREDENTIALS_B64).
    Does NOT write secret manifests to disk.
    """
    if cfg["USE_AZURE_KEYVAULT"]:
        return {"created": False, "reason": "using_azure_keyvault"}

    b64 = cfg["CLOUDFLARE_TUNNEL_CREDENTIALS_B64"]
    token = cfg["CLOUDFLARE_API_TOKEN"]
    if not b64 and not token:
        return {"created": False, "reason": "no_credentials"}

    secret_name = "cloudflared-tunnel-creds"
    ns = cfg["NAMESPACE"]

    if b64:
        try:
            decoded = base64.b64decode(b64.encode("utf-8"))
            # ensure valid JSON
            _ = json.loads(decoded.decode("utf-8"))
            # Build secret manifest with stringData (not written to disk)
            secret_manifest = {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {"name": secret_name, "namespace": ns},
                "stringData": {"credentials.json": decoded.decode("utf-8")},
                "type": "Opaque",
            }
            yaml_docs = yaml.safe_dump(secret_manifest, sort_keys=False)
            if dry_run:
                return {"created": False, "reason": "dry_run", "cmd": "kubectl apply --dry-run=client -f -", "manifest": yaml_docs}
            res = kubectl_apply_stream(yaml_docs, dry_run=False)
            if not res.get("applied", False):
                return {"created": False, "stderr": res.get("stderr")}
            return {"created": True, "stdout": res.get("stdout")}
        except Exception as e:
            return {"created": False, "stderr": f"invalid_credentials_json:{e}"}
    else:
        # If only API token provided, we cannot generate credentials JSON here (requires cloudflare API flow).
        # Instead create a simple secret holding the token (key: token). The cloudflared container can use token env.
        secret_manifest = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": secret_name, "namespace": ns},
            "stringData": {"token": token},
            "type": "Opaque",
        }
        yaml_docs = yaml.safe_dump(secret_manifest, sort_keys=False)
        if dry_run:
            return {"created": False, "reason": "dry_run", "cmd": "kubectl apply --dry-run=client -f -", "manifest": yaml_docs}
        res = kubectl_apply_stream(yaml_docs, dry_run=False)
        if not res.get("applied", False):
            return {"created": False, "stderr": res.get("stderr")}
        return {"created": True, "stdout": res.get("stdout")}

# ---------------------------
# Leak detection guard (never write secret content into non-secret manifests)
# ---------------------------
def detect_secret_leak(rendered: str, secret_b64: Optional[str]) -> Optional[str]:
    if not secret_b64:
        return None
    try:
        decoded = base64.b64decode(secret_b64.encode("utf-8")).decode("utf-8")
    except Exception:
        return None
    if len(decoded) >= 8 and decoded in rendered:
        return "CLOUDFLARE_TUNNEL_CREDENTIALS_B64 decoded payload"
    return None

# ---------------------------
# Generate / Apply / Delete
# ---------------------------
def generate(cfg: Dict[str, Any], dry_run: bool = False) -> None:
    ensure_dir(cfg["MANIFESTS_DIR"])
    ihash = canonical_inputs_hash(cfg)
    existing = None
    try:
        if cfg["INPUTS_HASH_PATH"].exists():
            existing = cfg["INPUTS_HASH_PATH"].read_text(encoding="utf-8").strip()
    except Exception:
        existing = None
    if existing == ihash and not dry_run:
        info("No non-secret changes; skipping generation.")
        return

    # render manifests and ensure no secret leak
    dep_yaml = render_deployment(cfg)
    leak = detect_secret_leak(dep_yaml, cfg["CLOUDFLARE_TUNNEL_CREDENTIALS_B64"])
    if leak:
        die(f"Secret leak detected in deployment YAML: {leak}; refusing to generate.")

    # write non-secret manifests
    atomic_write(cfg["FILES"]["namespace"], render_namespace(cfg))
    atomic_write(cfg["FILES"]["sa"], render_serviceaccount(cfg))
    atomic_write(cfg["FILES"]["deployment"], dep_yaml)

    # ExternalSecret manifest only if using KeyVault
    if cfg["USE_AZURE_KEYVAULT"]:
        es = render_external_secret(cfg)
        if es:
            atomic_write(cfg["FILES"]["externalsecret"], es)
    else:
        try:
            cfg["FILES"]["externalsecret"].unlink()
        except Exception:
            pass

    cfg["INPUTS_HASH_PATH"].write_text(ihash, encoding="utf-8")
    info(f"Wrote manifests to {cfg['MANIFESTS_DIR']} (secrets are NOT written to disk)")

def apply(cfg: Dict[str, Any], confirm: bool = False, dry_run: bool = False) -> None:
    if not confirm:
        die("Refusing to apply without --confirm")
    if not which("kubectl"):
        die("kubectl not found in PATH")

    # generate non-secret manifests
    generate(cfg, dry_run=dry_run)

    # create/update secret in-cluster if provided and not using KeyVault
    secret_res = create_or_update_secret_in_cluster(cfg, dry_run=dry_run)
    if secret_res.get("created") is False and secret_res.get("reason") == "no_credentials":
        if not cfg["ALLOW_MISSING_SECRETS"]:
            die("No Cloudflare tunnel credentials provided. Set CLOUDFLARE_TUNNEL_CREDENTIALS_B64 or CLOUDFLARE_API_TOKEN, or enable USE_AZURE_KEYVAULT.")
        else:
            info("No credentials provided but ALLOW_MISSING_SECRETS=true; continuing.")
    elif secret_res.get("created") is False and secret_res.get("reason") == "using_azure_keyvault":
        info("Using Azure KeyVault; ExternalSecret manifest will be applied")
    elif secret_res.get("created") is False and secret_res.get("reason") == "dry_run":
        info(f"Dry run secret creation: {secret_res.get('cmd')}")
    elif secret_res.get("created") is False and secret_res.get("reason") == "kubectl-not-found":
        die("kubectl required to create secret in cluster; install kubectl or run --generate and create secret manually.")
    elif secret_res.get("created"):
        info("Created/updated in-cluster secret from environment variables.")

    # apply non-secret manifests and ExternalSecret if present
    parts: List[str] = []
    for key in ("namespace", "sa", "deployment", "externalsecret"):
        p = cfg["FILES"].get(key)
        if p and p.exists():
            parts.append(p.read_text(encoding="utf-8"))
    if not parts:
        die("No manifests to apply (generate first)")

    combined = "\n---\n".join(parts)
    res = kubectl_apply_stream(combined, dry_run=dry_run)
    if not res.get("applied", False):
        die(f"kubectl apply failed: {res.get('stderr')}")
    info("Applied manifests to cluster.")

    summary = {"generated_at": datetime.datetime.utcnow().isoformat() + "Z", "tunnel": cfg["TUNNEL_NAME"], "namespace": cfg["NAMESPACE"], "replicas": cfg["REPLICAS"]}
    atomic_write(cfg["MANIFESTS_DIR"] / "last_deploy_summary.json", json.dumps(summary, indent=2))

def delete(cfg: Dict[str, Any], confirm: bool = False) -> None:
    if not confirm:
        die("Refusing to delete without --confirm")
    # remove manifest files locally (non-secret)
    if cfg["MANIFESTS_DIR"].exists():
        for p in sorted(cfg["MANIFESTS_DIR"].glob("*")):
            try:
                p.unlink()
            except Exception:
                pass
        info(f"Deleted local manifests in {cfg['MANIFESTS_DIR']}")
    # attempt to delete resources from cluster (best-effort)
    if which("kubectl"):
        ns = cfg["NAMESPACE"]
        cmds = [
            ["kubectl", "delete", "deployment", "cloudflared", "-n", ns, "--ignore-not-found"],
            ["kubectl", "delete", "serviceaccount", cfg["SERVICE_ACCOUNT_NAME"], "-n", ns, "--ignore-not-found"],
        ]
        if cfg["USE_AZURE_KEYVAULT"]:
            cmds.append(["kubectl", "delete", "externalsecret", "cloudflared-externalsecret", "-n", ns, "--ignore-not-found"])
        # do not delete secrets created by other means
        for c in cmds:
            try:
                subprocess.run(c, check=False, capture_output=True, timeout=30)
            except Exception:
                pass
        info("Requested deletion of k8s resources (ignore-not-found).")
    else:
        info("kubectl not present; local manifests removed but cluster resources remain.")

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generator for cloudflared tunnel manifests (secret-safe).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--delete", action="store_true")
    p.add_argument("--confirm", action="store_true", help="required to apply/delete")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config()

    # Basic validation (fail-fast)
    if cfg["MODE"] not in ("cloudflare_quick", "cloudflare_named"):
        die("FRONTEND_HOST_MODE must be 'cloudflare_quick' or 'cloudflare_named'")
    if cfg["MODE"] == "cloudflare_named" and not (cfg["CLOUDFLARE_TUNNEL_CREDENTIALS_B64"] or cfg["CLOUDFLARE_API_TOKEN"] or cfg["CLOUDFLARE_ACCOUNT_ID"]):
        die("cloudflare_named mode requires CLOUDFLARE_TUNNEL_CREDENTIALS_B64 or CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID")
    if args.generate:
        generate(cfg, dry_run=args.dry_run)
        return
    if args.apply:
        apply(cfg, confirm=args.confirm, dry_run=args.dry_run)
        return
    if args.delete:
        delete(cfg, confirm=args.confirm)
        return

if __name__ == "__main__":
    main()
