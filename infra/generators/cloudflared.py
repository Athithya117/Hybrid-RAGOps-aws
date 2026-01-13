#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import base64
import subprocess
import argparse
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS_DIR_DEFAULT = ROOT / "infra" / "manifests" / "cloudflared"


def die(msg: str):
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(2)


def info(msg: str):
    print("INFO:", msg)


def safe_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
    try:
        path.chmod(0o644)
    except Exception:
        pass


def run_kubectl(args: List[str], input_text: Optional[str] = None, timeout: int = 60) -> int:
    cmd = ["kubectl"] + args
    try:
        proc = subprocess.run(
            cmd,
            input=(input_text.encode("utf-8") if input_text else None),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
        out = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            if err:
                print("kubectl error:", err, file=sys.stderr)
        else:
            if out:
                print(out)
        return proc.returncode
    except FileNotFoundError:
        print("kubectl not found in PATH", file=sys.stderr)
        return 127
    except subprocess.TimeoutExpired:
        print("kubectl timed out", file=sys.stderr)
        return 124


def ensure_kubectl_available() -> None:
    rc = subprocess.run(["kubectl", "version", "--client"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    if rc != 0:
        die("kubectl client not available in PATH")


def ensure_namespace(ns: str) -> None:
    rc = subprocess.run(["kubectl", "get", "namespace", ns], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    if rc != 0:
        info(f"Namespace '{ns}' not present; creating")
        rc2 = run_kubectl(["create", "namespace", ns])
        if rc2 != 0:
            die(f"Failed to create namespace {ns} (kubectl create returned {rc2})")
        info(f"Namespace '{ns}' created")


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def render_configmap(tunnel_name: str, credentials_path: str, ingress_rules: List[Dict[str, Any]], use_token: bool) -> Tuple[Dict[str, Any], str]:
    """
    Returns (ConfigMap dict, rendered_config_text).
    """
    cfg: Dict[str, Any] = {"ingress": ingress_rules}
    if not use_token:
        if tunnel_name:
            cfg["tunnel"] = tunnel_name
        cfg["credentials-file"] = credentials_path
    # Keep ordering deterministic: yaml.safe_dump with sort_keys=False is acceptable because ingress_rules is constructed deterministically
    config_text = yaml.safe_dump(cfg, sort_keys=False)
    cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "cloudflared-config", "namespace": NAMESPACE},
        "data": {"config.yml": config_text},
    }
    return cm, config_text


def render_serviceaccount() -> Dict[str, Any]:
    return {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": {"name": "cloudflared-sa", "namespace": NAMESPACE}}


def render_deployment(replicas: int, image: str, use_token: bool, tunnel_name: str, mount_creds: bool, config_checksum: str = "", config_key: str = "config.yml", creds_key: str = "credentials.json") -> Dict[str, Any]:
    container = {
        "name": "cloudflared",
        "image": image,
        "resources": {"requests": {"cpu": "50m", "memory": "64Mi"}, "limits": {"cpu": "200m", "memory": "256Mi"}},
        "livenessProbe": {"exec": {"command": ["cloudflared", "--version"]}, "initialDelaySeconds": 15, "periodSeconds": 60, "timeoutSeconds": 5},
        "readinessProbe": {"exec": {"command": ["cloudflared", "--version"]}, "initialDelaySeconds": 10, "periodSeconds": 10, "timeoutSeconds": 3},
    }
    volume_mounts = [{"name": "config-volume", "mountPath": "/etc/cloudflared/config.yml", "subPath": config_key}]
    if use_token:
        args_list = ["tunnel", "--no-autoupdate", "run"]
        container["args"] = args_list
        container["env"] = [
            {"name": "TUNNEL_TOKEN", "valueFrom": {"secretKeyRef": {"name": "cloudflared-token", "key": "CLOUDFLARE_TUNNEL_TOKEN"}}},
            {"name": "CLOUDFLARE_TUNNEL_TOKEN", "valueFrom": {"secretKeyRef": {"name": "cloudflared-token", "key": "CLOUDFLARE_TUNNEL_TOKEN"}}},
        ]
    else:
        args_list = ["tunnel", "--no-autoupdate", "run"]
        if tunnel_name:
            args_list.append(tunnel_name)
        container["args"] = args_list
    if mount_creds:
        volume_mounts.append({"name": "creds-volume", "mountPath": "/etc/cloudflared/creds/credentials.json", "subPath": creds_key})
    container["volumeMounts"] = volume_mounts
    pod_spec = {"serviceAccountName": "cloudflared-sa", "containers": [container]}
    volumes = [{"name": "config-volume", "configMap": {"name": "cloudflared-config"}}]
    if mount_creds:
        volumes.append({"name": "creds-volume", "secret": {"secretName": "cloudflared-tunnel-credentials"}})
    pod_spec["volumes"] = volumes

    template_meta: Dict[str, Any] = {"labels": {"app": "cloudflared"}}
    if config_checksum:
        # annotation triggers rollout when config changes
        template_meta["annotations"] = {"cloudflared/config-checksum": config_checksum}

    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "cloudflared", "namespace": NAMESPACE},
        "spec": {"replicas": replicas, "selector": {"matchLabels": {"app": "cloudflared"}}, "template": {"metadata": template_meta, "spec": pod_spec}},
    }
    return deployment


def render_secret_token(token: str) -> Dict[str, Any]:
    data_b64 = base64.b64encode(token.encode("utf-8")).decode("utf-8")
    return {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": "cloudflared-token", "namespace": NAMESPACE}, "type": "Opaque", "data": {"CLOUDFLARE_TUNNEL_TOKEN": data_b64}}


def render_secret_credentials(creds_b64: str) -> Dict[str, Any]:
    try:
        raw = base64.b64decode(creds_b64.encode("utf-8"))
        data_b64 = base64.b64encode(raw).decode("utf-8")
    except Exception:
        data_b64 = creds_b64
    return {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": "cloudflared-tunnel-credentials", "namespace": NAMESPACE}, "type": "Opaque", "data": {"credentials.json": data_b64}}


def write_manifests(replicas: int, image: str, use_token: bool, tunnel_name: str, embed_secrets: bool, upstream_service: str, hostname: str, credentials_path: str, dashboard_hostname: str = "", dashboard_upstream: str = ""):
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    sa = render_serviceaccount()
    ingress_rules: List[Dict[str, Any]] = []
    seen = set()

    # Frontend hostname (if provided)
    if hostname:
        key = (hostname, upstream_service)
        if key not in seen:
            ingress_rules.append({"hostname": hostname, "service": upstream_service})
            seen.add(key)
            info(f"Added ingress rule: {hostname} -> {upstream_service}")

    # Dashboard hostname -> route to Grafana upstream when supplied
    if dashboard_hostname:
        dsvc = dashboard_upstream or DEFAULT_DASHBOARD_UPSTREAM
        key = (dashboard_hostname, dsvc)
        if key not in seen:
            # ensure Grafana sees the external host so redirects / asset URLs use the public hostname
            ingress_rules.append({
                "hostname": dashboard_hostname,
                "service": dsvc,
                "originRequest": {"httpHostHeader": dashboard_hostname},
            })
            seen.add(key)
            info(f"Added ingress rule: {dashboard_hostname} -> {dsvc}")

    # Default 404 catch-all
    ingress_rules.append({"service": "http_status:404"})

    cm, config_text = render_configmap(tunnel_name, credentials_path, ingress_rules, use_token)
    config_checksum = sha256_str(config_text)

    deploy = render_deployment(replicas, image, use_token, tunnel_name, mount_creds=not use_token, config_checksum=config_checksum)

    # Write manifests in stable order
    safe_write(MANIFESTS_DIR / "00-serviceaccount.yaml", yaml.safe_dump(sa, sort_keys=False))
    safe_write(MANIFESTS_DIR / "01-configmap.yaml", yaml.safe_dump(cm, sort_keys=False))
    safe_write(MANIFESTS_DIR / "02-deployment.yaml", yaml.safe_dump(deploy, sort_keys=False))

    if embed_secrets:
        idx = 3
        if CLOUDFLARE_TUNNEL_TOKEN:
            safe_write(MANIFESTS_DIR / f"{idx:02d}-secret-cloudflared-token.yaml", yaml.safe_dump(render_secret_token(CLOUDFLARE_TUNNEL_TOKEN), sort_keys=False))
            idx += 1
        if CLOUDFLARE_TUNNEL_CREDENTIALS_B64:
            safe_write(MANIFESTS_DIR / f"{idx:02d}-secret-cloudflared-credentials.yaml", yaml.safe_dump(render_secret_credentials(CLOUDFLARE_TUNNEL_CREDENTIALS_B64), sort_keys=False))
            idx += 1

    info(f"Wrote manifests to {MANIFESTS_DIR}")
    return config_checksum


def apply_secret_yaml(yaml_text: str, attempts: int = 3) -> int:
    for i in range(1, attempts + 1):
        rc = run_kubectl(["apply", "-f", "-"], input_text=yaml_text, timeout=30)
        if rc == 0:
            return 0
        info(f"apply secret attempt {i} failed")
        time.sleep(1)
    return rc


def apply_manifests() -> None:
    ensure_namespace(NAMESPACE)
    info("Applying manifests")
    rc = run_kubectl(["apply", "-f", str(MANIFESTS_DIR)], timeout=120)
    if rc != 0:
        die("kubectl apply failed")


def wait_for_rollout(deployment_name: str, namespace: str, timeout: int = 120) -> int:
    rc = run_kubectl(["rollout", "status", f"deployment/{deployment_name}", "-n", namespace, f"--timeout={timeout}s"], timeout=timeout + 10)
    return rc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--write-manifests", action="store_true")
    p.add_argument("--rollout", action="store_true", help="Create or converge resources to desired state (preferred over --apply)")
    p.add_argument("--apply", action="store_true", help="Legacy alias for --rollout (deprecated)")
    p.add_argument("--delete", action="store_true")
    p.add_argument("--replicas", type=int, default=int(os.getenv("CLOUDFLARE_TUNNEL_REPLICAS", "2")))
    p.add_argument("--namespace", type=str, default=os.getenv("CLOUDFLARE_NAMESPACE", "inference"))
    p.add_argument("--image", type=str, default=os.getenv("CLOUDFLARED_IMAGE", f"cloudflare/cloudflared:{os.getenv('CLOUDFLARED_VERSION','2025.11.1')}"))
    p.add_argument("--embed-secrets", action="store_true")
    args = p.parse_args()

    FRONTEND_HOSTNAME = os.getenv("FRONTEND_HOSTNAME", "").strip()
    if not FRONTEND_HOSTNAME:
        die("FRONTEND_HOSTNAME must be set")

    DASHBOARDS_HOSTNAME = os.getenv("DASHBOARDS_HOSTNAME", "").strip()  # optional; empty => no public route for Grafana

    # Accept either a full upstream url or fall back to service name in inference namespace
    UPSTREAM_SERVICE = os.getenv("UPSTREAM_SERVICE", "").strip()
    if not UPSTREAM_SERVICE:
        UPSTREAM_SERVICE = "http://frontend-svc.inference.svc.cluster.local:8000"

    # Dashboard upstream defaults to Grafana service in monitoring namespace, overridable by env
    GRAFANA_NAMESPACE = os.getenv("GRAFANA_NAMESPACE", "monitoring").strip()
    DEFAULT_DASHBOARD_UPSTREAM = f"http://grafana.{GRAFANA_NAMESPACE}.svc.cluster.local:3000"
    DASHBOARDS_UPSTREAM = os.getenv("DASHBOARDS_UPSTREAM", DEFAULT_DASHBOARD_UPSTREAM).strip()

    CLOUDFLARE_TUNNEL_TOKEN = os.getenv("CLOUDFLARE_TUNNEL_TOKEN", "").strip()
    CLOUDFLARE_TUNNEL_CREDENTIALS_B64 = os.getenv("CLOUDFLARE_TUNNEL_CREDENTIALS_B64", "").strip()
    CLOUDFLARE_TUNNEL_NAME = os.getenv("CLOUDFLARE_TUNNEL_NAME", "rag-frontend").strip()

    NAMESPACE = args.namespace
    MANIFESTS_DIR = MANIFESTS_DIR_DEFAULT
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    ensure_kubectl_available()

    use_token = bool(CLOUDFLARE_TUNNEL_TOKEN)
    replicas = args.replicas
    image = args.image
    credentials_path = "/etc/cloudflared/creds/credentials.json"

    if not (args.write_manifests or args.rollout or args.apply or args.delete):
        die("One of --write-manifests, --rollout, --apply, or --delete is required")

    if args.delete:
        if MANIFESTS_DIR.exists():
            run_kubectl(["delete", "-f", str(MANIFESTS_DIR)])
        info("deleted (where present)")
        sys.exit(0)

    # Render manifests and compute checksum (write_manifests returns checksum)
    config_checksum = write_manifests(
        replicas=replicas,
        image=image,
        use_token=use_token,
        tunnel_name=CLOUDFLARE_TUNNEL_NAME,
        embed_secrets=args.embed_secrets,
        upstream_service=UPSTREAM_SERVICE,
        hostname=FRONTEND_HOSTNAME,
        credentials_path=credentials_path,
        dashboard_hostname=DASHBOARDS_HOSTNAME,
        dashboard_upstream=DASHBOARDS_UPSTREAM,
    )

    if args.rollout or args.apply:
        if args.apply and not args.rollout:
            info("--apply is deprecated; using --rollout semantics (prefer --rollout)")

        ensure_namespace(NAMESPACE)
        if not args.embed_secrets:
            # apply secrets first (if provided via env) so Deployment can reference them immediately
            if CLOUDFLARE_TUNNEL_TOKEN:
                info("Applying cloudflared-token Secret from env (not written to repo).")
                sc_yaml = yaml.safe_dump(render_secret_token(CLOUDFLARE_TUNNEL_TOKEN), sort_keys=False)
                rc = apply_secret_yaml(sc_yaml, attempts=3)
                if rc != 0:
                    die("Failed to apply cloudflared-token secret")
            if CLOUDFLARE_TUNNEL_CREDENTIALS_B64 and not use_token:
                info("Applying cloudflared-tunnel-credentials Secret from env (not written to repo).")
                sc_yaml = yaml.safe_dump(render_secret_credentials(CLOUDFLARE_TUNNEL_CREDENTIALS_B64), sort_keys=False)
                rc = apply_secret_yaml(sc_yaml, attempts=3)
                if rc != 0:
                    die("Failed to apply cloudflared-tunnel-credentials secret")
        else:
            info("Secrets embedded into manifests (--embed-secrets).")

        # Apply manifests (configmap + deployment). Because deployment contains the config-checksum annotation,
        # any config changes will create a new ReplicaSet and trigger a rollout.
        apply_manifests()

        rc = wait_for_rollout("cloudflared", NAMESPACE, timeout=120)
        if rc != 0:
            print("Rollout failed or timed out; printing diagnostics", file=sys.stderr)
            run_kubectl(["get", "pods", "-n", NAMESPACE])
            run_kubectl(["describe", "pod", "-l", "app=cloudflared", "-n", NAMESPACE])
            run_kubectl(["logs", "-l", "app=cloudflared", "-n", NAMESPACE, "--tail=200"])
            die("cloudflared deployment failed to rollout")
        info("cloudflared deployment rolled out successfully")
        sys.exit(0)

    info("manifests written")
    sys.exit(0)
