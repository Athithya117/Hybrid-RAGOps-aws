#!/usr/bin/env python3
"""
gen_query.py

Deterministic generator for the inference retrieval (query) Kubernetes manifests.
Writes manifests to infra/manifests/inference/

Usage:
  python3 gen_query.py --generate
  python3 gen_query.py --apply
  python3 gen_query.py --delete

Behaviour:
- When ENV (QUERY_ENV/ENV) == "STAGING", the generator will create/update a k8s Secret
  named by QUERY_AWS_SECRET (default: aws-credentials) from the local env vars
  AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY before applying the deployment.
- The secret YAML is never stored on disk (created/applied in-memory via kubectl).
"""

from pathlib import Path
import os
import sys
import json
import yaml
import shutil
import subprocess
import argparse
import hashlib
import uuid
import datetime
import logging
import shlex

logging.basicConfig(level=os.environ.get("GEN_QUERY_LOGLEVEL", "INFO"))
log = logging.getLogger("gen_query")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def atomic_write(path: Path, content: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)


def canonical_inputs_hash(cfg: dict):
    serial = {}
    for k in sorted(cfg.keys()):
        if k == "INPUTS_HASH_PATH":
            continue
        v = cfg.get(k)
        try:
            json.dumps(v)
            serial[k] = v
        except Exception:
            serial[k] = str(v)
    j = json.dumps(serial, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode("utf-8")).hexdigest()


def kubectl_apply_yaml(yaml_str: str, dry_run=False):
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return {"applied": False, "error": "kubectl-not-found"}
    cmd = [kubectl, "apply"]
    if dry_run:
        cmd += ["--dry-run=client", "-f", "-"]
    else:
        cmd += ["-f", "-"]
    try:
        proc = subprocess.run(cmd, input=yaml_str.encode("utf-8"), capture_output=True, check=True, timeout=120)
        return {"applied": True, "stdout": proc.stdout.decode() if proc.stdout else ""}
    except subprocess.CalledProcessError as e:
        return {"applied": False, "stderr": e.stderr.decode() if e.stderr else str(e)}
    except subprocess.TimeoutExpired as e:
        return {"applied": False, "stderr": f"timeout: {e}"}


def load_config():
    cfg = {}
    cfg["ENV"] = os.environ.get("QUERY_ENV", os.environ.get("ENV", "STAGING")).upper()
    cfg["MANIFESTS_DIR"] = Path(os.environ.get("MANIFESTS_DIR", "infra/manifests/retriever"))
    cfg["INPUTS_HASH_PATH"] = cfg["MANIFESTS_DIR"] / ".inputs_hash"
    cfg["IMAGE"] = os.environ.get("QUERY_IMAGE", "athithya5354/retrieval:amd64-arm64-v1")
    cfg["NAMESPACE"] = os.environ.get("QUERY_NAMESPACE", "inference")
    cfg["SERVICE_NAME"] = os.environ.get("QUERY_SERVICE_NAME", "retrieval")
    cfg["CONTAINER_PORT"] = int(os.environ.get("QUERY_PORT", "8001"))
    cfg["HOST"] = os.environ.get("QUERY_HOST", "0.0.0.0")
    cfg["LOGLEVEL"] = os.environ.get("QUERY_LOGLEVEL", "INFO")
    # replicas & resources
    if cfg["ENV"] == "PROD":
        cfg.update({
            "REPLICAS": int(os.environ.get("QUERY_REPLICAS", "3")),
            "CPU_REQUEST": os.environ.get("QUERY_CPU_REQUEST", "1000m"),
            "CPU_LIMIT": os.environ.get("QUERY_CPU_LIMIT", "4000m"),
            "MEMORY_REQUEST": os.environ.get("QUERY_MEMORY_REQUEST", "1Gi"),
            "MEMORY_LIMIT": os.environ.get("QUERY_MEMORY_LIMIT", "4Gi"),
            "STARTUP_FAILURE_THRESHOLD": int(os.environ.get("QUERY_STARTUP_FAILURE_THRESHOLD", "24")),
        })
    else:
        cfg.update({
            "REPLICAS": int(os.environ.get("QUERY_REPLICAS", "1")),
            "CPU_REQUEST": os.environ.get("QUERY_CPU_REQUEST", "250m"),
            "CPU_LIMIT": os.environ.get("QUERY_CPU_LIMIT", "1000m"),
            "MEMORY_REQUEST": os.environ.get("QUERY_MEMORY_REQUEST", "512Mi"),
            "MEMORY_LIMIT": os.environ.get("QUERY_MEMORY_LIMIT", "1Gi"),
            "STARTUP_FAILURE_THRESHOLD": int(os.environ.get("QUERY_STARTUP_FAILURE_THRESHOLD", "60")),
        })
    # probes
    cfg["PROBE_PERIOD_SECONDS"] = int(os.environ.get("QUERY_PROBE_PERIOD_SECONDS", "5"))
    cfg["READINESS_INITIAL_DELAY"] = int(os.environ.get("QUERY_READINESS_INITIAL_DELAY", "5"))
    cfg["LIVENESS_INITIAL_DELAY"] = int(os.environ.get("QUERY_LIVENESS_INITIAL_DELAY", "10"))
    cfg["PROBE_TIMEOUT_SECONDS"] = int(os.environ.get("QUERY_PROBE_TIMEOUT_SECONDS", "3"))
    # gpu
    cfg["ENABLE_GPU"] = os.environ.get("QUERY_ENABLE_GPU", "false").lower() in ("1", "true", "yes")
    cfg["GPU_RESOURCE_NAME"] = os.environ.get("QUERY_GPU_RESOURCE", "nvidia.com/gpu")
    cfg["GPU_COUNT"] = os.environ.get("QUERY_GPU_COUNT", "1")
    cfg["GPU_NODE_SELECTOR"] = os.environ.get("QUERY_GPU_NODE_SELECTOR", "")
    # hpa
    cfg["HPA_ENABLED"] = os.environ.get("QUERY_HPA_ENABLED", "false").lower() in ("1", "true", "yes")
    cfg["HPA_MIN"] = int(os.environ.get("QUERY_HPA_MIN_REPLICAS", "1"))
    cfg["HPA_MAX"] = int(os.environ.get("QUERY_HPA_MAX_REPLICAS", "10"))
    cfg["HPA_TARGET_CPU"] = int(os.environ.get("QUERY_HPA_TARGET_CPU", "60"))
    # service specifics (propagate app envs)
    cfg["QDRANT_URL"] = os.environ.get("QDRANT_URL", "http://qdrant.qdrant.svc.cluster.local:6333")
    cfg["QDRANT_API_KEY"] = os.environ.get("QDRANT_API_KEY", "")
    cfg["COLLECTION_NAME"] = os.environ.get("COLLECTION_NAME", "default_rag_collection1")
    cfg["DENSE_URL"] = os.environ.get("DENSE_URL", "http://dense-svc.models.svc.cluster.local:8200")
    cfg["SPARSE_URL"] = os.environ.get("SPARSE_URL", "http://sparse-svc.models.svc.cluster.local:8201")
    cfg["RERANKER_URL"] = os.environ.get("RERANKER_URL", "http://reranker-svc.models.svc.cluster.local:8202")
    cfg["DENSE_DIM"] = int(os.environ.get("DENSE_DIM", "384"))
    cfg["HTTP_TIMEOUT"] = float(os.environ.get("HTTP_TIMEOUT", "10.0"))
    cfg["SPARSE_BATCH_FALLBACK"] = int(os.environ.get("SPARSE_BATCH_FALLBACK", "8"))
    cfg["API_KEY"] = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY") or ""
    cfg["LLM_MODEL"] = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
    cfg["LLM_MAX_TOKENS"] = int(os.environ.get("LLM_MAX_TOKENS", "512"))
    cfg["LLM_TEMPERATURE"] = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
    # AWS secret handling for staging: name of the k8s Secret containing AWS keys
    cfg["AWS_SECRET_NAME"] = os.environ.get("QUERY_AWS_SECRET", "aws-credentials")
    cfg["AWS_REGION"] = os.environ.get("AWS_REGION", "")
    # rbac/serviceaccount names
    cfg["SA_NAME"] = os.environ.get("QUERY_SA_NAME", f"{cfg['SERVICE_NAME']}-sa")
    cfg["ROLE_NAME"] = os.environ.get("QUERY_ROLE_NAME", f"{cfg['SERVICE_NAME']}-role")
    cfg["ROLEBIND_NAME"] = os.environ.get("QUERY_ROLEBIND_NAME", f"{cfg['SERVICE_NAME']}-rb")
    # labels
    cfg["LABELS"] = {
        "app.kubernetes.io/name": cfg["SERVICE_NAME"],
        "app.kubernetes.io/component": "inference",
        "app.kubernetes.io/managed-by": "gen_query",
        "app.kubernetes.io/instance": cfg["SERVICE_NAME"],
        "env": cfg["ENV"].lower(),
    }
    cfg["FILES"] = {
        "namespace": cfg["MANIFESTS_DIR"] / "00-namespace.yaml",
        "sa_role": cfg["MANIFESTS_DIR"] / "01-sa-role.yaml",
        "deployment": cfg["MANIFESTS_DIR"] / "02-deployment.yaml",
        "service": cfg["MANIFESTS_DIR"] / "03-service.yaml",
        "hpa": cfg["MANIFESTS_DIR"] / "04-hpa.yaml",
    }
    cfg["UUID_SHORT"] = str(uuid.uuid4())[:8]
    return cfg


def render_namespace(cfg):
    ns = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": cfg["NAMESPACE"], "labels": {"app.kubernetes.io/managed-by": "gen_query"}},
    }
    return yaml.safe_dump(ns, sort_keys=False)


def render_sa_role(cfg):
    sa = {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {"name": cfg["SA_NAME"], "namespace": cfg["NAMESPACE"]},
    }
    role = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {"name": cfg["ROLE_NAME"], "namespace": cfg["NAMESPACE"]},
        "rules": [
            {"apiGroups": [""], "resources": ["pods", "services", "endpoints", "configmaps"], "verbs": ["get", "list", "watch"]},
            {"apiGroups": [""], "resources": ["secrets"], "verbs": ["get"]},
        ],
    }
    rb = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": {"name": cfg["ROLEBIND_NAME"], "namespace": cfg["NAMESPACE"]},
        "subjects": [{"kind": "ServiceAccount", "name": cfg["SA_NAME"], "namespace": cfg["NAMESPACE"]}],
        "roleRef": {"kind": "Role", "name": cfg["ROLE_NAME"], "apiGroup": "rbac.authorization.k8s.io"},
    }
    return "\n---\n".join([yaml.safe_dump(x, sort_keys=False) for x in (sa, role, rb)])


def render_deployment(cfg):
    labels = cfg["LABELS"].copy()
    container_env = [
        {"name": "QDRANT_URL", "value": cfg["QDRANT_URL"]},
        {"name": "QDRANT_API_KEY", "value": cfg["QDRANT_API_KEY"]},
        {"name": "COLLECTION_NAME", "value": cfg["COLLECTION_NAME"]},
        {"name": "DENSE_URL", "value": cfg["DENSE_URL"]},
        {"name": "SPARSE_URL", "value": cfg["SPARSE_URL"]},
        {"name": "RERANKER_URL", "value": cfg["RERANKER_URL"]},
        {"name": "DENSE_DIM", "value": str(cfg["DENSE_DIM"])},
        {"name": "HTTP_TIMEOUT", "value": str(cfg["HTTP_TIMEOUT"])},
        {"name": "SPARSE_BATCH_FALLBACK", "value": str(cfg["SPARSE_BATCH_FALLBACK"])},
        {"name": "GROQ_API_KEY", "value": cfg["API_KEY"]},
        {"name": "OPENAI_API_KEY", "value": cfg["API_KEY"]},
        {"name": "LLM_MODEL", "value": cfg["LLM_MODEL"]},
        {"name": "LLM_MAX_TOKENS", "value": str(cfg["LLM_MAX_TOKENS"])},
        {"name": "LLM_TEMPERATURE", "value": str(cfg["LLM_TEMPERATURE"])},
        {"name": "ENV", "value": cfg["ENV"]},
        {"name": "LOG_LEVEL", "value": cfg["LOGLEVEL"]},
    ]

    # If staging — add AWS env entries sourced from a Kubernetes Secret (name configurable)
    if cfg["ENV"] == "STAGING":
        aws_secret = cfg.get("AWS_SECRET_NAME", "aws-credentials")
        container_env.extend([
            {"name": "AWS_ACCESS_KEY_ID", "valueFrom": {"secretKeyRef": {"name": aws_secret, "key": "AWS_ACCESS_KEY_ID"}}},
            {"name": "AWS_SECRET_ACCESS_KEY", "valueFrom": {"secretKeyRef": {"name": aws_secret, "key": "AWS_SECRET_ACCESS_KEY"}}},
        ])
        if cfg.get("AWS_REGION"):
            container_env.append({"name": "AWS_REGION", "value": cfg["AWS_REGION"]})

    container = {
        "name": cfg["SERVICE_NAME"],
        "image": cfg["IMAGE"],
        "ports": [{"containerPort": cfg["CONTAINER_PORT"]}],
        "env": container_env,
        "livenessProbe": {
            "httpGet": {"path": "/healthz", "port": cfg["CONTAINER_PORT"]},
            "initialDelaySeconds": cfg["LIVENESS_INITIAL_DELAY"],
            "periodSeconds": cfg["PROBE_PERIOD_SECONDS"],
            "timeoutSeconds": cfg["PROBE_TIMEOUT_SECONDS"],
            "failureThreshold": 3,
        },
        "readinessProbe": {
            "httpGet": {"path": "/readyz", "port": cfg["CONTAINER_PORT"]},
            "initialDelaySeconds": cfg["READINESS_INITIAL_DELAY"],
            "periodSeconds": cfg["PROBE_PERIOD_SECONDS"],
            "timeoutSeconds": cfg["PROBE_TIMEOUT_SECONDS"],
            "failureThreshold": 3,
        },
        "startupProbe": {
            "httpGet": {"path": "/healthz", "port": cfg["CONTAINER_PORT"]},
            "periodSeconds": cfg["PROBE_PERIOD_SECONDS"],
            "timeoutSeconds": cfg["PROBE_TIMEOUT_SECONDS"],
            "failureThreshold": cfg["STARTUP_FAILURE_THRESHOLD"],
        },
        "resources": {
            "requests": {"cpu": cfg["CPU_REQUEST"], "memory": cfg["MEMORY_REQUEST"]},
            "limits": {"cpu": cfg["CPU_LIMIT"], "memory": cfg["MEMORY_LIMIT"]},
        },
    }

    if cfg["ENABLE_GPU"]:
        try:
            gcount = int(cfg["GPU_COUNT"])
        except Exception:
            gcount = 1
        container["resources"]["limits"][cfg["GPU_RESOURCE_NAME"]] = gcount

    pod_spec = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-deployment", "namespace": cfg["NAMESPACE"], "labels": labels},
        "spec": {
            "replicas": cfg["REPLICAS"],
            "selector": {"matchLabels": {"app.kubernetes.io/name": cfg["SERVICE_NAME"]}},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "serviceAccountName": cfg["SA_NAME"],
                    "containers": [container],
                },
            },
        },
    }

    if cfg["ENABLE_GPU"] and cfg["GPU_NODE_SELECTOR"]:
        pod_spec["spec"]["template"]["spec"]["nodeSelector"] = {k: v for k, v in [cfg["GPU_NODE_SELECTOR"].split("=", 1)]} if "=" in cfg["GPU_NODE_SELECTOR"] else {cfg["GPU_NODE_SELECTOR"]: "true"}

    pod_spec["spec"]["template"]["metadata"].setdefault("annotations", {})
    pod_spec["spec"]["template"]["metadata"]["annotations"].update({
        "prometheus.io/scrape": "true",
        "prometheus.io/port": str(cfg["CONTAINER_PORT"]),
        "prometheus.io/path": "/metrics",
    })

    return yaml.safe_dump(pod_spec, sort_keys=False)


def render_service(cfg):
    svc = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-svc", "namespace": cfg["NAMESPACE"], "labels": cfg["LABELS"]},
        "spec": {
            "type": "ClusterIP",
            "ports": [{"port": cfg["CONTAINER_PORT"], "targetPort": cfg["CONTAINER_PORT"], "protocol": "TCP", "name": "http"}],
            "selector": {"app.kubernetes.io/name": cfg["SERVICE_NAME"]},
        },
    }
    return yaml.safe_dump(svc, sort_keys=False)


def render_hpa(cfg):
    hpa = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-hpa", "namespace": cfg["NAMESPACE"]},
        "spec": {
            "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": f"{cfg['SERVICE_NAME']}-deployment"},
            "minReplicas": cfg["HPA_MIN"],
            "maxReplicas": cfg["HPA_MAX"],
            "metrics": [
                {"type": "Resource", "resource": {"name": "cpu", "target": {"type": "Utilization", "averageUtilization": cfg["HPA_TARGET_CPU"]}}}
            ],
        },
    }
    return yaml.safe_dump(hpa, sort_keys=False)


def create_or_update_aws_secret_in_cluster(cfg, dry_run=False):
    """
    Create/update the AWS secret directly in the cluster using kubectl pipeline.
    This avoids writing the secret YAML to disk.
    Requires environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in the process running this script.
    """
    if cfg["ENV"] != "STAGING":
        log.debug("Not STAGING; skipping AWS secret creation.")
        return {"created": False, "reason": "not_staging"}

    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        log.warning("AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY not present in environment; skipping secret creation.")
        return {"created": False, "reason": "missing_env"}

    kubectl = shutil.which("kubectl")
    if not kubectl:
        log.error("kubectl not found; cannot create secret")
        return {"created": False, "reason": "kubectl_missing"}

    secret_name = cfg.get("AWS_SECRET_NAME", "aws-credentials")
    ns = cfg["NAMESPACE"]

    # Build safe, quoted args
    q_secret = shlex.quote(secret_name)
    q_ns = shlex.quote(ns)
    q_ak = shlex.quote(access_key)
    q_sk = shlex.quote(secret_key)

    # Use kubectl pipeline to create secret YAML and apply it in-memory.
    cmd = (
        f"{kubectl} create secret generic {q_secret} --namespace {q_ns} "
        f"--from-literal=AWS_ACCESS_KEY_ID={q_ak} "
        f"--from-literal=AWS_SECRET_ACCESS_KEY={q_sk} "
        f"--dry-run=client -o yaml | {kubectl} apply -f -"
    )

    if dry_run:
        log.info("Dry-run: would run secret create/apply command (not executed).")
        return {"created": False, "reason": "dry_run"}

    try:
        # run the pipeline; avoid logging secrets
        proc = subprocess.run(cmd, shell=True, capture_output=True, check=True, timeout=30)
        log.info("AWS secret '%s' created/updated in namespace '%s'.", secret_name, ns)
        return {"created": True, "stdout": proc.stdout.decode() if proc.stdout else ""}
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode() if e.stderr else str(e)
        log.error("Failed to create/update AWS secret: %s", err)
        return {"created": False, "stderr": err}
    except Exception as ex:
        log.error("Unexpected error creating secret: %s", ex)
        return {"created": False, "stderr": str(ex)}


def generate_manifests(cfg, dry_run=False, verbose=False):
    ensure_dir(cfg["MANIFESTS_DIR"])
    inputs_hash = canonical_inputs_hash(cfg)
    existing = None
    if cfg["INPUTS_HASH_PATH"].exists():
        existing = cfg["INPUTS_HASH_PATH"].read_text().strip()
    if existing == inputs_hash and not dry_run:
        log.info("No non-secret changes detected; generation skipped.")
        return

    ns_yaml = render_namespace(cfg)
    sa_role_yaml = render_sa_role(cfg)
    deploy_yaml = render_deployment(cfg)
    svc_yaml = render_service(cfg)
    atomic_write(cfg["FILES"]["namespace"], ns_yaml)
    atomic_write(cfg["FILES"]["sa_role"], sa_role_yaml)
    atomic_write(cfg["FILES"]["deployment"], deploy_yaml)
    atomic_write(cfg["FILES"]["service"], svc_yaml)
    if cfg["HPA_ENABLED"]:
        hpa_yaml = render_hpa(cfg)
        atomic_write(cfg["FILES"]["hpa"], hpa_yaml)
    cfg["INPUTS_HASH_PATH"].write_text(inputs_hash)
    log.info("Wrote manifests to %s", str(cfg["MANIFESTS_DIR"]))
    if verbose:
        log.info("Deployment (head):\n%s", deploy_yaml.splitlines()[:120])


def apply_to_cluster(cfg, dry_run=False, verbose=False):
    kubectl = shutil.which("kubectl")
    if not kubectl:
        log.error("kubectl not found in PATH; cannot apply")
        sys.exit(2)

    # Generate manifests (files)
    generate_manifests(cfg, dry_run=dry_run, verbose=verbose)

    # Create/update AWS secret in-cluster if STAGING and creds available (no file written)
    secret_res = create_or_update_aws_secret_in_cluster(cfg, dry_run=dry_run)
    if secret_res.get("created") is False and secret_res.get("reason") == "missing_env":
        log.warning("AWS secret was not created because env vars missing. Pods may fail if they require the secret.")

    if dry_run:
        log.info("Dry-run: skipping kubectl apply for manifests.")
        return

    files = [cfg["FILES"]["namespace"], cfg["FILES"]["sa_role"], cfg["FILES"]["deployment"], cfg["FILES"]["service"]]
    if cfg["HPA_ENABLED"]:
        files.append(cfg["FILES"]["hpa"])
    combined = ""
    for p in files:
        combined += f"---\n# source: {p.name}\n" + p.read_text() + "\n"
    res = kubectl_apply_yaml(combined, dry_run=False)
    if not res.get("applied", False):
        log.error("kubectl apply failed: %s", res.get("stderr") or res.get("error"))
        sys.exit(2)
    summary = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "image": cfg["IMAGE"],
        "namespace": cfg["NAMESPACE"],
        "replicas": cfg["REPLICAS"],
        "files": {k: str(v) for k, v in cfg["FILES"].items()},
    }
    atomic_write(cfg["MANIFESTS_DIR"] / "last_deploy_summary.json", json.dumps(summary, indent=2))
    log.info("Applied manifests to cluster and wrote deploy summary")


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
        log.info("Deleted manifests at %s", str(cfg["MANIFESTS_DIR"]))
    else:
        log.info("Manifests dir not present: %s", str(cfg["MANIFESTS_DIR"]))


def parse_args():
    p = argparse.ArgumentParser(description="Generate/apply Query (retrieval) Kubernetes manifests.")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--generate", action="store_true", help="Generate manifests to MANIFESTS_DIR.")
    grp.add_argument("--apply", action="store_true", help="Generate manifests and apply to cluster (requires kubectl).")
    grp.add_argument("--delete", action="store_true", help="Delete generated manifests.")
    p.add_argument("--dry-run", action="store_true", help="Render and validate but do not write or apply.")
    p.add_argument("--verbose", action="store_true", help="Print extra debug info.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    if args.delete:
        delete_manifests(cfg)
        return
    if args.generate:
        generate_manifests(cfg, dry_run=args.dry_run, verbose=args.verbose)
        return
    if args.apply:
        generate_manifests(cfg, dry_run=args.dry_run, verbose=args.verbose)
        if args.dry_run:
            log.info("Dry-run mode: skipping kubectl apply.")
            return
        apply_to_cluster(cfg, dry_run=args.dry_run, verbose=args.verbose)
        return


if __name__ == "__main__":
    main()
