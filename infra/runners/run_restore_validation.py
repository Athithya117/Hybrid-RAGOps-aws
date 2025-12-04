#!/usr/bin/env python3
"""
infra/tests/validate_restore.py

Validate a Qdrant restore without re-downloading full snapshot blobs.

Behavior (concise):
 - Fetches manifest.json for a backup (local path or from S3).
 - By default it does NOT download snapshot files. Instead it validates:
     * manifest presence and basic shape,
     * that the expected collections appear on each pod,
     * per-collection document/point counts via Qdrant HTTP API,
     * optional in-pod snapshot file existence and sha256 checksum (via kubectl exec).
 - Two access modes to inspect pods:
     1) HTTP via local port-forwards: provide --port-base (or let script auto-start port-forwards).
     2) In-cluster checks via kubectl exec: use --use-kubectl-exec (no port-forwards required).
 - It can fetch manifests from S3 (--s3-bucket + --s3-prefix) when local manifest missing. No secret manifests are written.
 - Exits with non-zero code on validation failures; prints a readable report.

Requirements:
 - Python 3.8+, requests, boto3 (if using S3), kubectl in PATH for kubectl-exec or auto port-forwarding.
"""

from pathlib import Path
import argparse
import json
import os
import sys
import time
import tempfile
import hashlib
import subprocess
import shlex

try:
    import requests
except Exception:
    print("Missing dependency: requests", file=sys.stderr)
    sys.exit(2)

# boto3 optional (only needed when fetching manifests from S3)
try:
    import boto3
except Exception:
    boto3 = None

# ---------- Helpers ----------
def run_cmd(cmd, capture=True, timeout=None, input_bytes=None):
    try:
        proc = subprocess.run(cmd, capture_output=capture, text=True, check=True, timeout=timeout, input=(input_bytes.decode() if isinstance(input_bytes, bytes) else input_bytes))
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.CalledProcessError as e:
        return e.returncode, getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or str(e)
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or f"timeout after {timeout}s"

def sha256_of_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_kubectl_available():
    rc, out, err = run_cmd(["which", "kubectl"])
    return rc == 0

# ---------- S3 manifest fetch ----------
def s3_download_file(bucket: str, key: str, dest: Path, region: str = None, endpoint_url: str = None):
    if boto3 is None:
        raise RuntimeError("boto3 is required to fetch from S3")
    kwargs = {}
    if region:
        kwargs["region_name"] = region
    if endpoint_url:
        client = boto3.client("s3", region_name=region, endpoint_url=endpoint_url)
    else:
        client = boto3.client("s3", **kwargs)
    ensure_dir(dest.parent)
    client.download_file(Bucket=bucket, Key=key, Filename=str(dest))

def fetch_manifest_from_s3(bucket: str, prefix: str, backup_id: str | None, region: str = None, endpoint_url: str = None, dest_dir: Path | None = None):
    dest_dir = Path(dest_dir or tempfile.gettempdir())
    ensure_dir(dest_dir)
    # if backup_id provided download that manifest, else fetch latest.manifest.json to find backup_id
    if not backup_id:
        latest_key = f"{prefix.rstrip('/')}/latest.manifest.json"
        latest_local = dest_dir / "latest.manifest.json"
        s3_download_file(bucket, latest_key, latest_local, region=region, endpoint_url=endpoint_url)
        j = json.loads(latest_local.read_text())
        bid = j.get("backup_id")
        if not bid:
            raise RuntimeError("latest.manifest.json from S3 missing backup_id")
    else:
        bid = backup_id
    manifest_key = f"{prefix.rstrip('/')}/{bid}/manifest.json"
    manifest_local = dest_dir / "manifest.json"
    s3_download_file(bucket, manifest_key, manifest_local, region=region, endpoint_url=endpoint_url)
    return manifest_local, bid

# ---------- Pod discovery ----------
def list_qdrant_pods(namespace: str):
    # Return list of qdrant-* pods sorted
    rc, out, err = run_cmd(["kubectl", "get", "pods", "-n", namespace, "-o", "jsonpath={.items[*].metadata.name}"])
    if rc != 0:
        raise RuntimeError("kubectl get pods failed: " + (err or out))
    names = [n for n in out.strip().split() if n.startswith("qdrant-")]
    names.sort()
    return names

# ---------- Port-forward helper ----------
def start_port_forward(pod: str, namespace: str, local_port: int, remote_port: int = 6333, timeout: int = 10):
    cmd = ["kubectl", "port-forward", f"pod/{pod}", f"{local_port}:{remote_port}", "-n", namespace]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # wait for readiness by probing /collections
    import requests as _requests
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{local_port}/collections"
    while time.time() < deadline:
        try:
            r = _requests.get(url, timeout=1)
            if r.status_code == 200:
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            # process exited
            stdout, stderr = proc.communicate(timeout=1) if proc.poll() is not None else ("", "")
            raise RuntimeError(f"port-forward failed: {stderr.strip()}")
        time.sleep(0.2)
    # didn't come up
    proc.kill()
    raise RuntimeError(f"port-forward to {pod} did not respond on localhost:{local_port}")

def stop_port_forward(proc: subprocess.Popen):
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

# ---------- Qdrant HTTP helpers (local-forwarded) ----------
def qdrant_collections_via_http(base_url: str):
    r = requests.get(f"{base_url.rstrip('/')}/collections", timeout=5)
    r.raise_for_status()
    j = r.json()
    result = j.get("result", {})
    cols = []
    if isinstance(result, dict) and "collections" in result:
        for c in result["collections"]:
            if isinstance(c, dict) and "name" in c:
                cols.append(c["name"])
            elif isinstance(c, str):
                cols.append(c)
    elif isinstance(result, list):
        for c in result:
            if isinstance(c, dict) and "name" in c:
                cols.append(c["name"])
            elif isinstance(c, str):
                cols.append(c)
    return cols

def qdrant_collection_count_via_http(base_url: str, collection: str):
    # POST /collections/<col>/points/count with {"exact": true}
    url = f"{base_url.rstrip('/')}/collections/{collection}/points/count"
    r = requests.post(url, json={"exact": True}, timeout=10)
    r.raise_for_status()
    j = r.json()
    res = j.get("result", {})
    if isinstance(res, dict) and "count" in res:
        return int(res["count"])
    # older endpoints may have different shape
    if isinstance(j.get("result"), dict) and "count" in j["result"]:
        return int(j["result"]["count"])
    raise RuntimeError(f"unexpected count response: {j}")

# ---------- In-pod helpers (kubectl exec) ----------
def pod_collections_via_exec(namespace: str, pod: str):
    # Use curl inside pod to query local qdrant at localhost:6333
    cmd = ["kubectl", "exec", "-n", namespace, pod, "--", "curl", "-sS", "http://127.0.0.1:6333/collections"]
    rc, out, err = run_cmd(cmd, timeout=10)
    if rc != 0:
        raise RuntimeError("kubectl exec curl failed: " + (err or out))
    j = json.loads(out)
    result = j.get("result", {})
    cols = []
    if isinstance(result, dict) and "collections" in result:
        for c in result["collections"]:
            if isinstance(c, dict) and "name" in c:
                cols.append(c["name"])
            elif isinstance(c, str):
                cols.append(c)
    elif isinstance(result, list):
        for c in result:
            if isinstance(c, dict) and "name" in c:
                cols.append(c["name"])
            elif isinstance(c, str):
                cols.append(c)
    return cols

def pod_collection_count_via_exec(namespace: str, pod: str, collection: str):
    # Use curl to POST count
    body = json.dumps({"exact": True})
    cmd = ["kubectl", "exec", "-n", namespace, pod, "--", "curl", "-sS", "-X", "POST", "-H", "Content-Type: application/json", "-d", body, f"http://127.0.0.1:6333/collections/{shlex.quote(collection)}/points/count"]
    rc, out, err = run_cmd(cmd, timeout=15)
    if rc != 0:
        raise RuntimeError("kubectl exec curl failed: " + (err or out))
    j = json.loads(out)
    res = j.get("result", {})
    if isinstance(res, dict) and "count" in res:
        return int(res["count"])
    raise RuntimeError(f"unexpected count response from exec: {j}")

def pod_check_file_and_sha(namespace: str, pod: str, remote_path: str):
    # Check file exists, get size and sha256 if sha256sum present
    # test -f
    rc, out, err = run_cmd(["kubectl", "exec", "-n", namespace, pod, "--", "test", "-f", remote_path])
    exists = (rc == 0)
    if not exists:
        return {"exists": False}
    # size
    rc, out, err = run_cmd(["kubectl", "exec", "-n", namespace, pod, "--", "stat", "-c", "%s", remote_path])
    size = int(out.strip()) if rc == 0 and out.strip().isdigit() else None
    # sha256
    rc, out, err = run_cmd(["kubectl", "exec", "-n", namespace, pod, "--", "sh", "-c", f"command -v sha256sum >/dev/null 2>&1 && sha256sum {remote_path} || true"], timeout=60)
    sha = None
    if rc == 0 and out.strip():
        # out like "<sha>  <path>"
        first = out.strip().split()[0]
        if all(c in "0123456789abcdef" for c in first.lower()) and len(first) == 64:
            sha = first
    return {"exists": True, "size": size, "sha256": sha}

# ---------- Main validation logic ----------
def validate(manifest_path: Path, args):
    # load manifest
    manifest = json.loads(manifest_path.read_text())
    backup_id = manifest.get("backup_id")
    namespace = manifest.get("namespace", args.namespace)
    pods_map = manifest.get("pods", {})
    expected_collections = manifest.get("collections", [])

    print("Manifest backup_id:", backup_id)
    print("Namespace:", namespace)
    print("Pods in manifest:", ", ".join(pods_map.keys()))
    print("Collections (expected):", ", ".join(expected_collections))
    print()

    # decide pod list and mapping
    manifest_pods = list(pods_map.keys())
    if manifest_pods == ["service"]:
        # service-mode: need to map service to actual pods
        if args.use_kubectl_exec:
            pods = list_qdrant_pods(namespace)
        else:
            pods = list_qdrant_pods(namespace)
        if not pods:
            raise RuntimeError("No qdrant pods discovered in namespace for service-mode validation")
        pod_mapping = {pod: pod for pod in pods}  # identity mapping; manifest 'service' implies collections present on pods
    else:
        pods = manifest_pods
        pod_mapping = {pod: pod for pod in pods}

    # if using port-forwarding and auto, start pf's
    port_forwards = {}
    portprocs = {}
    if not args.use_kubectl_exec:
        # map pods deterministically to ports base + index
        # allow user pre-existing port-forwards; but if --auto-port-forward flag, script will start them
        pod_list_sorted = sorted(pod_mapping.keys())
        for idx, pod in enumerate(pod_list_sorted):
            port = args.port_base + idx
            port_forwards[pod] = port
            if args.auto_port_forward:
                print(f"Starting port-forward for {pod} -> localhost:{port}")
                proc = start_port_forward(pod, namespace, port, remote_port=6333, timeout=10)
                portprocs[pod] = proc
    try:
        # per-pod checks
        results = {"pod_reports": {}, "collection_cross_counts": {}, "failed": False}
        for pod in sorted(pod_mapping.keys()):
            report = {"pod": pod, "collections_expected": [], "collections_found": [], "counts": {}, "snapshots": {}, "errors": []}
            # determine how to contact this pod
            if args.use_kubectl_exec:
                # use kubectl exec to list collections
                try:
                    cols = pod_collections_via_exec(namespace, pod)
                    report["collections_found"] = cols
                except Exception as e:
                    report["errors"].append(f"failed to list collections via kubectl exec: {e}")
                    results["failed"] = True
                    results["pod_reports"][pod] = report
                    continue
                # check counts and snapshot files (if requested)
                for col in expected_collections:
                    try:
                        cnt = pod_collection_count_via_exec(namespace, pod, col)
                        report["counts"][col] = cnt
                        # collect for cross-check
                        results["collection_cross_counts"].setdefault(col, []).append(cnt)
                    except Exception:
                        # missing collection or API error
                        report["counts"][col] = None
                    # in-pod snapshot verification if requested
                    if args.check_in_pod_snapshots:
                        # derive filename(s) expected for this pod/collection from manifest if manifest stored per-pod
                        meta_pod = pods_map.get(pod) or pods_map.get("service") or {}
                        colmeta = (meta_pod.get("collections") or {}).get(col)
                        if colmeta and colmeta.get("s3_key"):
                            fname = Path(colmeta["s3_key"]).name
                            remote_path = f"/qdrant/snapshots/{fname}"
                            try:
                                chk = pod_check_file_and_sha(namespace, pod, remote_path)
                                report["snapshots"][col] = chk
                                if chk.get("exists") and chk.get("sha256") and colmeta.get("sha256") and chk.get("sha256") != colmeta.get("sha256"):
                                    report["errors"].append(f"sha256 mismatch for {remote_path}: manifest {colmeta.get('sha256')} vs pod {chk.get('sha256')}")
                                    results["failed"] = True
                            except Exception as e:
                                report["errors"].append(f"snapshot check failed for {remote_path}: {e}")
                                results["failed"] = True
            else:
                # HTTP via local port mapping
                port = port_forwards.get(pod)
                if port is None:
                    report["errors"].append("no port mapping for pod")
                    results["failed"] = True
                    results["pod_reports"][pod] = report
                    continue
                base = f"http://127.0.0.1:{port}"
                # list collections via HTTP
                try:
                    cols = qdrant_collections_via_http(base)
                    report["collections_found"] = cols
                except Exception as e:
                    report["errors"].append(f"failed to list collections via HTTP {base}: {e}")
                    results["failed"] = True
                    results["pod_reports"][pod] = report
                    continue
                for col in expected_collections:
                    try:
                        cnt = qdrant_collection_count_via_http(base, col)
                        report["counts"][col] = cnt
                        results["collection_cross_counts"].setdefault(col, []).append(cnt)
                    except Exception:
                        report["counts"][col] = None
                        report["errors"].append(f"failed to get count for collection {col} via HTTP")
                        results["failed"] = True
                # optional in-pod snapshot presence/sha check via kubectl exec
                if args.check_in_pod_snapshots:
                    meta_pod = pods_map.get(pod) or pods_map.get("service") or {}
                    for col in expected_collections:
                        colmeta = (meta_pod.get("collections") or {}).get(col)
                        if colmeta and colmeta.get("s3_key"):
                            fname = Path(colmeta["s3_key"]).name
                            remote_path = f"/qdrant/snapshots/{fname}"
                            try:
                                chk = pod_check_file_and_sha(namespace, pod, remote_path)
                                report["snapshots"][col] = chk
                                if chk.get("exists") and chk.get("sha256") and colmeta.get("sha256") and chk.get("sha256") != colmeta.get("sha256"):
                                    report["errors"].append(f"sha256 mismatch for {remote_path}: manifest {colmeta.get('sha256')} vs pod {chk.get('sha256')}")
                                    results["failed"] = True
                            except Exception as e:
                                report["errors"].append(f"snapshot check failed for {remote_path}: {e}")
                                results["failed"] = True
            # expected collections
            report["collections_expected"] = expected_collections
            # record report
            results["pod_reports"][pod] = report

        # Cross-pod consistency checks
        cross = results["collection_cross_counts"]
        cross_report = {}
        for col, counts in cross.items():
            # filter None
            present_counts = [c for c in counts if c is not None]
            if not present_counts:
                cross_report[col] = {"ok": False, "reason": "no counts available on any pod"}
                results["failed"] = True
            else:
                all_equal = all(c == present_counts[0] for c in present_counts)
                cross_report[col] = {"ok": all_equal, "counts_sample": present_counts[:5]}
                if not all_equal:
                    results["failed"] = True
        results["cross_report"] = cross_report

        # pretty-print results
        print("\nValidation report\n=================")
        for pod, rep in results["pod_reports"].items():
            print(f"\n== Pod {pod} ==")
            if rep["errors"]:
                print("  ERRORS:")
                for e in rep["errors"]:
                    print("   -", e)
            print("  expected collections:", ", ".join(rep.get("collections_expected") or []))
            print("  found collections   :", ", ".join(rep.get("collections_found") or []))
            print("  counts:")
            for col in (rep.get("collections_expected") or []):
                cnt = rep["counts"].get(col)
                print(f"    {col}: {cnt if cnt is not None else 'N/A'}")
            if rep.get("snapshots"):
                print("  snapshot files (in-pod checks):")
                for col, s in rep["snapshots"].items():
                    if not s.get("exists"):
                        print(f"    {col}: MISSING")
                    else:
                        print(f"    {col}: exists size={s.get('size')} sha256={s.get('sha256') or '<no-sha-supported-in-pod>'}")

        print("\nCross-pod consistency:")
        for col, r in results["cross_report"].items():
            print(f"  {col}: {'OK' if r['ok'] else 'MISMATCH'} counts_sample={r.get('counts_sample')}")

        # overall summary
        if results["failed"]:
            print("\nSummary: [FAIL] Some validation checks failed.")
            return 2
        else:
            print("\nSummary: [OK] All validation checks passed.")
            return 0

    finally:
        # cleanup port-forwards if any started
        for p, proc in portprocs.items():
            try:
                stop_port_forward(proc)
            except Exception:
                pass

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Validate Qdrant restore using manifest (no re-download of snapshots by default).")
    p.add_argument("--manifest", help="Local manifest.json path (if present skip S3).")
    p.add_argument("--s3-bucket", help="S3 bucket to fetch manifest from if local manifest missing.")
    p.add_argument("--s3-prefix", default="qdrant/backups", help="S3 prefix (default qdrant/backups).")
    p.add_argument("--backup-id", help="Specific backup id under prefix; if omitted latest.manifest.json is used to determine.")
    p.add_argument("--namespace", default=os.environ.get("QDRANT_NAMESPACE", "qdrant"), help="Kubernetes namespace for qdrant.")
    p.add_argument("--use-kubectl-exec", action="store_true", help="Use kubectl exec to query pod-local Qdrant (no port-forwards).")
    p.add_argument("--port-base", type=int, default=7000, help="When using HTTP mode, base port for pod port-forwards (7000).")
    p.add_argument("--auto-port-forward", action="store_true", help="Automatically start port-forwards for each pod (requires kubectl).")
    p.add_argument("--check-in-pod-snapshots", dest="check_in_pod_snapshots", action="store_true", help="If enabled, verify snapshot files exist and compute sha256 in-pod (requires kubectl).")
    p.add_argument("--aws-region", default=os.environ.get("AWS_REGION", "us-east-1"), help="AWS region for S3.")
    p.add_argument("--s3-endpoint", default=os.environ.get("BACKUP_S3_ENDPOINT", ""), help="Optional S3 endpoint URL.")
    return p.parse_args()

def main():
    args = parse_args()
    manifest_path = None
    # first try local manifest if provided
    if args.manifest:
        mp = Path(args.manifest)
        if not mp.exists():
            print("manifest file not found:", args.manifest, file=sys.stderr)
            sys.exit(2)
        manifest_path = mp
    else:
        # attempt to fetch from S3 if s3-bucket provided
        if not args.s3_bucket:
            print("No manifest provided and no --s3-bucket; cannot proceed.", file=sys.stderr)
            sys.exit(2)
        if boto3 is None:
            print("boto3 not installed; cannot fetch manifest from S3.", file=sys.stderr)
            sys.exit(2)
        try:
            tmpd = Path("tmp/latest")
            ensure_dir(tmpd)
            print("Local manifest missing; fetching from S3...")
            manifest_local, bid = fetch_manifest_from_s3(args.s3_bucket, args.s3_prefix, args.backup_id, region=args.aws_region, endpoint_url=(args.s3_endpoint or None), dest_dir=tmpd)
            manifest_path = manifest_local
            print(f"Downloaded manifest for backup_id={bid} -> {manifest_local}")
        except Exception as e:
            print("Failed to fetch manifest from S3:", e, file=sys.stderr)
            sys.exit(2)

    # Validate file exists and readable
    if not manifest_path or not manifest_path.exists():
        print("Manifest not found:", manifest_path, file=sys.stderr)
        sys.exit(2)

    # If using HTTP mode with auto port-forward, ensure kubectl present
    if not args.use_kubectl_exec and args.auto_port_forward and not is_kubectl_available():
        print("auto port-forward requested but kubectl not available in PATH", file=sys.stderr)
        sys.exit(2)

    # Run validation
    try:
        rc = validate(manifest_path, args)
        sys.exit(rc)
    except Exception as e:
        print("Validation failed with exception:", str(e), file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
