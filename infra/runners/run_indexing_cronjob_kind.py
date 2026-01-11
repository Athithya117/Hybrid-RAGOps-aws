#!/usr/bin/env python3
"""
run_indexing_cronjob_kind.py

Kind-local variant of run_indexing_cronjob with robust DNS probing and conservative
CoreDNS remediation. Makes the runner resilient to:
 - stale probe pods / name collisions
 - probing the correct service name when QDRANT runs in a non-default namespace
 - avoiding unnecessary CoreDNS restarts for NXDOMAIN (wrong name) results
 - retrying useful transient failures (timeouts / SERVFAIL) and attempting a
   safe rollout restart of the coredns deployment if needed

Behavior and envs (sane defaults):
 - NAMESPACE (default "indexing")
 - CRONJOB (default "indexing-backup-cronjob")
 - RUNNER_TIMEOUT (default 3600)
 - DEBUG_INDEXING_POD / NO_CLEANUP
 - QDRANT_URL (default "http://qdrant:6333") -- can be full host or host:port
 - QDRANT_NAMESPACE (default "qdrant") -- used when QDRANT_URL host is short name
 - DNS_FIX_ENABLED (default "true")
 - DNS_PROBE_RETRIES (default 6)
 - DNS_PROBE_BACKOFF_BASE (default 1.5)
 - DNS_PROBE_TIMEOUT (default 6)
 - DNS_FIX_RESTART_ATTEMPTS (default 2)
 - DNS_FIX_WAIT_FOR_READY (default 60)
"""

from __future__ import annotations

import os
import sys
import time
import json
import signal
import uuid
import datetime
import subprocess
import threading
from typing import List, Tuple, Optional, Dict

# ----- helpers -----------------------------------------------------------------

def run_cmd(cmd: List[str], input_text: Optional[str] = None, timeout: Optional[int] = None) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, input=input_text, capture_output=True, text=True, timeout=timeout, check=True)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout or "", e.stderr or ""
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or f"timeout after {timeout}s"
    except Exception as e:
        return 255, "", str(e)

def now_ts() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def info(msg: str, *args) -> None:
    print(f"{now_ts()} [info] " + (msg % args), flush=True)

def warn(msg: str, *args) -> None:
    print(f"{now_ts()} [warn] " + (msg % args), file=sys.stderr, flush=True)

def err(msg: str, *args) -> None:
    print(f"{now_ts()} [error] " + (msg % args), file=sys.stderr, flush=True)

def kubectl_available() -> bool:
    rc, _, _ = run_cmd(["kubectl", "version", "--client=true"], timeout=5)
    return rc == 0

# ----- CoreDNS helpers --------------------------------------------------------

def list_coredns_pods() -> List[str]:
    # Common label selectors; try several and aggregate results
    selectors = [
        ("-l", "k8s-app=kube-dns"),
        ("-l", "app.kubernetes.io/name=coredns"),
        ("-l", "k8s-app=coredns"),
        ("-l", "k8s-app=kube-dns,kubernetes.io/name=coredns"),
    ]
    pods = []
    for arg, sel in selectors:
        rc, out, _ = run_cmd(["kubectl", "get", "pods", "-n", "kube-system", arg, sel, "-o", "jsonpath={.items[*].metadata.name}"], timeout=5)
        if rc == 0 and out.strip():
            for n in out.strip().split():
                if n not in pods:
                    pods.append(n)
    return pods

def coredns_deployment_exists() -> bool:
    rc, _, _ = run_cmd(["kubectl", "get", "deploy", "-n", "kube-system", "coredns"], timeout=5)
    return rc == 0

def rollout_restart_coredns() -> Tuple[bool, str]:
    if not coredns_deployment_exists():
        warn("coredns deployment not found in kube-system; skipping rollout restart")
        return False, "no-deployment"
    rc, out, errout = run_cmd(["kubectl", "rollout", "restart", "deployment/coredns", "-n", "kube-system"], timeout=20)
    if rc != 0:
        warn("rollout restart coredns failed: %s", errout or out)
        return False, errout or out
    info("triggered rollout restart for coredns")
    return True, out

def wait_for_coredns_ready(timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        pods = list_coredns_pods()
        if not pods:
            time.sleep(1)
            continue
        all_ok = True
        for p in pods:
            rc, out, _ = run_cmd(["kubectl", "get", "pod", p, "-n", "kube-system", "-o", "jsonpath={.status.containerStatuses[*].ready}"], timeout=5)
            if rc != 0 or "true" not in out:
                all_ok = False
                break
        if all_ok:
            info("coredns pods reported ready: %s", ", ".join(pods))
            return True
        time.sleep(1)
    warn("timed out waiting for coredns ready (timeout=%ds)", timeout)
    return False

# ----- DNS probe helpers -----------------------------------------------------

def _unique_probe_pod_name(prefix: str = "dns-probe") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

def dns_probe_nslookup(host: str, probe_images: Optional[List[str]] = None, probe_timeout: int = 6) -> Tuple[bool, str, str]:
    """
    Run nameserver probe using a transient pod that executes 'nslookup <host>'.
    Returns (ok, reason, combined_output).
    reason values: "OK", "NXDOMAIN", "SERVFAIL", "TIMEOUT", "CMD_ERROR", "OTHER_ERROR"
    """
    if probe_images is None:
        probe_images = ["busybox:1.36", "infoblox/dnstools:latest", "nicolaka/netshoot:latest"]
    last_out = ""
    for image in probe_images:
        pod_name = _unique_probe_pod_name()
        # Try to run nslookup in a temporary pod with a short timeout
        cmd = ["kubectl", "run", pod_name, "--restart=Never", "--image", image, "--", "sh", "-c", f"nslookup {host} || true"]
        rc, out, errout = run_cmd(cmd, timeout=probe_timeout + 2)
        combined = (out or "") + "\n" + (errout or "")
        last_out = combined.strip()
        # Clean up any created pod object if it still exists (kubectl run --rm may not delete Error pods)
        _ = run_cmd(["kubectl", "delete", "pod", pod_name, "-n", "default", "--ignore-not-found"], timeout=5)
        # Interpret results
        if rc == 0:
            # success; examine output for NXDOMAIN / SERVER failures just in case
            lo = combined.lower()
            if "server can't find" in lo or "nxdomain" in lo:
                info("nslookup returned NXDOMAIN (image=%s)", image)
                return False, "NXDOMAIN", last_out
            if "servfail" in lo:
                info("nslookup returned SERVFAIL (image=%s)", image)
                return False, "SERVFAIL", last_out
            # normal positive lookup
            return True, "OK", last_out
        else:
            # rc != 0: could be timeout, image not present, command not found, etc.
            if "command not found" in combined.lower() or "no such file or directory" in combined.lower():
                warn("probe image %s missing nslookup; trying next", image)
                continue
            if "timeout" in combined.lower() or rc == 124:
                warn("probe image %s timed out", image)
                return False, "TIMEOUT", last_out
            # If busybox reports NXDOMAIN but rc != 0, catch NXDOMAIN
            if "nxdomain" in combined.lower() or "server can't find" in combined.lower():
                info("nslookup (rc!=0) produced NXDOMAIN")
                return False, "NXDOMAIN", last_out
            # else, unknown error; return OTHER_ERROR to consider remedial action
            warn("nslookup probe using image=%s failed rc=%s stderr=%s stdout=%s", image, rc, errout[:300] if errout else "", out[:300] if out else "")
            return False, "OTHER_ERROR", last_out
    # exhausted images
    return False, "OTHER_ERROR", last_out

def http_probe_url(url: str, probe_image: str = "curlimages/curl:8.3.0", timeout: int = 10) -> Tuple[bool, str, str]:
    """
    Simple HTTP GET probe using curl in an ephemeral pod.
    Returns (ok, reason, output).
    """
    pod_name = _unique_probe_pod_name("http-probe")
    cmd = ["kubectl", "run", pod_name, "--restart=Never", "--image", probe_image, "--", "sh", "-c", f"curl -fsS --max-time {timeout} {url} || true"]
    rc, out, errout = run_cmd(cmd, timeout=timeout + 5)
    combined = (out or "") + "\n" + (errout or "")
    _ = run_cmd(["kubectl", "delete", "pod", pod_name, "-n", "default", "--ignore-not-found"], timeout=5)
    if rc == 0 and out:
        return True, "OK", combined
    if rc == 124 or "timeout" in combined.lower():
        return False, "TIMEOUT", combined
    return False, "HTTP_ERROR", combined

# ----- high-level DNS/service readiness with conservative remediation ---------

def normalize_service_host(host: str, namespace: str) -> str:
    """
    If host already contains dots, assume fully-qualified and return as-is.
    Otherwise return <host>.<namespace>.svc.cluster.local
    """
    if "." in host:
        return host
    return f"{host}.{namespace}.svc.cluster.local"

def ensure_dns_and_service_reachable(service_host: str,
                                     service_http_url: Optional[str],
                                     max_probe_attempts: int = 6,
                                     backoff_base: float = 1.5,
                                     probe_timeout: int = 6,
                                     enable_fix: bool = True,
                                     restart_attempts: int = 2,
                                     wait_for_ready: int = 60) -> bool:
    """
    Probe DNS and optional HTTP endpoint. If DNS failures are TIMEOUT/SERVFAIL/OTHER_ERROR
    and enable_fix is true, try a safe rollout restart of coredns up to restart_attempts.
    Do NOT restart on NXDOMAIN (that indicates wrong name / missing service).
    Returns True when DNS resolves and HTTP (if provided) returns OK.
    """
    attempt = 0
    restart_count = 0
    last_reason = None
    while attempt < max_probe_attempts:
        attempt += 1
        info("DNS/HTTP probe attempt %d/%d for host=%s url=%s", attempt, max_probe_attempts, service_host, service_http_url or "<none>")
        dns_ok, dns_reason, dns_out = dns_probe_nslookup(service_host, probe_timeout=probe_timeout)
        last_reason = dns_reason
        if dns_ok:
            http_ok = True
            http_reason = "SKIPPED"
            http_out = ""
            if service_http_url:
                http_ok, http_reason, http_out = http_probe_url(service_http_url, timeout=probe_timeout)
            if dns_ok and http_ok:
                info("Service reachable: dns=%s http=%s", dns_reason, http_reason)
                return True
            # DNS good but HTTP failed -> treat as remedial candidate
            warn("DNS OK but HTTP probe failed: http_reason=%s; http_out=%s", http_reason, (http_out or "")[:300])
            last_reason = http_reason
        else:
            info("DNS probe result=%s output=%s", dns_reason, (dns_out or "")[:300])

        # Decide if we should attempt remediation: only for TIMEOUT, SERVFAIL, OTHER_ERROR
        if enable_fix and dns_reason in ("TIMEOUT", "SERVFAIL", "OTHER_ERROR", "HTTP_ERROR") and restart_count < restart_attempts:
            info("Attempting coredns remediation (%d/%d) due to reason=%s", restart_count + 1, restart_attempts, dns_reason)
            ok, msg = rollout_restart_coredns()
            if ok:
                restart_count += 1
                info("Waiting %ds for coredns pods to become ready after restart", wait_for_ready)
                if wait_for_coredns_ready(timeout=wait_for_ready):
                    info("coredns ready after restart; will re-probe")
                    time.sleep(2)
                    continue
                else:
                    warn("coredns did not become ready after restart attempt; will retry probes")
            else:
                warn("failed to trigger rollout restart of coredns: %s", msg)
        else:
            if dns_reason == "NXDOMAIN":
                # wrong name / wrong namespace: don't remediate
                info("DNS returned NXDOMAIN for %s - likely wrong host or namespace; will NOT attempt CoreDNS remediation", service_host)
                return False
        # backoff before next probe
        sleep_for = min(60.0, backoff_base * (2 ** (attempt - 1)))
        jitter = sleep_for * (0.3 + 0.7 * ((time.time() * 1000) % 1))
        info("Probe failed; sleeping %.1fs before next attempt (last_reason=%s)", jitter, last_reason)
        time.sleep(jitter)
    warn("DNS/HTTP probes exhausted; last_reason=%s", last_reason)
    return False

# ----- Job orchestration adapted from original runner -------------------------

def create_secret_from_env(namespace: str, secret_name: str, mapping: Dict[str, str]) -> bool:
    literals = []
    for envvar, key in mapping.items():
        val = os.environ.get(envvar)
        if val:
            literals += ["--from-literal", f"{key}={val}"]
    if not literals:
        return False
    cmd = ["kubectl", "create", "secret", "generic", secret_name, "-n", namespace, "--dry-run=client", "-o", "yaml"] + literals
    rc, out, errout = run_cmd(cmd, timeout=20)
    if rc != 0:
        err("rendering secret YAML failed: %s", errout or out)
        return False
    rc2, out2, err2 = run_cmd(["kubectl", "apply", "-f", "-"], input_text=out, timeout=20)
    if rc2 != 0:
        err("applying secret failed: %s", err2 or out2)
        return False
    info("secret/%s created/updated in namespace %s", secret_name, namespace)
    return True

def safe_job_name(cronjob_name: str) -> str:
    ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    base = f"{cronjob_name}-manual-{ts}".lower()
    base = "".join(c if (c.isalnum() or c == "-") else "-" for c in base)
    base = base[:63].strip("-")
    if not base:
        raise SystemExit("generated job name empty (unexpected)")
    return base

def _fetch_cronjob_json(namespace: str, cronjob: str) -> Optional[dict]:
    rc, out, _ = run_cmd(["kubectl", "get", "cronjob", cronjob, "-n", namespace, "-o", "json"], timeout=10)
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except Exception:
        return None

def _pos_int_or_none(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    try:
        v = int(s)
        if v > 0:
            return v
    except Exception:
        pass
    return None

def _create_job_from_cronjob_spec(namespace: str, jobname: str, cj_json: dict, parallelism: Optional[int], completions: Optional[int]) -> None:
    if "spec" not in cj_json:
        raise RuntimeError("CronJob JSON missing spec")
    jt = cj_json["spec"].get("jobTemplate")
    if not jt or "spec" not in jt:
        raise RuntimeError("CronJob jobTemplate.spec missing")
    job_spec = jt["spec"]
    job_spec = json.loads(json.dumps(job_spec))
    if parallelism is not None:
        job_spec["parallelism"] = parallelism
    if completions is not None:
        job_spec["completions"] = completions
    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": jobname,
            "namespace": namespace,
            "labels": {"created-by": "manual-runner", "cronjob": cj_json.get("metadata", {}).get("name", "")},
        },
        "spec": job_spec,
    }
    payload = json.dumps(job_manifest)
    rc, out, errout = run_cmd(["kubectl", "apply", "-f", "-"], input_text=payload, timeout=15)
    if rc != 0:
        raise RuntimeError(f"kubectl apply job manifest failed: {errout or out}")
    info("Created Job: %s (from CronJob jobTemplate.spec)", jobname)

def create_job_from_cronjob(namespace: str, cronjob: str, jobname: str) -> None:
    p_env = _pos_int_or_none(os.environ.get("CRONJOB_PARALLELISM"))
    c_env = _pos_int_or_none(os.environ.get("CRONJOB_COMPLETIONS"))
    cj_json = _fetch_cronjob_json(namespace, cronjob)
    if cj_json:
        try:
            _create_job_from_cronjob_spec(namespace, jobname, cj_json, p_env, c_env)
            return
        except Exception as e:
            warn("creating Job from CronJob spec failed: %s", e)
    rc, out, errout = run_cmd(["kubectl", "create", "job", jobname, "--from=cronjob/" + cronjob, "-n", namespace], timeout=20)
    if rc != 0:
        raise RuntimeError(f"kubectl create job failed: {errout or out}")
    info("Created Job: %s (fallback create)", jobname)

def list_pods_for_job(namespace: str, jobname: str) -> List[str]:
    rc, out, _ = run_cmd(["kubectl", "get", "pods", "-n", namespace, "-l", f"job-name={jobname}", "-o", "jsonpath={.items[*].metadata.name}"], timeout=10)
    if rc != 0:
        return []
    s = out.strip()
    return s.split() if s else []

def pod_phase(namespace: str, pod: str) -> Optional[str]:
    rc, out, _ = run_cmd(["kubectl", "get", "pod", pod, "-n", namespace, "-o", "jsonpath={.status.phase}"], timeout=5)
    return out.strip() if rc == 0 else None

def job_status_counts(namespace: str, jobname: str) -> Tuple[int, int, int]:
    rc, out, _ = run_cmd(["kubectl", "get", "job", jobname, "-n", namespace, "-o", "jsonpath={.status.active}{'|'}{.status.succeeded}{'|'}{.status.failed}"], timeout=5)
    if rc != 0 or not out:
        return 0, 0, 0
    parts = out.strip().split("|")
    def safe_int(i: int) -> int:
        try:
            p = parts[i]
            if p and p != "<none>":
                return int(p)
        except Exception:
            pass
        return 0
    return safe_int(0), safe_int(1), safe_int(2)

def get_container_name_for_pod(namespace: str, pod: str) -> Optional[str]:
    rc, out, _ = run_cmd(["kubectl", "get", "pod", pod, "-n", namespace, "-o", "jsonpath={.spec.containers[*].name}"], timeout=5)
    if rc != 0 or not out:
        return None
    names = out.strip().split()
    if "indexer" in names:
        return "indexer"
    return names[0] if names else None

class PodLogStreamer:
    def __init__(self, namespace: str, pod: str, container: Optional[str] = None):
        self.namespace = namespace
        self.pod = pod
        self.container = container
        self.proc = None
        self.thread = None
        self.stop_event = threading.Event()
        self.started_event = threading.Event()

    def _build_cmd(self) -> List[str]:
        cmd = ["kubectl", "logs", "-n", self.namespace, "-f", self.pod]
        if self.container:
            cmd += ["-c", self.container]
        return cmd

    def _wait_for_pod_running(self, timeout: int = 300) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline and not self.stop_event.is_set():
            ph = pod_phase(self.namespace, self.pod)
            if not ph:
                time.sleep(0.5)
                continue
            if ph in ("Running", "Succeeded", "Failed"):
                return True
            time.sleep(0.5)
        return False

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.started_event.wait(timeout=5)

    def _run(self) -> None:
        ok = self._wait_for_pod_running(timeout=300)
        if not ok:
            warn("pod %s did not become ready within timeout", self.pod)
            self.started_event.set()
            return
        cmd = self._build_cmd()
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except Exception as e:
            err("failed to start log stream for %s: %s", self.pod, e)
            self.started_event.set()
            return
        prefix = f"[{self.pod}/{self.container or 'main'}]"
        self.started_event.set()
        try:
            assert self.proc is not None
            for line in self.proc.stdout:
                if line is None:
                    break
                if self.stop_event.is_set():
                    break
                sys.stdout.write(prefix + " " + line)
            self.proc.wait()
        except Exception as e:
            err("log stream error for %s: %s", self.pod, e)
        finally:
            try:
                if self.proc and self.proc.poll() is None:
                    self.proc.terminate()
            except Exception:
                pass

    def stop(self, timeout: int = 5) -> None:
        self.stop_event.set()
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=timeout)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
        except Exception:
            pass
        if self.thread:
            self.thread.join(timeout=timeout)

# ----- main -------------------------------------------------------------------

def main() -> None:
    if not kubectl_available():
        err("kubectl required in PATH")
        sys.exit(2)

    ns = os.environ.get("NAMESPACE", "indexing")
    cronjob = os.environ.get("CRONJOB", "indexing-backup-cronjob")
    runner_timeout = int(os.environ.get("RUNNER_TIMEOUT", "3600"))
    debug_flag = os.environ.get("DEBUG_INDEXING_POD", "false").strip().lower() in ("1", "true", "yes")
    no_cleanup_flag = os.environ.get("NO_CLEANUP", "").strip().lower() in ("1", "true", "yes")
    keep_job_for_debug = debug_flag or no_cleanup_flag

    # create secrets from env if present
    created_any = False
    if os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
        ok = create_secret_from_env(ns, "indexer-azure-creds", {"AZURE_STORAGE_CONNECTION_STRING": "AZURE_STORAGE_CONNECTION_STRING"})
        created_any = created_any or ok
    if os.environ.get("QDRANT_API_KEY"):
        ok = create_secret_from_env(ns, "qdrant-api-key", {"QDRANT_API_KEY": "QDRANT_API_KEY"})
        created_any = created_any or ok
    if created_any:
        info("created/updated secrets in-cluster")

    # DNS probe settings
    dns_fix_enabled = os.environ.get("DNS_FIX_ENABLED", "true").strip().lower() in ("1", "true", "yes")
    dns_probe_retries = int(os.environ.get("DNS_PROBE_RETRIES", "6"))
    dns_probe_backoff_base = float(os.environ.get("DNS_PROBE_BACKOFF_BASE", "1.5"))
    dns_probe_timeout = int(os.environ.get("DNS_PROBE_TIMEOUT", "6"))
    dns_fix_restart_attempts = int(os.environ.get("DNS_FIX_RESTART_ATTEMPTS", "2"))
    dns_fix_wait_for_ready = int(os.environ.get("DNS_FIX_WAIT_FOR_READY", "60"))

    # Qdrant host resolution: use QDRANT_URL and QDRANT_NAMESPACE
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333").strip()
    qdrant_namespace = os.environ.get("QDRANT_NAMESPACE", "qdrant").strip()
    # parse host from URL in a simple manner
    host_port = qdrant_url.replace("http://", "").replace("https://", "").split("/", 1)[0]
    host = host_port.split(":")[0].strip()
    # If the host looks short (no dots) assume <host>.<namespace>.svc.cluster.local
    normalized_host = normalize_service_host(host, qdrant_namespace)
    service_http_url = qdrant_url.rstrip("/")

    info("provisioning manual job for cronjob=%s namespace=%s", cronjob, ns)
    info("probing service host=%s url=%s dns_fix_enabled=%s", normalized_host, service_http_url, dns_fix_enabled)

    # Pre-create step: ensure DNS and Qdrant reachable (with remediation if enabled)
    dns_ok = ensure_dns_and_service_reachable(service_host=normalized_host,
                                              service_http_url=service_http_url,
                                              max_probe_attempts=dns_probe_retries,
                                              backoff_base=dns_probe_backoff_base,
                                              probe_timeout=dns_probe_timeout,
                                              enable_fix=dns_fix_enabled,
                                              restart_attempts=dns_fix_restart_attempts,
                                              wait_for_ready=dns_fix_wait_for_ready)
    if not dns_ok:
        warn("DNS/service probe failed after remediation attempts. Proceeding to create job anyway; indexing may fail.")

    # create job and stream logs
    jobname = safe_job_name(cronjob)
    try:
        create_job_from_cronjob(ns, cronjob, jobname)
    except Exception as e:
        err("creating Job from CronJob failed: %s", e)
        rc, out, errout = run_cmd(["kubectl", "get", "cronjob", cronjob, "-n", ns], timeout=10)
        print("-- cronjob check --")
        if rc == 0:
            print(out)
        else:
            print(errout or out)
        sys.exit(3)
    info("Created Job: %s", jobname)

    stop_requested = False
    def _sig(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        warn("signal received, attempting graceful shutdown...")
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    start_time = time.time()
    streamers: Dict[str, PodLogStreamer] = {}
    seen_pods = set()

    try:
        while True:
            if stop_requested:
                info("stop requested; breaking watch loop")
                break
            if time.time() - start_time > runner_timeout:
                err("runner timeout after %ds", runner_timeout)
                break
            pods = list_pods_for_job(ns, jobname)
            for pod in pods:
                if pod in seen_pods:
                    continue
                cont = get_container_name_for_pod(ns, pod)
                streamer = PodLogStreamer(ns, pod, cont)
                streamer.start()
                streamers[pod] = streamer
                seen_pods.add(pod)
                info("started streaming logs for pod %s (container=%s)", pod, cont or "default")
            active, succeeded, failed = job_status_counts(ns, jobname)
            if succeeded > 0 or failed > 0:
                wait_deadline = time.time() + 60
                while time.time() < wait_deadline:
                    pods_now = list_pods_for_job(ns, jobname)
                    active_pods = []
                    for p in pods_now:
                        ph = pod_phase(ns, p) or ""
                        if ph not in ("Succeeded", "Failed"):
                            active_pods.append(p)
                    if not active_pods:
                        break
                    time.sleep(1)
                info("job finished (succeeded=%d, failed=%d)", succeeded, failed)
                break
            time.sleep(1)
    finally:
        for p, s in list(streamers.items()):
            try:
                s.stop(timeout=3)
            except Exception:
                pass
        if keep_job_for_debug:
            info("DEBUG_INDEXING_POD=true or NO_CLEANUP set; keeping job/pods for debugging")
        else:
            rc, out, errout = run_cmd(["kubectl", "delete", "job", jobname, "-n", ns, "--cascade=foreground"], timeout=60)
            if rc == 0:
                info("deleted job/%s", jobname)
            else:
                warn("failed to delete job: %s", errout or out)

    _, succeeded, failed = job_status_counts(ns, jobname)
    if succeeded > 0:
        info("job succeeded")
        sys.exit(0)
    if failed > 0:
        err("job failed")
        sys.exit(3)
    info("exiting (no explicit succeeded/failed status available)")
    sys.exit(0)

if __name__ == "__main__":
    main()
