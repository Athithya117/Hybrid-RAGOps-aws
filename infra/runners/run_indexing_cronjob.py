#!/usr/bin/env python3
"""
Run a one-off Job from a CronJob and stream logs from all pods.

Behavior controlled exclusively via environment variables (no CLI arguments):
- NAMESPACE (default: indexing)
- CRONJOB (default: indexing-backup-cronjob)
- RUNNER_TIMEOUT (seconds, default: 3600)
- DEBUG_INDEXING_POD (true/false, default: false)
- NO_CLEANUP (true/false)  - legacy; if set to "true" it's treated same as DEBUG_INDEXING_POD=true

Secrets auto-created when envs available:
- AZURE_STORAGE_CONNECTION_STRING -> secret indexer-azure-creds
- QDRANT_API_KEY -> secret qdrant-api-key
"""
from __future__ import annotations

import os
import sys
import time
import signal
import threading
import datetime
import subprocess
import json
from typing import List, Tuple, Optional, Dict

# ----- helpers -----
def run_cmd(cmd: List[str],
            input_text: Optional[str] = None,
            timeout: Optional[int] = None) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout or "", e.stderr or ""
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or f"timeout after {timeout}s"
    except Exception as e:
        return 255, "", str(e)


def kubectl_available() -> bool:
    rc, _, _ = run_cmd(["kubectl", "version", "--client=true"], timeout=5)
    return rc == 0


def create_secret_from_env(namespace: str,
                           secret_name: str,
                           mapping: Dict[str, str]) -> bool:
    # mapping: env var -> key name in secret
    literals: List[str] = []
    for envvar, key in mapping.items():
        val = os.environ.get(envvar)
        if val:
            literals += ["--from-literal", f"{key}={val}"]
    if not literals:
        return False
    cmd = ["kubectl", "create", "secret", "generic", secret_name,
           "-n", namespace, "--dry-run=client", "-o", "yaml"] + literals
    rc, out, err = run_cmd(cmd, timeout=20)
    if rc != 0:
        print("[error] rendering secret YAML failed:", err or out, file=sys.stderr)
        return False
    rc2, out2, err2 = run_cmd(["kubectl", "apply", "-f", "-"],
                              input_text=out,
                              timeout=20)
    if rc2 != 0:
        print("[error] applying secret failed:", err2 or out2, file=sys.stderr)
        return False
    print(f"[ok] secret/{secret_name} created/updated in namespace {namespace}")
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
    rc, out, err = run_cmd(
        ["kubectl", "get", "cronjob", cronjob, "-n", namespace, "-o", "json"],
        timeout=10,
    )
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


def _create_job_from_cronjob_spec(namespace: str,
                                  jobname: str,
                                  cj_json: dict,
                                  parallelism: Optional[int],
                                  completions: Optional[int]) -> None:
    """
    Build a Job manifest from the CronJob's jobTemplate.spec and apply it.
    This creates a Job resource with the desired parallelism/completions before
    Kubernetes schedules pods. Avoids patching immutable fields.
    """
    if "spec" not in cj_json:
        raise RuntimeError("CronJob JSON missing spec")
    jt = cj_json["spec"].get("jobTemplate")
    if not jt or "spec" not in jt:
        raise RuntimeError("CronJob jobTemplate.spec missing")
    job_spec = jt["spec"]
    # Ensure we operate on a copy
    job_spec = json.loads(json.dumps(job_spec))

    # Apply overrides if provided
    if parallelism is not None:
        job_spec["parallelism"] = parallelism
    if completions is not None:
        job_spec["completions"] = completions

    # Build Job manifest
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

    # Create job via kubectl apply -f - (JSON is acceptable)
    payload = json.dumps(job_manifest)
    rc, out, err = run_cmd(["kubectl", "apply", "-f", "-"], input_text=payload, timeout=15)
    if rc != 0:
        raise RuntimeError(f"kubectl apply job manifest failed: {err or out}")
    print(f"[ok] Created Job: {jobname} (from CronJob jobTemplate.spec)")


def create_job_from_cronjob(namespace: str, cronjob: str, jobname: str) -> None:
    """
    Create a Job by attempting to manufacture a Job manifest from the CronJob's
    jobTemplate.spec (so we can set parallelism/completions). If that fails,
    fall back to kubectl create job --from=cronjob/<cronjob>.
    """
    p_env = _pos_int_or_none(os.environ.get("CRONJOB_PARALLELISM"))
    c_env = _pos_int_or_none(os.environ.get("CRONJOB_COMPLETIONS"))

    cj_json = _fetch_cronjob_json(namespace, cronjob)
    if cj_json:
        try:
            _create_job_from_cronjob_spec(namespace, jobname, cj_json, p_env, c_env)
            return
        except Exception as e:
            print("[warn] creating Job from CronJob spec failed:", e, file=sys.stderr)
            # fallback to kubectl create job --from=cronjob/...
    # Fallback behavior
    cmd = ["kubectl", "create", "job", jobname, "--from=cronjob/" + cronjob, "-n", namespace]
    rc, out, err = run_cmd(cmd, timeout=20)
    if rc != 0:
        raise RuntimeError(f"kubectl create job failed: {err or out}")
    print(f"[ok] Created Job: {jobname} (fallback create)")


def list_pods_for_job(namespace: str, jobname: str) -> List[str]:
    rc, out, _ = run_cmd([
        "kubectl", "get", "pods", "-n", namespace,
        "-l", f"job-name={jobname}",
        "-o", "jsonpath={.items[*].metadata.name}"
    ], timeout=10)
    if rc != 0:
        return []
    s = out.strip()
    return s.split() if s else []


def pod_phase(namespace: str, pod: str) -> Optional[str]:
    rc, out, _ = run_cmd([
        "kubectl", "get", "pod", pod, "-n", namespace,
        "-o", "jsonpath={.status.phase}"
    ], timeout=5)
    return out.strip() if rc == 0 else None


def job_status_counts(namespace: str, jobname: str) -> Tuple[int, int, int]:
    rc, out, _ = run_cmd([
        "kubectl", "get", "job", jobname, "-n", namespace,
        "-o", "jsonpath={.status.active}{'|'}{.status.succeeded}{'|'}{.status.failed}"
    ], timeout=5)
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


def describe_pod(namespace: str, pod: str) -> str:
    rc, out, err = run_cmd(["kubectl", "describe", "pod", pod, "-n", namespace],
                           timeout=15)
    return out if rc == 0 else (err or out)


def tail_events(namespace: str) -> str:
    rc, out, err = run_cmd(["kubectl", "get", "events", "-n", namespace,
                            "--sort-by=.metadata.creationTimestamp"],
                           timeout=15)
    return out if rc == 0 else (err or out)


def get_container_name_for_pod(namespace: str, pod: str) -> Optional[str]:
    rc, out, _ = run_cmd([
        "kubectl", "get", "pod", pod, "-n", namespace,
        "-o", "jsonpath={.spec.containers[*].name}"
    ], timeout=5)
    if rc != 0 or not out:
        return None
    names = out.strip().split()
    if "indexer" in names:
        return "indexer"
    return names[0] if names else None


# ----- Pod log streamer -----
class PodLogStreamer:
    def __init__(self, namespace: str, pod: str, container: Optional[str] = None):
        self.namespace = namespace
        self.pod = pod
        self.container = container
        self.proc: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.started_event = threading.Event()

    def _build_cmd(self) -> List[str]:
        cmd = ["kubectl", "logs", "-n", self.namespace, "-f", self.pod]
        if self.container:
            cmd += ["-c", self.container]
        return cmd

    def _wait_for_pod_running(self, timeout: int = 300) -> bool:
        # wait until pod phase is Running or Succeeded/Failed
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
            print(f"[warn] pod {self.pod} did not become ready within timeout",
                  file=sys.stderr)
            self.started_event.set()
            return
        cmd = self._build_cmd()
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
        except Exception as e:
            print(f"[error][{self.pod}] failed to start log stream: {e}",
                  file=sys.stderr)
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
            print(f"[error][{self.pod}] log stream error: {e}", file=sys.stderr)
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


# ----- main flow (env-driven) -----
def main() -> None:
    if not kubectl_available():
        print("ERROR: kubectl required in PATH", file=sys.stderr)
        sys.exit(2)

    # env/config
    ns = os.environ.get("NAMESPACE", "indexing")
    cronjob = os.environ.get("CRONJOB", "indexing-backup-cronjob")
    runner_timeout = int(os.environ.get("RUNNER_TIMEOUT", "3600"))
    debug_flag = os.environ.get("DEBUG_INDEXING_POD", "false").strip().lower() == "true"
    no_cleanup_flag = os.environ.get("NO_CLEANUP", "").strip().lower() == "true"
    keep_job_for_debug = debug_flag or no_cleanup_flag

    # create secrets from env if present
    created_any = False
    if os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
        ok = create_secret_from_env(ns, "indexer-azure-creds",
                                    {"AZURE_STORAGE_CONNECTION_STRING": "AZURE_STORAGE_CONNECTION_STRING"})
        created_any = created_any or ok
    if os.environ.get("QDRANT_API_KEY"):
        ok = create_secret_from_env(ns, "qdrant-api-key",
                                    {"QDRANT_API_KEY": "QDRANT_API_KEY"})
        created_any = created_any or ok
    if created_any:
        print("[info] created/updated secrets in-cluster")

    # create job name
    jobname = safe_job_name(cronjob)

    # create job (prefer rendering Job manifest from CronJob so we can set
    # parallelism/completions before creation)
    try:
        create_job_from_cronjob(ns, cronjob, jobname)
    except Exception as e:
        print("[error] creating Job from CronJob failed:", e, file=sys.stderr)
        rc, out, err = run_cmd(["kubectl", "get", "cronjob", cronjob, "-n", ns], timeout=10)
        print("-- cronjob check --")
        if rc == 0:
            print(out)
        else:
            print(err or out)
        sys.exit(3)

    print(f"[ok] Created Job: {jobname}")

    # signal handling
    stop_requested = False

    def _sig(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("\n[info] signal received, attempting graceful shutdown...", file=sys.stderr)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    start_time = time.time()
    streamers: Dict[str, PodLogStreamer] = {}
    seen_pods: set = set()

    try:
        # main watch loop: attach to pods as they appear and wait
        while True:
            if stop_requested:
                print("[info] stop requested by signal; breaking watch loop")
                break
            # timeout guard
            if time.time() - start_time > runner_timeout:
                print(f"[error] runner timeout after {runner_timeout}s", file=sys.stderr)
                break
            # list pods for job
            pods = list_pods_for_job(ns, jobname)
            # attach streamers for new pods
            for pod in pods:
                if pod in seen_pods:
                    continue
                cont = get_container_name_for_pod(ns, pod)
                streamer = PodLogStreamer(ns, pod, cont)
                streamer.start()
                streamers[pod] = streamer
                seen_pods.add(pod)
                print(f"[info] started streaming logs for pod {pod} (container={cont or 'default'})")

            # evaluate job status: only finish when controller reports success/failure
            active, succeeded, failed = job_status_counts(ns, jobname)
            # If job succeeded/failed, wait for all pods to terminate (no active pods)
            if succeeded > 0 or failed > 0:
                # wait up to a short window for pods to finish and logs to flush
                wait_deadline = time.time() + 60
                while time.time() < wait_deadline:
                    pods_now = list_pods_for_job(ns, jobname)
                    # active pods are those with phase not Succeeded/Failed
                    active_pods = []
                    for p in pods_now:
                        ph = pod_phase(ns, p) or ""
                        if ph not in ("Succeeded", "Failed"):
                            active_pods.append(p)
                    if not active_pods:
                        break
                    time.sleep(1)
                print(f"[info] job finished (succeeded={succeeded}, failed={failed})")
                break

            # otherwise keep watching, short sleep
            time.sleep(1)

    finally:
        # stop streamers
        for p, s in list(streamers.items()):
            try:
                s.stop(timeout=3)
            except Exception:
                pass

        # cleanup decision
        if keep_job_for_debug:
            print("[info] DEBUG_INDEXING_POD=true or NO_CLEANUP set; keeping job/pods for debugging")
        else:
            # delete job and let Kubernetes garbage-collect pods
            rc, out, err = run_cmd(["kubectl", "delete", "job", jobname,
                                    "-n", ns, "--cascade=foreground"], timeout=60)
            if rc == 0:
                print(f"[ok] deleted job/{jobname}")
            else:
                print("[warn] failed to delete job:", err or out, file=sys.stderr)

    # determine exit code: prefer job status if available
    _, succeeded, failed = job_status_counts(ns, jobname)
    if succeeded > 0:
        print("[ok] job succeeded")
        sys.exit(0)
    if failed > 0:
        print("[error] job failed", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
