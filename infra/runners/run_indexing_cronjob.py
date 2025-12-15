#!/usr/bin/env python3
"""run_indexing_cronjob.py
- Create a one-off Job from CronJob and stream logs from all pods concurrently.
- Auto-creates secrets from envs (AZURE_STORAGE_CONNECTION_STRING -> indexer-azure-creds, QDRANT_API_KEY -> qdrant-api-key).
- By default cleans up Job/pods; use --no-cleanup to keep them for debugging.
"""
from __future__ import annotations
import subprocess, sys, time, argparse, os, shlex, datetime, typing, threading, signal
from typing import Optional, Dict, Set, Tuple, List
def run_cmd(cmd: typing.List[str], input_text: str | None = None, timeout: int | None = None) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, input=input_text, capture_output=True, text=True, timeout=timeout, check=True)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout or "", e.stderr or ""
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or f"timeout after {timeout}s"
def kubectl_exists() -> bool:
    return subprocess.run(["kubectl","version","--client"], capture_output=True).returncode == 0
def create_secret_from_env(namespace: str, secret_name: str, mapping: Dict[str,str]) -> bool:
    literals: List[str] = []
    for envvar, key in mapping.items():
        val = os.environ.get(envvar)
        if val:
            literals += ["--from-literal", f"{key}={val}"]
    if not literals:
        return False
    cmd = ["kubectl", "create", "secret", "generic", secret_name, "-n", namespace, "--dry-run=client", "-o", "yaml"] + literals
    rc, out, err = run_cmd(cmd, timeout=20)
    if rc != 0:
        print("[error] rendering secret YAML failed:", err or out, file=sys.stderr); return False
    rc2, out2, err2 = run_cmd(["kubectl","apply","-f","-"], input_text=out, timeout=20)
    if rc2 != 0:
        print("[error] applying secret failed:", err2 or out2, file=sys.stderr); return False
    print(f"[ok] secret/{secret_name} created/updated in namespace {namespace}"); return True
def safe_job_name(cronjob_name: str) -> str:
    ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    base = f"{cronjob_name}-manual-{ts}".lower()
    base = "".join(c if (c.isalnum() or c == "-") else "-" for c in base)
    base = base[:63].strip("-")
    if not base:
        raise SystemExit("generated job name empty (unexpected)")
    return base
def create_job_from_cronjob(namespace: str, cronjob: str, jobname: str):
    cmd = ["kubectl","create","job", f"--from=cronjob/{cronjob}", jobname, "-n", namespace]
    rc, out, err = run_cmd(cmd, timeout=20)
    if rc != 0:
        raise RuntimeError(f"kubectl create job failed: {err or out}")
    print(f"[ok] Created Job: {jobname}"); return True
def list_pods_for_job(namespace: str, jobname: str) -> List[str]:
    rc, out, err = run_cmd(["kubectl","get","pods","-n",namespace,"-l",f"job-name={jobname}","-o","jsonpath={.items[*].metadata.name}"], timeout=10)
    if rc != 0:
        return []
    s = out.strip()
    return s.split() if s else []
def pod_phase(namespace: str, pod: str) -> Optional[str]:
    rc, out, err = run_cmd(["kubectl","get","pod", pod, "-n", namespace, "-o", "jsonpath={.status.phase}"], timeout=5)
    return out.strip() if rc == 0 else None
def describe_pod(namespace: str, pod: str) -> str:
    rc, out, err = run_cmd(["kubectl","describe","pod", pod, "-n", namespace], timeout=15)
    return out if rc == 0 else err or out
def tail_events(namespace: str, since_seconds: int = 3600) -> str:
    rc, out, err = run_cmd(["kubectl","get","events","-n",namespace,"--sort-by=.metadata.creationTimestamp"], timeout=15)
    return out if rc == 0 else err or out
def job_status_counts(namespace: str, jobname: str) -> Tuple[int,int,int]:
    rc, out, err = run_cmd(["kubectl","get","job", jobname, "-n", namespace, "-o", "jsonpath={.status.active}{'|'}{.status.succeeded}{'|'}{.status.failed}"], timeout=5)
    if rc != 0 or not out:
        return 0,0,0
    parts = out.strip().split("|")
    try:
        a = int(parts[0]) if parts[0] and parts[0] != "<none>" else 0
    except Exception:
        a = 0
    try:
        s = int(parts[1]) if len(parts) > 1 and parts[1] and parts[1] != "<none>" else 0
    except Exception:
        s = 0
    try:
        f = int(parts[2]) if len(parts) > 2 and parts[2] and parts[2] != "<none>" else 0
    except Exception:
        f = 0
    return a,s,f
class PodLogStreamer:
    def __init__(self, namespace: str, pod: str, container: Optional[str] = None):
        self.namespace = namespace; self.pod = pod; self.container = container; self.proc = None; self.thread = None; self.stopped = threading.Event()
    def _build_cmd(self) -> List[str]:
        cmd = ["kubectl","logs","-n",self.namespace,"-f",self.pod]
        if self.container:
            cmd += ["-c", self.container]
        return cmd
    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, daemon=True); self.thread.start()
    def _run(self):
        cmd = self._build_cmd()
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except Exception as e:
            print(f"[error][{self.pod}] failed to start log stream: {e}", file=sys.stderr); return
        prefix = f"[{self.pod}/{self.container or 'main'}]"
        try:
            for line in self.proc.stdout:
                if not line:
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
    def stop(self, timeout: int = 5):
        self.stopped.set()
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
def get_container_name_for_pod(namespace: str, pod: str) -> Optional[str]:
    rc, out, err = run_cmd(["kubectl","get","pod", pod, "-n", namespace, "-o", "jsonpath={.spec.containers[*].name}"], timeout=5)
    if rc != 0 or not out:
        return None
    names = out.strip().split()
    if "indexer" in names:
        return "indexer"
    return names[0] if names else None
def delete_job(namespace: str, jobname: str):
    rc, out, err = run_cmd(["kubectl","delete","job", jobname, "-n", namespace, "--cascade=foreground"], timeout=60)
    if rc == 0:
        print(f"[ok] deleted job/{jobname}")
    else:
        print("[warn] failed to delete job:", err or out)
def graceful_shutdown(streamers: Dict[str, PodLogStreamer], jobname: str, namespace: str, cleanup: bool):
    for p, s in list(streamers.items()):
        try:
            s.stop()
        except Exception:
            pass
    if cleanup:
        try:
            delete_job(namespace, jobname)
        except Exception:
            pass
def main():
    parser = argparse.ArgumentParser(description="Create one-off Job from CronJob and stream logs from all pods.")
    parser.add_argument("--namespace", default=os.environ.get("NAMESPACE","indexing"))
    parser.add_argument("--cronjob", default=os.environ.get("CRONJOB","indexing-backup-cronjob"))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--idle-timeout", type=int, default=30, help="seconds to wait with no active pods before concluding Job finished")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--create-secrets", action="store_true")
    args = parser.parse_args()
    if not kubectl_exists():
        print("ERROR: kubectl required in PATH", file=sys.stderr); sys.exit(2)
    ns = args.namespace; cj = args.cronjob
    created_any = False
    if os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
        ok = create_secret_from_env(ns, "indexer-azure-creds", {"AZURE_STORAGE_CONNECTION_STRING":"AZURE_STORAGE_CONNECTION_STRING"})
        created_any = created_any or ok
    if os.environ.get("QDRANT_API_KEY"):
        ok = create_secret_from_env(ns, "qdrant-api-key", {"QDRANT_API_KEY":"QDRANT_API_KEY"})
        created_any = created_any or ok
    if args.create_secrets and not created_any:
        print("[info] create-secrets requested but no known secret env vars present; nothing created.")
    jobname = safe_job_name(cj)
    try:
        create_job_from_cronjob(ns, cj, jobname)
    except Exception as e:
        print("[error] creating Job from CronJob failed:", e, file=sys.stderr)
        rc, out, err = run_cmd(["kubectl","get","cronjob", cj, "-n", ns], timeout=10)
        print("-- cronjob check --")
        if rc == 0:
            print(out)
        else:
            print(err or out)
        sys.exit(3)
    print(f"[ok] Created Job: {jobname}")
    deadline = time.time() + args.timeout
    streamers: Dict[str, PodLogStreamer] = {}
    seen_pods: Set[str] = set()
    last_active_time = time.time()
    stop_requested = False
    def _sigint_handler(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("\n[info] SIGINT received, attempting graceful shutdown...")
    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)
    try:
        while True:
            if stop_requested:
                print("[info] stop requested by signal"); break
            pods = list_pods_for_job(ns, jobname)
            active_pods = []
            for pod in pods:
                phase = pod_phase(ns, pod) or ""
                if phase.lower() not in ("succeeded","failed"):
                    active_pods.append(pod)
                if pod not in seen_pods:
                    c = get_container_name_for_pod(ns, pod)
                    streamer = PodLogStreamer(ns, pod, c)
                    streamer.start()
                    streamers[pod] = streamer
                    seen_pods.add(pod)
                    print(f"[info] started streaming logs for pod {pod} (container={c or 'default'})")
            if active_pods:
                last_active_time = time.time()
            a,s,f = job_status_counts(ns, jobname)
            if time.time() > deadline:
                print(f"[error] timeout waiting for Job pods after {args.timeout}s", file=sys.stderr)
                print("-- recent events --"); print(tail_events(ns))
                break
            if not active_pods:
                idle_elapsed = time.time() - last_active_time
                if idle_elapsed >= args.idle_timeout and pods:
                    print(f"[info] no active pods for {args.idle_timeout}s and pods exist; assuming job finished"); break
                if not pods:
                    if time.time() - last_active_time > args.idle_timeout:
                        print(f"[info] no pods observed for job yet after waiting {int(time.time()-last_active_time)}s"); pass
            time.sleep(1)
    finally:
        graceful_shutdown(streamers, jobname, ns, cleanup=not args.no_cleanup)
        print("\n-- final pod descriptions --")
        for p in sorted(seen_pods):
            print(f"\n--- describe {p} ---"); print(describe_pod(ns, p))
        print("\n-- recent events --"); print(tail_events(ns))
    sys.exit(0)
if __name__ == "__main__":
    main()
