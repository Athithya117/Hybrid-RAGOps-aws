"""
python3 run_indexing_cronjob.py

Run a CronJob manually by creating a Job from it, stream logs from the Job pod,
and clean up afterwards.

Defaults:
  --namespace indexing
  --cronjob indexing-backup-cronjob

Options:
  --create-secrets  (for staging) create secrets in-cluster from env vars (qdrant + aws) before creating the Job
  --timeout SECS     max seconds to wait for pod to appear / start (default 180)
  --no-cleanup       keep Job & pods after completion for debugging

"""

from __future__ import annotations
import subprocess, sys, time, argparse, os, shlex, datetime, typing

def run_cmd(cmd: typing.List[str], input_text: str | None = None, timeout: int | None = None):
    try:
        proc = subprocess.run(cmd, input=input_text, capture_output=True, text=True, timeout=timeout, check=True)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout or "", e.stderr or ""
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or f"timeout after {timeout}s"

def kubectl_exists():
    return subprocess.run(["kubectl","version","--client"], capture_output=True).returncode == 0

def create_secret_from_env(namespace: str, secret_name: str, mapping: dict):
    """
    mapping: {ENV_VAR_NAME: secret_key_name}
    Only includes keys present in the environment.
    """
    literals = []
    for envvar, key in mapping.items():
        val = os.environ.get(envvar)
        if val:
            literals += ["--from-literal", f"{key}={val}"]
    if not literals:
        print(f"[info] no env vars present for secret {secret_name}; skipping creation.")
        return False
    cmd = ["kubectl", "create", "secret", "generic", secret_name, "-n", namespace, "--dry-run=client", "-o", "yaml"] + literals
    rc, out, err = run_cmd(cmd, timeout=20)
    if rc != 0:
        print("[error] rendering secret YAML failed:", err or out, file=sys.stderr)
        return False
    rc2, out2, err2 = run_cmd(["kubectl","apply","-f","-"], input_text=out, timeout=20)
    if rc2 != 0:
        print("[error] applying secret failed:", err2 or out2, file=sys.stderr)
        return False
    print(f"[ok] secret/{secret_name} created/updated in namespace {namespace}")
    return True

def safe_job_name(cronjob_name: str):
    ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    base = f"{cronjob_name}-manual-{ts}"
    # ensure RFC1123: lower, alnum, '-' and start/end alnum
    base = base.lower()
    # collapse invalid chars (shouldn't be present but be safe)
    base = "".join(c if (c.isalnum() or c == "-") else "-" for c in base)
    # trim to 63 chars and ensure start/end alnum
    base = base[:63].strip("-")
    if not base:
        raise SystemExit("generated job name empty (unexpected)")
    return base

def create_job_from_cronjob(namespace: str, cronjob: str, jobname: str):
    cmd = ["kubectl","create","job", f"--from=cronjob/{cronjob}", jobname, "-n", namespace]
    rc, out, err = run_cmd(cmd, timeout=20)
    if rc != 0:
        raise RuntimeError(f"kubectl create job failed: {err or out}")
    print(f"[ok] Created Job: {jobname}")
    return True

def find_pod_for_job(namespace: str, jobname: str, timeout: int = 120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        rc, out, err = run_cmd([
            "kubectl","get","pods","-n",namespace,
            "-l", f"job-name={jobname}",
            "-o", "jsonpath={.items[0].metadata.name}"
        ], timeout=5)
        if rc == 0 and out.strip():
            return out.strip()
        time.sleep(1)
    return None

def pod_phase(namespace: str, pod: str):
    rc, out, err = run_cmd(["kubectl","get","pod", pod, "-n", namespace, "-o", "jsonpath={.status.phase}"])
    return out.strip() if rc == 0 else None

def container_state_reason(namespace: str, pod: str, container_index: int = 0):
    # returns state reason or full state json fragment for diagnostics
    rc, out, err = run_cmd(["kubectl","get","pod", pod, "-n", namespace, "-o", "jsonpath={.status.containerStatuses[%d].state}" % container_index])
    if rc != 0:
        return None
    return out.strip()

def describe_pod(namespace: str, pod: str):
    rc, out, err = run_cmd(["kubectl","describe","pod", pod, "-n", namespace])
    return out if rc == 0 else err or out

def tail_events(namespace: str, since_seconds: int = 3600):
    rc, out, err = run_cmd(["kubectl","get","events","-n",namespace,"--sort-by=.metadata.creationTimestamp"])
    if rc == 0:
        return out
    return err

def wait_for_container_running(namespace: str, pod: str, timeout: int = 180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        phase = pod_phase(namespace, pod)
        if not phase:
            time.sleep(1); continue
        # If phase is Running, good
        if phase.lower() == "running":
            # ensure container state.running exists
            state = container_state_reason(namespace, pod)
            if state and "running" in state.lower():
                return True
        # If the pod already finished/succeeded/failed, return (nothing to stream)
        if phase.lower() in ("succeeded", "failed"):
            return False
        # If image pull backoff or err, detect via describe events quickly
        state = container_state_reason(namespace, pod)
        if state and ("imagepullbackoff" in state.lower() or "errimagepull" in state.lower() or "back-off" in state.lower()):
            return False
        time.sleep(1)
    return False

def stream_pod_logs(namespace: str, pod: str, container: str | None = None):
    cmd = ["kubectl","logs","-n",namespace,"-f",pod]
    if container:
        cmd += ["-c", container]
    print(f"[info] streaming logs: {' '.join(shlex.quote(x) for x in cmd)}")
    # spawn and forward stdout/stderr live
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
        proc.wait()
        return proc.returncode
    except KeyboardInterrupt:
        try:
            proc.terminate()
        except Exception:
            pass
        raise

def delete_job(namespace: str, jobname: str):
    rc, out, err = run_cmd(["kubectl","delete","job", jobname, "-n", namespace, "--cascade=foreground"], timeout=60)
    if rc == 0:
        print(f"[ok] deleted job/{jobname}")
    else:
        print("[warn] failed to delete job:", err or out)

def main():
    p = argparse.ArgumentParser(description="Create one-off Job from CronJob and stream logs.")
    p.add_argument("--namespace", default=os.environ.get("NAMESPACE","indexing"))
    p.add_argument("--cronjob", default=os.environ.get("CRONJOB","indexing-backup-cronjob"))
    p.add_argument("--create-secrets", action="store_true", help="create qdrant/aws secrets from env before running")
    p.add_argument("--timeout", type=int, default=180, help="seconds to wait for pod to appear/start")
    p.add_argument("--no-cleanup", action="store_true", help="do not delete Job/pods after completion")
    args = p.parse_args()

    if not kubectl_exists():
        print("ERROR: kubectl required in PATH", file=sys.stderr); sys.exit(2)

    ns = args.namespace
    cj = args.cronjob

    if args.create_secrets:
        # create sample secrets used by CronJob: qdrant-api-key and indexer-aws-creds
        # only create keys that exist in env
        create_secret_from_env(ns, "qdrant-api-key", {"QDRANT_API_KEY":"QDRANT_API_KEY"})
        create_secret_from_env(ns, "indexer-aws-creds", {"AWS_ACCESS_KEY_ID":"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY":"AWS_SECRET_ACCESS_KEY"})

    jobname = safe_job_name(cj)
    try:
        create_job_from_cronjob(ns, cj, jobname)
    except Exception as e:
        print("[error] creating Job from CronJob failed:", e, file=sys.stderr)
        # show cronjob existence diagnostic
        rc, out, err = run_cmd(["kubectl","get","cronjob", cj, "-n", ns], timeout=10)
        print("-- cronjob check --")
        if rc == 0:
            print(out)
        else:
            print(err or out)
        sys.exit(3)

    # find pod for job
    pod = find_pod_for_job(ns, jobname, timeout=args.timeout)
    if not pod:
        print(f"[error] no pod created for job {jobname} within {args.timeout}s", file=sys.stderr)
        print("-- recent events --")
        print(tail_events(ns))
        if not args.no_cleanup:
            delete_job(ns, jobname)
        sys.exit(4)

    print(f"[ok] Created Job: {jobname}")
    print(f"[info] Pod created: {pod}")

    # wait until container running (or detect failures like ImagePullBackOff)
    started = wait_for_container_running(ns, pod, timeout=args.timeout)
    if not started:
        phase = pod_phase(ns, pod)
        print(f"[error] container did not reach Running within {args.timeout}s (pod phase: {phase})", file=sys.stderr)
        print("\n-- kubectl describe pod --")
        print(describe_pod(ns, pod))
        print("\n-- recent events --")
        print(tail_events(ns))
        if not args.no_cleanup:
            delete_job(ns, jobname)
        sys.exit(5)

    # stream logs (container name default: indexer if present)
    # try to detect container name
    rc, conts, err = run_cmd(["kubectl","get","pod", pod, "-n", ns, "-o", "jsonpath={.spec.containers[*].name}"])
    container_name = None
    if rc == 0 and conts.strip():
        # prefer 'indexer' container
        names = conts.strip().split()
        if "indexer" in names:
            container_name = "indexer"
        else:
            container_name = names[0]
    try:
        rc = stream_pod_logs(ns, pod, container_name)
    except KeyboardInterrupt:
        print("\n[info] log streaming interrupted by user (Ctrl-C).")
        rc = 130

    # after logs finish, show pod final status and exit code if container ended
    print("\n-- pod final describe --")
    print(describe_pod(ns, pod))
    if not args.no_cleanup:
        delete_job(ns, jobname)

    # exit with kubectl logs return code if available
    if isinstance(rc, int) and rc != 0:
        sys.exit(rc)
    sys.exit(0)

if __name__ == "__main__":
    main()
