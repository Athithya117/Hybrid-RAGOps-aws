# python3 indexing_pipeline/ray_indexing_pipeline.py --libreoffice-ready-timeout 30

from __future__ import annotations
import argparse
import logging
import os
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ray
from ray.util.placement_group import placement_group, remove_placement_group

DEFAULT_WORKDIR = "/workspace/indexing_pipeline"
LIBREOFFICE_SCRIPT = "parse_chunk/pre_conversions/libreoffice_server.sh"
PRE_CONVERSION_SCRIPTS = [
    "parse_chunk/pre_conversions/group_similar_raw_files.py",
    "parse_chunk/pre_conversions/all_audio_to_wav.py",
    "parse_chunk/pre_conversions/doc_docx_to_pdf.py",
    "parse_chunk/pre_conversions/spreadsheets_to_csv.py",
    "parse_chunk/pre_conversions/ppt_to_pptx.py",
]
ROUTER = "parse_chunk/router.py"
INDEX = "index.py"
LIBREOFFICE_LOG = "/tmp/libreoffice.log"
SOFFICE_LOG = "/tmp/libreoffice_soffice.log"

# logging to stdout so wrapper doesn't re-label as ERROR
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s")
handler.setFormatter(formatter)
root = logging.getLogger()
for h in list(root.handlers):
    root.removeHandler(h)
root.addHandler(handler)
root.setLevel(os.getenv("LOG_LEVEL", "INFO"))
# quiet noisy libraries
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)

logger = logging.getLogger("ray_indexing_pipeline")


def log_and_exit(msg: str, code: int = 1, extra: Optional[Dict] = None):
    logger.error(msg)
    if extra:
        for k, v in extra.items():
            logger.error("%s: %s", k, v)
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass
    sys.exit(code)


@ray.remote
def run_cmd(cmd: List[str], cwd: str = ".", env: dict = None, timeout: int = None) -> Tuple[int, str, str]:
    env_used = os.environ.copy()
    if env:
        env_used.update(env)
    try:
        proc = subprocess.run(cmd, cwd=cwd, env=env_used, capture_output=True, text=True, check=False, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, "", f"TimeoutExpired: {e}"
    except Exception as e:
        return 1, "", f"Exception while running {cmd}: {e}"


@ray.remote
class LibreOfficeActor:
    def __init__(self, workdir: str, script_relpath: str, port: int, log_path: str = LIBREOFFICE_LOG, force_direct: bool = False):
        self.workdir = Path(workdir)
        self.script = self.workdir / script_relpath
        self.port = int(port)
        self.log_path = log_path
        self.soffice_log = SOFFICE_LOG
        self.proc = None
        self.mode = None
        self.force_direct = bool(force_direct)
        self._start()

    def _which_soffice(self) -> Optional[str]:
        for name in ("soffice", "libreoffice"):
            p = shutil.which(name)
            if p:
                return p
        return None

    def _start_soffice(self, explicit_bin: Optional[str] = None):
        soffice_bin = explicit_bin if explicit_bin else self._which_soffice()
        if not soffice_bin:
            raise FileNotFoundError("soffice/libreoffice not found in PATH")
        try:
            open(self.log_path, "a").close()
        except Exception:
            pass
        logf = open(self.soffice_log, "a")
        cmd = [
            soffice_bin,
            "--headless",
            "--nologo",
            "--invisible",
            "--nodefault",
            "--norestore",
            f"--accept=socket,host=127.0.0.1,port={self.port};urp;"
        ]
        self.proc = subprocess.Popen(cmd, cwd=str(self.workdir), stdout=logf, stderr=logf, preexec_fn=os.setsid)
        self.mode = "soffice"
        self.pid = getattr(self.proc, "pid", None)

    def _start_wrapper(self):
        try:
            os.chmod(self.script, os.stat(self.script).st_mode | 0o111)
        except Exception:
            pass
        try:
            open(self.log_path, "a").close()
        except Exception:
            pass
        logf = open(self.log_path, "a")
        self.proc = subprocess.Popen(["bash", str(self.script), str(self.port)], cwd=str(self.workdir), stdout=logf, stderr=logf, preexec_fn=os.setsid)
        self.mode = "script"
        self.pid = getattr(self.proc, "pid", None)

    def _start(self):
        try:
            if self.force_direct:
                self._start_soffice()
                return
            if self.script.exists():
                try:
                    self._start_wrapper()
                except Exception:
                    self._start_soffice()
                    return
                time.sleep(0.2)
                if self.proc and self.proc.poll() is not None:
                    self._start_soffice()
                    return
            else:
                self._start_soffice()
        except Exception:
            raise

    def info(self):
        return {"pid": getattr(self, "pid", None), "port": self.port, "log": self.log_path, "mode": self.mode}

    def wait_until_ready(self, timeout: int = 30, interval: float = 0.5) -> bool:
        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            if getattr(self, "proc", None) and self.proc.poll() is not None:
                return False
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=1):
                    return True
            except Exception:
                time.sleep(interval)
        return False

    def stop(self):
        if getattr(self, "proc", None):
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        return True


def _follow_file(path: str, stop_event: threading.Event, prefix: str = "[libreoffice-log]"):
    waited = 0.0
    while not stop_event.is_set() and not os.path.exists(path) and waited < 5.0:
        time.sleep(0.1); waited += 0.1
    try:
        if not os.path.exists(path):
            open(path, "a").close()
    except Exception:
        pass
    try:
        with open(path, "r", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            while not stop_event.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.25)
                    continue
                logger.info("%s %s", prefix, line.rstrip())
    except Exception as e:
        logger.exception("Log follower for %s failed: %s", path, e)


def start_log_followers(paths: List[str]) -> Tuple[threading.Event, List[threading.Thread]]:
    stop = threading.Event()
    threads = []
    for p in paths:
        t = threading.Thread(target=_follow_file, args=(p, stop, "[libreoffice-log]"), daemon=True)
        t.start()
        threads.append(t)
    return stop, threads


def stop_log_followers(stop_event: threading.Event, threads: List[threading.Thread], timeout: float = 1.0):
    stop_event.set()
    for t in threads:
        t.join(timeout=timeout)


def connect_or_start_ray(runtime_env=None):
    try:
        ray.init(address="auto")
        logger.info("Connected to existing Ray cluster (address=auto).")
    except Exception:
        logger.info("Starting local Ray instance.")
        if runtime_env:
            ray.init(runtime_env=runtime_env)
        else:
            ray.init()


def run_local_and_stream(script_path: Path, workdir: str) -> int:
    cmd = [sys.executable, str(script_path)]
    logger.info("Starting local script: %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    except Exception as e:
        logger.exception("Failed to start %s: %s", script_path, e)
        return 1
    out_lines = []
    err_lines = []

    def reader(stream, collect, level):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                collect.append(line)
                # log both stdout and stderr as INFO so wrapper won't label them as ERROR
                logger.info("[%s] %s", script_path.name, line.rstrip())
        except Exception:
            pass

    t_out = threading.Thread(target=reader, args=(proc.stdout, out_lines, "out"), daemon=True)
    t_err = threading.Thread(target=reader, args=(proc.stderr, err_lines, "err"), daemon=True)
    t_out.start()
    t_err.start()
    proc.wait()
    t_out.join(timeout=1.0)
    t_err.join(timeout=1.0)
    return proc.returncode


def run_pipeline(workdir: str, libreoffice_port: int, skip_install: bool, livreoffice_ready_timeout: int):
    workdir = str(Path(workdir).resolve())
    runtime_env = {"working_dir": workdir}
    connect_or_start_ray(runtime_env if runtime_env else None)
    stop_event, threads = start_log_followers([LIBREOFFICE_LOG, SOFFICE_LOG])
    actor_cpus = 1
    task_cpus = 1
    bundle_cpu = actor_cpus + task_cpus
    try:
        pg = placement_group([{"CPU": bundle_cpu}], strategy="STRICT_PACK")
        ray.get(pg.ready())
    except Exception:
        logger.exception("placement_group failed")
        stop_log_followers(stop_event, threads)
        log_and_exit("placement_group ready failed", 1)
    try:
        libre_actor = LibreOfficeActor.options(
            placement_group=pg,
            placement_group_bundle_index=0,
            num_cpus=actor_cpus,
            runtime_env=runtime_env,
        ).remote(workdir, LIBREOFFICE_SCRIPT, libreoffice_port, LIBREOFFICE_LOG, False)
    except Exception:
        logger.exception("Failed to create LibreOfficeActor")
        try:
            remove_placement_group(pg)
        except Exception:
            pass
        stop_log_followers(stop_event, threads)
        log_and_exit("Failed to start LibreOfficeActor", 1)
    try:
        info = ray.get(libre_actor.info.remote(), timeout=30)
        logger.info("LibreOffice actor info: %s", info)
    except Exception:
        logger.exception("Failed to get actor info")
        try:
            ray.get(libre_actor.stop.remote(), timeout=10)
        except Exception:
            pass
        try:
            remove_placement_group(pg)
        except Exception:
            pass
        stop_log_followers(stop_event, threads)
        log_and_exit("LibreOffice actor init failed", 1)
    ready = False
    try:
        ready = ray.get(libre_actor.wait_until_ready.remote(timeout=livreoffice_ready_timeout), timeout=livreoffice_ready_timeout + 5)
    except Exception:
        logger.exception("Exception while waiting for libreoffice")
    if not ready:
        logger.warning("LibreOffice not ready after %s seconds. Attempting fallback (direct soffice) and will not block forever.", livreoffice_ready_timeout)
        try:
            ray.get(libre_actor.stop.remote(), timeout=10)
        except Exception:
            pass
        try:
            libre_actor = LibreOfficeActor.options(
                placement_group=pg,
                placement_group_bundle_index=0,
                num_cpus=actor_cpus,
                runtime_env=runtime_env,
            ).remote(workdir, LIBREOFFICE_SCRIPT, libreoffice_port, LIBREOFFICE_LOG, True)
            info = ray.get(libre_actor.info.remote(), timeout=30)
            logger.info("LibreOffice actor after fallback info: %s", info)
            ready = ray.get(libre_actor.wait_until_ready.remote(timeout=livreoffice_ready_timeout), timeout=livreoffice_ready_timeout + 5)
        except Exception:
            logger.exception("Fallback soffice start failed.")
            try:
                ray.get(libre_actor.stop.remote(), timeout=10)
            except Exception:
                pass
            try:
                remove_placement_group(pg)
            except Exception:
                pass
            stop_log_followers(stop_event, threads)
            log_and_exit("LibreOffice not ready after fallback", 1)
    if not ready:
        logger.warning("LibreOffice still not ready after fallback. Proceeding anyway (foreground), but conversions may fail. Check logs: %s and %s", LIBREOFFICE_LOG, SOFFICE_LOG)
    for script_rel in PRE_CONVERSION_SCRIPTS:
        script_path = Path(workdir) / script_rel
        if not script_path.exists():
            logger.error("Missing pre-conversion script: %s", script_path)
            try:
                ray.get(libre_actor.stop.remote(), timeout=30)
            except Exception:
                pass
            try:
                remove_placement_group(pg)
            except Exception:
                pass
            stop_log_followers(stop_event, threads)
            log_and_exit(f"Missing pre-conversion script: {script_path}", 1)
        rc = run_local_and_stream(script_path, workdir)
        if rc != 0:
            logger.error("Pre-conversion failed: %s (rc=%s)", script_path, rc)
            try:
                ray.get(libre_actor.stop.remote(), timeout=30)
            except Exception:
                pass
            try:
                remove_placement_group(pg)
            except Exception:
                pass
            stop_log_followers(stop_event, threads)
            log_and_exit(f"Pre-conversion failed: {script_path} rc={rc}", rc)
        else:
            logger.info("Pre-conversion succeeded: %s", script_path)
    router_path = Path(workdir) / ROUTER
    if not router_path.exists():
        logger.error("Router missing: %s", router_path)
        try:
            ray.get(libre_actor.stop.remote(), timeout=30)
        except Exception:
            pass
        try:
            remove_placement_group(pg)
        except Exception:
            pass
        stop_log_followers(stop_event, threads)
        log_and_exit("Router missing", 1)
    rc = run_local_and_stream(router_path, workdir)
    if rc != 0:
        logger.error("Router failed (rc=%s).", rc)
        try:
            ray.get(libre_actor.stop.remote(), timeout=30)
        except Exception:
            pass
        try:
            remove_placement_group(pg)
        except Exception:
            pass
        stop_log_followers(stop_event, threads)
        log_and_exit(f"Router failed rc={rc}", rc)
    else:
        logger.info("Router completed successfully.")
    index_path = Path(workdir) / INDEX
    if not index_path.exists():
        logger.error("Index missing: %s", index_path)
        try:
            ray.get(libre_actor.stop.remote(), timeout=30)
        except Exception:
            pass
        try:
            remove_placement_group(pg)
        except Exception:
            pass
        stop_log_followers(stop_event, threads)
        log_and_exit("Index missing", 1)
    rc = run_local_and_stream(index_path, workdir)
    if rc != 0:
        logger.error("Index failed (rc=%s).", rc)
        try:
            ray.get(libre_actor.stop.remote(), timeout=30)
        except Exception:
            pass
        try:
            remove_placement_group(pg)
        except Exception:
            pass
        stop_log_followers(stop_event, threads)
        log_and_exit(f"Index failed rc={rc}", rc)
    else:
        logger.info("Index completed successfully.")
    try:
        ray.get(libre_actor.stop.remote(), timeout=30)
    except Exception:
        logger.exception("Exception while stopping libre actor")
    try:
        remove_placement_group(pg)
    except Exception:
        logger.exception("Exception while removing placement_group")
    stop_log_followers(stop_event, threads)
    logger.info("Pipeline completed successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--libreoffice-port", type=int, default=7003)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--libreoffice-ready-timeout", type=int, default=10, help="seconds to wait for libreoffice before proceeding")
    args = parser.parse_args()
    try:
        run_pipeline(args.workdir, args.libreoffice_port, args.skip_install, args.libreoffice_ready_timeout)
    except SystemExit as e:
        logger.error("Exiting with SystemExit: %s", getattr(e, "code", None))
        raise
    except Exception:
        logger.exception("Unhandled exception in main")
        raise
    finally:
        try:
            ray.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    def _handler(sig, frame):
        logger.info("Signal %s received, exiting.", sig)
        try:
            ray.shutdown()
        except Exception:
            pass
        sys.exit(1)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    main()

