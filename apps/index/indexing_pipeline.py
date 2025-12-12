from __future__ import annotations
import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import resource
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----- Configuration -----
DEFAULT_WORKDIR = "/indexing_pipeline"
ROUTER = "parse_chunk/router.py"
INDEX = "index.py"
PRE_CONVERSIONS = "pre_conversions.py"

# ----- Logging setup -----
class ColoredFormatter(logging.Formatter):
    RESET = "\x1b[0m"
    COLORS = {
        "DEBUG": "\x1b[38;20m",
        "INFO": "\x1b[32;20m",
        "WARNING": "\x1b[33;20m",
        "ERROR": "\x1b[31;20m",
        "CRITICAL": "\x1b[41;30;20m",
    }

    def __init__(self, fmt=None, datefmt=None, use_colors: Optional[bool] = None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        if use_colors is None:
            use_colors = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        self.use_colors = use_colors

    def format(self, record):
        levelname = record.levelname
        if self.use_colors and levelname in self.COLORS:
            color = self.COLORS[levelname]
            record.levelname = f"{color}{levelname}{self.RESET}"
        return super().format(record)

handler = logging.StreamHandler(sys.stdout)
base_fmt = "%(asctime)s.%(msecs)03d %(levelname)s %(message)s"
formatter = ColoredFormatter(fmt=base_fmt)
handler.setFormatter(formatter)
root = logging.getLogger()
for h in list(root.handlers):
    root.removeHandler(h)
root.addHandler(handler)
root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logger = logging.getLogger("indexing_pipeline")

# ----- Utilities -----
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

def try_raise_nofile(limit: int = 524288):
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(limit, hard) if hard != resource.RLIM_INFINITY else limit
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        logger.debug("Set RLIMIT_NOFILE -> soft=%s hard=%s", new_soft, hard)
    except Exception as e:
        logger.debug("Unable to raise RLIMIT_NOFILE: %s", e)

def run_cmd(cmd: List[str], cwd: str = ".", env: dict = None, timeout: int = None) -> Tuple[int, str, str]:
    env_used = os.environ.copy()
    if env:
        env_used.update(env)
    try:
        proc = subprocess.run(cmd, cwd=cwd, env=env_used, capture_output=True, text=True, check=False, timeout=timeout)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        return 124, "", f"TimeoutExpired: {e}"
    except Exception as e:
        return 1, "", f"Exception while running {cmd}: {e}"

def connect_or_start_local():
    logger.info("Running pipeline in local mode (no Ray).")

def run_local_and_stream(script_path: Path, workdir: str, timeout: Optional[int] = None) -> int:
    cmd = [sys.executable, str(script_path)]
    logger.info("Starting local script: %s (cwd=%s)", " ".join(cmd), workdir)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
    except Exception as e:
        logger.exception("Failed to start %s: %s", script_path, e)
        return 1

    out_lines: List[str] = []
    err_lines: List[str] = []

    def reader(stream, collect, prefix):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                collect.append(line)
                logger.info("[%s] %s", prefix, line.rstrip())
        except Exception:
            pass

    t_out = threading.Thread(target=reader, args=(proc.stdout, out_lines, script_path.name), daemon=True)
    t_err = threading.Thread(target=reader, args=(proc.stderr, err_lines, script_path.name + ":err"), daemon=True)
    t_out.start()
    t_err.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error("Script %s timed out after %s seconds", script_path, timeout)
        try:
            proc.kill()
        except Exception:
            pass
        return 124
    t_out.join(timeout=1.0)
    t_err.join(timeout=1.0)
    return proc.returncode

# ----- Pipeline steps -----
def run_pre_conversions(workdir: str) -> None:
    workdir_path = Path(workdir).resolve()
    script = workdir_path / PRE_CONVERSIONS
    if not script.exists():
        logger.info("No pre_conversions script found at %s, skipping.", script)
        return
    try:
        timeout_env = os.getenv("PRE_CONVERSIONS_TIMEOUT", "")
        try:
            timeout = int(timeout_env) if timeout_env else None
        except Exception:
            timeout = None
        logger.info("Running pre_conversions (python): %s (timeout=%s)", script, timeout)
        # ensure readable
        if not os.access(str(script), os.R_OK):
            logger.debug("Making pre_conversions.py readable")
            try:
                script.chmod(script.stat().st_mode | 0o444)
            except Exception:
                pass
        rc, out, err = run_cmd([sys.executable, str(script)], cwd=str(workdir_path), timeout=timeout)
        if out:
            for line in out.splitlines():
                logger.info("[pre_conversions] %s", line)
        if err:
            for line in err.splitlines():
                logger.warning("[pre_conversions:err] %s", line)
        if rc != 0:
            logger.error("pre_conversions script failed with rc=%s", rc)
            log_and_exit(f"pre_conversions failed (rc={rc})", rc)
        logger.info("pre_conversions completed successfully.")
    except SystemExit:
        raise
    except Exception:
        logger.exception("Exception while running pre_conversions")
        log_and_exit("pre_conversions raised exception", 2)

def run_pipeline(workdir: str):
    try:
        try_raise_nofile()
        workdir = str(Path(workdir).resolve())
        if not Path(workdir).exists():
            log_and_exit(f"Workdir not found: {workdir}", 2)
        logger.info("Pipeline start order: 1) pre_conversions 2) router 3) index")
        run_pre_conversions(workdir)
        connect_or_start_local()
        router_path = Path(workdir) / ROUTER
        if not router_path.exists():
            logger.error("Router missing: %s", router_path)
            log_and_exit("Router missing", 1)
        rc = run_local_and_stream(router_path, workdir)
        if rc != 0:
            logger.error("Router failed (rc=%s).", rc)
            log_and_exit(f"Router failed rc={rc}", rc)
        logger.info("Router completed successfully.")
        index_path = Path(workdir) / INDEX
        if not index_path.exists():
            logger.error("Index missing: %s", index_path)
            log_and_exit("Index missing", 1)
        rc = run_local_and_stream(index_path, workdir)
        if rc != 0:
            logger.error("Index failed (rc=%s).", rc)
            log_and_exit(f"Index failed rc={rc}", rc)
        logger.info("Index completed successfully.")
        logger.info("Pipeline completed successfully.")
    except SystemExit:
        raise
    except Exception:
        logger.exception("Unhandled exception in pipeline")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=os.getenv("WORKDIR", DEFAULT_WORKDIR))
    args = parser.parse_args()
    def _handler(sig, frame):
        logger.info("Signal %s received, exiting.", sig)
        try:
            pass
        except Exception:
            pass
        sys.exit(1)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    try:
        run_pipeline(args.workdir)
    except SystemExit as e:
        logger.error("Exiting with SystemExit: %s", getattr(e, "code", None))
        raise
    except Exception:
        logger.exception("Unhandled exception in main")
        raise

if __name__ == "__main__":
    main()
