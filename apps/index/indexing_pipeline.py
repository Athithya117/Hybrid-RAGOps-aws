from __future__ import annotations
import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_WORKDIR = "/indexing_pipeline"
ROUTER = "parse_chunk/router.py"
INDEX = "index.py"

class ColoredFormatter(logging.Formatter):
    RESET = "\x1b[0m"
    COLORS = {
        "DEBUG": "\x1b[38;20m",
        "INFO": "\x1b[32;20m",
        "WARNING": "\x1b[33;20m",
        "ERROR": "\x1b[31;20m",
        "CRITICAL": "\x1b[41;30;20m"
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

def connect_or_start_local():
    logger.info("Running in local mode (no Ray).")

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
    def reader(stream, collect, prefix):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                collect.append(line)
                # keep logs short
                logger.info("[%s] %s", prefix, line.rstrip())
        except Exception:
            pass
    t_out = threading.Thread(target=reader, args=(proc.stdout, out_lines, script_path.name), daemon=True)
    t_err = threading.Thread(target=reader, args=(proc.stderr, err_lines, script_path.name + ":err"), daemon=True)
    t_out.start()
    t_err.start()
    proc.wait()
    t_out.join(timeout=1.0)
    t_err.join(timeout=1.0)
    return proc.returncode

def run_pipeline(workdir: str):
    workdir = str(Path(workdir).resolve())
    connect_or_start_local()
    router_path = Path(workdir) / ROUTER
    if not router_path.exists():
        logger.error("Router missing: %s", router_path)
        log_and_exit("Router missing", 1)
    rc = run_local_and_stream(router_path, workdir)
    if rc != 0:
        logger.error("Router failed (rc=%s).", rc)
        log_and_exit(f"Router failed rc={rc}", rc)
    else:
        logger.info("Router completed successfully.")
    index_path = Path(workdir) / INDEX
    if not index_path.exists():
        logger.error("Index missing: %s", index_path)
        log_and_exit("Index missing", 1)
    rc = run_local_and_stream(index_path, workdir)
    if rc != 0:
        logger.error("Index failed (rc=%s).", rc)
        log_and_exit(f"Index failed rc={rc}", rc)
    else:
        logger.info("Index completed successfully.")
    logger.info("Pipeline completed successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=os.getenv("WORKDIR", DEFAULT_WORKDIR))
    args = parser.parse_args()
    try:
        run_pipeline(args.workdir)
    except SystemExit as e:
        logger.error("Exiting with SystemExit: %s", getattr(e, "code", None))
        raise
    except Exception:
        logger.exception("Unhandled exception in main")
        raise

if __name__ == "__main__":
    def _handler(sig, frame):
        logger.info("Signal %s received, exiting.", sig)
        try:
            pass
        except Exception:
            pass
        sys.exit(1)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    main()
