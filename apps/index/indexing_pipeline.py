#!/usr/bin/env python3
from __future__ import annotations
import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import resource
import json
import time
import random
from collections import deque
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
    try:
        root.removeHandler(h)
    except Exception:
        pass
root.addHandler(handler)
root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logger = logging.getLogger("indexing_pipeline")

# ----- Failure handling policy -----
STRICT_MODE = os.getenv("INDEXING_STRICT", "").strip().lower() in ("1", "true", "yes")

REQUESTED_EXIT_CODE = 0

def _record_exit(code: int) -> None:
    global REQUESTED_EXIT_CODE
    try:
        ival = int(code) if code is not None else 1
    except Exception:
        ival = 1
    if ival and ival > REQUESTED_EXIT_CODE:
        REQUESTED_EXIT_CODE = ival

def log_and_exit(msg: str, code: int = 1, extra: Optional[Dict] = None) -> None:
    logger.error(msg)
    if extra:
        for k, v in extra.items():
            logger.error("%s: %s", k, v)
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass
    if STRICT_MODE:
        sys.exit(code)
    _record_exit(code)

# ----- Utilities -----
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
        return 124, (getattr(e, "stdout", "") or ""), (getattr(e, "stderr", "") or f"TimeoutExpired: {e}")
    except Exception as e:
        return 1, "", f"Exception while running {cmd}: {e}"

def connect_or_start_local():
    logger.info("Running pipeline in local mode (no Ray).")

def run_local_and_stream(script_path: Path, workdir: str, timeout: Optional[int] = None, extra_env: Optional[Dict[str,str]] = None) -> int:
    """
    Start a Python script as a subprocess and stream its stdout/stderr line-by-line.
    Returns subprocess return code.
    """
    cmd = [sys.executable, str(script_path)]
    logger.info("Starting local script: %s (cwd=%s)", " ".join(cmd), workdir)
    env_used = os.environ.copy()
    if extra_env:
        env_used.update(extra_env)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            env=env_used,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
    except Exception as e:
        logger.exception("Failed to start %s: %s", script_path, e)
        return 1

    def reader(stream, is_err: bool, prefix: str):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                text = line.rstrip("\n")
                if is_err:
                    logger.warning("[%s] %s", prefix, text)
                else:
                    logger.info("[%s] %s", prefix, text)
        except Exception:
            logger.exception("Reader thread for %s failed", prefix)

    prefix_out = script_path.name
    prefix_err = f"{script_path.name}:err"
    t_out = threading.Thread(target=reader, args=(proc.stdout, False, prefix_out), daemon=True)
    t_err = threading.Thread(target=reader, args=(proc.stderr, True, prefix_err), daemon=True)
    t_out.start()
    t_err.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error("Script %s timed out after %s seconds", script_path, timeout)
        try:
            proc.kill()
        except Exception:
            logger.exception("Failed to kill timed-out process %s", script_path)
        return 124
    t_out.join(timeout=2.0)
    t_err.join(timeout=2.0)
    return proc.returncode

# run_local_and_capture with bounded memory for stdout/stderr tail
def run_local_and_capture(script_path: Path, workdir: str, timeout: Optional[int] = None, extra_env: Optional[Dict[str,str]] = None, max_lines: int = 2000) -> Tuple[int, List[str], List[str]]:
    """
    Run script, stream logs live, and return (rc, stdout_lines, stderr_lines).
    Keeps only the last `max_lines` lines for each stream to avoid unbounded memory growth.
    """
    cmd = [sys.executable, str(script_path)]
    logger.info("Starting local script (capture): %s (cwd=%s)", " ".join(cmd), workdir)
    env_used = os.environ.copy()
    if extra_env:
        env_used.update(extra_env)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            env=env_used,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
    except Exception as e:
        logger.exception("Failed to start %s: %s", script_path, e)
        return 1, [], [str(e)]

    out_deque: "deque[str]" = deque(maxlen=max_lines)
    err_deque: "deque[str]" = deque(maxlen=max_lines)

    def reader(stream, collect_deque: "deque[str]", is_err: bool, prefix: str):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                text = line.rstrip("\n")
                collect_deque.append(text)
                if is_err:
                    logger.warning("[%s] %s", prefix, text)
                else:
                    logger.info("[%s] %s", prefix, text)
        except Exception:
            logger.exception("Reader thread for %s failed", prefix)

    prefix_out = script_path.name
    prefix_err = f"{script_path.name}:err"
    t_out = threading.Thread(target=reader, args=(proc.stdout, out_deque, False, prefix_out), daemon=True)
    t_err = threading.Thread(target=reader, args=(proc.stderr, err_deque, True, prefix_err), daemon=True)
    t_out.start()
    t_err.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error("Script %s timed out after %s seconds", script_path, timeout)
        try:
            proc.kill()
        except Exception:
            logger.exception("Failed to kill timed-out process %s", script_path)
        return 124, list(out_deque), list(err_deque)
    t_out.join(timeout=2.0)
    t_err.join(timeout=2.0)
    return proc.returncode, list(out_deque), list(err_deque)

# ----- Pipeline steps -----
def run_pre_conversions(workdir: str) -> bool:
    workdir_path = Path(workdir).resolve()
    script = workdir_path / PRE_CONVERSIONS
    if not script.exists():
        logger.info("No pre_conversions script found at %s, skipping.", script)
        return True
    try:
        timeout_env = os.getenv("PRE_CONVERSIONS_TIMEOUT", "")
        try:
            timeout = int(timeout_env) if timeout_env else None
        except Exception:
            timeout = None
        logger.info("Running pre_conversions (python): %s (timeout=%s)", script, timeout)
        if not os.access(str(script), os.R_OK):
            logger.debug("Making pre_conversions.py readable")
            try:
                script.chmod(script.stat().st_mode | 0o444)
            except Exception:
                logger.debug("Failed to chmod pre_conversions.py; continuing")
        rc = run_local_and_stream(script, str(workdir_path), timeout=timeout)
        if rc != 0:
            logger.error("pre_conversions script failed with rc=%s", rc)
            log_and_exit(f"pre_conversions failed (rc={rc})", rc)
            return False
        logger.info("pre_conversions completed successfully.")
        return True
    except SystemExit:
        raise
    except Exception:
        logger.exception("Exception while running pre_conversions")
        log_and_exit("pre_conversions raised exception", 2)
        return False

# parse the final JSON summary emitted by index.py (single-line JSON expected as last non-empty line)
def parse_index_summary(stdout_lines: List[str]) -> Optional[Dict]:
    if not stdout_lines:
        logger.warning("Index stdout empty; cannot parse summary.")
        return None
    for line in reversed(stdout_lines):
        s = line.strip()
        if not s:
            continue
        # best-effort JSON parse
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            # try to find JSON substring
            start = s.find("{")
            end = s.rfind("}")
            if 0 <= start < end:
                try:
                    parsed = json.loads(s[start:end+1])
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            continue
    logger.warning("Failed to locate JSON summary in index stdout.")
    return None

# environment helpers
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y")

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(v)
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default

def should_run_backup_from_summary(summary: Dict) -> Tuple[bool, str]:
    enable = _env_bool("ENABLE_QDRANT_BACKUP", True)
    if not enable:
        return False, "ENABLE_QDRANT_BACKUP=false"

    force = _env_bool("FORCE_QDRANT_BACKUP", False)
    avoid_empty = _env_bool("AVOID_BACKUP_AFTER_EMPTY_INDEXING", True)
    min_points = _env_int("MIN_INDEXED_POINTS_FOR_BACKUP", 100)
    min_delta_ratio = _env_float("MIN_INDEX_DELTA_RATIO_FOR_BACKUP", 0.0)

    indexed = int(summary.get("indexed_points", 0) or 0)
    total_input_chunks = int(summary.get("total_input_chunks", 0) or 0)
    skipped_existing = int(summary.get("skipped_existing", 0) or 0)

    existing_points = None
    if "existing_points" in summary:
        try:
            existing_points = int(summary.get("existing_points", 0) or 0)
        except Exception:
            existing_points = None
    else:
        existing_points = skipped_existing if skipped_existing > 0 else None

    if force:
        return True, "FORCE_QDRANT_BACKUP=true"

    if avoid_empty and indexed == 0:
        return False, "no points indexed (empty) and AVOID_BACKUP_AFTER_EMPTY_INDEXING=true"

    if indexed < min_points:
        return False, f"indexed_points {indexed} < MIN_INDEXED_POINTS_FOR_BACKUP {min_points}"

    if min_delta_ratio and min_delta_ratio > 0.0:
        if existing_points is None or existing_points <= 0:
            logger.warning("MIN_INDEX_DELTA_RATIO_FOR_BACKUP set but existing_points unknown; skipping ratio check.")
        else:
            ratio = indexed / float(existing_points)
            if ratio < min_delta_ratio:
                return False, f"indexed/existing ratio {ratio:.6f} < MIN_INDEX_DELTA_RATIO_FOR_BACKUP {min_delta_ratio}"

    return True, "passes all guards"

def _sleep_with_backoff(base: float, attempt: int, cap: float = 60.0):
    backoff = min(cap, base * (2 ** max(0, attempt - 1)))
    jittered = backoff * (0.5 + random.random() * 0.5)
    time.sleep(jittered)

def _find_backup_script(workdir: str) -> Optional[str]:
    """
    Resolve RUN_QDRANT_BACKUP_PATH with sensible fallbacks:
      - explicit env var (absolute or relative)
      - workdir/run_qdrant_backup.py
      - workdir/infra/runners/run_qdrant_backup_service.py
      - workdir/infra/runners/run_qdrant_backup.py
      - script dir (repo) variants
    Returns first existing path or None.
    """
    candidates: List[str] = []
    env_path = os.getenv("RUN_QDRANT_BACKUP_PATH")
    if env_path:
        candidates.append(env_path)
    # common locations relative to workdir
    candidates.extend([
        os.path.join(workdir, "run_qdrant_backup.py"),
        os.path.join(workdir, "run_qdrant_backup_service.py"),
        os.path.join(workdir, "infra", "runners", "run_qdrant_backup_service.py"),
        os.path.join(workdir, "infra", "runners", "run_qdrant_backup.py"),
        os.path.join(workdir, "infra", "runners", "run_qdrant_backup.py"),
    ])
    # location relative to this file (generator may produce different layout)
    here = Path(__file__).resolve().parent
    candidates.extend([
        str(here / "run_qdrant_backup.py"),
        str(here / "infra" / "runners" / "run_qdrant_backup_service.py"),
        str(here / "infra" / "runners" / "run_qdrant_backup.py"),
    ])
    # unique-ify and test
    seen = set()
    for c in candidates:
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        p = Path(c)
        # if relative path, resolve relative to workdir
        if not p.is_absolute():
            p = (Path(workdir) / c).resolve()
        if p.exists() and p.is_file():
            logger.debug("Found backup script candidate: %s", str(p))
            return str(p)
    return None

def invoke_service_backup(workdir: str) -> None:
    """
    Invoke the data-plane service backup script with retries.
    Treat backup failure as fatal after retries exhausted.
    """
    backup_script = _find_backup_script(workdir)
    if not backup_script:
        logger.error("Backup script not found. Tried RUN_QDRANT_BACKUP_PATH and common locations.")
        sys.exit(3)

    az_container = os.getenv("BACKUP_AZ_CONTAINER") or os.getenv("BACKUP_AZ_CONTAINER".upper())
    az_prefix = os.getenv("BACKUP_PREFIX")
    if not az_container or not az_prefix:
        logger.error("BACKUP_AZ_CONTAINER and BACKUP_PREFIX must be set for backups (found: container=%s prefix=%s)", az_container, az_prefix)
        sys.exit(3)

    retries = _env_int("BACKUP_INVOKE_RETRIES", 3)
    base = _env_float("BACKUP_INVOKE_RETRY_BASE", 2.0)
    timeout = _env_int("BACKUP_TIMEOUT", 300)
    env = os.environ.copy()
    cmd = [sys.executable, backup_script]

    last_err = None
    for attempt in range(1, retries + 1):
        logger.info("Invoking service-mode backup (attempt %d/%d): script=%s container=%s prefix=%s", attempt, retries, backup_script, az_container, az_prefix)
        rc, out, err = run_cmd(cmd, cwd=workdir, env=env, timeout=timeout + 30)
        if rc == 0:
            logger.info("Backup script completed successfully. stdout=%s", (out[:2000] if out else ""))
            return
        last_err = (rc, out, err)
        logger.warning("Backup attempt %d failed rc=%s stdout(last)=%.200s stderr(last)=%.200s", attempt, rc, (out[-200:] if out else ""), (err[-200:] if err else ""))
        if attempt < retries:
            _sleep_with_backoff(base, attempt)
    # exhausted retries
    rc, out, err = last_err if last_err else (3, "", "unknown error")
    logger.error("Backup failed after %d attempts. last rc=%s stdout=%s stderr=%s", retries, rc, (out[:2000] if out else ""), (err[:2000] if err else ""))
    sys.exit(rc or 3)

def run_pipeline(workdir: str) -> None:
    try:
        try_raise_nofile()
        workdir = str(Path(workdir).resolve())
        if not Path(workdir).exists():
            log_and_exit(f"Workdir not found: {workdir}", 2)
            return
        logger.info("Pipeline start order: 1) pre_conversions 2) router 3) index")
        if not run_pre_conversions(workdir):
            logger.warning("pre_conversions step failed; skipping remaining steps.")
            return
        connect_or_start_local()
        router_path = Path(workdir) / ROUTER
        if not router_path.exists():
            logger.error("Router missing: %s", router_path)
            log_and_exit("Router missing", 1)
            return
        rc = run_local_and_stream(router_path, workdir)
        if rc != 0:
            logger.error("Router failed (rc=%s).", rc)
            log_and_exit(f"Router failed rc={rc}", rc)
            logger.warning("Router step failed; skipping index step.")
            return
        logger.info("Router completed successfully.")
        index_path = Path(workdir) / INDEX
        if not index_path.exists():
            logger.error("Index missing: %s", index_path)
            log_and_exit("Index missing", 1)
            return

        # Run index.py and capture the tail of stdout/stderr
        index_timeout = _env_int("INDEX_TIMEOUT", 1800)
        rc, stdout_lines, stderr_lines = run_local_and_capture(index_path, workdir, timeout=index_timeout, max_lines=_env_int("INDEX_STDOUT_TAIL_LINES", 2000))
        if rc != 0:
            logger.error("Index failed (rc=%s). stdout(last)=%.200s stderr(last)=%.200s", rc, (stdout_lines[-1] if stdout_lines else ""), (stderr_lines[-1] if stderr_lines else ""))
            log_and_exit(f"Index failed rc={rc}", rc)
            return
        logger.info("Index completed successfully. Parsing summary...")

        summary = parse_index_summary(stdout_lines)
        if summary is None:
            logger.warning("Index summary missing or unparsable; skipping backup decision (no backup will be performed).")
            logger.debug("Index stdout (tail):\n%s", "\n".join(stdout_lines[-50:]))
            return

        should_backup, reason = should_run_backup_from_summary(summary)
        logger.info("Backup decision: should_backup=%s reason=%s summary=%s", bool(should_backup), reason, summary)

        if should_backup:
            invoke_service_backup(workdir)
        else:
            logger.info("Skipping backup: %s", reason)

        logger.info("Pipeline completed successfully.")
    except SystemExit:
        raise
    except Exception:
        logger.exception("Unhandled exception in pipeline")
        if STRICT_MODE:
            raise
        else:
            log_and_exit("Unhandled exception in pipeline", 2)

def _finalize_and_exit() -> None:
    global REQUESTED_EXIT_CODE
    if REQUESTED_EXIT_CODE != 0:
        if STRICT_MODE:
            logger.error("Exiting (strict) with code %s", REQUESTED_EXIT_CODE)
            sys.exit(REQUESTED_EXIT_CODE)
        else:
            logger.warning("Non-fatal errors recorded (rc=%s). Exiting 0 because INDEXING_STRICT!=1", REQUESTED_EXIT_CODE)
            sys.exit(0)
    sys.exit(0)

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
        if STRICT_MODE:
            raise
        else:
            log_and_exit("Unhandled exception in main", 2)
    _finalize_and_exit()

if __name__ == "__main__":
    main()
