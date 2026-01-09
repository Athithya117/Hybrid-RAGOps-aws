import json
import os
import sys
import logging
from datetime import datetime, timezone

# =============================================================================
# APPLICATION LOGGING CONTRACT (AUTHORITATIVE)
#
# 1. This logger is the ONLY supported way to emit application logs.
# 2. Logs MUST be:
#    - Single-line JSON
#    - Written to stdout
#    - One JSON object per log event
# 3. Log volume is controlled ONLY by LOG_LEVEL (never by infra).
# 4. Severity semantics are owned by the application, not Vector.
# 5. WARN and ERROR MUST always pass when LOG_LEVEL=INFO.
# 6. No ANSI colors, no multiline output, no plain text.
# =============================================================================

SERVICE = os.getenv("SERVICE_NAME", "frontend").strip()
if not SERVICE:
    # Hard fail early to prevent unlabeled logs; caller can catch and handle if needed.
    sys.stderr.write("FATAL: SERVICE_NAME must be set and non-empty\n")
    raise RuntimeError("SERVICE_NAME must be set and non-empty")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
ALLOWED = {"DEBUG", "INFO", "WARN", "ERROR"}
if LOG_LEVEL not in ALLOWED:
    # fallback to INFO but surface to stderr
    sys.stderr.write(f"invalid LOG_LEVEL '{LOG_LEVEL}', defaulting to INFO\n")
    LOG_LEVEL = "INFO"

# Configure stdlib logging (library logs to stderr). App JSON logs go to stdout.
logging.basicConfig(stream=sys.stderr, level=getattr(logging, LOG_LEVEL, logging.INFO))

_LEVEL_MAP = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warn",
    logging.ERROR: "error",
}

def _iso_ts():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

class JsonLogger:
    """
    JsonLogger emits single-line JSON logs to stdout.
    Methods: debug/info/warn/error(msg, **fields)
    """

    def __init__(self):
        # Use stdlib logger for gating decisions (configured above)
        self._std = logging.getLogger("app-json-logger")
        self._level_map = _LEVEL_MAP

    def _emit(self, level_int: int, message: str, **fields) -> None:
        rec = {
            "timestamp": _iso_ts(),
            "level": self._level_map.get(level_int, "info"),
            "message": message or "",
            "service": SERVICE,
        }
        # Put structured data under "fields" to preserve stable top-level schema
        if fields:
            rec["fields"] = fields
        # Guarantee single-line JSON to stdout
        try:
            sys.stdout.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except Exception:
            # As a last resort, emit minimal fallback to stderr so operator can see problem
            try:
                sys.stderr.write(f"logger: failed to emit json log for message={message}\n")
            except Exception:
                pass

    def debug(self, msg: str, **kw) -> None:
        if self._std.isEnabledFor(logging.DEBUG):
            self._emit(logging.DEBUG, msg, **kw)

    def info(self, msg: str, **kw) -> None:
        if self._std.isEnabledFor(logging.INFO):
            self._emit(logging.INFO, msg, **kw)

    def warn(self, msg: str, **kw) -> None:
        # Intentionally check WARNING level so WARN logs are visible when LOG_LEVEL=INFO
        if self._std.isEnabledFor(logging.WARNING):
            self._emit(logging.WARNING, msg, **kw)

    def error(self, msg: str, **kw) -> None:
        # Always emit errors (never gate)
        self._emit(logging.ERROR, msg, **kw)

# application-global singleton
log = JsonLogger()
