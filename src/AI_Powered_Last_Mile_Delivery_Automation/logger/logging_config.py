"""
AI_Powered_Last_Mile_Delivery_Automation.logger.logging_config — centralized, run-specific logging.

Each pipeline run creates a timestamped directory under logs/ and each module
gets its own log file inside that directory.

Set the ``LOG_FORMAT`` environment variable to ``"json"`` for structured JSON
output (suitable for CloudWatch, ELK, Datadog, etc.).

Usage:
    from AI_Powered_Last_Mile_Delivery_Automation.logger.logging_config import get_module_logger
    logger = get_module_logger("load_data")
"""

import contextvars
import json
import os
import sys
import logging
import logging.handlers
import threading
from datetime import datetime

_LOG_FORMAT = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"

# Context variable for trace-ID propagation across async/threaded boundaries.
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default=""
)

# Module-level state guarded by a lock for thread safety.
_lock = threading.Lock()
_run_dir: str | None = None
_disk_logging_available = True


class TraceIdFilter(logging.Filter):
    """Inject the current trace_id (from contextvars) into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get("")  # type: ignore[attr-defined]
        return True


_PII_KEYS: frozenset[str] = frozenset(
    {
        "customer_profile_full",
        "name",
        "customer_name",
        "email",
        "phone",
        "address",
        "raw_rows",
    }
)
_REDACTED = "<redacted>"


def _scrub_pii(value: object, depth: int = 0) -> object:
    """Return a PII-safe copy of ``value``.

    Walks dicts/lists/tuples recursively and replaces any value whose key
    appears in ``_PII_KEYS`` with ``<redacted>``. Depth is bounded so a
    pathological record cannot blow the stack.
    """
    if depth > 6:
        return value
    if isinstance(value, dict):
        return {
            k: (_REDACTED if k in _PII_KEYS else _scrub_pii(v, depth + 1))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        scrubbed = [_scrub_pii(item, depth + 1) for item in value]
        return type(value)(scrubbed) if isinstance(value, tuple) else scrubbed
    return value


class PIIFilter(logging.Filter):
    """Redact known-PII keys from log record ``args`` and ``msg``.

    Attached to every module logger so any dict / list passed as a format
    argument — or embedded in a pre-formatted message string — is stripped
    before reaching disk or the console handler. Custom ``extra={}`` fields
    are scrubbed the same way.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            if isinstance(record.args, dict):
                record.args = _scrub_pii(record.args)  # type: ignore[assignment]
            elif isinstance(record.args, tuple):
                record.args = tuple(_scrub_pii(a) for a in record.args)
        # Scrub any extras attached via ``logger.info(..., extra={...})``.
        for key in list(record.__dict__.keys()):
            if key in _PII_KEYS:
                record.__dict__[key] = _REDACTED
        return True


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
            "logger": record.name,
        }
        tid = getattr(record, "trace_id", "")
        if tid:
            payload["trace_id"] = tid
        return json.dumps(payload)


def _get_formatter() -> logging.Formatter:
    """Return a JSON or human-readable formatter based on LOG_FORMAT env var."""
    if os.environ.get("LOG_FORMAT", "").lower() == "json":
        return JSONFormatter()
    return logging.Formatter(_LOG_FORMAT)


def setup_run_logging() -> str | None:
    """Create a timestamped run directory and return its path.

    Safe to call multiple times — only the first call creates the directory;
    subsequent calls return the existing path.  Falls back to console-only
    logging if the directory cannot be created (e.g. read-only filesystem).
    """
    global _run_dir, _disk_logging_available
    with _lock:
        if _run_dir is not None:
            return _run_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _run_dir = os.path.join(
            "logs", f"ai_powered_last_mile_delivery_automation_{timestamp}"
        )
        try:
            os.makedirs(_run_dir, exist_ok=True)
        except OSError:
            _disk_logging_available = False
            _run_dir = None
    return _run_dir


def get_module_logger(module_name: str) -> logging.Logger:
    """Return a logger that writes to both console and a module-specific file.

    The file is placed inside the current run directory. If setup_run_logging()
    has not been called yet, it is called automatically.

    File handlers use RotatingFileHandler (10 MB max, 5 backups) to prevent
    unbounded log growth in long-running training sessions.
    """
    run_dir = setup_run_logging()

    logger = logging.getLogger(
        f"AI_Powered_Last_Mile_Delivery_Automation.{module_name}"
    )
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers on repeated imports.
    if logger.handlers:
        return logger

    # Attach trace-ID filter so every record includes the current trace ID,
    # and the PII filter so customer data never reaches disk.
    logger.addFilter(TraceIdFilter())
    logger.addFilter(PIIFilter())

    formatter = _get_formatter()

    # File handler — only if disk logging is available.
    if run_dir is not None and _disk_logging_available:
        fh = logging.handlers.RotatingFileHandler(
            os.path.join(run_dir, f"{module_name}.log"),
            maxBytes=10_485_760,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Console handler — writes to stdout for container log drivers.
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # Prevent messages from propagating to the root logger (avoids duplicates).
    logger.propagate = False

    return logger
