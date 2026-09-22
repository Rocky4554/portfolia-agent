"""
Structured logging for the whole agent.

Every step of a valuation logs what it did, how long it took, and what it
decided. The goal is that a run which went wrong in production can be
reconstructed from its logs alone, without a reproduction — because the
inputs (a live market service's response at a point in time) usually cannot
be reproduced.

Five decisions here are deliberate, and each avoids a specific failure:

**Only the `portfolio_agent` logger is configured, never the root logger.**
This package is imported into nameaiv1's LangGraph server. Calling
`logging.basicConfig()` from a library hijacks the host application's
logging, which is rude at best and silently loses their logs at worst. The
package installs a NullHandler and emits nothing until someone calls
`setup_logging()`.

**A run id travels through `contextvars`, not a parameter.** Evidence
gathering runs its tools concurrently under asyncio, so a `threading.local`
would leak one valuation's id into another's log lines. `contextvars` is the
one mechanism that survives `asyncio.gather` correctly. Every line in a run
carries the same `run_id`, so `grep run=ab12cd34` reconstructs it.

**Structured fields, not formatted sentences.** Values go in `extra`, not
interpolated into the message, so the text renderer can print them for a
human and the JSON renderer can emit them for a log aggregator without
either having to parse the other's output.

**Durations are on every step.** "Which tool made this valuation slow" is
the question that actually gets asked, and it cannot be answered
retroactively from logs that did not record it.

**Secrets and untrusted payloads are never logged.** Credentials are
redacted by a filter as a backstop, but the real rule is upstream: tools log
summaries and counts, never raw upstream bodies. Those bodies carry both PII
(registrant names, emails) and whatever text an attacker put on a web page,
and a log viewer is just another place that text gets read.
"""

import json
import logging
import os
import re
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

LOGGER_NAME = "portfolio_agent"

# Per-run correlation, set by `run_context` and read by the filter below.
_run_id: ContextVar[str] = ContextVar("portfolio_agent_run_id", default="")
_run_domain: ContextVar[str] = ContextVar("portfolio_agent_run_domain", default="")

# Field names whose values never reach a log line, whatever the caller does.
#
# Anchored deliberately. A loose substring match on "token" redacts
# `max_tokens` and a token *count*, and a redaction that fires on harmless
# fields trains everyone to ignore it — which is how the one that mattered
# gets ignored too.
_SECRET_KEY_PATTERN = re.compile(
    r"^(key|token|secret|password|credential|authorization|auth)$"
    r"|(^|_)(api_?key|access_token|refresh_token|secret|password|passwd|bearer)(_|$)",
    re.IGNORECASE,
)
# Bearer tokens and long opaque strings appearing inside a message.
_SECRET_VALUE_PATTERN = re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{12,}", re.IGNORECASE)

REDACTED = "***redacted***"

# Anything longer than this is truncated before it is logged. Applies to
# model output and error bodies — enough to diagnose, not enough to dump a
# document into a log aggregator.
MAX_VALUE_CHARS = 300

_configured = False


def get_logger(name: str) -> logging.Logger:
    """A logger under the package namespace.

    Call as `get_logger(__name__)`; module names already start with
    `portfolio_agent.`, so the hierarchy — and therefore per-module level
    control — works out of the box.
    """
    return logging.getLogger(name)


# ── Context ────────────────────────────────────────────────────────────────


@contextmanager
def run_context(domain: str = "", run_id: str | None = None):
    """Tag every log line inside this block with one run id.

    Nested calls keep the outer id, so a valuation invoked from a batch job
    stays correlated with the batch rather than starting a new trace.
    """
    existing = _run_id.get()
    token_id = _run_id.set(existing or run_id or uuid.uuid4().hex[:8])
    token_domain = _run_domain.set(domain or _run_domain.get())
    try:
        yield _run_id.get()
    finally:
        _run_id.reset(token_id)
        _run_domain.reset(token_domain)


def current_run_id() -> str:
    return _run_id.get()


# ── Emitting ───────────────────────────────────────────────────────────────


def log_event(logger: logging.Logger, level: int, event: str, exc_info: bool = False, **fields) -> None:
    """Log one structured event.

    `event` is a stable dotted name (`tool.done`, `tier.computed`) meant to
    be grepped and counted; the varying detail belongs in `fields`. Keeping
    them apart is what makes "how often does the comps tool fail" answerable
    without a regex over free text.
    """
    logger.log(level, event, extra={"ctx": _clean(fields)}, exc_info=exc_info)


def debug(logger, event, **fields):
    log_event(logger, logging.DEBUG, event, **fields)


def info(logger, event, **fields):
    log_event(logger, logging.INFO, event, **fields)


def warn(logger, event, **fields):
    log_event(logger, logging.WARNING, event, **fields)


def error(logger, event, exc_info: bool = False, **fields):
    log_event(logger, logging.ERROR, event, exc_info=exc_info, **fields)


class Step:
    """Handle for a `log_step` block, for attaching results to its end line."""

    def __init__(self):
        self.fields: dict = {}

    def set(self, **fields) -> None:
        self.fields.update(fields)


@contextmanager
def log_step(logger: logging.Logger, event: str, level: int = logging.INFO, **fields):
    """Time a step and log its outcome.

    Emits `<event>.start` at DEBUG, then exactly one of `<event>.done` or
    `<event>.failed` carrying `duration_ms`. The failure path re-raises — the
    point is to record what happened, never to swallow it.
    """
    step = Step()
    started = time.perf_counter()
    log_event(logger, logging.DEBUG, f"{event}.start", **fields)
    try:
        yield step
    except Exception as e:
        log_event(
            logger, logging.ERROR, f"{event}.failed",
            exc_info=True,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            error=f"{type(e).__name__}: {e}",
            **fields, **step.fields,
        )
        raise
    log_event(
        logger, level, f"{event}.done",
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
        **fields, **step.fields,
    )


def _clean(fields: dict) -> dict:
    """Redact credential-shaped keys and truncate long values."""
    cleaned = {}
    for key, value in fields.items():
        if _SECRET_KEY_PATTERN.search(key):
            cleaned[key] = REDACTED
        elif isinstance(value, str) and len(value) > MAX_VALUE_CHARS:
            cleaned[key] = value[:MAX_VALUE_CHARS] + f"...[{len(value)} chars]"
        else:
            cleaned[key] = value
    return cleaned


# ── Filters and formatters ─────────────────────────────────────────────────


class ContextFilter(logging.Filter):
    """Attaches the current run id and domain to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id.get()
        record.domain = _run_domain.get()
        if not hasattr(record, "ctx"):
            record.ctx = {}
        return True


class RedactionFilter(logging.Filter):
    """Last line of defence against a credential reaching a log.

    The real protection is that nothing logs a secret in the first place;
    this catches the case where a third-party library's exception message
    contains an Authorization header, which does happen.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _SECRET_VALUE_PATTERN.sub(r"\1" + REDACTED, record.msg)
        ctx = getattr(record, "ctx", None)
        if isinstance(ctx, dict):
            record.ctx = {
                k: (_SECRET_VALUE_PATTERN.sub(r"\1" + REDACTED, v) if isinstance(v, str) else v)
                for k, v in ctx.items()
            }
        return True


class TextFormatter(logging.Formatter):
    """Human-readable, one line per event, fields trailing."""

    default_time_format = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        ctx = getattr(record, "ctx", {}) or {}
        fields = " ".join(f"{k}={_render(v)}" for k, v in ctx.items())
        run = getattr(record, "run_id", "") or "-"
        domain = getattr(record, "domain", "") or "-"
        module = record.name.removeprefix(LOGGER_NAME + ".")
        line = (
            f"{self.formatTime(record)} {record.levelname:<7} "
            f"[{run} {domain}] {module}: {record.getMessage()}"
        )
        if fields:
            line += "  " + fields
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for a log aggregator.

    Hand-rolled rather than pulling in a dependency: it is twenty lines, and
    the alternative adds a package to a valuation service for formatting.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "run_id": getattr(record, "run_id", "") or None,
            "domain": getattr(record, "domain", "") or None,
        }
        payload.update(getattr(record, "ctx", {}) or {})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _render(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    text = str(value)
    return f'"{text}"' if " " in text else text


# ── Setup ──────────────────────────────────────────────────────────────────


def setup_logging(level: str | int | None = None, fmt: str | None = None,
                  stream=None, force: bool = False) -> logging.Logger:
    """Configure the package logger. Idempotent.

    Env defaults: PORTFOLIO_AGENT_LOG_LEVEL (INFO), PORTFOLIO_AGENT_LOG_FORMAT
    (text | json), PORTFOLIO_AGENT_LOG_FILE (also write to this path).

    Touches only the `portfolio_agent` logger and sets `propagate = False`,
    so importing this package never changes how the host application logs.
    """
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    if _configured and not force:
        return logger

    level = level or os.environ.get("PORTFOLIO_AGENT_LOG_LEVEL", "INFO")
    fmt = (fmt or os.environ.get("PORTFOLIO_AGENT_LOG_FORMAT", "text")).lower()
    formatter = JsonFormatter() if fmt == "json" else TextFormatter()

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    log_file = os.environ.get("PORTFOLIO_AGENT_LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        # Always JSON on disk: a file is read by a tool, a terminal by a
        # person, and JSON is the format that survives being grepped a month
        # later by something that is not a human eye.
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)

    logger.addFilter(ContextFilter())
    logger.addFilter(RedactionFilter())
    for h in logger.handlers:
        h.addFilter(ContextFilter())
        h.addFilter(RedactionFilter())

    logger.setLevel(level if isinstance(level, int) else str(level).upper())
    # Do not hand records to the root logger: this package is embedded in
    # another service, and duplicating into their handlers would double every
    # line and apply their formatting to our structured records.
    logger.propagate = False
    _configured = True
    return logger


def auto_setup() -> None:
    """Configure logging when this package is the deployed entrypoint.

    Called from nodes/graph.py, which is what langgraph.json points at, so a
    `langgraph dev` server logs without anyone wiring it up. Skipped when
    PORTFOLIO_AGENT_LOG_SETUP=off, and skipped when handlers already exist,
    so a host application that configured us keeps its configuration.
    """
    if os.environ.get("PORTFOLIO_AGENT_LOG_SETUP", "").lower() == "off":
        return
    if logging.getLogger(LOGGER_NAME).handlers:
        return
    setup_logging()
