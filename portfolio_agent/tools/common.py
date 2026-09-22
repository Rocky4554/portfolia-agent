"""
Shared plumbing for the valuation tools that call Namekart's internal
services (DVS, the Namebio sales service, the Comparable AI Agent).

Every such tool returns the same ToolResult envelope and never raises: a
service being down, slow or rate-limited is recoverable (the valuation just
has less evidence, and can say so), a crashed graph is not. Unlike
tools/namebio.py's mock mode, these tools never substitute sample numbers on
failure — a missing appraisal is honest, an invented one gets priced.
"""

import asyncio
import secrets
import time
from datetime import datetime, timezone
from typing import Any, TypedDict

import httpx

from portfolio_agent import config
from portfolio_agent.logging_setup import debug, get_logger, warn

log = get_logger(__name__)


class ToolResult(TypedDict):
    source: str  # stable label, also used as the evidence source name
    ok: bool
    data: dict | None  # normalized fields only — never a raw upstream payload
    data_age_days: int | None  # how old the underlying data is, when known
    error: str | None
    mock: bool  # True when this came from a fixture, not the live service


def ok_result(source: str, data: dict, data_age_days: int | None = None, *, mock: bool = False) -> ToolResult:
    return {"source": source, "ok": True, "data": data, "data_age_days": data_age_days, "error": None, "mock": mock}


def error_result(source: str, error: str) -> ToolResult:
    return {"source": source, "ok": False, "data": None, "data_age_days": None, "error": error, "mock": False}


def mock_result(source: str, data: dict, data_age_days: int | None = None) -> ToolResult:
    """A fixture answer, flagged so it cannot be mistaken for a live one.

    Every consumer — the evidence items, the warnings list, the rationale —
    reads this flag, so a mock number is labeled everywhere it surfaces.
    """
    return ok_result(source, data, data_age_days, mock=True)


def new_traceparent() -> str:
    """A W3C traceparent, so a single call can be found in the service's Grafana/Loki logs."""
    return f"00-{secrets.token_hex(16)}-{secrets.token_hex(8)}-01"


def age_in_days(timestamp: str | None) -> int | None:
    """Whole days since an ISO-8601 date or timestamp; None if missing or unparseable."""
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0, (datetime.now(timezone.utc) - parsed).days)


def _make_client(timeout: float, *, follow_redirects: bool = False) -> httpx.AsyncClient:
    # Single construction point so the evals can swap in an httpx.MockTransport.
    # Redirects are off by default and opt-in per call: most of these APIs
    # answer directly, and silently following a redirect to somewhere
    # unexpected is not a property to hand every caller. RDAP needs it —
    # rdap.org is a bootstrap service whose whole job is to redirect to the
    # registry that actually holds the record.
    return httpx.AsyncClient(timeout=timeout, follow_redirects=follow_redirects)


async def request_json(
    method: str,
    url: str,
    *,
    timeout: float,
    params: dict | None = None,
    json: Any = None,
    headers: dict | None = None,
    max_attempts: int | None = None,
    follow_redirects: bool = False,
) -> tuple[Any, str | None]:
    """One logical HTTP call → (parsed JSON body, None) or (None, short error).

    Never raises. Retries the failures that are worth retrying — timeouts,
    dropped connections, 429s and 5xx — with exponential backoff, and gives
    up immediately on a 4xx, which is a bad request that a second attempt
    would only spend the rate limit on twice.

    The returned error names the last failure and how many attempts it took,
    because "timed out after 30s (3 attempts)" and "timed out after 30s" mean
    very different things when reading a warning in a valuation.
    """
    attempts = max_attempts if max_attempts is not None else config.HTTP_MAX_ATTEMPTS
    attempts = max(1, attempts)
    error = "no attempt was made"

    made = 0
    for attempt in range(attempts):
        made += 1
        started = time.perf_counter()
        body, error, retryable = await _attempt(
            method, url, timeout, params, json, headers, follow_redirects)
        elapsed = round((time.perf_counter() - started) * 1000, 2)

        # The endpoint only, never the query string: DVS and the sales
        # service put domains and keywords in there, and a log is one more
        # place that data gets read.
        endpoint = _endpoint(url)
        if error is None:
            debug(log, "http.ok", method=method, endpoint=endpoint, attempt=made, duration_ms=elapsed)
            return body, None

        warn(log, "http.failed", method=method, endpoint=endpoint, attempt=made,
             duration_ms=elapsed, error=error, retryable=retryable,
             will_retry=retryable and attempt < attempts - 1)

        if not retryable or attempt == attempts - 1:
            break
        await asyncio.sleep(config.HTTP_BACKOFF_BASE_S * (2 ** attempt))

    # Report attempts actually made, not the ceiling: a 4xx gives up after
    # one try, and saying "3 attempts" there would misdescribe the failure.
    return None, error if made == 1 else f"{error} ({made} attempts)"


async def _attempt(
    method: str, url: str, timeout: float, params: dict | None, json: Any, headers: dict | None,
    follow_redirects: bool = False,
) -> tuple[Any, str | None, bool]:
    """A single HTTP round trip → (body, error, retryable)."""
    try:
        async with _make_client(timeout, follow_redirects=follow_redirects) as client:
            resp = await client.request(method, url, params=params, json=json, headers=headers)
    except httpx.TimeoutException:
        return None, f"timed out after {timeout:.0f}s", True
    except httpx.HTTPError as e:
        return None, f"connection failed ({type(e).__name__})", True

    if resp.status_code == 429:
        return None, _describe_rate_limit(resp), True
    if resp.status_code >= 500:
        return None, f"HTTP {resp.status_code}", True
    if resp.status_code >= 400:
        return None, f"HTTP {resp.status_code}", False
    if not resp.content:
        # An empty 200 is a service bug, not a transient one (DVS's
        # /estibot/leads/domain does this today) — retrying won't fill it in.
        return None, f"HTTP {resp.status_code} with an empty body", False
    try:
        return resp.json(), None, False
    except ValueError:
        return None, f"HTTP {resp.status_code} with a non-JSON body", False


def _endpoint(url: str) -> str:
    """Scheme, host and path — the part safe to log. The query string carries
    the domain or keyword being valued, which belongs in the tool's own
    summary line, not in every HTTP record."""
    return url.split("?", 1)[0]


def _describe_rate_limit(resp: httpx.Response) -> str:
    """DVS's 429 body names the exhausted bucket and window — that's the actionable part, so keep it."""
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    details = [f"{k}={body[k]}" for k in ("apiType", "window", "bucket", "retryAfter") if k in body]
    return "rate limited" + (f" ({', '.join(details)})" if details else "")
