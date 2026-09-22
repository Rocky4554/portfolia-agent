"""
get_site_history — what this domain used to be, from the Internet Archive.

**Why it is worth having.** Everything else in this package describes the
domain today. The archive knows whether it was ever on a sale lander (and at
what asking price, which is a historical ASK nobody is charging for), when a
site went dormant, and how long it has carried real content. A domain with
twenty years of continuous pages is a different asset from one registered
the same year and parked ever since, and RDAP cannot tell them apart — both
are simply "registered in 2005".

**Why it is NOT in the default evidence plan.** The CDX endpoint took 35
seconds on every attempt while this was written, twice in a row, having
first refused with a 429. The valuation budget is 60 seconds and a live run
currently takes about twelve. Adding a reliable 35-second call to every
valuation to gain a historical nice-to-have is a bad trade made silently, so
it is opt-in via PORTFOLIO_AGENT_ENABLE_WAYBACK and off by default. The
right home for it is a cached backfill job — a domain's 1998 snapshot does
not change — rather than the hot path of a valuation.

**A slow or throttled archive is a failed lookup, never an empty history.**
This is the same discipline as everywhere else here: "we could not ask" and
"there is nothing there" are different answers, and only one of them is a
fact about the domain. Getting that backwards would let a rate limit
masquerade as evidence that a domain has never been developed.
"""

from datetime import date, datetime

from portfolio_agent import config
from portfolio_agent.logging_setup import get_logger, info, warn
from portfolio_agent.tools.common import ToolResult, error_result, mock_result, ok_result, request_json

log = get_logger(__name__)

SOURCE = "wayback"

# Generous because the endpoint genuinely is this slow, not because we are
# being patient with a flaky one. Measured at ~35s repeatedly.
_TIMEOUT_S = 45.0
_CDX = "http://web.archive.org/cdx/search/cdx"

# No content captured in this long and the site is dormant, whatever the
# registration says. Two years is past any plausible redesign gap.
_DORMANT_AFTER_DAYS = 730


async def get_site_history(domain: str) -> ToolResult:
    """Archive timeline for a domain.

    Returns `{first_seen, last_seen, snapshot_count, years_archived,
    dormant, dormant_since_days}`.
    """
    domain = (domain or "").strip().lower().rstrip(".")
    if not domain or "." not in domain:
        return error_result(SOURCE, "domain is empty or has no TLD")

    if config.MOCK_MODE:
        return mock_result(SOURCE, _mock_history(domain))

    if not config.ENABLE_WAYBACK:
        return error_result(
            SOURCE,
            "archive history is disabled (PORTFOLIO_AGENT_ENABLE_WAYBACK) — the CDX endpoint "
            "runs ~35s per call, which would dominate the valuation's latency budget")

    body, error = await request_json(
        "GET", _CDX,
        params={
            "url": domain,
            "output": "json",
            "fl": "timestamp,statuscode",
            # One row per month at most: the shape of the timeline is the
            # signal, and a busy domain has tens of thousands of captures.
            "collapse": "timestamp:6",
            "limit": "600",
        },
        timeout=_TIMEOUT_S,
    )
    if error:
        warn(log, "wayback.unavailable", domain=domain, error=error)
        return error_result(SOURCE, error)

    # CDX returns a header row followed by data rows, or an empty list for a
    # domain it has never captured — which IS an answer.
    if not isinstance(body, list):
        return error_result(SOURCE, f"CDX returned {type(body).__name__}, expected a list")
    rows = [r for r in body[1:] if isinstance(r, list) and r]
    if not rows:
        info(log, "wayback.never_captured", domain=domain)
        return ok_result(SOURCE, _result(domain, stamps=[]))

    stamps = []
    for row in rows:
        parsed = _as_date(row[0])
        if parsed:
            stamps.append(parsed)
    data = _result(domain, stamps=sorted(stamps))
    info(log, "wayback.resolved", domain=domain, snapshots=data["snapshot_count"],
         years=data["years_archived"], dormant=data["dormant"])
    return ok_result(SOURCE, data)


async def find_historical_listings(domain: str, samples: int = 4) -> ToolResult:
    """Whether this domain has ever sat on a sale lander, and at what ask.

    The one signal here that nobody is charging for. A domain that carried
    an Afternic lander at $12,000 in 2021 and a working site today tells you
    two things no current-state check can: that the owner has been willing
    to sell, and roughly where they started. A domain that has been parked
    for nine straight years tells you the opposite.

    **A historical ask is the weakest kind of price evidence here and is
    labelled as such.** It is not a sale — nobody paid it — and it is not
    even a current ask, because the seller may have moved or given up since.
    It reaches `historical_asks` and nothing that feeds comps, for the same
    reason marketplace listings do not: a comp is what a buyer paid.

    Slow by construction. Each snapshot is a separate archive fetch on an
    endpoint that measured ~35s, so this samples a handful of captures
    across the timeline rather than walking it, and belongs in a backfill.
    """
    domain = (domain or "").strip().lower().rstrip(".")
    if not domain or "." not in domain:
        return error_result(SOURCE, "domain is empty or has no TLD")
    if config.MOCK_MODE:
        return mock_result(SOURCE, _mock_listings(domain))
    if not config.ENABLE_WAYBACK:
        return error_result(SOURCE, "archive history is disabled (PORTFOLIO_AGENT_ENABLE_WAYBACK)")

    timeline = await get_site_history(domain)
    if not timeline["ok"]:
        return timeline
    stamps = _timeline_stamps(timeline["data"] or {})
    if not stamps:
        return ok_result(SOURCE, _listings_result(domain, [], checked=0))

    from portfolio_agent.tools.lander import classify_page

    # Spread across the timeline rather than clustered at either end: a
    # domain's recent captures all look alike, and the lander years are
    # usually in the middle.
    picks = _spread_samples(stamps, samples)
    found, checked = [], 0
    for stamp in picks:
        html, error = await _fetch_snapshot(domain, stamp)
        if error:
            warn(log, "wayback.snapshot_failed", domain=domain, stamp=stamp, error=error)
            continue
        checked += 1
        verdict = classify_page(domain, html, f"https://{domain}/", 200)
        if verdict["status"] == "listed_for_sale":
            found.append({
                "captured": stamp,
                "venue": verdict.get("venue"),
                # Named asking_price for the same reason it is everywhere
                # else: so nothing downstream can mistake it for a sale.
                "asking_price": verdict.get("asking_price"),
                "confidence": verdict.get("confidence"),
            })
    info(log, "wayback.listings", domain=domain, checked=checked, found=len(found))
    return ok_result(SOURCE, _listings_result(domain, found, checked=checked))


def _listings_result(domain: str, found: list[dict], *, checked: int) -> dict:
    priced = [f for f in found if f.get("asking_price")]
    return {
        "domain": domain,
        "historical_asks": found,
        "snapshots_checked": checked,
        "ever_listed": bool(found),
        "lowest_historical_ask": min((f["asking_price"] for f in priced), default=None),
        "highest_historical_ask": max((f["asking_price"] for f in priced), default=None),
        "note": (
            "historical asking prices, not sales — nobody paid these, and the seller may have "
            "moved or given up since. They show that an owner was once willing to sell and "
            "roughly where they started, and they must not reach the comp set"
            if found else
            f"no sale lander found in the {checked} snapshot(s) sampled. That is a sample, not "
            f"a search: the archive does not capture every day and a lander may have come and "
            f"gone between captures"
        ),
    }


def _timeline_stamps(data: dict) -> list[str]:
    first, last = data.get("first_seen"), data.get("last_seen")
    return [s.replace("-", "") for s in (first, last) if s]


def _spread_samples(stamps: list[str], samples: int) -> list[str]:
    """Evenly spaced capture dates between the first and last seen.

    The CDX call already collapsed to one row a month, and re-querying per
    sample would multiply a 35s call. So the sample points are synthesised
    from the span and the archive redirects each to its nearest capture,
    which is what its timestamp URLs do by design.
    """
    if len(stamps) < 2:
        return stamps[:samples]
    start, end = int(stamps[0][:4]), int(stamps[-1][:4])
    if end <= start:
        return stamps[:samples]
    step = max(1, (end - start) // max(1, samples - 1))
    years = list(range(start, end + 1, step))[:samples]
    return [f"{year}0601000000" for year in years]


async def _fetch_snapshot(domain: str, stamp: str) -> tuple[str, str | None]:
    """One archived capture. `id_` asks for the original bytes rather than
    the archive's rewritten page, so the lander markup we match on survives."""
    import httpx

    url = f"https://web.archive.org/web/{stamp}id_/http://{domain}/"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S, follow_redirects=True) as client:
            response = await client.get(
                url, headers={"User-Agent": config.CRAWLER_USER_AGENT})
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}"
    if response.status_code != 200:
        return "", f"HTTP {response.status_code}"
    return response.text[:400_000], None


def _mock_listings(domain: str) -> dict:
    from portfolio_agent.tools.mocks import _seed, _spread

    seed = _seed(domain, "wayback_listings")
    if seed % 3 == 0:
        return _listings_result(domain, [], checked=4)
    found = [{
        "captured": f"{2016 + (seed % 6)}0601000000",
        "venue": ("Afternic", "Sedo", "Dan.com")[seed % 3],
        "asking_price": int(_spread(seed, 0, 2_500, 90_000)) // 100 * 100,
        "confidence": "high",
    }]
    return _listings_result(domain, found, checked=4)


def _result(domain: str, *, stamps: list[date]) -> dict:
    if not stamps:
        return {
            "domain": domain, "first_seen": None, "last_seen": None, "snapshot_count": 0,
            "years_archived": None, "dormant": None, "dormant_since_days": None,
            "note": "the archive has never captured this domain, which usually means it has "
                    "never carried a public site",
        }
    first, last = stamps[0], stamps[-1]
    since = (date.today() - last).days
    return {
        "domain": domain,
        "first_seen": first.isoformat(),
        "last_seen": last.isoformat(),
        "snapshot_count": len(stamps),
        "years_archived": round((last - first).days / 365.25, 1),
        "dormant": since > _DORMANT_AFTER_DAYS,
        "dormant_since_days": since,
        "note": None,
    }


def _as_date(stamp) -> date | None:
    try:
        return datetime.strptime(str(stamp)[:8], "%Y%m%d").date()
    except (ValueError, TypeError):
        return None


def _mock_history(domain: str) -> dict:
    from datetime import timedelta

    from portfolio_agent.tools.mocks import _seed, _spread

    seed = _seed(domain, "wayback")
    if seed % 9 == 0:
        return _result(domain, stamps=[])
    span = int(_spread(seed, 0, 400, 9000))
    gap = int(_spread(seed, 1, 5, 1400))
    last = date.today() - timedelta(days=gap)
    first = last - timedelta(days=span)
    months = max(1, span // 30)
    return _result(domain, stamps=[first + timedelta(days=30 * i) for i in range(months)] + [last])
