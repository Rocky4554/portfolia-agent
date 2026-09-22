"""
Keyword sales history from our Namebio service — the in-house mirror of
NameBio's daily sales feed (~3-4k sales a day, backfilled to 2024-01-01).

Meant to replace tools/namebio.py, which calls NameBio's paid public API
directly: this service already pays for the same data, so querying it costs
nothing. The valuation graph still uses namebio.py until it's rewired.

One hard limit, straight from the service's code: search is a plain
substring match (SQL LIKE '%keyword%'). That finds keyword and exact-match
comps well ("insurance" → autoinsurance.net), but a coined brandable has no
keyword to search on — use tools/comparables.py for those. The search is
also unindexed: ~5-14s over the full history, hence the long timeout.

Stats and comps come from ONE scan, so the pipeline never pays for that
slow search twice.
"""

from datetime import date

from portfolio_agent import config
from portfolio_agent.config import NAMEBIO_SERVICE_BASE_URL
from portfolio_agent.tools import mocks
from portfolio_agent.tools.common import ToolResult, age_in_days, error_result, mock_result, ok_result, request_json
from portfolio_agent.tools.pricing import price_stats

SOURCE = "namebio_sales"

_HISTORY_START = date(2024, 1, 1)  # earliest date the service has backfilled
_PAGE_SIZE = 500  # verified the service accepts it — fewer slow round-trips
_MAX_PAGES = 10  # caps a very common keyword at 5,000 sales
_TIMEOUT_S = 45.0
_MIN_KEYWORD_LEN = 3  # a 2-letter substring ("ai") matches a huge share of the table
_LENGTH_BAND = 2  # same ±2 SLD-length band the Comparable Agent filters on
_MIN_COMPS_BEFORE_RELAXING = 3


async def get_keyword_sales(
    keyword: str,
    *,
    tld: str | None = None,
    sld_length: int | None = None,
    min_price: float = 100,
    since: date | None = None,
    top_n: int = 20,
) -> ToolResult:
    """Sales whose domain contains `keyword`: price stats plus the closest comps.

    Comps are narrowed to the same `tld` and to an SLD length within ±2 of
    `sld_length` when given. If the length band leaves fewer than 3 comps, it
    is dropped (the TLD filter is kept) and `comp_filters` says so.
    """
    keyword = (keyword or "").strip().lower()
    if len(keyword) < _MIN_KEYWORD_LEN:
        return error_result(SOURCE, f"keyword '{keyword}' is too short for a substring search")
    tld = _normalize_tld(tld)
    since = max(since or _HISTORY_START, _HISTORY_START)
    if config.MOCK_MODE:
        return mock_result(SOURCE, mocks.mock_keyword_sales(keyword, tld, sld_length), data_age_days=11)

    rows, total, truncated, error = await _search_all(keyword, since, min_price)
    if error:
        return error_result(SOURCE, error)

    sales = [_to_sale(row) for row in rows]
    same_tld = [s for s in sales if s["tld"] == tld] if tld else sales

    comps, length_band_applied = same_tld, False
    if sld_length is not None:
        banded = [s for s in same_tld if abs(s["sld_length"] - sld_length) <= _LENGTH_BAND]
        if len(banded) >= _MIN_COMPS_BEFORE_RELAXING:
            comps, length_band_applied = banded, True

    latest_sale = max((s["date"] for s in sales if s["date"]), default=None)
    return ok_result(
        SOURCE,
        {
            "keyword": keyword,
            "date_from": since.isoformat(),
            "min_price": min_price,
            "total_matches": total,
            "truncated": truncated,
            "stats_all_tlds": price_stats(s["price"] for s in sales),
            "stats_same_tld": price_stats(s["price"] for s in same_tld) if tld else None,
            "comp_count": len(comps),
            "comps": sorted(comps, key=lambda s: s["price"], reverse=True)[:top_n],
            "comp_filters": {"tld": tld, "sld_length_band": _LENGTH_BAND if length_band_applied else None},
        },
        data_age_days=age_in_days(latest_sale),
    )


async def get_tld_cohort_sales(
    tld: str, *, min_price: float = 1000, since: date | None = None, top_n: int = 60
) -> ToolResult:
    """Recent sales on one extension, regardless of keyword.

    The comp ladder's cohort rung needs sales of structurally SIMILAR names,
    not of this name — and a keyword search cannot supply them, because
    everything it returns contains the keyword by construction. Filtering
    that pool for a cohort can never produce one.

    This queries the extension itself. The mirror's search is a substring
    match on the domain, so ".ai" matches every .ai sale (and a handful of
    false positives like x.ai.com, which the caller filters on the TLD
    field). Shape filtering — length, dictionary word, no hyphens — is the
    caller's job, because what counts as a cohort depends on the category.

    Slow, like every search on this service. Call it only when the stronger
    rungs came up short.
    """
    tld = _normalize_tld(tld)
    if not tld:
        return error_result(SOURCE, "no TLD given")
    if config.MOCK_MODE:
        return mock_result(SOURCE, mocks.mock_tld_cohort(tld), data_age_days=14)

    since = max(since or _HISTORY_START, _HISTORY_START)
    rows, total, truncated, error = await _search_all(tld, since, min_price)
    if error:
        return error_result(SOURCE, error)

    sales = [s for s in (_to_sale(row) for row in rows) if s["tld"] == tld]
    return ok_result(
        SOURCE,
        {
            "tld": tld,
            "total_matches": total,
            "truncated": truncated,
            "comp_count": len(sales),
            "comps": sorted(sales, key=lambda s: s["price"], reverse=True)[:top_n],
            "stats": price_stats(s["price"] for s in sales),
        },
        data_age_days=age_in_days(max((s["date"] for s in sales if s["date"]), default=None)),
    )


async def _search_all(keyword: str, since: date, min_price: float) -> tuple[list[dict], int, bool, str | None]:
    """Every matching sale, price-descending, up to _MAX_PAGES pages.

    Returns (rows, total_matches, truncated, error).
    """
    url = f"{NAMEBIO_SERVICE_BASE_URL}/namebio/sales/search"
    rows: list[dict] = []
    total = 0
    for page in range(_MAX_PAGES):
        body, error = await request_json(
            "GET",
            url,
            timeout=_TIMEOUT_S,
            params={
                "dateFrom": since.isoformat(),
                # Must be explicit: the service reads a missing dateTo as
                # "same day as dateFrom" and quietly searches a single day.
                "dateTo": date.today().isoformat(),
                "domainSearch": keyword,
                "minPrice": min_price,
                "sortField": "price",
                "sortDir": "desc",
                "page": page,
                "size": _PAGE_SIZE,
            },
        )
        if error or not isinstance(body, dict):
            if rows:  # a later page failed — keep what arrived, flagged as partial
                return rows, total, True, None
            return [], 0, False, error or "unexpected response shape"
        rows.extend(body.get("content") or [])
        total = body.get("totalElements", len(rows))
        if body.get("last", True):
            return rows, total, False, None
    return rows, total, True, None


def _to_sale(row: dict) -> dict:
    domain = str(row.get("domain") or "").lower()
    sld, _, tld = domain.partition(".")
    return {
        "domain": domain,
        "price": float(row.get("price") or 0),
        "date": row.get("saleDate"),
        "marketplace": row.get("marketplace"),
        "tld": f".{tld}" if tld else "",
        "sld_length": len(sld),
    }


def _normalize_tld(tld: str | None) -> str | None:
    if not tld:
        return None
    tld = tld.strip().lower()
    return tld if tld.startswith(".") else f".{tld}"
