"""
scan_market_trends — APVA §3.3's trend scanner.

Two layers, and the split matters:

  1. The **verdict** comes from knowledge/market_trends.py: deterministic
     rules over curated hot/risk lists. This is what the tier engine reads.
     It works offline, it is reproducible, and it is regression-tested.
  2. The **corroboration** comes from a web search (Jina, or a Google CSE),
     which returns headlines and links. These are attached as `sources` so a
     rationale can cite them.

Search results never move the number. They are third-party text fetched from
the open web — exactly the input a prompt-injection attack arrives on — and
they reach the model wrapped as evidence. A search snippet that says "this
category is worth $5M" is a snippet that says that, not a fact about price.

Without a search key configured the tool still returns its full verdict with
an empty `sources` list and a note saying so. Degrading to "no corroboration"
is honest; degrading to "no trend analysis" would silently drop a modifier
the valuation is expecting.
"""

from datetime import date

from portfolio_agent import config
from portfolio_agent.knowledge import market_trends
from portfolio_agent.tools.common import ToolResult, mock_result, ok_result, request_json

SOURCE = "market_trends"

_TIMEOUT_S = 20.0
_MAX_SOURCES = 5


async def scan_market_trends(
    category: str,
    keywords: list[str] | None = None,
    *,
    tld: str = "com",
    today: date | None = None,
) -> ToolResult:
    """Trend verdict for a category/keyword set, plus corroborating links.

    Returns `{hot_category, risk_category, trend_velocity, matched_rules,
    sources, stale, as_of}`. `trend_velocity` is -1..1 and is a modifier on
    comps, never a price.
    """
    keywords = [k for k in (keywords or []) if k]
    signal = market_trends.evaluate(category, tld, keywords, today)

    payload = {
        "category": category,
        "keywords": keywords,
        "tld": tld,
        "hot_category": signal.hot_category,
        "risk_category": signal.risk_category,
        "trend_velocity": signal.trend_velocity,
        "matched_rules": signal.matched_rules,
        "stale": signal.stale,
        "as_of": market_trends.AS_OF.isoformat(),
        "staleness_warning": market_trends.staleness_warning(today),
        "sources": [],
    }

    if config.MOCK_MODE:
        payload["sources"] = _mock_sources(category, keywords)
        payload["sources_note"] = "fixture links — not real search results"
        return mock_result(SOURCE, payload)

    payload["sources"], note = await _search_sources(category, keywords, tld)
    if note:
        payload["sources_note"] = note
    return ok_result(SOURCE, payload)


def _query_for(category: str, keywords: list[str], tld: str) -> str:
    head = keywords[0] if keywords else category
    return f"{head} .{tld} domain sales trend {date.today().year}"


async def _search_sources(category: str, keywords: list[str], tld: str) -> tuple[list[dict], str | None]:
    """Corroborating links from whichever search backend is configured.

    Never raises and never blocks the verdict: a failed search costs the
    output its citations, not its trend reading.
    """
    query = _query_for(category, keywords, tld)

    if config.JINA_API_KEY:
        body, error = await request_json(
            "GET",
            f"{config.JINA_SEARCH_URL}/{query}",
            headers={"Authorization": f"Bearer {config.JINA_API_KEY}", "Accept": "application/json"},
            timeout=_TIMEOUT_S,
        )
        if error:
            return [], f"search unavailable ({error}) — verdict is rule-based only"
        return _normalize_jina(body), None

    if config.GOOGLE_CSE_KEY and config.GOOGLE_CSE_ID:
        body, error = await request_json(
            "GET",
            "https://www.googleapis.com/customsearch/v1",
            params={"key": config.GOOGLE_CSE_KEY, "cx": config.GOOGLE_CSE_ID, "q": query},
            timeout=_TIMEOUT_S,
        )
        if error:
            return [], f"search unavailable ({error}) — verdict is rule-based only"
        return _normalize_cse(body), None

    return [], "no search backend configured — verdict is rule-based only, with no corroborating sources"


def _normalize_jina(body) -> list[dict]:
    rows = (body or {}).get("data") if isinstance(body, dict) else None
    return [
        {"title": r.get("title"), "link": r.get("url"), "snippet": (r.get("description") or "")[:300]}
        for r in (rows or [])[:_MAX_SOURCES]
        if isinstance(r, dict)
    ]


def _normalize_cse(body) -> list[dict]:
    rows = (body or {}).get("items") if isinstance(body, dict) else None
    return [
        {"title": r.get("title"), "link": r.get("link"), "snippet": (r.get("snippet") or "")[:300]}
        for r in (rows or [])[:_MAX_SOURCES]
        if isinstance(r, dict)
    ]


def _mock_sources(category: str, keywords: list[str]) -> list[dict]:
    head = keywords[0] if keywords else category
    return [
        {
            "title": f"[FIXTURE] {head} domain sales report",
            "link": f"https://example.invalid/fixtures/{head}",
            "snippet": "Fixture source. Contains no real market reporting.",
        }
    ]
