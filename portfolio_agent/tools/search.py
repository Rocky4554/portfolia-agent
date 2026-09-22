"""
The search layer, built to be used as little as possible.

**Why the design is about spending, not searching.** Tavily gives 1,000
credits a month and Exa about 1,400 searches for $10. A valuation that
casually issues three searches burns the month in a hundred domains, and
the failure is quiet: you find out when a batch run comes back empty and
the tier that depended on it silently goes unpriced. So the client is built
around four savings, in order of how much they actually save.

**1. Cache, because the answers do not change.** The domain a company uses
and the shape of a keyword's search results are stable facts. Resolving
"The Hartford" to thehartford.com twice is paying twice for one fact. The
cache is on disk, survives restarts, and is checked before anything else —
it is worth more than the other three together.

**2. Ask only where the free path is silent.** The intent lexicon already
classifies most keywords for nothing; search runs only for the ones it
returns `unclassified` for. EDGAR already names incumbents for nothing;
search runs only to resolve the domain EDGAR does not report. Never as a
first resort.

**3. A hard budget that refuses rather than overspends.** Counted, persisted,
and checked before every call. Over the cap, the tool reports itself
unavailable — the same state as a missing key, which every consumer here
already handles. An engine that quietly runs up a bill is worse than one
that stops.

**4. The right provider for the question.** Exa is neural and is good at
"which site belongs to this company", which is a similarity problem.
Tavily returns a conventional result mix, which is what reading commercial
intent off a SERP requires. Using either for the other's job costs the same
credit and answers worse.

**A throttle is not an answer.** Every failure here returns an error, never
an empty result: a rate limit that reads as "this company has no website"
or "this keyword has no commercial intent" would corrupt a price in the
direction nobody checks.
"""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from portfolio_agent import config
from portfolio_agent.logging_setup import get_logger, info, warn
from portfolio_agent.tools.common import ToolResult, error_result, mock_result, ok_result, request_json

log = get_logger(__name__)

SOURCE = "web_search"

_TIMEOUT_S = 30.0
_TAVILY_URL = "https://api.tavily.com/search"
_EXA_URL = "https://api.exa.ai/search"
_SEARXNG_TIMEOUT_S = 15.0

# Hosts that are never a company's own site, so a resolver that returns one
# has failed rather than answered.
_NOT_A_HOMEPAGE = (
    "wikipedia.org", "linkedin.com", "facebook.com", "twitter.com", "x.com",
    "crunchbase.com", "bloomberg.com", "reuters.com", "sec.gov", "glassdoor.com",
    "indeed.com", "youtube.com", "instagram.com", "zoominfo.com", "dnb.com",
    "pitchbook.com", "owler.com", "yahoo.com", "google.com", "amazon.com",
)

# Positive evidence that a result is a place to transact. Required, rather
# than inferred from the absence of an encyclopaedia: an invented word
# returns pages that are simply unrelated, none of them Wikipedia, and
# reading that as "no editorial, therefore commercial" scored zylantix at
# 100% commercial intent. Absence of one thing is not presence of another.
_COMMERCIAL_MARKERS = (
    "/pricing", "/price", "/prices", "/buy", "/shop", "/store", "/quote",
    "/quotes", "/plans", "/plan", "/product", "/products", "/services",
    "/signup", "/sign-up", "/demo", "/order", "/cart", "/checkout", "/booking",
    "/book", "/compare", "/deals", "/offers", "/subscribe", "/get-started",
    "/rates", "/apply", "/insurance", "/loans", "/hire", "/rent",
)
_COMMERCIAL_TITLE_WORDS = (
    "pricing", "price", "buy", "shop", "quote", "quotes", "compare", "deals",
    "discount", "cheap", "best", "book", "hire", "rent", "order", "free trial",
    "get started", "sign up", "plans", "rates", "save",
)

# Result hosts that mark a SERP as informational rather than commercial.
_EDITORIAL = (
    "wikipedia.org", "britannica.com", ".gov", ".edu", "investopedia.com",
    "businessinsider.com", "nytimes.com", "theguardian.com", "bbc.co", "cnn.com",
    "forbes.com", "medium.com", "quora.com", "reddit.com", "stackexchange.com",
    "healthline.com", "webmd.com", "mayoclinic.org", "howstuffworks.com",
)


@dataclass
class Usage:
    calls: int = 0
    dollars: float = 0.0


def _state_path() -> Path:
    return Path(config.SEARCH_STATE_DIR).expanduser() / "search_state.json"


def _load_state() -> dict:
    path = _state_path()
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {"cache": {}, "usage": {}}


def _save_state(state: dict) -> None:
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state))
    except OSError as exc:  # noqa: BLE001
        warn(log, "search.state_unwritable", error=str(exc))


def _month() -> str:
    return time.strftime("%Y-%m")


def usage_this_month(provider: str) -> Usage:
    raw = (_load_state().get("usage") or {}).get(f"{provider}:{_month()}") or {}
    return Usage(calls=int(raw.get("calls") or 0), dollars=float(raw.get("dollars") or 0.0))


def _record(provider: str, dollars: float = 0.0) -> None:
    state = _load_state()
    key = f"{provider}:{_month()}"
    entry = (state.setdefault("usage", {})).setdefault(key, {"calls": 0, "dollars": 0.0})
    entry["calls"] += 1
    entry["dollars"] = round(float(entry.get("dollars") or 0.0) + dollars, 4)
    _save_state(state)


def budget_remaining(provider: str) -> int:
    cap = config.SEARCH_MONTHLY_CAP.get(provider, 0)
    return max(0, cap - usage_this_month(provider).calls)


# Bumped whenever the meaning of a cached value changes. Without it a fixed
# classifier keeps serving its own old mistakes: the first SERP reader
# scored an invented word at 100% commercial intent, and that verdict would
# have outlived the fix by ninety days.
_CACHE_VERSION = "v2"


def _cached(key: str):
    key = f"{_CACHE_VERSION}:{key}"
    entry = (_load_state().get("cache") or {}).get(key)
    if not entry:
        return None
    age_days = (time.time() - float(entry.get("at") or 0)) / 86400.0
    if age_days > config.SEARCH_CACHE_TTL_DAYS:
        return None
    return entry.get("value")


def _cache(key: str, value) -> None:
    key = f"{_CACHE_VERSION}:{key}"
    state = _load_state()
    state.setdefault("cache", {})[key] = {"at": time.time(), "value": value}
    _save_state(state)


async def _search(provider: str, query: str, limit: int) -> tuple[list[dict], str | None, float]:
    """One provider call, budget-checked. Returns (results, error, dollars)."""
    key = config.TAVILY_API_KEY if provider == "tavily" else config.EXA_API_KEY
    if not key:
        return [], f"no {provider.upper()}_API_KEY configured", 0.0
    if budget_remaining(provider) <= 0:
        used = usage_this_month(provider)
        return [], (f"{provider} monthly budget exhausted ({used.calls} calls, "
                    f"cap {config.SEARCH_MONTHLY_CAP.get(provider)}) — refusing rather than "
                    f"overspending; raise SEARCH_MONTHLY_CAP to continue"), 0.0

    if provider == "tavily":
        body, error = await request_json(
            "POST", _TAVILY_URL, timeout=_TIMEOUT_S,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"query": query, "max_results": limit, "search_depth": "basic"})
        dollars = 0.0
    else:
        body, error = await request_json(
            "POST", _EXA_URL, timeout=_TIMEOUT_S,
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json={"query": query, "numResults": limit, "type": "auto"})
        dollars = float(((body or {}).get("costDollars") or {}).get("total") or 0.0) \
            if isinstance(body, dict) else 0.0

    if error:
        return [], error, 0.0
    if not isinstance(body, dict):
        return [], f"{provider} returned {type(body).__name__}, expected an object", 0.0

    _record(provider, dollars)
    results = [r for r in (body.get("results") or []) if isinstance(r, dict)]
    return results, None, dollars


async def _search_searxng(query: str, limit: int) -> tuple[list[dict], str | None]:
    """The self-hosted instance, tried before Tavily/Exa spend a credit.

    Unmetered, so it carries none of the budget/cost bookkeeping the paid
    providers need — just a result list or an error, so a caller falls
    back to them exactly as it would on a missing API key.
    """
    base = config.SEARXNG_BASE_URL
    if not base:
        return [], "no SEARXNG_BASE_URL configured"

    body, error = await request_json(
        "GET", f"{base.rstrip('/')}/search", timeout=_SEARXNG_TIMEOUT_S,
        params={"q": query, "format": "json"})
    if error:
        return [], error
    if not isinstance(body, dict):
        return [], f"searxng returned {type(body).__name__}, expected an object"

    results = [r for r in (body.get("results") or []) if isinstance(r, dict)]
    return results[:limit], None


def _registrable(url: str) -> str | None:
    match = re.match(r"^https?://([^/:?#]+)", (url or "").strip(), re.IGNORECASE)
    if not match:
        return None
    host = match.group(1).lower().removeprefix("www.")
    return host if "." in host else None


async def resolve_company_domain(company: str) -> ToolResult:
    """The domain a company actually uses.

    This is what turns an EDGAR candidate into a lead: the targeted tier
    prices the gap between the domain a company is ON and the one being
    valued, and EDGAR does not report websites. Exa rather than Tavily
    because "which site belongs to this company" is a similarity question.
    """
    company = (company or "").strip()
    if not company:
        return error_result(SOURCE, "company name is empty")
    if config.MOCK_MODE:
        return mock_result(SOURCE, {"company": company, "domain": _mock_domain(company),
                                    "cached": False, "provider": "mock"})

    key = f"company_domain:{company.lower()}"
    hit = _cached(key)
    if hit is not None:
        return ok_result(SOURCE, {**hit, "cached": True})

    query = f"{company} official company website"

    # SearXNG first: free, so a hit here is a credit Exa never spends. Only
    # an error or a page with no plausible homepage falls through to Exa —
    # same as a missing SEARXNG_BASE_URL.
    searxng_results, searxng_error = await _search_searxng(query, 5)
    host = next((h for r in searxng_results
                 if (h := _registrable(r.get("url", ""))) and not any(bad in h for bad in _NOT_A_HOMEPAGE)),
                None)
    if host:
        value = {"company": company, "domain": host, "provider": "searxng", "cost_dollars": 0.0}
        _cache(key, value)
        info(log, "search.company_resolved", company=company, domain=host, provider="searxng", cost=0.0)
        return ok_result(SOURCE, {**value, "cached": False})
    if searxng_error:
        info(log, "search.searxng_unavailable", error=searxng_error)

    results, error, dollars = await _search("exa", query, 3)
    if error:
        return error_result(SOURCE, error)

    for result in results:
        host = _registrable(result.get("url", ""))
        # A directory listing is not the company's site. Returning
        # linkedin.com as "their domain" would make every lead look like it
        # had the same credibility gap.
        if host and not any(bad in host for bad in _NOT_A_HOMEPAGE):
            value = {"company": company, "domain": host, "provider": "exa",
                     "cost_dollars": dollars}
            _cache(key, value)
            info(log, "search.company_resolved", company=company, domain=host, provider="exa", cost=dollars)
            return ok_result(SOURCE, {**value, "cached": False})

    value = {"company": company, "domain": None, "provider": "exa", "cost_dollars": dollars}
    _cache(key, value)
    return ok_result(SOURCE, {**value, "cached": False,
                              "note": "no result looked like a company homepage"})


async def serp_composition(keyword: str) -> ToolResult:
    """What the results for a bare keyword actually are.

    The instrument F9 wants. "Is this word's demand commercial" is answered
    by what ranks for it — product and comparison pages, or encyclopaedia
    entries — and not by a trend index, which measures attention without
    saying what kind.

    Called only where the intent lexicon returns `unclassified`, so the
    common cases cost nothing.
    """
    keyword = (keyword or "").strip().lower()
    if not keyword:
        return error_result(SOURCE, "keyword is empty")
    if config.MOCK_MODE:
        return mock_result(SOURCE, _mock_composition(keyword))

    key = f"serp:{keyword}"
    hit = _cached(key)
    if hit is not None:
        return ok_result(SOURCE, {**hit, "cached": True})

    # SearXNG first: free, so a hit here is a credit Tavily never spends.
    # Only an error or an empty page falls through to Tavily — same as a
    # missing SEARXNG_BASE_URL.
    results, searxng_error = await _search_searxng(keyword, 8)
    provider = "searxng"
    if not results:
        if searxng_error:
            info(log, "search.searxng_unavailable", error=searxng_error)
        results, error, _ = await _search("tavily", keyword, 8)
        if error:
            return error_result(SOURCE, error)
        provider = "tavily"

    if not results:
        # No results is a real answer about an obscure keyword, but it is
        # not evidence of commercial intent either way.
        return ok_result(SOURCE, {"keyword": keyword, "intent": "unclassified",
                                  "commercial_share": None, "sampled": 0, "cached": False})

    commercial, editorial, unknown = _classify_results(results)
    classified = commercial + editorial
    # Too little classified evidence is no evidence. An unrelated SERP for a
    # coined word lands here, which is where it belongs — unclassified costs
    # the domain nothing.
    if classified < config.SERP_MIN_CLASSIFIED:
        value = {"keyword": keyword, "intent": "unclassified", "commercial_share": None,
                 "sampled": len(results), "commercial": commercial, "editorial": editorial,
                 "unknown": unknown, "provider": provider,
                 "note": f"only {classified} of {len(results)} results could be read as either "
                         f"commercial or editorial — not enough to call the intent"}
        _cache(key, value)
        return ok_result(SOURCE, {**value, "cached": False})

    share = round(commercial / classified, 3)
    intent = ("commercial" if share >= config.SERP_COMMERCIAL_SHARE
              else "informational" if share <= 1 - config.SERP_COMMERCIAL_SHARE
              else "unclassified")
    value = {"keyword": keyword, "intent": intent, "commercial_share": share,
             "sampled": len(results), "commercial": commercial, "editorial": editorial,
             "unknown": unknown, "provider": provider}
    _cache(key, value)
    info(log, "search.serp_intent", keyword=keyword, intent=intent, share=share, provider=provider)
    return ok_result(SOURCE, {**value, "cached": False})


async def web_search(query: str, limit: int = 8) -> ToolResult:
    """Raw search results for whatever a caller is looking up.

    resolve_company_domain and serp_composition are shaped for this agent's
    own valuation logic — a domain, a commercial-intent verdict. This is the
    generic path underneath the public /search API (portfolio_agent/api.py):
    a domain lead, a company name, anything a caller outside this project
    wants results for, with no valuation-specific interpretation attached.

    Same SearXNG-first, Tavily-fallback order and the same disk cache as the
    rest of this module, so a public caller re-asking a question that was
    already answered costs nothing either.
    """
    query = (query or "").strip()
    if not query:
        return error_result(SOURCE, "query is empty")
    limit = max(1, min(int(limit or 8), 20))
    if config.MOCK_MODE:
        return mock_result(SOURCE, {"query": query, "provider": "mock",
                                    "results": _mock_web_results(query, limit)})

    key = f"web:{limit}:{query.lower()}"
    hit = _cached(key)
    if hit is not None:
        return ok_result(SOURCE, {**hit, "cached": True})

    results, searxng_error = await _search_searxng(query, limit)
    provider = "searxng"
    if not results:
        if searxng_error:
            info(log, "search.searxng_unavailable", error=searxng_error)
        results, error, _ = await _search("tavily", query, limit)
        if error:
            return error_result(SOURCE, error)
        provider = "tavily"

    normalized = [{"title": r.get("title"), "url": r.get("url"),
                   "content": r.get("content") or r.get("snippet")} for r in results]
    value = {"query": query, "provider": provider, "results": normalized}
    _cache(key, value)
    info(log, "search.web_search", query=query, provider=provider, results=len(normalized))
    return ok_result(SOURCE, {**value, "cached": False})


def _mock_web_results(query: str, limit: int) -> list[dict]:
    from portfolio_agent.tools.mocks import _seed

    seed = _seed(query, "web")
    slug = re.sub(r"[^a-z0-9]", "", query.lower())[:18] or "example"
    return [{"title": f"{query} — result {i + 1}", "url": f"https://{slug}{i}.example.com",
             "content": f"mock result {i + 1} for {query!r}"} for i in range((seed % limit) + 1)]


def _classify_results(results: list[dict]) -> tuple[int, int, int]:
    """Each result as commercial, editorial, or neither.

    Neither is the common case for a coined word and is counted separately
    on purpose: it must not be silently folded into commercial, which is how
    an unrelated SERP came to read as pure buying intent.
    """
    commercial = editorial = unknown = 0
    for result in results:
        url = str(result.get("url") or "").lower()
        host = _registrable(url) or ""
        title = str(result.get("title") or "").lower()
        if any(e in host for e in _EDITORIAL):
            editorial += 1
        elif (any(m in url for m in _COMMERCIAL_MARKERS)
              or any(w in title for w in _COMMERCIAL_TITLE_WORDS)):
            commercial += 1
        else:
            unknown += 1
    return commercial, editorial, unknown


def _mock_domain(company: str) -> str:
    slug = re.sub(r"[^a-z0-9]", "", company.lower())[:18] or "example"
    return f"{slug}.com"


def _mock_composition(keyword: str) -> dict:
    from portfolio_agent.tools.mocks import _seed

    seed = _seed(keyword, "serp")
    share = round(0.25 + (seed % 60) / 100.0, 3)
    intent = ("commercial" if share >= config.SERP_COMMERCIAL_SHARE
              else "informational" if share <= 1 - config.SERP_COMMERCIAL_SHARE
              else "unclassified")
    return {"keyword": keyword, "intent": intent, "commercial_share": share,
            "sampled": 8, "editorial_hosts": [], "provider": "mock", "cached": False}
