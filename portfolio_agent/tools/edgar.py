"""
find_public_incumbents — established companies in a domain's market, from SEC filings.

**The lead type this exists for.** `find_potential_end_users` looks for
recently-funded startups on compromise domains, which is one buyer. The
incumbent is another and is invisible to it: a profitable company that has
been in the market for twenty years, has more cash than most Series A
startups, has never raised a round anybody wrote about, and sits on a
domain it settled for. The funding-news collector structurally cannot see
them — they are not news. EDGAR can, because a public company files whether
or not anyone is paying attention.

**Free, keyless, and authoritative.** data.sec.gov is the filing record
itself rather than a database about it. SEC asks for a User-Agent naming who
is calling and rate-limits at ten requests a second; both are honoured here,
because being blocked from a public good for impoliteness would be a
self-inflicted outage.

**What it gives and what it does not.** Entity name, CIK, tickers, exchange,
the SIC code (which is the industry, and maps onto knowledge/industries.py),
the business address, and revenue from the XBRL facts. Revenue is a better
qualification for an incumbent than funding is: a company with $400M of
revenue and no venture round is a stronger buyer than one that raised $20M,
and the funding threshold would have rejected it.

It does NOT give the company's domain. The `website` field exists in the
schema and is empty for most filers — Forge Group returns "". So an EDGAR
hit is a candidate, not a lead: the credibility gap this tier prices is the
distance between the domain they are ON and the one being valued, and until
that is resolved there is a company, not a lead. Resolving it needs a search
call, which is the one part of this that is not free.
"""

import re

from portfolio_agent import config
from portfolio_agent.logging_setup import debug, get_logger, info, warn
from portfolio_agent.tools.common import ToolResult, error_result, mock_result, ok_result, request_json

log = get_logger(__name__)

SOURCE = "sec_edgar"

_TIMEOUT_S = 25.0
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# SIC ranges to the verticals in knowledge/industries.py. Ranges rather than
# codes because SIC is hierarchical and the leaves are far more specific than
# anything a domain valuation cares about.
_SIC_RANGES = [
    ((6300, 6411), "insurance"),
    ((6000, 6299), "finance"),
    ((6700, 6799), "finance"),
    ((8110, 8111), "legal"),
    ((8000, 8099), "health"),
    ((2830, 2836), "health"),
    ((6500, 6599), "real_estate"),
    ((1520, 1799), "home_services"),
    ((7370, 7379), "software"),
    ((3570, 3579), "software"),
    ((4810, 4899), "software"),
    # Travel first: SIC nests passenger air inside a transport block that is
    # otherwise freight, so a broad logistics range checked first would file
    # scheduled airlines under haulage.
    ((7000, 7011), "travel"),
    ((4512, 4512), "travel"),
    ((4522, 4522), "travel"),
    ((4700, 4799), "travel"),
    ((4200, 4299), "logistics"),
    ((4400, 4499), "logistics"),
    ((4513, 4513), "logistics"),
    ((4600, 4699), "logistics"),
    ((8200, 8299), "education"),
    # Food before the broad retail range: eating places (5812) sit inside
    # it, and a restaurant chain is not an ecommerce buyer.
    ((2000, 2099), "food"),
    ((5810, 5819), "food"),
    ((5200, 5999), "ecommerce"),
    ((7310, 7319), "marketing"),
    ((7990, 7999), "gambling"),
    ((3711, 3716), "automotive"),
    ((5500, 5599), "automotive"),
    ((7360, 7369), "recruiting"),
]

# Tried in order: filers report revenue under whichever tag their accountants
# chose, and a company is not missing revenue merely because it does not use
# the first one.
_REVENUE_CONCEPTS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
)

_index_cache: dict[int, dict] | None = None


def industry_for_sic(sic) -> str | None:
    """The vertical a SIC code belongs to, or None where we do not map it."""
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return None
    for (low, high), industry in _SIC_RANGES:
        if low <= code <= high:
            return industry
    return None


def _headers() -> dict:
    # SEC blocks callers who do not identify themselves, and rightly.
    return {"User-Agent": config.SEC_USER_AGENT, "Accept": "application/json"}


async def find_public_incumbents(keyword: str, industry: str | None = None,
                                 limit: int = 5) -> ToolResult:
    """Public companies whose name carries `keyword`, with their industry and revenue.

    Returns `{candidates: [...], searched_for, industry_filter}`. Each
    candidate carries `company, cik, ticker, exchange, sic, industry,
    revenue, city`. They are candidates rather than leads until the domain
    each is on has been resolved.
    """
    keyword = (keyword or "").strip().lower()
    if not keyword or len(keyword) < 3:
        return error_result(SOURCE, "keyword is empty or too short to match a company name")

    if config.MOCK_MODE:
        return mock_result(SOURCE, _mock_incumbents(keyword, industry))

    if not config.SEC_USER_AGENT:
        return error_result(
            SOURCE,
            "no SEC_USER_AGENT configured — data.sec.gov requires a User-Agent naming a real "
            "contact address (e.g. 'yourcompany (you@example.com)') and returns 403 without "
            "one. Free, but not anonymous.")

    index, error = await _company_index()
    if error:
        return error_result(SOURCE, error)

    # Word-boundary matching. A substring search for "forge" returns
    # Forgent Power Solutions, which is a different word and a different
    # company, and a lead list built from coincidental letter runs is worse
    # than an empty one.
    # Word boundaries on BOTH sides.
    #
    # This was a trailing boundary only, on the reasoning that English
    # compounds append — MindForge is Mind + Forge with the word intact. It
    # is, and the rule still cost more than it earned: searching "hop"
    # returned Build-A-Bear WorkSHOP as the strongest incumbent for
    # hop.xyz. A three-letter keyword lives inside a great many words, and
    # a fabricated lead does not merely clutter a list — it qualifies, and
    # prices the highest tier in the system off a company with no
    # connection to the domain.
    #
    # Losing MindForge is a real cost and the cheaper one. Missing a lead
    # leaves a tier honestly unpriced; inventing one puts a number on it.
    pattern = re.compile(rf"(?<![a-z]){re.escape(keyword)}(?![a-z])", re.IGNORECASE)
    matches = [row for row in index.values()
               if pattern.search(str(row.get("title", "")))]
    if not matches:
        info(log, "edgar.no_name_match", keyword=keyword)
        return ok_result(SOURCE, {"candidates": [], "searched_for": keyword,
                                  "industry_filter": industry})

    candidates = []
    # Bounded: each detail is a separate request, and SEC's courtesy limit is
    # ten a second. A wider net belongs in a backfill, not in a valuation.
    for row in matches[: limit * 3]:
        detail, detail_error = await _company_detail(int(row["cik_str"]))
        if detail_error or not detail:
            debug(log, "edgar.detail_failed", cik=row.get("cik_str"), error=detail_error)
            continue
        matched_industry = industry_for_sic(detail.get("sic"))
        # An industry filter is a filter: a fire-and-casualty insurer is not
        # a lead for a software domain however well its name matches.
        if industry and matched_industry != industry:
            continue
        revenue, _ = await _revenue(int(row["cik_str"]))
        candidates.append({
            "company": detail.get("name"),
            "cik": detail.get("cik"),
            "ticker": (detail.get("tickers") or [None])[0],
            "exchange": (detail.get("exchanges") or [None])[0],
            "sic": detail.get("sic"),
            "sic_description": detail.get("sicDescription"),
            "industry": matched_industry,
            "revenue": revenue,
            "city": ((detail.get("addresses") or {}).get("business") or {}).get("city"),
            # Empty for most filers. Named explicitly so no consumer mistakes
            # its absence for "they have no website".
            "website": (detail.get("website") or "").strip() or None,
            "lead_type": "incumbent",
            "current_domain": None,  # needs a search call to resolve
        })
        if len(candidates) >= limit:
            break

    candidates.sort(key=lambda c: c.get("revenue") or 0, reverse=True)
    info(log, "edgar.resolved", keyword=keyword, matched=len(matches),
         candidates=len(candidates), industry=industry)
    return ok_result(SOURCE, {
        "candidates": candidates,
        "searched_for": keyword,
        "industry_filter": industry,
        "name_matches": len(matches),
        "note": "candidates, not leads — the domain each company currently uses is unresolved, "
                "and the credibility gap this tier prices is the distance between that domain "
                "and the one being valued",
    })


async def _company_index() -> tuple[dict, str | None]:
    """The whole ticker/CIK index, fetched once per process.

    Cached because it is 800KB, changes daily at most, and every lookup
    needs it — refetching per valuation would be the slowest part of the run
    and the rudest thing we do to a free service.
    """
    global _index_cache
    if _index_cache is not None:
        return _index_cache, None
    body, error = await request_json("GET", _TICKERS_URL, headers=_headers(), timeout=_TIMEOUT_S)
    if error:
        if "403" in error:
            error = (f"{error} — SEC rejected the User-Agent {config.SEC_USER_AGENT!r}. It must "
                     f"name a real contact address.")
        return {}, error
    if not isinstance(body, dict):
        return {}, f"company index returned {type(body).__name__}, expected an object"
    _index_cache = body
    return body, None


async def _company_detail(cik: int) -> tuple[dict | None, str | None]:
    body, error = await request_json(
        "GET", _SUBMISSIONS.format(cik=cik), headers=_headers(), timeout=_TIMEOUT_S)
    if error:
        return None, error
    return (body, None) if isinstance(body, dict) else (None, "unexpected submissions shape")


async def _revenue(cik: int) -> tuple[int | None, str | None]:
    """Most recent annual revenue, under whichever tag the filer used.
    
    Uses the companyfacts bulk endpoint rather than per-concept queries
    to avoid 404s on companies that report revenue under different tags.
    """
    body, error = await request_json(
        "GET", _COMPANYFACTS.format(cik=cik),
        headers=_headers(), timeout=_TIMEOUT_S, max_attempts=1)
    if error or not isinstance(body, dict):
        return None, f"companyfacts fetch failed: {error}"
    
    facts = body.get("facts", {}).get("us-gaap", {})
    for concept in _REVENUE_CONCEPTS:
        if concept not in facts:
            continue
        rows = [r for r in ((facts[concept].get("units") or {}).get("USD") or [])
                if isinstance(r, dict) and r.get("form") == "10-K" and r.get("val")]
        if not rows:
            continue
        # Filers restate, so the same fiscal year appears more than once.
        # Take the latest filing rather than the largest number.
        latest = max(rows, key=lambda r: (str(r.get("filed") or ""), r.get("end") or ""))
        return int(latest["val"]), None
    return None, "no revenue concept reported"


def _mock_incumbents(keyword: str, industry: str | None) -> dict:
    from portfolio_agent.tools.mocks import _seed, _spread

    seed = _seed(keyword, "edgar")
    count = seed % 4
    candidates = [
        {
            "company": f"{keyword.title()} {suffix}",
            "cik": f"{1000000 + seed % 800000:010d}",
            "ticker": keyword[:3].upper(),
            "exchange": ("NASDAQ", "NYSE", "OTC")[seed % 3],
            "sic": 7372,
            "sic_description": "Services-Prepackaged Software",
            "industry": industry or "software",
            "revenue": int(_spread(seed, i, 12_000_000, 2_400_000_000)),
            "city": "SAN FRANCISCO",
            "website": None,
            "lead_type": "incumbent",
            "current_domain": None,
        }
        for i, suffix in enumerate(("Group, Inc.", "Holdings Corp", "Industries Inc")[:count])
    ]
    return {"candidates": candidates, "searched_for": keyword, "industry_filter": industry,
            "name_matches": count, "note": "candidates, not leads — the domain each company "
            "currently uses is unresolved"}
