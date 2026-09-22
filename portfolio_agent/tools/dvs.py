"""
Domain Valuation Service (DVS): our cache and rate limiter in front of
Estibot, GoDaddy GoValue and DotDB (namekart/domain-valuation-service).

Verified against DVS's source and live responses on 2026-09-14:
  - Every call sends X-System-Actor, so DVS's rate limiter bills a stable
    "SYSTEM:portfolio-agent" principal instead of an anonymous IP.
  - Freshness is always explicit (maxAgeDays): an unbounded Estibot lookup
    happily returns a year-old appraisal.
  - DVS has no authentication today; nothing here relies on that.

State of the upstreams on 2026-09-14, which the ok/error results reflect:
  - GoDaddy: live and cached values both work.
  - Estibot: cached data works, but live appraisals and every leads call fail
    upstream, so a domain Estibot hasn't already cached returns no profile.
  - investorScore: its three LLM-scored factors all came back at their exact
    fallback values, i.e. DVS's Groq calls are failing and the score is
    rule-based only (insurance.com scored 27/100, tier D). The result flags
    this as llm_scoring_likely_failed.
"""

from portfolio_agent import config
from portfolio_agent.tools import mocks
from portfolio_agent.config import DVS_BASE_URL, DVS_ESTIBOT_MAX_AGE_DAYS, DVS_GDV_MAX_AGE_DAYS, DVS_SYSTEM_ACTOR
from portfolio_agent.tools.common import (
    ToolResult,
    age_in_days,
    error_result,
    mock_result,
    new_traceparent,
    ok_result,
    request_json,
)

SOURCE_ESTIBOT = "dvs_estibot"
SOURCE_GODADDY = "dvs_godaddy"
SOURCE_INVESTOR_SCORE = "dvs_investor_score"
SOURCE_DOTDB = "dvs_dotdb"
SOURCE_LEADS = "dvs_estibot_leads"

_TIMEOUT_S = 30.0
_INVESTOR_SCORE_TIMEOUT_S = 90.0  # DVS makes three LLM calls per uncached score

# Sent on every call. `source` feeds DVS's usage stats, and the leads and
# DotDB routes reject requests without it. OTHER is DVS's own platform type
# for callers that aren't AMP or a registrar.
_BASE_PARAMS = {"source": DVS_SYSTEM_ACTOR, "platformType": "OTHER"}

# Allowlist, not a denylist: leads rows also carry registrant names, emails,
# phone numbers and scraped site emails. Only business-level fields pass, so
# a field Estibot adds later can't leak into a prompt or a log.
#
# `domain` is the LEAD's own site and is the single most important field
# here — it is what makes a credibility gap measurable. It was missing for a
# long time, and `source_domain` was read in its place: that is the domain
# being SEARCHED, identical across every row, so every lead appeared to be
# sitting on the very domain being valued, scored a gap of zero, and was
# discarded. Both are kept now, because the pair is what shows the
# relationship.
#
# `site_title` is a public web page title and carries the business name.
# `whois_reg_org` is empty in this data, which is why it alone was never
# going to identify anyone.
_LEAD_FIELDS = (
    "domain",
    "sld",
    "tld",
    "source_domain",
    "site_title",
    "type",
    "lead_score",
    "whois_reg_org",
    "whois_reg_country",
    "site_status",
    "site_total_pages",
    "estibot_alexa_traffic_rank",
    "site_has_online_store",
    "site_language",
    "whois_age",
)


async def get_domain_profile(domain: str, *, max_age_days: int = DVS_ESTIBOT_MAX_AGE_DAYS) -> ToolResult:
    """Estibot's profile of a domain: category, keywords and structure,
    trademark risk, keyword economics, whois, and Estibot's automated value."""
    if config.MOCK_MODE:
        return mock_result(SOURCE_ESTIBOT, mocks.mock_estibot_profile(domain), data_age_days=3)
    raw, error = await _estibot_lookup(domain, max_age_days)
    if error:
        return error_result(SOURCE_ESTIBOT, error)
    return ok_result(SOURCE_ESTIBOT, _profile_from(raw), data_age_days=age_in_days(raw.get("fetchedAt")))


async def get_base_appraisal(
    domain: str, *, max_age_days: int = DVS_GDV_MAX_AGE_DAYS, cache_only: bool = False
) -> ToolResult:
    """GoDaddy GoValue: value, wholesale and list price, and sale probabilities.

    GoValue models the mass-market registrar tail and badly underprices
    premium names (insurance.com: $25,000, against Estibot's $36.49M and a
    real $35.6M sale) — one signal, never the anchor. `cache_only=True`
    never spends a GoDaddy credit, for bulk portfolio runs.
    """
    domain = _clean_domain(domain)
    if not domain:
        return error_result(SOURCE_GODADDY, "domain is empty")
    if config.MOCK_MODE:
        return mock_result(SOURCE_GODADDY, mocks.mock_godaddy(domain), data_age_days=7)
    body, error = await _get("/getgdv", {"domain": domain, "maxAgeDays": max_age_days, "cacheOnly": cache_only})
    if error:
        return error_result(SOURCE_GODADDY, error)
    if not isinstance(body, dict):
        return error_result(SOURCE_GODADDY, "unexpected response shape")
    if str(body.get("status") or "").lower() == "quota_exceeded":
        return error_result(SOURCE_GODADDY, "GoDaddy quota exceeded")

    go_value = _money(body, "goValue")
    if not body.get("ok") or go_value is None:
        return error_result(SOURCE_GODADDY, f"no GoDaddy value (status={body.get('status')})")
    return ok_result(
        SOURCE_GODADDY,
        {
            "domain": domain,
            "go_value": go_value,
            "wholesale_value": _money(body, "goValueWholesale"),
            "list_price": _money(body, "listPrice"),
            "min_price": _money(body, "minPrice"),
            "max_price": _money(body, "maxPrice"),
            "sales_probability": body.get("salesProbability"),
            "sales_probability_at_500": body.get("salesProbability500"),
            "sales_probability_wholesale": body.get("salesProbabilityWholesale"),
            "data_source": body.get("dataSource"),
        },
        data_age_days=age_in_days(body.get("fetchedAt")),
    )


async def get_investor_score(domain: str) -> ToolResult:
    """DVS's 0-100 investor score, A-D tier and 10-factor breakdown.

    A quality signal, not a price. Scores are cached for 7 days on DVS's
    side, so one computed while its LLM calls were failing sticks around
    until it expires — check llm_scoring_likely_failed.
    """
    if config.MOCK_MODE:
        return mock_result(SOURCE_INVESTOR_SCORE, mocks.mock_investor_score(_clean_domain(domain)))
    # The score endpoint takes Estibot's full profile as its body; this
    # lookup is normally a local-DB hit on DVS, since the profile was just
    # fetched.
    raw, error = await _estibot_lookup(domain, DVS_ESTIBOT_MAX_AGE_DAYS)
    if error:
        return error_result(SOURCE_INVESTOR_SCORE, f"needs an Estibot profile first: {error}")
    body, error = await request_json(
        "POST",
        f"{DVS_BASE_URL}/api/investorScore",
        json=raw,
        headers=_headers(),
        timeout=_INVESTOR_SCORE_TIMEOUT_S,
    )
    if error:
        return error_result(SOURCE_INVESTOR_SCORE, error)
    if not isinstance(body, dict) or body.get("totalScore") is None:
        return error_result(SOURCE_INVESTOR_SCORE, "unexpected response shape")

    breakdown = body.get("breakdown") or {}
    return ok_result(
        SOURCE_INVESTOR_SCORE,
        {
            "domain": body.get("domain"),
            "total_score": body.get("totalScore"),
            "tier": body.get("tier"),
            "breakdown": breakdown,
            "llm_scoring_likely_failed": _llm_scoring_likely_failed(breakdown, raw),
            "from_cache": bool(body.get("fromCache")),
        },
    )


async def get_keyword_demand(keyword: str) -> ToolResult:
    """DotDB: how widely a keyword is registered with an active website, across names and TLDs."""
    keyword = (keyword or "").strip().lower()
    if not keyword:
        return error_result(SOURCE_DOTDB, "keyword is empty")
    if config.MOCK_MODE:
        return mock_result(SOURCE_DOTDB, mocks.mock_keyword_demand(keyword))
    body, error = await _get("/dotdb/getleads", {"keyword": keyword})
    if error:
        return error_result(SOURCE_DOTDB, error)
    if not isinstance(body, dict) or body.get("total_name") is None:
        return error_result(SOURCE_DOTDB, "unexpected response shape")
    return ok_result(
        SOURCE_DOTDB,
        {
            "keyword": keyword,
            # e.g. insurance.<tld> is an active site in 347 TLDs
            "exact_keyword_active_tld_count": body.get("exact_match_total_suffix"),
            "active_names_containing_keyword": body.get("total_name"),
            "active_registrations_containing_keyword": body.get("total_suffix"),
            "top_matches": [
                {"name": m.get("name"), "tld_count": m.get("count")} for m in (body.get("matches") or [])[:10]
            ],
        },
    )


async def find_end_user_leads(domain: str, *, limit: int = 15) -> ToolResult:
    """Estibot leads: businesses using the same term on other domains — potential end-user buyers.

    Takes a domain, not a keyword: Estibot rejects keyword-only lead
    requests ("invalid domain."), and /estibot/leads is the DVS route that
    reports upstream failures explicitly instead of returning an empty body.
    Only business-level fields are returned — see _LEAD_FIELDS.
    """
    domain = _clean_domain(domain)
    if not domain:
        return error_result(SOURCE_LEADS, "domain is empty")
    if config.MOCK_MODE:
        return mock_result(SOURCE_LEADS, mocks.mock_leads(domain))
    body, error = await _get("/estibot/leads", {"d": domain})
    if error:
        return error_result(SOURCE_LEADS, error)
    if not isinstance(body, dict):
        return error_result(SOURCE_LEADS, "unexpected response shape")
    if str(body.get("success") or "").lower() not in ("true", "1"):
        reason = body.get("reason") or body.get("message") or "no reason given"
        return error_result(SOURCE_LEADS, f"Estibot leads failed: {reason}")

    results = body.get("results") or {}
    rows = results.get("data") or []
    leads = [{field: row.get(field) for field in _LEAD_FIELDS} for row in rows[:limit]]
    # Normalized names, so a caller does not have to know that Estibot calls
    # the lead's own site `domain` and the searched term `source_domain`.
    for lead in leads:
        lead["current_domain"] = lead.get("domain")
        lead["business_name"] = lead.get("site_title") or lead.get("whois_reg_org")
    return ok_result(
        SOURCE_LEADS,
        {
            "domain": domain,
            "total": results.get("total", len(rows)),
            "leads": leads,
        },
    )


async def _estibot_lookup(domain: str, max_age_days: int) -> tuple[dict | None, str | None]:
    """The raw Estibot row via DVS's lookup waterfall: DVS's DB → Estibot cache → Estibot live."""
    domain = _clean_domain(domain)
    if not domain:
        return None, "domain is empty"
    body, error = await _get("/estibot/appraisal/lookup", {"d": domain, "maxAgeDays": max_age_days})
    if error:
        return None, error
    if not isinstance(body, dict):
        return None, "unexpected response shape"
    # DVS sets ok=true only when there's a value > 0; with no category either,
    # Estibot has nothing on the domain at all.
    if not body.get("ok") and not body.get("category"):
        return None, "Estibot returned no data for this domain"
    return body, None


def _profile_from(raw: dict) -> dict:
    return {
        "domain": raw.get("domain"),
        "sld": raw.get("sld"),
        "tld": f".{raw['tld']}" if raw.get("tld") else None,
        "category": raw.get("category"),
        "category_root": raw.get("category_root"),
        "words": (raw.get("words") or "").split(),
        "num_words": raw.get("num_words"),
        "sld_length": raw.get("sld_length"),
        "has_numbers": bool(raw.get("has_numbers")),
        "num_hyphens": raw.get("num_hyphens"),
        "language": raw.get("language"),
        "is_adult": bool(raw.get("is_adult")),
        "trademark_risk": {
            "has_trademark": bool(raw.get("has_trademark")),
            "udrp_or_wipo_case": bool(raw.get("has_udrp_wipo")) or bool(raw.get("udrp")),
        },
        "estibot_value": _positive(raw.get("appraised_value")),
        "estibot_value_ignoring_trademark": _positive(raw.get("appraised_ignore_tm_value")),
        "keyword_metrics": {
            "cpc": _known(raw.get("keyword_exact_cpc")),
            "competition": _known(raw.get("keyword_exact_competition")),
            "global_monthly_searches": _known(raw.get("keyword_exact_global_search_volume")),
            "local_monthly_searches": _known(raw.get("keyword_exact_local_search_volume")),
        },
        "whois": {
            "created": raw.get("whois_create_date"),
            "expires": raw.get("whois_expire_date"),
            "registrar": raw.get("whois_registrar"),
            "private": bool(raw.get("whois_is_private")),
        },
        "first_archived_year": _positive(raw.get("wayback_age")),
        "sld_registered_tld_count": _known(raw.get("tld_count")),
        "data_source": raw.get("dataSource"),
    }


def _llm_scoring_likely_failed(breakdown: dict, raw: dict) -> bool:
    """True when all three LLM-scored factors equal the fallbacks DVS uses
    when its LLM call fails (InvestorScoringService): buyer_count falls back
    to a mapping of Estibot's end_user_buyers, startup_usage to 3,
    acquisition_prob to 2. One matching could be a real answer; all three
    matching is the failure signature."""
    end_user_buyers = raw.get("end_user_buyers") or 0
    buyer_fallback = 16 if end_user_buyers >= 10 else 12 if end_user_buyers >= 4 else 6 if end_user_buyers >= 1 else 0
    return (
        breakdown.get("buyer_count") == buyer_fallback
        and breakdown.get("startup_usage") == 3
        and breakdown.get("acquisition_prob") == 2
    )


async def _get(path: str, params: dict) -> tuple[object, str | None]:
    return await request_json(
        "GET", f"{DVS_BASE_URL}{path}", params={**_BASE_PARAMS, **params}, headers=_headers(), timeout=_TIMEOUT_S
    )


def _headers() -> dict:
    return {"X-System-Actor": DVS_SYSTEM_ACTOR, "X-Source": DVS_SYSTEM_ACTOR, "traceparent": new_traceparent()}


def _clean_domain(domain: str | None) -> str:
    domain = (domain or "").strip().lower()
    domain = domain.removeprefix("http://").removeprefix("https://").removeprefix("www.")
    return domain.split("/")[0]


def _money(body: dict, key: str):
    value = (body.get(key) or {}).get("value") if isinstance(body.get(key), dict) else None
    return value if isinstance(value, (int, float)) and value > 0 else None


def _positive(value):
    """Estibot uses 0 for "no value" on appraisals and archive years."""
    return value if isinstance(value, (int, float)) and value > 0 else None


def _known(value):
    """Estibot uses -1 for "unknown" on metrics where 0 is a real answer."""
    return value if isinstance(value, (int, float)) and value >= 0 else None
