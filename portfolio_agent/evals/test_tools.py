"""
Offline regression eval for the valuation tools — no network, no services,
no API keys. Run with: uv run python -m portfolio_agent.evals.test_tools

Every HTTP call is answered by an httpx.MockTransport with responses shaped
like the real ones captured from DVS, the Namebio service and the Comparable
Agent on 2026-09-14. For a check against the live services, use
smoke_tools.py instead.
"""

import asyncio
import json
import sys
from contextlib import contextmanager

import httpx

from portfolio_agent.config import DVS_SYSTEM_ACTOR
from portfolio_agent.tools import common, comparables, dvs, pricing, sales_history
from portfolio_agent.tools.registry import VALUATION_TOOLS

# ── Captured response shapes (trimmed to the fields the tools read) ─────────

ESTIBOT_INSURANCE = {
    "ok": True, "dataSource": "local_db", "fetchedAt": "2025-08-21T08:47:01Z",
    "domain": "insurance.com", "sld": "insurance", "tld": "com", "words": "insurance",
    "num_words": 1, "sld_length": 9, "category": "Finance -- Insurance", "category_root": "Finance",
    "appraised_value": 36490000, "appraised_ignore_tm_value": 36490000,
    "has_trademark": 0, "has_udrp_wipo": 0, "udrp": 0,
    "keyword_exact_cpc": 14.77, "keyword_exact_competition": 1,
    "keyword_exact_global_search_volume": 7480000, "keyword_exact_local_search_volume": 6120000,
    "whois_create_date": "1994-06-30", "whois_expire_date": "2027-01-08",
    "whois_registrar": "GoDaddy.com, LLC", "whois_is_private": 0,
    "wayback_age": 1998, "tld_count": 351, "end_user_buyers": -1,
}

GDV_INSURANCE = {
    "status": "DB", "domain": "insurance.com", "ok": True, "dataSource": "cache", "fetchedAt": "2026-09-11T06:15:32Z",
    "goValue": {"currencyCode": "USD", "value": 25000}, "goValueWholesale": {"currencyCode": "USD", "value": 1000},
    "listPrice": {"currencyCode": "USD", "value": 50000}, "minPrice": {"currencyCode": "USD", "value": 100},
    "maxPrice": {"currencyCode": "USD", "value": 25000},
    "salesProbability": 0.4604, "salesProbability500": 1.0, "salesProbabilityWholesale": 0.8255,
}

# The real 2026-09-14 score: all three LLM factors at their fallback values.
INVESTOR_SCORE_AT_FALLBACK = {
    "domain": "insurance.com", "totalScore": 27, "tier": "D", "fromCache": False, "cot": None,
    "breakdown": {"buyer_count": 0, "industry_momentum": 7, "brandability": 5, "industry_growth": 3,
                  "startup_usage": 3, "acquisition_prob": 2, "scarcity": 2, "linguistic": 1,
                  "radio_test": 2, "domain_age": 2},
}

DOTDB_INSURANCE = {
    "items_per_page": 30, "estimate_total": False,
    "exact_match_total_suffix": 347, "total_name": 191664, "total_suffix": 258909,
    "matches": [{"name": "insurance", "count": 347, "suffixes": [".com"]},
                {"name": "007insurance", "count": 1, "suffixes": [".com"]}],
}

LEADS_UPSTREAM_ERROR = {"success": "false", "message": None, "reason": "upstream_error",
                        "results": {"total": 0, "count": 0, "data": []}}

LEADS_WITH_PII = {"success": "true", "results": {"total": 1, "count": 1, "data": [{
    "source_domain": "insurancequotes.com", "type": "domain", "lead_score": 80,
    "whois_reg_org": "Quotes Inc", "whois_reg_country": "US", "site_status": "active",
    "whois_reg_name": "Jane Doe", "whois_reg_email": "jane@example.com",
    "whois_reg_phone": "+15550100", "site_emails": "info@example.com",
}]}}

COMPARABLE_FAILED = {"success": False, "data": None, "error": "Scoring failed: 'NoneType' object is not iterable"}

COMPARABLE_WITH_WARNING = {"success": True, "error": None, "data": {
    "input_domain": "onepay.ai", "primary_category": "Brandable", "secondary_category": "Service-based",
    "descriptions": ["A payments brand."], "total_comparables": 1, "confidence": "low",
    "warning": "Used mock data due to API issues",
    "comparables": [{"domain": "paynow.ai", "price": 12000, "date": "2025-11-01", "platform": "Sedo",
                     "semantic_sim": 0.71, "cat_match": 1.0, "recency": 0.5, "score": 0.73}],
}}

# Target insurance.com: .com, SLD length 9 → length band 7-11.
NAMEBIO_ROWS = [
    {"domain": "autoinsurance.net", "price": 34500.0, "saleDate": "2025-08-28", "marketplace": "GoDaddy"},
    {"domain": "longtermcareinsurance.com", "price": 12900.0, "saleDate": "2025-08-04", "marketplace": "GoDaddy"},
    {"domain": "insurances.com", "price": 9000.0, "saleDate": "2025-07-01", "marketplace": "Sedo"},
    {"domain": "myinsurance.com", "price": 7000.0, "saleDate": "2026-01-10", "marketplace": "Afternic"},
    {"domain": "insuranceai.com", "price": 5550.0, "saleDate": "2025-03-15", "marketplace": "DropCatch"},
]


def namebio_page(rows, *, last=True, total=None):
    return {"content": rows, "totalElements": len(rows) if total is None else total, "last": last}


# ── Harness ─────────────────────────────────────────────────────────────────

class Checks:
    def __init__(self):
        self.total = 0
        self.failures: list[str] = []

    def expect(self, condition, message: str):
        self.total += 1
        if not condition:
            self.failures.append(message)


@contextmanager
def mock_http(route):
    """Answer every tool HTTP call with route(request) -> httpx.Response; yields the recorded requests."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return route(request)

    original = common._make_client
    common._make_client = lambda timeout, **kwargs: httpx.AsyncClient(
        transport=httpx.MockTransport(handler), timeout=timeout, **kwargs)
    try:
        yield calls
    finally:
        common._make_client = original


def respond(body, status=200):
    return lambda request: httpx.Response(status, json=body)


def run(coro):
    return asyncio.run(coro)


# ── Checks ──────────────────────────────────────────────────────────────────

def check_http_plumbing(c: Checks):
    rate_limit = {"error": "Rate limit exceeded", "apiType": "gdv", "window": "day", "bucket": "principal", "retryAfter": 86400}
    with mock_http(respond(rate_limit, status=429)):
        _, error = run(common.request_json("GET", "https://dvs.test/x", timeout=5))
    c.expect(error and "bucket=principal" in error and "retryAfter=86400" in error,
             f"429 should name the exhausted bucket and retry time, got {error!r}")

    with mock_http(lambda r: httpx.Response(200, content=b"")):
        _, error = run(common.request_json("GET", "https://dvs.test/x", timeout=5))
    c.expect(error and "empty body" in error, f"an empty 200 (DVS's /estibot/leads/domain today) should be an error, got {error!r}")

    with mock_http(lambda r: httpx.Response(200, content=b"<html>")):
        _, error = run(common.request_json("GET", "https://dvs.test/x", timeout=5))
    c.expect(error and "non-JSON" in error, f"a non-JSON body should be an error, got {error!r}")

    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    with mock_http(timeout):
        _, error = run(common.request_json("GET", "https://dvs.test/x", timeout=5))
    c.expect(error and "timed out" in error, f"a timeout should be reported, not raised, got {error!r}")

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    with mock_http(refused):
        _, error = run(common.request_json("GET", "https://dvs.test/x", timeout=5))
    c.expect(error and "connection failed" in error, f"a connection error should be reported, not raised, got {error!r}")

    c.expect(common.age_in_days("nan") is None and common.age_in_days(None) is None,
             "age_in_days should return None for unparseable or missing dates")


def check_dvs(c: Checks):
    with mock_http(respond(ESTIBOT_INSURANCE)) as calls:
        result = run(dvs.get_domain_profile("https://www.Insurance.com/"))
    req = calls[0]
    c.expect(req.url.path == "/estibot/appraisal/lookup" and req.url.params.get("d") == "insurance.com",
             f"profile should look up the cleaned domain, got {req.url}")
    c.expect(req.headers.get("X-System-Actor") == DVS_SYSTEM_ACTOR and req.url.params.get("maxAgeDays") == "30",
             "profile must identify itself to DVS and bound freshness to 30 days")
    data = result["data"] or {}
    c.expect(result["ok"] and data.get("estibot_value") == 36490000 and data.get("words") == ["insurance"]
             and data.get("tld") == ".com" and data.get("sld_registered_tld_count") == 351,
             f"profile fields not normalized as expected: {data}")

    with mock_http(respond({"ok": False, "domain": "zzqx.com"})):
        result = run(dvs.get_domain_profile("zzqx.com"))
    c.expect(not result["ok"], "a lookup with no Estibot data must be ok=False")

    with mock_http(respond(GDV_INSURANCE)) as calls:
        result = run(dvs.get_base_appraisal("insurance.com"))
    data = result["data"] or {}
    c.expect(result["ok"] and data.get("go_value") == 25000 and data.get("sales_probability_at_500") == 1.0,
             f"GoDaddy fields not normalized as expected: {data}")
    c.expect(calls[0].url.params.get("maxAgeDays") == "180" and calls[0].url.params.get("cacheOnly") == "false",
             f"GoDaddy call should default to 180 days and allow live calls, got {calls[0].url}")

    with mock_http(respond(GDV_INSURANCE)) as calls:
        run(dvs.get_base_appraisal("insurance.com", cache_only=True))
    c.expect(calls[0].url.params.get("cacheOnly") == "true", "cache_only=True must be sent as cacheOnly=true")

    with mock_http(respond({"status": "quota_exceeded", "ok": False})):
        result = run(dvs.get_base_appraisal("insurance.com"))
    c.expect(not result["ok"] and "quota" in (result["error"] or ""), "GoDaddy quota exhaustion must be ok=False")

    def score_route(breakdown_override=None):
        def route(request):
            if request.url.path == "/estibot/appraisal/lookup":
                return httpx.Response(200, json=ESTIBOT_INSURANCE)
            body = json.loads(request.content)
            if body.get("domain") != "insurance.com":
                return httpx.Response(400)
            score = dict(INVESTOR_SCORE_AT_FALLBACK)
            if breakdown_override:
                score["breakdown"] = {**score["breakdown"], **breakdown_override}
            return httpx.Response(200, json=score)
        return route

    with mock_http(score_route()) as calls:
        result = run(dvs.get_investor_score("insurance.com"))
    c.expect(result["ok"] and calls[-1].url.path == "/api/investorScore" and calls[-1].method == "POST",
             "investor score should POST the Estibot profile to /api/investorScore")
    c.expect((result["data"] or {}).get("llm_scoring_likely_failed") is True,
             "the real 2026-09-14 breakdown must be flagged as LLM fallback")

    with mock_http(score_route({"buyer_count": 14})):
        result = run(dvs.get_investor_score("insurance.com"))
    c.expect((result["data"] or {}).get("llm_scoring_likely_failed") is False,
             "a breakdown with a real LLM buyer_count must not be flagged")

    with mock_http(respond(DOTDB_INSURANCE)) as calls:
        result = run(dvs.get_keyword_demand("Insurance"))
    data = result["data"] or {}
    c.expect(result["ok"] and data.get("exact_keyword_active_tld_count") == 347
             and calls[0].url.params.get("platformType") == "OTHER" and calls[0].url.params.get("source"),
             f"DotDB demand not normalized, or required source/platformType missing: {data}")

    with mock_http(respond(LEADS_UPSTREAM_ERROR)) as calls:
        result = run(dvs.find_end_user_leads("insurance.com"))
    c.expect(not result["ok"] and "upstream_error" in (result["error"] or "")
             and calls[0].url.path == "/estibot/leads" and calls[0].url.params.get("d") == "insurance.com",
             f"leads failure must be ok=False with DVS's reason, via /estibot/leads?d=, got {result}")

    with mock_http(respond(LEADS_WITH_PII)):
        result = run(dvs.find_end_user_leads("insurance.com"))
    lead = ((result["data"] or {}).get("leads") or [{}])[0]
    leaked = {"whois_reg_name", "whois_reg_email", "whois_reg_phone", "site_emails"} & set(lead)
    c.expect(result["ok"] and not leaked and lead.get("whois_reg_org") == "Quotes Inc",
             f"leads must keep business fields and drop personal ones; leaked: {leaked}")


def check_sales_history(c: Checks):
    with mock_http(respond(namebio_page(NAMEBIO_ROWS))) as calls:
        result = run(sales_history.get_keyword_sales("ai"))
    c.expect(not result["ok"] and not calls, "a 2-letter keyword must be refused without calling the service")

    with mock_http(respond(namebio_page(NAMEBIO_ROWS))) as calls:
        result = run(sales_history.get_keyword_sales("Insurance", tld="com", sld_length=9))
    params = calls[0].url.params
    c.expect(params.get("domainSearch") == "insurance" and params.get("dateTo") and params.get("size") == "500",
             f"search must send the keyword, an explicit dateTo and page size 500, got {calls[0].url}")
    data = result["data"] or {}
    comp_domains = {s["domain"] for s in data.get("comps", [])}
    c.expect(comp_domains == {"insurances.com", "myinsurance.com", "insuranceai.com"}
             and data["comp_filters"] == {"tld": ".com", "sld_length_band": 2},
             f"comps should be .com within ±2 length of 9, got {comp_domains} / {data.get('comp_filters')}")
    c.expect(data["stats_all_tlds"]["count"] == 5 and data["stats_same_tld"]["count"] == 4,
             "stats should cover all TLDs and, separately, the same TLD")

    with mock_http(respond(namebio_page(NAMEBIO_ROWS))):
        result = run(sales_history.get_keyword_sales("insurance", tld=".com", sld_length=3))
    data = result["data"] or {}
    c.expect(data.get("comp_count") == 4 and data["comp_filters"]["sld_length_band"] is None,
             f"with <3 comps in the length band, the band should be dropped and the TLD kept, got {data.get('comp_filters')}")

    def paged(request):
        page = int(request.url.params["page"])
        return httpx.Response(200, json=namebio_page(NAMEBIO_ROWS[:3] if page == 0 else NAMEBIO_ROWS[3:], last=page == 1, total=5))

    with mock_http(paged) as calls:
        result = run(sales_history.get_keyword_sales("insurance"))
    c.expect(len(calls) == 2 and result["data"]["stats_all_tlds"]["count"] == 5 and not result["data"]["truncated"],
             "search should follow pages until the service says last=true")

    def second_page_fails(request):
        if request.url.params["page"] == "0":
            return httpx.Response(200, json=namebio_page(NAMEBIO_ROWS[:3], last=False, total=5))
        return httpx.Response(502)

    with mock_http(second_page_fails):
        result = run(sales_history.get_keyword_sales("insurance"))
    c.expect(result["ok"] and result["data"]["truncated"] and result["data"]["stats_all_tlds"]["count"] == 3,
             "a later page failing should keep the partial results, flagged truncated")


def check_comparables(c: Checks):
    with mock_http(respond(COMPARABLE_FAILED)) as calls:
        result = run(comparables.fetch_similar_sales("OnePay.ai"))
    c.expect(not result["ok"] and "Scoring failed" in (result["error"] or ""),
             f"today's failing search must be ok=False with the service's error, got {result}")
    c.expect(calls[0].method == "POST" and calls[0].url.path == "/api/v1/search"
             and json.loads(calls[0].content) == {"domain": "onepay.ai"},
             "should POST the lowercased domain to /api/v1/search")

    with mock_http(respond(COMPARABLE_WITH_WARNING)):
        result = run(comparables.fetch_similar_sales("onepay.ai"))
    data = result["data"] or {}
    comp = (data.get("comps") or [{}])[0]
    c.expect(result["ok"] and data.get("low_confidence") is True,
             "a response built on placeholder descriptions must be low_confidence")
    c.expect("recency" not in comp and isinstance(comp.get("days_since_sale"), int) and comp.get("similarity") == 0.71,
             f"their constant recency must be dropped for a computed days_since_sale, got {comp}")


def check_pricing(c: Checks):
    c.expect(pricing.price_stats([]) is None, "price_stats of no prices should be None")

    thin = pricing.compute_price_tiers([100, 200])
    c.expect(thin["tiers"] is None and thin["confidence"] == "insufficient", "2 sales must not produce tiers")

    tiers = pricing.compute_price_tiers([1000 * i for i in range(1, 21)])["tiers"]
    c.expect(tiers["floor"] < tiers["auction"] < tiers["end_user"] < tiers["list"] and tiers["auction"] == 10500,
             f"tiers should be ordered with auction at the median, got {tiers}")

    gdv_data = run(_gdv_data())
    points = pricing.compute_price_tiers([], gdv=gdv_data)["sale_probability_points"]
    c.expect([p["price"] for p in points] == [500, 1000, 25000],
             f"GoDaddy probability points should be sorted by price, got {points}")


async def _gdv_data():
    with mock_http(respond(GDV_INSURANCE)):
        return (await dvs.get_base_appraisal("insurance.com"))["data"]


def check_registry(c: Checks):
    names = [t.name for t in VALUATION_TOOLS]
    # Asserted structurally rather than as a count: every phase adds tools,
    # and a magic number here would fail on every one of them while catching
    # nothing that matters.
    c.expect(len(set(names)) == len(names), f"tool names must be unique, got {sorted(names)}")
    required = {"get_domain_profile", "get_base_appraisal", "get_keyword_sales", "fetch_similar_sales",
                "check_spelling_and_typos", "calculate_brand_score", "compute_price_tiers"}
    c.expect(required <= set(names), f"registry is missing {sorted(required - set(names))}")

    by_name = {t.name: t for t in VALUATION_TOOLS}
    c.expect({"keyword", "tld", "sld_length"} <= set(by_name["get_keyword_sales"].args),
             "get_keyword_sales should expose keyword, tld and sld_length")

    with mock_http(respond(ESTIBOT_INSURANCE)):
        wrapped = run(by_name["get_domain_profile"].ainvoke({"domain": "insurance.com"}))
    c.expect(wrapped.get("type") == "evidence" and wrapped.get("source") == "dvs_estibot",
             "service tool output must be wrapped as untrusted evidence")

    typo = by_name["check_spelling_and_typos"].invoke({"domain": "propertymanagment.com"})
    # typo_of names the correctly spelled SIBLING DOMAIN, not just the fixed
    # word, because that is the domain whose comps the deflator is measured
    # against ("propertymanagement.com has high value, this one does not").
    c.expect(typo.get("typo_of") == "propertymanagement.com",
             f"typo tool should name the corrected sibling domain, got {typo}")
    # And the misspelling must NOT have eaten the category: this is still
    # structurally an EMD and must run the EMD evidence plan.
    c.expect(typo.get("category") == "emd",
             f"a misspelling must not replace the structural category, got {typo.get('category')!r}")


def main() -> int:
    c = Checks()
    for check in (check_http_plumbing, check_dvs, check_sales_history, check_comparables, check_pricing, check_registry):
        check(c)

    if c.failures:
        print(f"FAILED {len(c.failures)}/{c.total} checks:")
        for failure in c.failures:
            print(f"  - {failure}")
        return 1

    print(f"PASSED {c.total}/{c.total} checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
