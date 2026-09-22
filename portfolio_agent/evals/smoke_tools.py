"""
Live smoke check of every valuation tool against the real services.

Makes real calls — possibly a GoDaddy credit, Estibot lookups, an
investorScore computation and one Comparable Agent LLM call per run — so run
it by hand, not on every change:

    uv run python -m portfolio_agent.evals.smoke_tools insurance.com

Prints one line per tool: OK with its headline numbers, or the error. Use it
to confirm a service is back, e.g. after the Comparable Agent is fixed.
Exits 1 if any tool failed.
"""

import asyncio
import sys

from portfolio_agent.pricing.tier_engine import check_trademark_conflict
from portfolio_agent.scoring.brand_score import calculate_brand_score
from portfolio_agent.tools import base_appraisal, comparables, dvs, end_users, sales_history, seo, trends
from portfolio_agent.tools.linguistics import classify_domain
from portfolio_agent.tools.pricing import compute_price_tiers


def _summary(name: str, data: dict) -> str:
    if name == "get_domain_profile":
        return f"estibot_value={data['estibot_value']} category={data['category']!r} words={data['words']} source={data['data_source']}"
    if name == "get_base_appraisal":
        return f"go_value={data['go_value']} list={data['list_price']} p(sale)={data['sales_probability']}"
    if name == "get_investor_score":
        return f"{data['total_score']}/100 tier {data['tier']} llm_scoring_likely_failed={data['llm_scoring_likely_failed']}"
    if name == "get_keyword_demand":
        return f"exact keyword active in {data['exact_keyword_active_tld_count']} TLDs, {data['active_names_containing_keyword']} names contain it"
    if name == "find_end_user_leads":
        return f"{data['total']} leads"
    if name == "get_keyword_sales":
        median = (data["stats_all_tlds"] or {}).get("median")
        return f"{data['total_matches']} sales containing '{data['keyword']}', median ${median}, {data['comp_count']} comps after filters"
    if name == "fetch_similar_sales":
        return f"{len(data['comps'])} comps, low_confidence={data['low_confidence']}"
    if name == "scan_market_trends":
        return (f"velocity={data['trend_velocity']} hot={data['hot_category']} risk={data['risk_category']} "
                f"rules={data['matched_rules']} sources={len(data['sources'])}")
    if name == "get_seo_and_traffic_metrics":
        return (f"DR={data['domain_rating']} band={data['authority_band']} "
                f"refdomains={data['refdomains_total']} traffic/mo={data['organic_traffic_monthly']}")
    if name == "get_base_appraisal_facade":
        return f"go_value={data['go_value']} wholesale={data['wholesale']} underprices_premium={data['underprices_premium']}"
    if name == "find_potential_end_users":
        best = (data["leads"] or [{}])[0]
        return (f"{len(data['leads'])} leads, {data['qualified_count']} qualified; "
                f"best: {best.get('company')} ${best.get('funding_total')} on {best.get('current_domain')}")
    return ""


async def main(domain: str) -> int:
    profile = await dvs.get_domain_profile(domain)
    words = (profile["data"] or {}).get("words") or []
    sld, _, tld = domain.lower().partition(".")
    keyword = max(words, key=len) if words else sld

    classification = classify_domain(domain)

    names = [
        "get_base_appraisal",
        "get_investor_score",
        "get_keyword_demand",
        "find_end_user_leads",
        "get_keyword_sales",
        "fetch_similar_sales",
        "get_base_appraisal_facade",
        "scan_market_trends",
        "get_seo_and_traffic_metrics",
        "find_potential_end_users",
    ]
    gathered = await asyncio.gather(
        dvs.get_base_appraisal(domain),
        dvs.get_investor_score(domain),
        dvs.get_keyword_demand(keyword),
        dvs.find_end_user_leads(domain),
        sales_history.get_keyword_sales(keyword, tld=tld, sld_length=len(sld)),
        comparables.fetch_similar_sales(domain),
        base_appraisal.get_base_appraisal(domain),
        trends.scan_market_trends(classification.category, classification.keywords, tld=classification.tld),
        seo.get_seo_and_traffic_metrics(domain),
        end_users.find_potential_end_users(domain, classification.category, classification.keywords),
    )
    results = {"get_domain_profile": profile, **dict(zip(names, gathered))}

    print(f"Smoke check for {domain} (keyword: {keyword!r})\n")
    for name, result in results.items():
        if result["ok"]:
            age = f" [data {result['data_age_days']}d old]" if result["data_age_days"] is not None else ""
            print(f"  OK   {name:22} {_summary(name, result['data'])}{age}")
        else:
            print(f"  ERR  {name:22} {result['error']}")

    # Local, deterministic tools. No service to be down, so these are
    # printed rather than counted in the service tally below.
    print(f"  OK   {'check_spelling_and_typos':22} category={classification.category} "
          f"typo_of={classification.typo_of}")

    brand = calculate_brand_score(domain)
    print(f"  OK   {'calculate_brand_score':22} {brand.total}/100 {brand.components} "
          f"typo_deflator={brand.typo_deflator}")

    trademark = check_trademark_conflict(domain)
    print(f"  OK   {'check_trademark_conflict':22} status={trademark['status']} (stub — never screens)")

    prices = [c["price"] for key in ("get_keyword_sales", "fetch_similar_sales") if results[key]["ok"] for c in results[key]["data"]["comps"]]
    tiers = compute_price_tiers(prices, gdv=results["get_base_appraisal"]["data"])
    print(f"  OK   {'compute_price_tiers':22} {tiers['sample_size']} prices → {tiers['tiers']} ({tiers['confidence']})")

    failed = [name for name, result in results.items() if not result["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} service tools OK" + (f" — failing: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m portfolio_agent.evals.smoke_tools <domain>")
        sys.exit(2)
    sys.exit(asyncio.run(main(sys.argv[1])))
