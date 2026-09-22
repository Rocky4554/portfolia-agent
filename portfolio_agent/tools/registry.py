"""
The valuation tools as LangChain tools, for an agent that chooses its own
tools (e.g. a chat assistant asked "what did pay-something .ai domains sell
for?"). The valuation graph calls the plain functions in each tool module
directly instead, so its evidence gathering stays fixed and reproducible.

Every service result is wrapped with wrap_external_evidence: these services
return third-party text — domain names, LLM-written descriptions — which must
reach a model as data to weigh, never as instructions.
"""

import json

from langchain_core.tools import tool

from portfolio_agent.guardrails import require_tier, wrap_external_evidence
from portfolio_agent.scoring.brand_score import brand_score_detail
from portfolio_agent.tools import comparables, dvs, lander, pricing, sales_history
from portfolio_agent.tools.common import ToolResult
from portfolio_agent.tools.linguistics import classify_domain


def _evidence(result: ToolResult) -> dict:
    return wrap_external_evidence(result["source"], json.dumps(result, default=str))


@tool("get_domain_profile")
async def get_domain_profile_tool(domain: str) -> dict:
    """Estibot's profile of a domain: category, keywords, length, trademark and UDRP risk, keyword CPC and
    search volume, whois dates and registrar, and Estibot's automated value. Call this first — its keywords
    feed get_keyword_sales and get_keyword_demand."""
    return _evidence(await dvs.get_domain_profile(domain))


@tool("get_base_appraisal")
async def get_base_appraisal_tool(domain: str) -> dict:
    """GoDaddy GoValue for a domain: value, wholesale and list price, and sale probabilities at three price
    points. Badly underprices premium domains — use it as one signal, never as the anchor."""
    return _evidence(await dvs.get_base_appraisal(domain))


@tool("get_investor_score")
async def get_investor_score_tool(domain: str) -> dict:
    """A 0-100 investor-quality score with an A-D tier and a 10-factor breakdown. A quality signal, not a
    price. If llm_scoring_likely_failed is true, the score is rule-based only and should be discounted."""
    return _evidence(await dvs.get_investor_score(domain))


@tool("get_keyword_demand")
async def get_keyword_demand_tool(keyword: str) -> dict:
    """How widely a keyword is registered with an active website: how many TLDs host the exact keyword, and
    how many names contain it. A scarcity and demand signal."""
    return _evidence(await dvs.get_keyword_demand(keyword))


@tool("find_end_user_leads")
async def find_end_user_leads_tool(domain: str) -> dict:
    """Businesses using the same term on other domains — potential end-user buyers, with a lead score.
    Business-level fields only."""
    return _evidence(await dvs.find_end_user_leads(domain))


@tool("get_keyword_sales")
async def get_keyword_sales_tool(keyword: str, tld: str | None = None, sld_length: int | None = None) -> dict:
    """Real historical sales (2024 onward) of domains containing a keyword: price stats plus the closest
    comps, optionally narrowed to one TLD (e.g. ".com") and a similar name length. Substring match only, so it
    works for keyword domains but not coined brandables — use fetch_similar_sales for those. Slow (5-15s)."""
    return _evidence(await sales_history.get_keyword_sales(keyword, tld=tld, sld_length=sld_length))


@tool("fetch_similar_sales")
async def fetch_similar_sales_tool(domain: str) -> dict:
    """Up to 10 sold domains most similar in meaning to this one, with price, date and similarity — works for
    brandables too. If low_confidence is true, treat the comps as weak."""
    return _evidence(await comparables.fetch_similar_sales(domain))


@tool("check_domain_status")
async def check_domain_status_tool(domain: str) -> dict:
    """What is served at the domain today, and whether it is offered for sale. A domain running a
    live site is not on the market, and a price for it is hypothetical; a domain on a sale lander
    has a public ASK, which is not a comp."""
    return _evidence(await lander.check_domain_status(domain))


@tool("check_spelling_and_typos")
def check_spelling_and_typos_tool(domain: str) -> dict:
    """Classify a domain's structure and flag a likely misspelling of a common word (e.g.
    propertymanagment.com). A typo domain must not be priced like its correctly spelled sibling."""
    result = classify_domain(domain)
    return {
        "category": result.category,
        "subcategory": result.subcategory,
        "typo_of": result.typo_of,
        "typo_distance": result.typo_distance,
        "notes": result.structural_notes,
    }


@tool("calculate_brand_score")
def calculate_brand_score_tool(domain: str) -> dict:
    """A 0-100 brand-potential score for a domain, with its four components (simplicity, pronounceability,
    memorability, sentiment), their weights, and the reasoning behind each. Also returns a typo_deflator: a
    multiplier below 1.0 means the domain misspells a commercially valuable term and its whole valuation
    should be scaled by it. Deterministic — no network, no model."""
    # Classified here rather than taking a category argument: this is the
    # standalone tool surface, and a caller who has not classified the
    # domain should not have to, nor be able to pass a category that
    # contradicts what the domain actually is.
    return brand_score_detail(domain, category=classify_domain(domain).category)


@tool("compute_price_tiers")
def compute_price_tiers_tool(sale_prices: list[float]) -> dict:
    """Deterministic floor, auction, end-user and list price tiers from comparable sale prices. Needs at least
    3 prices; returns tiers=None with fewer."""
    return pricing.compute_price_tiers(sale_prices)


VALUATION_TOOLS = [
    get_domain_profile_tool,
    get_base_appraisal_tool,
    get_investor_score_tool,
    get_keyword_demand_tool,
    find_end_user_leads_tool,
    get_keyword_sales_tool,
    fetch_similar_sales_tool,
    check_domain_status_tool,
    check_spelling_and_typos_tool,
    calculate_brand_score_tool,
    compute_price_tiers_tool,
]

# Same build-time guard as nodes/graph.py: a deployment that isn't allowed
# these tools fails at import, not mid-conversation.
for _tool in VALUATION_TOOLS:
    require_tier(_tool.name, "read_only")
