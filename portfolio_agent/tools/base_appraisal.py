"""
get_base_appraisal — APVA Table 7.1.1's GoDaddy GoValue integration.

A thin facade, not a second implementation. The real call already lives in
tools/dvs.py, behind our own cache and rate limiter, and calling GoDaddy
directly from here would spend credits DVS has already paid for and bypass
the freshness bounds config.py sets deliberately. This module exists so the
tool has the name and the shape the spec names it by, and so a deployment
without DVS still gets a baseline appraisal.

The one thing worth restating every time this number is used: GoValue models
the mass-market registrar tail and badly underprices premium names.
insurance.com appraises at $25,000 against a real $35.6M sale. It is one
signal among several and must never be the anchor — see
pricing/tier_engine.py, which reads it for the sale-probability curve and
for a floor, not for a price.
"""

from portfolio_agent import config
from portfolio_agent.tools import dvs, mocks
from portfolio_agent.tools.common import ToolResult, error_result, mock_result

SOURCE = "base_appraisal"

# Above this, GoValue is a wholesale-tail estimate rather than a market
# price, and the tier engine should say so out loud.
UNDERPRICING_SUSPECT_ABOVE = 25_000


async def get_base_appraisal(domain: str, *, cache_only: bool = False) -> ToolResult:
    """`{go_value, wholesale, sales_probability, ...}` plus an explicit
    `underprices_premium` flag when the value has hit GoValue's ceiling."""
    domain = (domain or "").strip().lower()
    if not domain:
        return error_result(SOURCE, "domain is empty")

    if config.MOCK_MODE:
        return mock_result(SOURCE, _annotate(mocks.mock_godaddy(domain)))

    result = await dvs.get_base_appraisal(domain, cache_only=cache_only)
    if not result["ok"] or not result["data"]:
        return error_result(SOURCE, result["error"] or "no baseline appraisal available")
    return {**result, "source": SOURCE, "data": _annotate(result["data"])}


def _annotate(data: dict) -> dict:
    go_value = data.get("go_value")
    return {
        **data,
        "wholesale": data.get("wholesale_value"),
        "underprices_premium": bool(go_value and go_value >= UNDERPRICING_SUSPECT_ABOVE),
    }
