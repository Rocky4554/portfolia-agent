"""
Similar past sales from the Comparable AI Agent (namekart/Comparable_AI_Agent).

That agent has an LLM describe the domain, then vector-searches sold domains
with similar descriptions — so unlike tools/sales_history.py it finds comps
for coined brandables too, which have no keyword to substring-match.

Known issues in that service (2026-09-14), handled here instead of trusted:
  - Every search currently fails ("Scoring failed: 'NoneType' object is not
    iterable"). This tool returns ok=False until the service is fixed; no
    change needed here afterwards.
  - Its recency score is a constant (a parameter-name typo in its scoring),
    so it's dropped and days_since_sale is computed here from the sale date.
  - If its own LLM call fails it silently matches on placeholder text and
    only says so in `warning` — surfaced here as low_confidence.
  - Its corpus was seeded with sales of $5,000+ only, so comps skew high for
    inexpensive domains.
"""

from portfolio_agent import config
from portfolio_agent.config import COMPARABLE_AGENT_BASE_URL
from portfolio_agent.tools import mocks
from portfolio_agent.logging_setup import get_logger, warn
from portfolio_agent.tools.common import ToolResult, age_in_days, error_result, mock_result, ok_result, request_json

log = get_logger(__name__)

SOURCE = "comparable_agent"

_TIMEOUT_S = 120.0  # one LLM call + embeddings + a vector search on their side


async def fetch_similar_sales(domain: str) -> ToolResult:
    """Up to 10 sold domains most similar to `domain`, with price, date and similarity."""
    domain = (domain or "").strip().lower()
    if not domain:
        return error_result(SOURCE, "domain is empty")
    if config.MOCK_MODE:
        return mock_result(SOURCE, mocks.mock_similar_sales(domain), data_age_days=24)

    body, error = await request_json(
        "POST", f"{COMPARABLE_AGENT_BASE_URL}/api/v1/search", json={"domain": domain}, timeout=_TIMEOUT_S
    )
    if error:
        return error_result(SOURCE, error)
    if not isinstance(body, dict):
        return error_result(SOURCE, "unexpected response shape")
    if not body.get("success"):
        return error_result(SOURCE, f"search failed: {body.get('error') or 'no error message returned'}")

    # Every field is type-checked before use, not merely defaulted. `or {}`
    # protects against null and against nothing else: a `data` that arrives
    # as a list, or a `comparables` that arrives as an object, sails through
    # it and raises several frames later, where the traceback names this
    # tool but not the shape that broke it. A malformed response is an
    # upstream fault and must degrade to a warning, never take the tool down
    # -- this one failed hard in production and cost the valuation its
    # semantic cross-check, which was the signal that would have tempered an
    # over-anchored comp set.
    data = body.get("data")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return error_result(SOURCE, f"'data' arrived as {type(data).__name__}, expected an object")

    raw_comps = data.get("comparables") or []
    if not isinstance(raw_comps, list):
        return error_result(
            SOURCE, f"'comparables' arrived as {type(raw_comps).__name__}, expected a list")

    # One malformed row must not discard the rows either side of it.
    comps, skipped = [], 0
    for row in raw_comps:
        if not isinstance(row, dict):
            skipped += 1
            continue
        try:
            comps.append(_to_comp(row))
        except (TypeError, ValueError):
            skipped += 1
    if skipped:
        warn(log, "comparables.rows_skipped", domain=domain, skipped=skipped, kept=len(comps))

    warning = data.get("warning")
    if skipped:
        warning = ((str(warning) + "; ") if warning else "") + (
            f"{skipped} comparable "
            f"{'rows were' if skipped > 1 else 'row was'} malformed and skipped")

    descriptions = data.get("descriptions")
    if not isinstance(descriptions, list):
        descriptions = []

    dated = [c for c in comps if c.get("days_since_sale") is not None]
    newest = min(dated, key=lambda c: c["days_since_sale"])["date"] if dated else None

    return ok_result(
        SOURCE,
        {
            "domain": domain,
            "categories": [c for c in (data.get("primary_category"), data.get("secondary_category")) if c],
            "descriptions": descriptions,
            "comps": comps,
            # Their own "confidence" only means "found 5+ comps"; placeholder
            # descriptions make any number of comps unreliable.
            "low_confidence": bool(warning) or data.get("confidence") != "high",
            "warning": warning,
        },
        data_age_days=age_in_days(newest),
    )


def _to_comp(comp: dict) -> dict:
    """One upstream row → our shape. Raises on a row it cannot read, so the
    caller can skip that row rather than lose the whole response."""
    return {
        "domain": comp.get("domain"),
        "price": float(comp.get("price") or 0),
        "date": comp.get("date"),
        "platform": comp.get("platform"),
        "similarity": comp.get("semantic_sim"),
        "category_match": comp.get("cat_match"),
        "days_since_sale": age_in_days(comp.get("date")),
    }
