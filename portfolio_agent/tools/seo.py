"""
get_seo_and_traffic_metrics — APVA Table 7.1.1's Ahrefs integration.

Mock-first, real key optional. The interface is Ahrefs-shaped (domain
rating, backlinks, referring domains, organic traffic) because that is what
the spec names and because SEMrush's alternative fields map onto the same
four numbers.

Worth being clear about what this measures, since it is easy to misread: for
most domains being valued these numbers are near zero, and that is not a
negative signal. An undeveloped domain has no backlinks because nobody has
built on it yet. SEO metrics only carry weight for an EMD or keyword domain
with an existing history — which is exactly the two categories the playbook
routes this tool to.
"""

from portfolio_agent import config
from portfolio_agent.tools.common import ToolResult, error_result, mock_result, ok_result, request_json
from portfolio_agent.tools.linguistics import split_domain

SOURCE = "seo_metrics"

_TIMEOUT_S = 25.0

# Domain Rating is 0-100 and logarithmic. These are the bands domain buyers
# actually reason in: below 10 is "undeveloped", above 40 is "an asset with
# a history someone will pay for".
_DR_UNDEVELOPED = 10
_DR_ESTABLISHED = 40


async def get_seo_and_traffic_metrics(domain: str) -> ToolResult:
    """SEO authority and traffic for a domain.

    Returns `{domain_rating, backlinks_total, refdomains_total,
    organic_traffic_monthly, authority_band}`. The band is computed here so
    every consumer reads the same thresholds.
    """
    domain = (domain or "").strip().lower()
    if not domain:
        return error_result(SOURCE, "domain is empty")

    if config.MOCK_MODE:
        return mock_result(SOURCE, _mock_metrics(domain))

    if not config.AHREFS_API_KEY:
        # An absent key is not a failure — it means this signal is simply
        # unavailable, and the tier engine must price without it rather than
        # substitute a number.
        return error_result(SOURCE, "no AHREFS_API_KEY configured — SEO signal unavailable")

    body, error = await request_json(
        "GET",
        f"{config.AHREFS_BASE_URL}/v3/site-explorer/metrics",
        params={"target": domain, "mode": "domain"},
        headers={"Authorization": f"Bearer {config.AHREFS_API_KEY}", "Accept": "application/json"},
        timeout=_TIMEOUT_S,
    )
    if error:
        return error_result(SOURCE, error)
    metrics = (body or {}).get("metrics") if isinstance(body, dict) else None
    if not isinstance(metrics, dict):
        return error_result(SOURCE, "unexpected response shape")

    return ok_result(SOURCE, _normalize(domain, metrics))


def _normalize(domain: str, metrics: dict) -> dict:
    rating = _number(metrics.get("domain_rating"))
    return {
        "domain": domain,
        "domain_rating": rating,
        "backlinks_total": _number(metrics.get("backlinks")),
        "refdomains_total": _number(metrics.get("refdomains")),
        "organic_traffic_monthly": _number(metrics.get("org_traffic")),
        "organic_keywords": _number(metrics.get("org_keywords")),
        "authority_band": authority_band(rating),
    }


def authority_band(domain_rating: float | None) -> str:
    """`unknown` when there is no rating — never `undeveloped`, because
    "we did not measure" and "we measured zero" price differently."""
    if domain_rating is None:
        return "unknown"
    if domain_rating < _DR_UNDEVELOPED:
        return "undeveloped"
    if domain_rating < _DR_ESTABLISHED:
        return "developing"
    return "established"


def _number(value):
    return value if isinstance(value, (int, float)) else None


def _mock_metrics(domain: str) -> dict:
    """Most domains at valuation time are undeveloped, so the fixture makes
    that the common case rather than handing every domain a flattering SEO
    profile the tier engine would then inflate on."""
    from portfolio_agent.tools.mocks import _seed, _spread

    sld, _ = split_domain(domain)
    seed = _seed(domain, "seo")
    developed = (seed % 5) == 0  # one domain in five has a history
    rating = round(_spread(seed, 0, 12, 68), 1) if developed else round(_spread(seed, 1, 0, 8), 1)
    refdomains = int(_spread(seed, 2, 50, 9000)) if developed else int(_spread(seed, 3, 0, 25))
    return {
        "domain": domain,
        "domain_rating": rating,
        "backlinks_total": refdomains * int(_spread(seed, 4, 3, 40)),
        "refdomains_total": refdomains,
        "organic_traffic_monthly": int(_spread(seed, 5, 100, 120_000)) if developed else 0,
        "organic_keywords": int(_spread(seed, 6, 10, 9000)) if developed else 0,
        "authority_band": authority_band(rating),
    }
