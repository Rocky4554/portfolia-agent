"""
Deterministic fixtures for every external service, used when
config.MOCK_MODE is on.

Two properties matter more than realism:

  1. Deterministic. Everything is derived from a SHA-256 of the domain, so a
     given domain produces byte-identical evidence on every run, on every
     machine. The tier and scoring evals assert exact numbers; they can only
     do that if the fixtures never move.
  2. Plausibly ordered. The anchor value is derived from the domain's
     category, length and TLD, so cloud.ai comes out expensive and
     zylantix.com comes out cheap. A fixture layer that returned the same
     $10,000 for everything would make every downstream eval vacuous — the
     tier invariants would pass without the tier engine doing anything.

These are fixtures, not a simulator. They are labeled mock=True everywhere
they surface, and nothing here should ever be read as a market estimate.
"""

import hashlib
from datetime import date, datetime, timedelta, timezone

from portfolio_agent.knowledge.lexicon import EMERGING_TLDS, PREMIUM_TLDS
from portfolio_agent.tools.linguistics import classify_domain

# A category's rough order of magnitude, before length and TLD adjustments.
# Ordering follows APVA Table 3.2.1's value drivers: scarcity and trust at
# the top (generic, acronym), residual coined names at the bottom.
_CATEGORY_ANCHOR = {
    "generic": 180_000,
    "acronym": 90_000,
    "emerging_tld": 60_000,
    "emd": 45_000,
    "combination": 28_000,
    "keyword": 22_000,
    "geo": 14_000,
    "brandable": 9_000,
}

_MARKETPLACES = ("GoDaddy", "Sedo", "Afternic", "DropCatch", "NameJet", "Dynadot")

# Fixed so fixtures don't drift as the calendar moves; the evals assert on
# ages derived from it.
_FIXTURE_TODAY = date(2026, 9, 1)


def _seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest(), 16)


def _spread(seed: int, index: int, low: float, high: float) -> float:
    """A stable pseudo-random float in [low, high) for slot `index` of `seed`."""
    bucket = (seed >> (index * 11)) & 0xFFFF
    return low + (high - low) * (bucket / 0xFFFF)


def anchor_value(domain: str) -> int:
    """The mock market's notion of what this domain is worth.

    Everything else in this module is scattered around this number, so the
    fixture set stays internally consistent: comps, appraisals and lead
    budgets all agree on roughly the same domain.
    """
    result = classify_domain(domain)
    base = _CATEGORY_ANCHOR.get(result.category, 10_000)

    # Shorter is scarcer — the inverse-length relationship APVA §4.3 makes
    # explicit for brand scoring.
    length = max(len(result.sld), 1)
    if length <= 3:
        base *= 4.0
    elif length <= 5:
        base *= 2.0
    elif length <= 8:
        base *= 1.0
    elif length <= 12:
        base *= 0.55
    else:
        base *= 0.3

    if result.tld in PREMIUM_TLDS:
        base *= 1.0
    elif result.tld in EMERGING_TLDS:
        base *= 0.8
    else:
        base *= 0.25

    if result.typo_of:
        # The propertymanagment.com case: near-zero liquidity. The fixture
        # reflects the market, not the deflator — the deflator is scoring's
        # job, and double-counting it here would hide bugs in that code.
        base *= 0.04

    if any(ch.isdigit() for ch in result.sld) or "-" in result.sld:
        base *= 0.4

    jitter = _spread(_seed(domain, "anchor"), 0, 0.85, 1.15)
    return max(100, int(base * jitter))


def mock_estibot_profile(domain: str) -> dict:
    result = classify_domain(domain)
    seed = _seed(domain, "estibot")
    value = anchor_value(domain)
    has_keywords = bool(result.keywords)
    return {
        "domain": result.domain,
        "sld": result.sld,
        "tld": f".{result.tld}",
        "category": {"generic": "Business", "emd": "Finance -- Insurance", "geo": "Local Services"}.get(
            result.category, "Technology"
        ),
        "category_root": "Business",
        "words": result.words,
        "num_words": len(result.words),
        "sld_length": len(result.sld),
        "has_numbers": any(ch.isdigit() for ch in result.sld),
        "num_hyphens": result.sld.count("-"),
        "language": "en",
        "is_adult": False,
        "trademark_risk": {"has_trademark": False, "udrp_or_wipo_case": False},
        "estibot_value": value,
        "estibot_value_ignoring_trademark": value,
        "keyword_metrics": {
            "cpc": round(_spread(seed, 1, 0.5, 22.0), 2) if has_keywords else None,
            "competition": round(_spread(seed, 2, 0.1, 1.0), 2) if has_keywords else None,
            "global_monthly_searches": int(_spread(seed, 3, 500, 900_000)) if has_keywords else None,
            "local_monthly_searches": int(_spread(seed, 4, 200, 400_000)) if has_keywords else None,
        },
        "whois": {
            "created": (_FIXTURE_TODAY - timedelta(days=int(_spread(seed, 5, 400, 9000)))).isoformat(),
            "expires": (_FIXTURE_TODAY + timedelta(days=int(_spread(seed, 6, 30, 700)))).isoformat(),
            "registrar": "GoDaddy.com, LLC",
            "private": False,
        },
        "first_archived_year": 2000 + int(_spread(seed, 7, 0, 24)),
        "sld_registered_tld_count": int(_spread(seed, 8, 1, 360)),
        "data_source": "mock",
    }


def mock_godaddy(domain: str) -> dict:
    """GoValue's shape — and its known bias. GoValue models the mass-market
    registrar tail and badly underprices premium names (insurance.com:
    $25,000 against a real $35.6M sale), so the fixture caps it rather than
    tracking the anchor upward. Downstream code that treats GoValue as the
    anchor instead of one signal should fail its eval here, not in
    production."""
    seed = _seed(domain, "godaddy")
    value = max(100, int(anchor_value(domain) * _spread(seed, 0, 0.7, 1.1)))
    value = min(value, 25_000)  # cap last, so the ceiling actually holds
    wholesale = max(50, int(value * _spread(seed, 1, 0.05, 0.2)))
    return {
        "domain": domain,
        "go_value": value,
        "wholesale_value": wholesale,
        "list_price": int(value * 2),
        "min_price": 100,
        "max_price": value,
        # Probability falls as the ask rises — the curve the floor and BIN
        # tiers read their sale_probability off.
        "sales_probability": round(_spread(seed, 2, 0.15, 0.65), 4),
        "sales_probability_at_500": round(_spread(seed, 3, 0.8, 1.0), 4),
        "sales_probability_wholesale": round(_spread(seed, 4, 0.6, 0.95), 4),
        "data_source": "mock",
    }


def mock_investor_score(domain: str) -> dict:
    seed = _seed(domain, "investor")
    total = int(_spread(seed, 0, 20, 92))
    tier = "A" if total >= 75 else "B" if total >= 55 else "C" if total >= 35 else "D"
    return {
        "domain": domain,
        "total_score": total,
        "tier": tier,
        "breakdown": {
            "buyer_count": int(_spread(seed, 1, 0, 16)),
            "industry_momentum": int(_spread(seed, 2, 0, 10)),
            "brandability": int(_spread(seed, 3, 0, 10)),
            "industry_growth": int(_spread(seed, 4, 0, 8)),
            "startup_usage": int(_spread(seed, 5, 0, 8)),
            "acquisition_prob": int(_spread(seed, 6, 0, 8)),
            "scarcity": int(_spread(seed, 7, 0, 8)),
            "linguistic": int(_spread(seed, 8, 0, 6)),
            "radio_test": int(_spread(seed, 9, 0, 6)),
            "domain_age": int(_spread(seed, 10, 0, 6)),
        },
        "llm_scoring_likely_failed": False,
        "from_cache": False,
    }


def mock_keyword_demand(keyword: str) -> dict:
    seed = _seed(keyword, "dotdb")
    names = int(_spread(seed, 0, 200, 250_000))
    return {
        "keyword": keyword,
        "exact_keyword_active_tld_count": int(_spread(seed, 1, 1, 350)),
        "active_names_containing_keyword": names,
        "active_registrations_containing_keyword": int(names * _spread(seed, 2, 1.1, 1.6)),
        "top_matches": [{"name": keyword, "tld_count": int(_spread(seed, 3, 1, 350))}],
    }


def _synthetic_sales(key: str, count: int, anchor: int, sld: str, tld: str) -> list[dict]:
    """A price-descending set of comparable sales scattered widely around
    `anchor` — real comp distributions are wide and right-skewed, and a tight
    scatter would let a broken percentile calculation pass its eval."""
    seed = _seed(key)
    sales = []
    for i in range(count):
        price = max(100, int(anchor * _spread(seed, i, 0.25, 3.2)))
        sold = _FIXTURE_TODAY - timedelta(days=int(_spread(seed, i + 20, 10, 620)))
        prefix = sld[: max(2, len(sld) // 2)] or "comp"
        sales.append(
            {
                "domain": f"{prefix}{'aeiou'[i % 5]}{i}.{tld}",
                "price": float(price),
                "date": sold.isoformat(),
                "marketplace": _MARKETPLACES[i % len(_MARKETPLACES)],
                "tld": f".{tld}",
                "sld_length": max(1, len(sld) + (i % 3) - 1),
            }
        )
    return sorted(sales, key=lambda s: s["price"], reverse=True)


def mock_keyword_sales(keyword: str, tld: str | None = None, sld_length: int | None = None) -> dict:
    """Anchored on the keyword's own exact-match domain, so
    get_keyword_sales("insurance") and a valuation of insurance.com agree
    about what that market is worth."""
    from portfolio_agent.tools.pricing import price_stats

    tld = (tld or ".com").lstrip(".")
    anchor = anchor_value(f"{keyword}.{tld}")
    sales = _synthetic_sales(f"keyword_sales:{keyword}:{tld}", 14, anchor, keyword, tld)
    if sld_length is not None:
        sales = [dict(s, sld_length=sld_length + (i % 3) - 1) for i, s in enumerate(sales)]
    return {
        "keyword": keyword,
        "date_from": "2024-01-01",
        "min_price": 100,
        "total_matches": len(sales),
        "truncated": False,
        "stats_all_tlds": price_stats(s["price"] for s in sales),
        "stats_same_tld": price_stats(s["price"] for s in sales),
        "comp_count": len(sales),
        "comps": sales,
        "comp_filters": {"tld": f".{tld}", "sld_length_band": 2 if sld_length is not None else None},
    }


def mock_similar_sales(domain: str) -> dict:
    result = classify_domain(domain)
    anchor = anchor_value(domain)
    sales = _synthetic_sales(f"similar:{domain}", 10, anchor, result.sld, result.tld)
    comps = [
        {
            "domain": s["domain"],
            "price": s["price"],
            "date": s["date"],
            "platform": s["marketplace"],
            "similarity": round(0.92 - 0.04 * i, 3),
            "category_match": 1.0,
            "days_since_sale": (_FIXTURE_TODAY - date.fromisoformat(s["date"])).days,
        }
        for i, s in enumerate(sales)
    ]
    return {
        "domain": domain,
        "categories": [result.category],
        "descriptions": [f"A {result.subcategory}."],
        "comps": comps,
        "low_confidence": False,
        "warning": None,
    }


def mock_leads(domain: str) -> dict:
    seed = _seed(domain, "leads")
    result = classify_domain(domain)
    count = 3 if result.keywords else 1
    return {
        "domain": domain,
        "total": count,
        "leads": [
            {
                "source_domain": f"{result.sld}{i}.net",
                "type": "domain",
                "lead_score": int(_spread(seed, i, 30, 95)),
                "whois_reg_org": f"Example Holdings {i}",
                "whois_reg_country": "US",
                "site_status": "active",
                "site_has_online_store": bool(i % 2),
                "site_language": "en",
                "whois_age": int(_spread(seed, i + 5, 1, 20)),
            }
            for i in range(count)
        ],
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mock_tld_cohort(tld: str) -> dict:
    """Single dictionary words sold on one extension.

    Deliberately shaped like the real thing: a wide, right-skewed spread
    from a few hundred dollars to the high six figures, because that is what
    a one-word extension cohort actually looks like and a tight cluster
    would let a broken percentile calculation pass.
    """
    from portfolio_agent.tools.pricing import price_stats

    tld = (tld or "com").lstrip(".")
    words = ("wisdom", "cloud", "genesis", "adapt", "orchestra", "rush", "signal",
             "atlas", "beacon", "compass", "harbor", "lantern", "meridian", "summit")
    seed = _seed("tld_cohort", tld)
    sales = []
    for i, word in enumerate(words):
        price = max(500, int(_spread(seed, i, 1_500, 700_000)))
        sold = _FIXTURE_TODAY - timedelta(days=int(_spread(seed, i + 30, 20, 700)))
        sales.append({
            "domain": f"{word}.{tld}",
            "price": float(price),
            "date": sold.isoformat(),
            "marketplace": _MARKETPLACES[i % len(_MARKETPLACES)],
            "tld": f".{tld}",
            "sld_length": len(word),
        })
    sales.sort(key=lambda s: s["price"], reverse=True)
    return {
        "tld": f".{tld}",
        "total_matches": len(sales),
        "truncated": False,
        "comp_count": len(sales),
        "comps": sales,
        "stats": price_stats(s["price"] for s in sales),
    }
