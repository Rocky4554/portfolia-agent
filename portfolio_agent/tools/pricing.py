"""
Deterministic price math over sale prices — plain arithmetic, no LLM, so the
same evidence always yields the same numbers. The synthesis step's job is to
explain and adjust these, not to invent them.

The tier mapping below is a v0 starting point, NOT the APVA spec's formulas
(which aren't in this repo). Calibrate it against real sale outcomes before
anyone treats a tier as a recommendation:

    floor     = 25th percentile   quick, wholesale-leaning sale
    auction   = median
    end_user  = 75th percentile
    list      = 90th percentile   buy-it-now asking price
"""

import statistics
from typing import Iterable

# Below this many sales, percentiles are noise rather than a price range.
_MIN_SALES_FOR_TIERS = 3
_HIGH_CONFIDENCE_SALES = 10


def price_stats(prices: Iterable[float]) -> dict | None:
    """Count, spread and volume of a set of sale prices; None when there are none."""
    clean = sorted(float(p) for p in prices if p and p > 0)
    if not clean:
        return None
    if len(clean) >= 2:
        p25, median, p75 = statistics.quantiles(clean, n=4, method="inclusive")
    else:
        p25 = median = p75 = clean[0]
    return {
        "count": len(clean),
        "min": clean[0],
        "p25": round(p25, 2),
        "median": round(median, 2),
        "p75": round(p75, 2),
        "max": clean[-1],
        "total_volume": round(sum(clean), 2),
    }


def compute_price_tiers(sale_prices: Iterable[float], gdv: dict | None = None) -> dict:
    """Price tiers from comparable sale prices, plus GoDaddy's sale-probability points.

    `gdv` is the `data` dict from tools.dvs.get_base_appraisal, if available.
    `tiers` is None when there are too few sales to support a range.
    """
    prices = sorted(float(p) for p in sale_prices if p and p > 0)
    result = {
        "sample_size": len(prices),
        "confidence": _confidence(len(prices)),
        "tiers": None,
        "sale_probability_points": _probability_points(gdv),
    }
    if len(prices) < _MIN_SALES_FOR_TIERS:
        return result

    # 19 cut points at 5% steps: [4]=p25, [9]=p50, [14]=p75, [17]=p90.
    cuts = statistics.quantiles(prices, n=20, method="inclusive")
    result["tiers"] = {
        "floor": round(cuts[4]),
        "auction": round(cuts[9]),
        "end_user": round(cuts[14]),
        "list": round(cuts[17]),
    }
    return result


def _confidence(sample_size: int) -> str:
    if sample_size >= _HIGH_CONFIDENCE_SALES:
        return "high"
    if sample_size >= _MIN_SALES_FOR_TIERS:
        return "medium"
    return "insufficient"


def _probability_points(gdv: dict | None) -> list[dict]:
    """GoDaddy reports sale probability at three prices — the start of a price→probability curve."""
    if not gdv:
        return []
    pairs = [
        (500, gdv.get("sales_probability_at_500")),
        (gdv.get("wholesale_value"), gdv.get("sales_probability_wholesale")),
        (gdv.get("go_value"), gdv.get("sales_probability")),
    ]
    points = [{"price": price, "probability": round(prob, 4)} for price, prob in pairs if price and prob is not None]
    return sorted(points, key=lambda p: p["price"])
