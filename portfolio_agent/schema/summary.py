"""
The one-screen digest of a valuation.

The full report is deliberately long: every adjustment states its multiplier,
every unfilled tier explains itself, and every missing tool says what it
would have contributed. That is right for someone auditing a number and
wrong for someone scanning twenty domains.

So this is the same object, compressed — and compressed rather than
simplified. Nothing here asserts anything the full output does not, the
warnings keep their teeth in short form, and a tier that is not priced stays
null rather than becoming a zero. A digest that rounded a caveat away would
be worse than no digest, because it would be the version people actually
read.
"""

from portfolio_agent.schema.valuation_output import TIER_ORDER, ValuationOutput

# Long warnings condensed to the fact they carry. Matched on a distinctive
# fragment of the full text; anything unmatched is truncated rather than
# dropped, so a new warning still appears.
_SHORTHAND = (
    ("no famous mark appears", "No trademark search run — famous-mark screen only"),
    ("trademark risk", "Famous-mark conflict detected — clear before relying on end-user tier"),
    ("comp basis 'cohort'", "No exact comp — priced from a structural cohort"),
    ("comp basis 'keyword_only'", "No individual sale found — priced from keyword aggregates"),
    ("category-defining domain:", "Anchored on comparable acquisitions, discounted to market level"),
    ("appraisal conflict", "Vendor appraisal and comps disagree by an order of magnitude"),
    ("only 1 vendor baseline", "Single vendor baseline — no consensus to check it against"),
    ("no vendor baseline", "No independent appraisal available"),
    ("median comparable sale is", "Comps predate this extension's evidence half-life"),
    ("every comp whose industry", "Comps sell into a different market — shape comparison only"),
    ("acquirability: a working site", "Domain is in use and not for sale"),
    ("acquirability: the domain is parked", "Parked — acquisition means approaching the owner"),
    ("acquirability: listed at", "Listed for sale — ask is not a sale price"),
    ("acquirability: this domain is not registered", "Unregistered — available at registration cost"),
    ("acquirability: could not be checked", "Availability unverified"),
    ("targeted_lead: no clear", "No qualified end-user buyer identified"),
    ("strategic_ma: not a category-defining", "Not category-defining — no strategic tier"),
    ("strategic_ma: no specific acquirer", "No named acquirer — strategic band widened, not discounted"),
    ("misspelling of", "Misspelling — every tier deflated"),
    ("unavailable (no ", "A data source was unavailable"),
)


def summarize(output: ValuationOutput) -> dict:
    """A compact dict of the whole valuation, safe to print or serialise."""
    signals = output.signals
    acq = output.acquirability
    brand = output.brand_score

    return {
        "domain": output.domain,
        "classification": {
            "category": output.category.value,
            "also_reads_as": [name.value if hasattr(name, "value") else str(name)
                              for name, _ in output.secondary_categories] or None,
            "confidence": round(output.category_confidence, 2),
            "tld": "." + output.domain.rsplit(".", 1)[-1],
            "buyer_intent": getattr(signals, "intent", None),
        },
        "brand_score": {
            "total": brand.total if brand else None,
            "weighted_for": brand.weighted_for if brand else None,
            "components": brand.components if brand else None,
            "typo_deflator": brand.typo_deflator if brand else None,
        } if brand else None,
        "comp_basis": {
            "basis": output.comp_basis.value,
            "exact_sale_found": output.comp_basis.value == "exact",
            "comps_used": len(output.comp_set),
            "confidence": round(output.comp_confidence, 2),
            "price_range": _comp_range(output),
        },
        # Null, never zero. An unpriced tier is a refusal to guess, and a
        # zero would read as a valuation of nothing.
        "valuation": {
            name: _tier(getattr(output.tiers, name)) for name in TIER_ORDER
        },
        "market_signals": {
            "listed_for_sale": acq.status == "listed_for_sale" if acq else None,
            "on_the_market": acq.on_the_market if acq else None,
            "public_ask": acq.asking_price if acq else None,
            "status": acq.status if acq else None,
            "registered_years": acq.age_years if acq else None,
            "dropping": acq.dropping if acq else False,
        },
        "independent_appraisal": {
            "value": output.independent_appraisal.value,
            "source": output.independent_appraisal.source,
            "baselines": output.independent_appraisal.all_baselines,
        } if output.independent_appraisal else None,
        "warnings": _short_warnings(output.warnings),
        "warning_count": len(output.warnings),
    }


def _tier(tier) -> dict | None:
    if tier is None:
        return None
    out = {"range": f"{tier.low}-{tier.high}", "midpoint": tier.point}
    if tier.sale_probability is not None:
        out["sale_probability"] = round(tier.sale_probability, 3)
    return out


def _comp_range(output: ValuationOutput) -> str | None:
    prices = sorted(c.price for c in output.comp_set if c.price > 0)
    return f"{prices[0]}-{prices[-1]}" if prices else None


def _short_warnings(warnings: list[str]) -> list[str]:
    """One line each, deduplicated, order preserved.

    Truncation rather than omission for anything unmatched: a warning this
    file has not seen before is exactly the one worth not losing.
    """
    seen, out = set(), []
    for warning in warnings:
        short = next((text for fragment, text in _SHORTHAND if fragment in warning), None)
        if short is None:
            short = warning.split(" — ")[0].split(". ")[0].strip()
            if len(short) > 110:
                short = short[:107].rstrip() + "..."
        if short not in seen:
            seen.add(short)
            out.append(short)
    return out
