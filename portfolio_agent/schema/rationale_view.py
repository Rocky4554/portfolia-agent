"""
What the rationale model is allowed to see.

Three times now a narrative has been factually wrong while every number in
it was real: it read `emd` as "emerging market domain", it called a GoValue
field a price tier, and it described an unfilled List/BIN tier as
"suggesting a range of $425". The numeric validator passed all three,
because a numeric validator checks figures and those were failures of
*claim*.

The fix is not a better checker. It is to stop handing the model material it
can be wrong about:

  - **An unpriced tier is not described as unpriced — it is absent.** A
    model cannot mischaracterise a tier that is not in its input. The
    reasons live in their own section, as supplied text, so the narrative
    can still say why something was not priced without inventing the why.
  - **No raw field names or enum values.** `emd` becomes "exact match domain
    (EMD)" before the model sees it. A model asked to expand an abbreviation
    will sometimes expand it wrongly; one handed the expansion cannot.
  - **Third-party baselines are labeled as external reference points**, not
    left adjacent to tiers where they look like tiers.
  - **Floats are pre-rounded.** A model shown 0.008949999697506428 writes
    0.0089, which is correct and was being rejected.

The view is also sliced per section, so a tier's section receives only that
tier. That is what makes "do not reference any other tier" enforceable
rather than merely requested.
"""

from dataclasses import dataclass, field

from portfolio_agent.pricing.relations import tier_relations
from portfolio_agent.schema.valuation_output import TIER_ORDER, ValuationOutput

# Every machine name a reader might otherwise meet. One dict, because the
# moment expansions live in two places they disagree.
LABELS: dict[str, str] = {
    # Categories (APVA Table 3.2.1)
    "generic": "generic — a single common word naming its own category",
    "brandable": "brandable — a coined or invented name",
    "keyword": "keyword — contains a commercial term without being an exact search phrase",
    "emd": "exact match domain (EMD) — it exactly matches a commercial search phrase",
    "acronym": "acronym — a short letter string valued for rarity and liquidity",
    "geo": "geo — a location term sold to local businesses and franchises",
    "emerging_tld": "emerging TLD — on a newer, trend-sensitive extension such as .ai or .io",
    "combination": "combination — a brand element fused to a keyword",
    # Tiers (APVA Table 4.1.1)
    "floor": "Floor price — the minimum for a quick wholesale exit",
    "auction": "Auction price — the expected result in a competitive investor auction",
    "end_user": "End-User price — a direct sale to a business that will develop the domain",
    "list_bin": "marketplace Buy-It-Now (BIN) price — a premium asking price",
    "targeted_lead": "Targeted Lead price — the value to one identified, funded buyer",
    "strategic_ma": "Strategic M&A price — the domain as a key asset in an acquisition",
    # Comp bases
    "exact": "priced from direct sales of this keyword on this extension",
    "sld_cross_tld": "priced from sales of this exact name on other extensions, with a transfer ratio applied",
    "cohort": "priced from the cohort of structurally similar domains on this extension",
    "keyword_only": "priced from aggregate keyword statistics — no individual comparable sale was found",
    "category_defining": ("priced from comparable acquisitions of category-defining domains, "
                          "brought down to the level these tiers describe — the keyword comps "
                          "were set aside as a different market"),
    "none": "no comparable sales were found at any level",
}


def label(key: str | None, fallback: str | None = None) -> str:
    if key is None:
        return fallback or ""
    return LABELS.get(str(key), fallback or str(key).replace("_", " "))


@dataclass
class Section:
    """One slice of the view, and the only thing its own LLM call receives."""

    key: str
    heading: str
    instruction: str
    content: dict = field(default_factory=dict)

    def numbers(self) -> set[float]:
        """Every figure this section is permitted to state."""
        found: set[float] = set()
        _collect_numbers(self.content, found)
        return found


@dataclass
class RationaleView:
    domain: str
    sections: list[Section] = field(default_factory=list)
    # Tier labels present anywhere in the view, so a section can be checked
    # for referring to a tier that is not its own.
    tier_labels: dict[str, str] = field(default_factory=dict)

    def section(self, key: str) -> Section | None:
        return next((s for s in self.sections if s.key == key), None)


# Rounding applied before the model sees anything. Probabilities to two
# significant figures, money to whole dollars.
def _round_for_prose(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if 0 < abs(value) < 1:
            return float(f"{value:.2g}")
        return round(value)
    if isinstance(value, dict):
        return {k: _round_for_prose(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_for_prose(v) for v in value]
    return value


def _collect_numbers(node, into: set[float]) -> None:
    if isinstance(node, bool) or node is None:
        return
    if isinstance(node, (int, float)):
        into.add(float(node))
        return
    if isinstance(node, dict):
        for value in node.values():
            _collect_numbers(value, into)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _collect_numbers(value, into)
    elif isinstance(node, str):
        from portfolio_agent.nodes.rationale import extract_numbers

        into.update(extract_numbers(node))


def build_rationale_view(output: ValuationOutput) -> RationaleView:
    """The output, transformed into what the model may see."""
    view = RationaleView(domain=output.domain)
    is_mock = any(item.mock for item in output.evidence)
    provenance = (
        "MOCK FIXTURES — these numbers describe no real market"
        if is_mock
        else "LIVE — real figures from live market services"
    )

    # ── 1. Classification ─────────────────────────────────────────────────
    # The opening section carries the provenance disclosure for the whole
    # narrative, so it is told to — a requirement the model is rejected for
    # missing but never asked to meet is a requirement that just produces
    # templates.
    disclosure = (
        " Begin by stating plainly that these figures come from fixtures and describe no real market."
        if is_mock else
        " These are live market figures; do not describe them as mock, fixture or simulated data."
    )
    view.sections.append(Section(
        key="classification",
        heading="Classification",
        instruction=(
            "Describe what kind of domain this is and why it was classified that way. "
            "Use the category description exactly as given — do not expand, abbreviate or "
            "reinterpret it. Where the domain also reads as another category, say so and say "
            "that evidence was gathered for both, because a domain can belong to more than "
            "one market at once." + disclosure
        ),
        content=_round_for_prose({
            "data_source": provenance,
            "domain": output.domain,
            "category": label(output.category.value),
            "classification_confidence": output.category_confidence,
            "also_reads_as": [
                {"category": label(name.value if hasattr(name, "value") else name),
                 "confidence": conf}
                for name, conf in output.secondary_categories
            ] or None,
            "brand_score_out_of_100": output.brand_score.total if output.brand_score else None,
            "brand_score_components": output.brand_score.components if output.brand_score else None,
            "misspelling_deflator_applied": (
                output.brand_score.typo_deflator
                if output.brand_score and output.brand_score.typo_deflator < 1.0 else None
            ),
        }),
    ))

    # ── 2. Evidence ───────────────────────────────────────────────────────
    view.sections.append(Section(
        key="evidence",
        heading="Evidence",
        instruction=(
            "Describe what evidence was gathered and how the comparable sales were found. "
            "Cite evidence by its id, like (ev3). Do not mention any price tier."
        ),
        content=_round_for_prose({
            "data_source": provenance,
            "how_comparables_were_found": label(output.comp_basis.value),
            "confidence_in_that_basis": output.comp_confidence,
            "how_many_comparable_sales": len(output.comp_set),
            "comparable_sales_price_range": _comp_range(output.comp_set),
            "comparable_sales_shown_below": _comp_sample_label(output.comp_set),
            "comparable_sales_sample": [
                {
                    "domain": c.domain,
                    "sold_for": c.price,
                    "counted_as": c.adjusted_price,
                    "why_adjusted": c.adjustment_reason,
                    "date": c.date,
                    "venue": c.venue,
                }
                for c in _comp_sample(output.comp_set)
            ],
            "tools_that_ran": [
                {"id": item.id, "tool": item.tool, "succeeded": item.ok, "key_data": item.key_datapoints}
                for item in output.evidence
            ],
        }),
    ))

    # ── 3. One section per POPULATED tier ─────────────────────────────────
    # An unpriced tier is absent, not marked. A model cannot mischaracterise
    # what it was never shown.
    for name, tier in output.tiers.ordered_populated():
        view.tier_labels[name] = label(name)
        view.sections.append(Section(
            key=f"tier:{name}",
            heading=label(name),
            instruction=(
                f"Describe this one price tier and the factors that set it. "
                f"Write only about {label(name)}. Do not mention, compare to, or name any other "
                f"price tier — no other tier's figures are available to you and any comparison "
                f"you make would be invented. A later section covers how the tiers relate to one "
                f"another, so the comparison is not missing from the report and does not belong "
                f"here."
            ),
            content=_round_for_prose({
                "data_source": provenance,
                "tier": label(name),
                "price": tier.point,
                "range_low": tier.low,
                "range_high": tier.high,
                "probability_of_sale_at_this_price": tier.sale_probability,
                "factors_that_raised_it": tier.inflationary_factors,
                "factors_that_lowered_it": tier.deflationary_factors,
                "evidence_ids": tier.evidence_refs,
            }),
        ))

    # ── 4. What was not priced, and why — supplied, not guessed ───────────
    not_priced = _not_priced(output)
    if not_priced:
        view.sections.append(Section(
            key="not_priced",
            heading="Not priced",
            instruction=(
                "State which valuations could not be produced and why, using the supplied reasons. "
                "Do not speculate about what these would have been worth — no figure for them exists."
            ),
            content={"data_source": provenance, "not_priced": not_priced},
        ))

    # ── 5. External reference points, clearly not tiers ───────────────────
    references = _third_party(output)
    if references:
        view.sections.append(Section(
            key="reference_points",
            heading="External reference points",
            instruction=(
                "Report these third-party estimates. State plainly that they are external "
                "reference points produced by other services, NOT price tiers from this valuation."
            ),
            content=_round_for_prose({
                "data_source": provenance,
                "third_party_reference_points": references,
                "note": "external reference — not one of this engine's price tiers",
            }),
        ))

    # ── 5b. How the tiers relate to one another ──────────────────────────
    # The one section allowed to name more than one tier. Every other
    # section is scoped to its own to stop cross-tier invention; the cost was
    # a report of six prices with nothing joining them, which is the question
    # a reader actually has. The comparisons are computed, so the narrative
    # is handed a finished relationship rather than asked to judge one.
    relations = tier_relations(output.tiers)
    if len(relations) >= 1:
        view.sections.append(Section(
            key="tier_relations",
            heading="How the tiers relate",
            instruction=(
                "Explain why the prices step up the way they do, using the comparisons given. "
                "Use each comparison's own magnitude word exactly — do not upgrade 'marginally' "
                "to 'far' or downgrade 'dramatically' to 'somewhat' — and never state that a "
                "lower tier exceeds a higher one. Do not restate any tier's range or "
                "probability; only the multiples given here."
            ),
            content=_round_for_prose({
                "data_source": provenance,
                "steps": [
                    {"from": LABELS.get(r.lower, r.lower), "to": LABELS.get(r.upper, r.upper),
                     "from_point": getattr(output.tiers, r.lower).point if getattr(output.tiers, r.lower) else 0,
                     "to_point": getattr(output.tiers, r.upper).point if getattr(output.tiers, r.upper) else 0,
                     "multiple": r.ratio, "magnitude": r.magnitude, "why": r.reason}
                    for r in relations
                ],
                "note": "these multiples are computed from the tier points, not estimated",
            }),
        ))

    # ── 6. Whether the domain can actually be bought ──────────────────────
    # Last because it qualifies everything above it. A reader who takes one
    # thing from the report should take the price; a reader who takes two
    # should know whether the price is reachable.
    if output.acquirability is not None:
        acq = output.acquirability
        view.sections.append(Section(
            key="acquirability",
            heading="Availability",
            instruction=(
                "State what is currently at this domain and whether it can be bought today. "
                "If it is not on the market, say so plainly and say that this valuation "
                "therefore describes a sale its owner has not offered. If an asking price is "
                "given, call it an asking price and never a comparable sale. Do not soften an "
                "unavailable domain into an available one, and do not speculate about the "
                "owner's willingness to sell. State no price except the asking price given "
                "here, and do not name or restate any price tier."
            ),
            content=_round_for_prose({
                "data_source": provenance,
                "status": acq.headline(),
                "on_the_market": acq.on_the_market,
                "asking_price_not_a_sale": acq.asking_price,
                "venue": acq.venue,
                "how_it_was_determined": acq.signals,
                "confidence_in_this_check": acq.confidence,
                "detail": acq.detail,
            }),
        ))

    return view


# How many comps reach the model. The full set can run to fifty; a
# narrative needs a handful.
_COMP_SAMPLE_SIZE = 8


def _comp_sample(comps: list) -> list:
    """A sample SPANNING the distribution, not its head.

    The set arrives price-sorted, so taking the first N handed the model the
    most expensive comps and nothing else. It then described a cohort whose
    median was $118,000 as ranging "$250,000 to $750,000" — accurately, about
    the slice it was shown, and wrongly about the basis it was describing.

    An evenly-spaced sample across the sorted set shows the shape instead of
    the top of it.
    """
    if len(comps) <= _COMP_SAMPLE_SIZE:
        return list(comps)
    ordered = sorted(comps, key=lambda c: c.adjusted_price)
    step = (len(ordered) - 1) / (_COMP_SAMPLE_SIZE - 1)
    return [ordered[round(i * step)] for i in range(_COMP_SAMPLE_SIZE)]


def _comp_sample_label(comps: list) -> str:
    """Says plainly whether the model is seeing all of them or a sample."""
    if len(comps) <= _COMP_SAMPLE_SIZE:
        return f"all {len(comps)} comparable sales"
    return (
        f"a sample of {_COMP_SAMPLE_SIZE} spanning the full range of {len(comps)} comparable sales — "
        f"describe the range and count given above, NOT the sample, as the basis"
    )


def _comp_range(comps: list) -> dict | None:
    if not comps:
        return None
    prices = sorted(c.adjusted_price for c in comps)
    return {
        "lowest": prices[0],
        "median": prices[len(prices) // 2],
        "highest": prices[-1],
    }


def _not_priced(output: ValuationOutput) -> list[dict]:
    """Unfilled tiers with the warning that explains each."""
    rows = []
    for name in TIER_ORDER:
        if getattr(output.tiers, name) is not None:
            continue
        reason = next(
            (w for w in output.warnings if w.startswith(f"{name}:")),
            f"{name}: no reason recorded",
        )
        rows.append({
            "valuation": label(name),
            "reason": reason.split(":", 1)[1].strip() if ":" in reason else reason,
        })
    return rows


_VENDOR_LABELS = {
    "estibot": "Estibot automated appraisal",
    "godaddy_govalue": "GoDaddy GoValue automated appraisal",
}


def _third_party(output: ValuationOutput) -> list[dict]:
    """Vendor appraisals, read from the SAME object the conflict check used.

    Previously assembled independently by scanning evidence, which is how
    the header and the warning came to name different numbers. One source of
    truth means they cannot disagree.
    """
    appraisal = output.independent_appraisal
    if not appraisal:
        return []
    return [
        {
            "source": _VENDOR_LABELS.get(name, name),
            "estimate": value,
            "used_as_the_reference": name == appraisal.source or len(appraisal.all_baselines) == 1,
        }
        for name, value in sorted(appraisal.all_baselines.items())
    ] + ([
        {
            "source": f"reconciled reference ({appraisal.method})",
            "estimate": appraisal.value,
            "used_as_the_reference": True,
        }
    ] if len(appraisal.all_baselines) > 1 else [])
