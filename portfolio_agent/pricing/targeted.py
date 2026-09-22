"""
The two conditional tiers of APVA Table 4.1.1 — Targeted Lead and Strategic
M&A.

These differ from the four market tiers in kind, not just in degree. A
market tier answers "what does this class of domain sell for". These answer
"what is it worth to one particular buyer", and that question has no answer
at all unless the buyer exists. So both tiers are conditional, and the
condition is enforced rather than softened:

  - **Targeted Lead** requires at least one qualified lead: funded, in the
    niche, and currently on a weaker domain. Table 4.1.1's deflationary
    factor for this row is literally "no clear, high-value end-user
    identified by the profiling tool" — which is not a discount, it is the
    absence of the tier. No lead, no price, and a warning saying so.
  - **Strategic M&A** requires the domain to be category-defining. Most
    domains are not, and pricing them as if they were is how a valuation
    becomes a fantasy.

Returning None here is the correct and common outcome. A system that always
produces a Strategic M&A price is a system whose Strategic M&A price means
nothing.
"""

from dataclasses import dataclass

from portfolio_agent import config
from portfolio_agent.knowledge.dictionary import frequency_per_million, is_common_noun, is_dictionary_word
from portfolio_agent.knowledge.lexicon import (
    COMMERCIAL_HEADS,
    GENERIC_SINGLE_WORDS,
    PREMIUM_TLDS,
    is_known_word,
    segment,
)
from portfolio_agent.knowledge.ma_cases import DISCLAIMER, comparable_cases, median_price
from portfolio_agent.logging_setup import get_logger, info
from portfolio_agent.pricing.tier_engine import (
    Adjustments,
    TierInputs,
    _prices,
    _tier,
    band_ratios,
    comp_stats,
)
from portfolio_agent.schema.valuation_output import PriceTier

log = get_logger(__name__)


@dataclass
class QualifiedLead:
    company: str
    funding_total: int
    current_domain: str | None
    credibility_gap: float
    # How the figure was arrived at. An incumbent's number is annual
    # revenue from SEC filings, and saying Hartford "has raised
    # $28,368,000,000" is simply false — they never raised it, they earn it.
    funding_basis: str | None = None

    @property
    def funding_phrase(self) -> str:
        return "reports annual revenue of" if self.funding_basis else "has raised"
    announced_on: str | None = None
    categories: tuple[str, ...] = ()


def qualify_leads(leads: list[dict], *, min_funding: int | None = None) -> list[QualifiedLead]:
    """The leads that actually support a targeted price.

    Three conditions, all required. Funding, because the tier is premised on
    a company that can write the cheque. A current domain, because without
    it there is no measurable gap — only a company that exists. And a
    non-zero gap, because a funded company already on an equally strong
    domain has no reason to pay a premium for this one.
    """
    threshold = config.MIN_LEAD_FUNDING if min_funding is None else min_funding
    qualified = []
    for lead in leads or []:
        funding = lead.get("funding_total") or 0
        current = lead.get("current_domain")
        gap = float(lead.get("credibility_gap") or 0.0)
        if funding >= threshold and current and gap > 0:
            qualified.append(
                QualifiedLead(
                    company=str(lead.get("company") or "an unnamed company"),
                    funding_total=int(funding),
                    current_domain=str(current),
                    credibility_gap=gap,
                    announced_on=lead.get("announced_on"),
                    categories=tuple(lead.get("categories") or ()),
                    funding_basis=lead.get("funding_basis"),
                )
            )
    # Best lead first: the tier is priced against the strongest single buyer,
    # not an average of everyone in the market.
    qualified.sort(key=lambda lead: (lead.funding_total * (1 + lead.credibility_gap)), reverse=True)
    return qualified


def funding_multiplier(funding_total: int) -> float:
    """The band a lead's funding puts them in.

    Stepped, not continuous, because the underlying effect is stepped: what
    changes a negotiation is which class of company you are talking to, not
    whether they raised $11M or $12M.
    """
    multiplier = 1.0
    for threshold, value in sorted(config.LEAD_FUNDING_BANDS):
        if funding_total >= threshold:
            multiplier = value
    return multiplier


def targeted_lead_price(
    inputs: TierInputs, end_user: PriceTier | None, leads: list[dict] | None
) -> tuple[PriceTier | None, list[str]]:
    """A price for one identified, funded buyer — or None and a warning."""
    if end_user is None:
        return None, ["targeted_lead: no end-user price to anchor to"]

    qualified = qualify_leads(leads or [])
    info(log, "targeted.leads_qualified", supplied=len(leads or []), qualified=len(qualified),
         threshold=config.MIN_LEAD_FUNDING)
    if not qualified:
        return None, [
            "targeted_lead: no clear, high-value end-user identified by the profiling tool "
            f"(need a company funded at ${config.MIN_LEAD_FUNDING:,}+, in the niche, and currently on a "
            "weaker domain) — tier left unpriced rather than estimated"
        ]

    lead = qualified[0]
    adjustments = Adjustments()

    adjustments.apply(
        funding_multiplier(lead.funding_total),
        f"{lead.company} {lead.funding_phrase} ${lead.funding_total:,} — a premium domain is a "
        f"small fraction of that",
    )

    low, high = config.CREDIBILITY_GAP_RANGE
    adjustments.apply(
        low + (high - low) * lead.credibility_gap,
        f"{lead.company} currently uses {lead.current_domain}; this domain closes a visible "
        f"credibility gap (score {lead.credibility_gap:.2f})",
    )

    if len(qualified) > 1:
        # Competing interest is real leverage, so it earns a multiplier
        # rather than a sentence. Small and capped: two interested parties
        # is not an auction.
        competition = min(config.MULTI_LEAD_MAX_LIFT, 1.0 + 0.04 * (len(qualified) - 1))
        adjustments.apply(
            competition,
            f"{len(qualified)} qualified buyers identified — competing interest strengthens the ask",
        )

    return (
        _tier(
            end_user.point * adjustments.multiplier,
            band_ratios(comp_stats(inputs.comps)),
            adjustments,
            inputs.refs("comps", "end_users", "brand_score"),
        ),
        [],
    )


def is_category_defining(inputs: TierInputs) -> tuple[bool, str]:
    """Whether this domain IS its category, and why or why not.

    The spec's test: a single dictionary word or an exact category term, on
    a premium TLD. Cars.com is the example — not "a good domain in the car
    category" but the domain a buyer would assume belongs to the category
    leader.
    """
    if inputs.tld not in PREMIUM_TLDS:
        return False, f".{inputs.tld} — the strategic tier is a .com phenomenon; no comparable case exists otherwise"

    tokens = segment(inputs.sld)
    known = [t for t in tokens if is_known_word(t)]

    # The spec's test is "a single dictionary word", but a dictionary holds
    # names as well as words. Curated terms pass on sight; everything else
    # has to look like an ordinary word in broad use rather than a name.
    # Without the second test, crusoe.com priced as a category killer on the
    # grounds that "crusoe" names its own category.
    if len(tokens) == 1:
        token = tokens[0]
        if token in GENERIC_SINGLE_WORDS or token in COMMERCIAL_HEADS:
            return True, f"'{inputs.sld}' is a recognized commercial category term"
        if is_common_noun(token):
            return True, (
                f"'{inputs.sld}' is a common noun in broad use "
                f"({frequency_per_million(token):.1f} per million words)"
            )
        if is_dictionary_word(token):
            return False, (
                f"'{inputs.sld}' is in the dictionary but appears only "
                f"{frequency_per_million(token):.2f} times per million words — that is a proper noun "
                "or a rare term, not a category everyone recognizes"
            )

    if inputs.sld.isalpha() and len(inputs.sld) <= 3:
        return True, f"'{inputs.sld}' is a {len(inputs.sld)}-character .com — the scarcest class of domain there is"

    if len(tokens) == 2 and len(known) == 2 and inputs.category in ("emd", "generic"):
        return True, f"'{' '.join(tokens)}' is the exact category term"

    return False, (
        f"'{inputs.sld}' is a {inputs.category} domain, not a category-defining term — "
        "complementary to a buyer's strategy rather than core to it"
    )


def strategic_ma_price(
    inputs: TierInputs, targeted_lead: PriceTier | None
) -> tuple[PriceTier | None, list[str]]:
    """The ceiling: the domain as a key asset in an acquisition.

    Anchored to the targeted lead price, sanity-checked against named
    comparable acquisitions, and published as a deliberately wide band. An
    M&A price is set by the acquirer's strategy rather than by the domain
    market, so a narrow range here would imply a precision that does not
    exist.
    """
    defining, reason = is_category_defining(inputs)
    info(log, "strategic_ma.category_defining", defining=defining, reason=reason)
    if not defining:
        return None, [f"strategic_ma: not a category-defining domain — {reason}"]

    cases = comparable_cases(inputs.sld, inputs.tld)
    adjustments = Adjustments()
    warnings: list[str] = []

    if targeted_lead is None:
        # The tier used to stop here, which made the single most valuable
        # slot in the system depend on holding a leads credential: no key,
        # no targeted lead, no strategic price — on exactly the
        # category-defining .com the tier exists for.
        #
        # It was never true that no anchor existed. We.com to Tencent and
        # Z.com to Nissan are strategic comparables, and neither required
        # identifying the buyer first; the acquisitions are the evidence.
        # So the tier anchors on them directly and says that it did.
        benchmark = median_price(cases) if cases else None
        if not benchmark:
            return None, [
                "strategic_ma: no identified buyer and no comparable acquisitions of "
                "structurally similar domains — nothing to anchor a strategic price on"
            ]
        adjustments.apply(1.0, f"category-defining domain: {reason}")
        point = float(benchmark)
        warnings.append(
            f"strategic_ma: no specific acquirer was identified, so this is anchored directly on "
            f"the median of {len(cases)} comparable acquisitions (${benchmark:,}) rather than on a "
            f"named buyer's funding. The price a strategic acquirer pays is set by their strategy, "
            f"and without knowing whose strategy it is the band is widened rather than the point "
            f"moved — an unidentified buyer makes the sale less likely, not the domain worth less"
        )
        low_ratio, high_ratio = config.STRATEGIC_MA_BAND
        widen = config.STRATEGIC_MA_NO_BUYER_WIDENING
        tier = _tier(point, (low_ratio / widen, high_ratio * widen), adjustments,
                     inputs.refs("comps"))
        citation = "; ".join(f"{case.domain} ${case.price:,} ({case.year})" for case in cases)
        return tier, warnings + [
            f"strategic_ma comparable acquisitions: {citation}",
            f"strategic_ma: a strategic price is set by the acquirer, not by the domain market — "
            f"treat the band as an order of magnitude, not an estimate. {DISCLAIMER}",
        ]

    adjustments.apply(config.STRATEGIC_MA_MULTIPLIER, f"category-defining domain: {reason}")

    point = targeted_lead.point * adjustments.multiplier

    citations: list[str] = []
    if cases:
        citation = "; ".join(f"{case.domain} ${case.price:,} ({case.year})" for case in cases)
        # A citation is evidence, not an adjustment — it moved no price, so
        # it is a warning rather than a factor.
        citations = adjustments.note(f"strategic_ma comparable acquisitions: {citation}")
        benchmark = median_price(cases)
        if benchmark and point > benchmark:
            point = adjustments.cap(
                point, float(benchmark),
                f"capped at the median of those comparable acquisitions (${benchmark:,}) — the "
                f"computed anchor exceeded every named case",
            )

    low_ratio, high_ratio = config.STRATEGIC_MA_BAND
    tier = _tier(point, (low_ratio, high_ratio), adjustments, inputs.refs("comps", "end_users"))

    return tier, citations + [
        f"strategic_ma: a strategic price is set by the acquirer, not by the domain market — treat the "
        f"band as an order of magnitude, not an estimate. {DISCLAIMER}"
    ]


def compute_targeted_tiers(
    inputs: TierInputs, end_user: PriceTier | None, leads: list[dict] | None
) -> tuple[PriceTier | None, PriceTier | None, list[str]]:
    """Both conditional tiers, with every warning they raised."""
    targeted, warnings = targeted_lead_price(inputs, end_user, leads)
    strategic, strategic_warnings = strategic_ma_price(inputs, targeted)
    return targeted, strategic, warnings + strategic_warnings
