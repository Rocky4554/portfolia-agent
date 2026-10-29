"""
The four market tiers of APVA Table 4.1.1 — Floor, Auction, End-User and
List/BIN — as pure functions over gathered evidence.

Each tier is anchored to the one below it, exactly as the table specifies,
so the chain runs bottom-up from a single comps-derived anchor:

    comps → auction → floor
                   └→ end_user → list_bin

Three rules hold throughout:

**Nothing is invented.** Every tier traces to comparable sales. Where there
are too few comps to form a distribution, the tier is None and a warning
says so. A tier computed from two sales is not a price, it is an anecdote
wearing a decimal point.

**Every adjustment is recorded.** A multiplier that fires writes itself into
`inflationary_factors` or `deflationary_factors` with the evidence ids
behind it. A tier that moved without saying why is a bug — the rationale
generator can only cite what these lists contain.

**Order is enforced, not assumed.** Adjustments are multiplicative and could
in principle push end-user below auction. The final pass clamps any
inversion and records that it did, because a valuation that tells a seller
their auction price exceeds their end-user price is worse than useless.
"""

import statistics
from dataclasses import dataclass, field
from datetime import date

from portfolio_agent import config
from portfolio_agent.knowledge.category_floors import get_floor
from portfolio_agent.knowledge.intent import classify_intent, volume_credit
from portfolio_agent.knowledge.lexicon import EMERGING_TLDS, PREMIUM_TLDS
from portfolio_agent.knowledge.tld_tier import get_tld_multiplier
from portfolio_agent.logging_setup import debug, get_logger, info, warn
from portfolio_agent.schema.valuation_output import (
    BrandScore,
    IndependentAppraisal,
    PriceTier,
    TierSet,
)

log = get_logger(__name__)


@dataclass
class TierInputs:
    """Everything the tier math reads. Assembled by nodes/synthesize.py from
    tool results; built directly by the evals from fixtures."""

    domain: str
    sld: str
    tld: str
    category: str
    comps: list[dict] = field(default_factory=list)
    brand_score: BrandScore | None = None
    trend: dict | None = None
    seo: dict | None = None
    base_appraisal: dict | None = None
    # An independent automated appraisal (Estibot, via get_domain_profile).
    # Used ONLY as a cross-check — never as an anchor, and it moves no
    # price. See appraisal_disagreement().
    independent_appraisal: IndependentAppraisal | None = None
    # Comparable ACQUISITIONS of structurally identical domains, supplied
    # only when this domain is category-defining. When present and numerous
    # enough these REPLACE the keyword comps as the anchor — see
    # anchor_comps().
    category_comps: list[dict] = field(default_factory=list)
    # Live ASKING prices. Deliberately a separate field from `comps`, and
    # nothing in this module merges the two: a comp is what a buyer paid, a
    # listing is what a seller wants, and averaging them would let one
    # seller's hope move an anchor built from transactions.
    listings: dict | None = None
    # How the comps were found, and how much that is worth. A weaker basis
    # does not merely lower a price — it widens the range and lowers the
    # stated probability of clearing, because the claim itself is broader.
    # Own-book demand: inquiries and the best offer received. NOT comps —
    # an offer is a price nobody paid.
    demand: dict | None = None
    comp_basis: str = "exact"
    comp_confidence: float = 1.0
    keyword_stats: dict | None = None  # merged keyword_metrics + demand
    # Length, word count, hyphens, digits. Supplied for EVERY category —
    # brand scoring is brandable-specific, structure is not.
    structural: dict | None = None
    phonetic_score: float | None = None
    sentiment_score: float | None = None
    # {intent, reasons, source} from the intent tool. None means the
    # lexicon is consulted directly, which is the offline path.
    intent: dict | None = None
    typo_deflator: float = 1.0
    acquisition_cost: int | None = None
    # Evidence ids per signal, so each tier cites only what it actually used.
    evidence_refs: dict[str, list[str]] = field(default_factory=dict)
    today: date | None = None

    def refs(self, *keys: str) -> list[str]:
        seen: list[str] = []
        for key in keys:
            for ref in self.evidence_refs.get(key, []):
                if ref not in seen:
                    seen.append(ref)
        return seen


class Adjustments:
    """A running multiplier that remembers why it is what it is."""

    def __init__(self) -> None:
        self.multiplier = 1.0
        self.inflationary: list[str] = []
        self.deflationary: list[str] = []

    def apply(self, factor: float, reason: str) -> None:
        """Record a multiplier and the reason it fired.

        A factor of exactly 1.0 is dropped: "this did not change the price"
        is not a reason, and listing it would bury the adjustments that did.
        """
        if factor is None or abs(factor - 1.0) < 1e-9:
            return
        self.multiplier *= factor
        entry = f"{reason} (×{factor:.2f})"
        (self.inflationary if factor > 1.0 else self.deflationary).append(entry)

    def cap(self, before: float, after: float, reason: str) -> float:
        """Record a ceiling as the multiplier it effectively is.

        A cap changes the price as surely as a multiplier does, and stating
        it without one leaves a reader unable to see how far the price moved
        — the same recorded-no-op problem in a different shape. Returns the
        capped value so the caller reads as an assignment.
        """
        if before <= 0 or after >= before:
            return before
        self.apply(after / before, reason)
        return after

    def note(self, observation: str) -> list[str]:
        """Something true that changed no price.

        Returns it for the caller to put in `warnings`. It is deliberately
        NOT appendable to a factor list — the schema rejects a factor with
        no multiplier, and this is the supported way to say something that
        did not move the number.
        """
        return [observation]


# ── Signals ────────────────────────────────────────────────────────────────


def _prices(comps: list[dict]) -> list[float]:
    return sorted(float(c["price"]) for c in comps if c.get("price") and float(c["price"]) > 0)


def recency_weight(age_days: int | None, half_life_days: int) -> float:
    """How much a sale of this age still counts, on this extension's clock.

    Exponential decay with a floor, not a cliff: a sale does not stop being
    evidence on a birthday, it stops being evidence gradually. The floor
    matters as much as the curve — an old sale is weaker evidence, never
    zero evidence, and dropping it entirely would silently shrink thin comp
    sets into nothing.

    Runs in both directions by construction. Where an extension is falling,
    old comps overstate today's value and get quieter; where it is rising,
    they understate it and the recent ones dominate. One mechanism.
    """
    if age_days is None or age_days <= 0 or half_life_days <= 0:
        return 1.0
    decayed = 0.5 ** (age_days / half_life_days)
    return max(config.MIN_RECENCY_WEIGHT, decayed)


def _pairs(comps: list[dict]) -> list[tuple[float, float]]:
    """(price, weight) for every usable comp, weight defaulting to 1.

    The weights come from the comp ladder, which sets them with some care:
    an exact comp counts double, an own-book sale one and a half, and a
    cross-band comp 0.15 because it was retrieved but describes a different
    quality of name. Until this function existed, every consumer read the
    price and dropped the weight, so none of that reached a price.
    """
    out = []
    for comp in comps:
        price = comp.get("price")
        if price and float(price) > 0:
            weight = comp.get("weight")
            weight = float(weight) if isinstance(weight, (int, float)) and weight > 0 else 1.0
            # Age was read only by liquidity_score until now, so a 2021 sale
            # counted exactly as much as a 2026 one in the distribution it
            # was pricing.
            half_life = comp.get("comp_half_life_days")
            if isinstance(half_life, (int, float)) and half_life > 0:
                weight *= recency_weight(comp.get("age_days"), int(half_life))
            out.append((float(price), weight))
    return sorted(out)


def weighted_percentile(pairs: list[tuple[float, float]], q: float) -> float:
    """The q-th percentile of a weighted distribution.

    A real weighted percentile rather than the repeat-by-weight trick the
    comp ladder used, because that trick can only ever UP-weight: rounding
    0.15 to an integer and taking `max(1, ...)` gives a cross-band comp the
    same single vote as a perfect one. Down-weighting is the half that
    matters, so the arithmetic has to carry fractional weights.
    """
    if not pairs:
        raise ValueError("no comps to take a percentile of")
    ordered = sorted(pairs)
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        raise ValueError("comp weights sum to zero")
    target = q * total
    cumulative = 0.0
    for value, weight in ordered:
        # Interpolating at the midpoint of each comp's weight band keeps the
        # median of an evenly weighted set equal to the ordinary median, so
        # turning weights on does not move a set that has no weighting.
        midpoint = cumulative + weight / 2.0
        if midpoint >= target:
            if cumulative <= 0:
                return value
            previous_value, previous_weight = _preceding(ordered, value, cumulative)
            span = midpoint - (cumulative - previous_weight / 2.0)
            if span <= 0:
                return value
            fraction = (target - (cumulative - previous_weight / 2.0)) / span
            return previous_value + (value - previous_value) * max(0.0, min(1.0, fraction))
        cumulative += weight
    return ordered[-1][0]


def _as_pairs(stats: "CompStats") -> list[tuple[float, float]]:
    """Only reached when BAND_PERCENTILES is retuned away from p10/p90; the
    common path reads the precomputed fields."""
    return [(stats.p10, 1.0), (stats.median, 1.0), (stats.p75, 1.0), (stats.p90, 1.0)]


def _preceding(ordered, value, cumulative):
    running = 0.0
    previous = ordered[0]
    for pair in ordered:
        if running >= cumulative:
            break
        previous = pair
        running += pair[1]
    return previous


@dataclass
class CompStats:
    """The distribution every tier reads, weighted and trophy-capped once.

    One object so the anchor and its own band are taken from the same
    numbers, which is how a point price and its range come to disagree.

    No outlier trimming here, and that is a measured decision rather than an
    omission. Winsorizing against a multiple of the median was written,
    tested and removed: percentile anchoring already excludes the tail (a
    $2,000,000 comp and a $70,000 one produce identical tiers), and the
    ceiling would have been computed from a median the outliers themselves
    inflate. TIER_BAND_LIMITS clamps the high end regardless. A cap on top
    of all three changed no number in any set size tested.
    """

    median: float
    p10: float
    p75: float
    p90: float
    count: int


def comp_stats(comps: list[dict]) -> CompStats | None:
    pairs = _pairs(comps)
    if not pairs:
        return None
    return CompStats(
        median=weighted_percentile(pairs, 0.5),
        p10=weighted_percentile(pairs, 0.10),
        p75=weighted_percentile(pairs, 0.75),
        p90=weighted_percentile(pairs, 0.90),
        count=len(pairs),
    )


def anchor_comps(inputs: TierInputs) -> tuple[list[dict], str | None]:
    """The comparable sales the tiers are built from, and why those ones.

    For almost every domain this is simply the keyword and semantic comps.
    For a category-defining domain it is not, and the difference is the
    whole point: a substring search for `insurance` returns
    autoinsurance.net and insurancer.com — real sales, in a different
    market. insurance.com is not a domain containing the word "insurance";
    it is the word. Pricing it against names that merely contain it produced
    $9,120 against a documented $35.6M sale.

    So a category killer is re-anchored on comparable ACQUISITIONS of
    structurally identical domains. The keyword comps are kept in the
    returned note, because a reader should be able to see both.

    **And the acquisitions are brought down a rung before the market tiers
    read them.** This function used to hand them over untouched, on the
    reasoning that no multiplier should be invented — but leaving the level
    alone was itself a claim, and a wrong one. These are completed
    acquisitions, in which the domain became a company's identity; the four
    market tiers price a sale between an investor and a buyer. Taking p75 of
    a set of strategic outcomes and publishing it as an end-user price
    applies the end-user premium twice, which is how hello.com reached
    $18.1M against a set whose p75 is $18.0M.

    The divisor is the engine's own chain read backwards rather than a
    haircut chosen to taste, and the strategic tier keeps the undiscounted
    cases, because for that tier they are already the right level.
    """
    cases = [c for c in (inputs.category_comps or []) if (c.get("price") or 0) > 0]
    if len(cases) < config.CATEGORY_KILLER_MIN_CASES:
        return inputs.comps, None
    divisor = config.STRATEGIC_TO_MARKET_DIVISOR
    cases = [dict(case, price=float(case["price"]) / divisor) for case in cases]

    keyword_median = statistics.median(_prices(inputs.comps)) if len(inputs.comps) >= 2 else None
    note = (
        f"anchored on {len(cases)} comparable acquisitions of category-defining domains "
        f"({', '.join(c['domain'] for c in cases)}) rather than on keyword comps"
    )
    note += (
        f", each brought down by {divisor:.2f}x to the level the market tiers describe — those "
        f"are completed acquisitions in which the domain became a company's identity, and "
        f"pricing an end-user sale at their face value would charge the strategic premium twice"
    )
    if keyword_median:
        note += (
            f". A substring search returns names CONTAINING this word, whose median sale is "
            f"${keyword_median:,.0f}, which is a different market from the word itself"
        )
    return cases, note


def auction_comps(comps: list[dict]) -> tuple[list[dict], bool, float]:
    """Comps from competitive auction venues, whether the filter held, and
    the share of the fallback set that is genuinely auction inventory.

    Falls back to the full set when too few auction sales survive — a
    narrower but empty filter is worse than a broader honest one — but the
    fallback is biased, not merely broader: brokered and retail sales clear
    above auction, so an anchor built from them overstates what an auction
    would fetch. The share is returned so that bias can be priced rather
    than merely mentioned.
    """
    auction_only = [
        c for c in comps
        if str(c.get("marketplace") or c.get("platform") or "").lower() in config.AUCTION_VENUES
    ]
    if len(auction_only) >= config.MIN_COMPS_FOR_TIERS:
        return auction_only, True, 1.0
    share = len(auction_only) / len(comps) if comps else 0.0
    return comps, False, share


def liquidity_score(inputs: TierInputs) -> float:
    """0-1: how quickly this domain converts to cash.

    Four signals, equally weighted, all of them things a buyer of last
    resort actually looks at: how short it is, whether its category has a
    standing bid, how many comparable sales exist, and how recent they are.
    A stale comp set means a market that has stopped clearing.
    """
    signals: list[float] = []

    length = len(inputs.sld)
    signals.append(1.0 if length <= 4 else 0.7 if length <= 8 else 0.4 if length <= 12 else 0.15)

    # Categories with a standing investor bid versus ones needing the right
    # single buyer to show up.
    signals.append(config.CATEGORY_LIQUIDITY.get(
        inputs.category, config.CATEGORY_LIQUIDITY_DEFAULT))

    count = len(inputs.comps)
    signals.append(1.0 if count >= 12 else 0.7 if count >= 6 else 0.4 if count >= 3 else 0.1)

    ages = [_age_days(c.get("date"), inputs.today) for c in inputs.comps]
    ages = [a for a in ages if a is not None]
    if ages:
        median_age = statistics.median(ages)
        signals.append(1.0 if median_age <= 120 else 0.7 if median_age <= 270 else 0.4 if median_age <= 540 else 0.2)
    else:
        signals.append(0.3)

    return round(sum(signals) / len(signals), 4)


def _age_days(sale_date, today: date | None) -> int | None:
    if not sale_date:
        return None
    try:
        parsed = date.fromisoformat(str(sale_date)[:10])
    except ValueError:
        return None
    return max(0, ((today or date.today()) - parsed).days)


def appraisal_disagreement(inputs: TierInputs, tiers: TierSet) -> str | None:
    """A warning when an independent appraisal and the comps disagree wildly.

    Deliberately does not reconcile them. Picking a winner would mean
    deciding, silently and per-domain, whether to trust a vendor's model or
    our own comps — and being wrong either way produces a confident number
    with no warning attached. Surfacing the disagreement hands that judgement
    to the person who has to act on it.

    The case this exists for: a category-defining domain. Comps search finds
    other names CONTAINING the term while the appraiser values the term
    itself, so the two are pricing different markets and the gap is orders of
    magnitude rather than percentages.
    """
    appraisal = inputs.independent_appraisal
    reference = tiers.end_user or tiers.auction
    if appraisal is None or reference is None:
        return None
    # Read the field directly. A getattr(..., None) here would turn a
    # wrong-typed appraisal into a silent no-warning, which is the exact
    # failure this function exists to prevent.
    value = appraisal.value
    if value <= 0 or reference.point <= 0:
        return None

    ratio = max(value / reference.point, reference.point / value)
    if ratio < config.APPRAISAL_DISAGREEMENT_FACTOR:
        return None

    direction = "far above" if value > reference.point else "far below"
    weaker, reasons = adjudicate_disagreement(inputs, reference.point, appraisal)

    note = (
        f"appraisal conflict: independent appraisals give {appraisal.describe()}, "
        f"{direction} the comps-derived end-user price of ${reference.point:,} ({ratio:.0f}x apart "
        f"from the ${value:,} reference). The tiers above are comps-anchored and have NOT been "
        f"adjusted."
    )

    # Which number to distrust, and on what grounds. Naming a weaker side is
    # not the same as adopting the other one: the tiers stay comps-anchored
    # either way, because reconciling them silently would hide exactly the
    # judgement the reader needs to make. What changes is that they are no
    # longer handed two numbers and left to guess.
    if weaker == "appraisal":
        note += (
            f" The appraisal is the weaker number here: {_join(reasons)}. Treat the comps-anchored "
            f"tiers as the better estimate and the vendor figure as a loose sanity check."
        )
    elif weaker == "comps":
        note += (
            f" The comp set is the weaker number here: {_join(reasons)}. The tiers are still what "
            f"this engine can defend from sales, but the vendor figure deserves real weight and "
            f"the domain is worth appraising by hand."
        )
    else:
        note += (
            f" Neither side is clearly stronger: {_join(reasons)}. This is a genuine disagreement "
            f"between two defensible methods, not an error in one of them, and it will not be "
            f"resolved by arithmetic."
        )

    # Both of these explain how the COMP SET could be wrong, so neither
    # belongs on a verdict that just named the comps the stronger side.
    # Appending them unconditionally had the note argue against its own
    # conclusion in the same breath.
    if weaker != "appraisal":
        if value > reference.point:
            note += (
                " A gap this size usually means the comparable sales describe a different market than "
                "the domain itself — the classic case being a single-word .com, where comps find other "
                "names containing the word rather than the word."
            )
        else:
            note += (
                " One outlier can carry a small comp set; check the individual sales before relying "
                "on them."
            )
    return note


def _join(reasons: list[str]) -> str:
    if not reasons:
        return "no distinguishing evidence either way"
    if len(reasons) == 1:
        return reasons[0]
    return "; ".join(reasons[:-1]) + f"; and {reasons[-1]}"


def adjudicate_disagreement(
    inputs: TierInputs, reference_point: int, appraisal: IndependentAppraisal
) -> tuple[str | None, list[str]]:
    """Which of the two numbers is the weaker, and the grounds for saying so.

    Returns ("comps" | "appraisal" | None, reasons). None means the evidence
    does not separate them, which is a real answer and more useful than a
    coin flip dressed as a finding.

    The signals are only the ones actually in hand. There is no model here
    of which vendor is right about what — that would be invention. What
    there is: how the comp set was built, and whether the vendors agree with
    each other, which F12 made visible by keeping every baseline rather than
    just the reconciled figure.
    """
    against_comps: list[str] = []
    against_appraisal: list[str] = []

    # --- the comp set's own account of itself -------------------------
    basis = (inputs.comp_basis or "").lower()
    comp_count = len([c for c in inputs.comps if c.get("price")])
    if basis in ("keyword_only", "none"):
        against_comps.append(
            f"the comps are {basis.replace('_', ' ')} — aggregate keyword statistics rather than "
            f"sales of this name")
    elif basis == "cohort":
        against_comps.append("the comps price the cohort this name belongs to, not the name itself")
    elif basis == "exact" and comp_count >= config.MIN_COMPS_FOR_TIERS * 2:
        against_appraisal.append(f"{comp_count} direct sales of this name support the comps figure")

    if inputs.comp_confidence and inputs.comp_confidence < 0.5:
        against_comps.append(f"comp confidence is only {inputs.comp_confidence:.2f}")
    elif inputs.comp_confidence and inputs.comp_confidence >= 0.85:
        against_appraisal.append(f"comp confidence is high at {inputs.comp_confidence:.2f}")

    if 0 < comp_count < config.MIN_COMPS_FOR_TIERS * 2:
        against_comps.append(f"only {comp_count} comparable sales carry the distribution")

    # --- whether the vendors agree with each other --------------------
    baselines = [v for v in (appraisal.all_baselines or {}).values() if v and v > 0]
    if len(baselines) >= 2:
        spread = max(baselines) / min(baselines)
        if spread >= config.VENDOR_SPREAD_UNRELIABLE:
            against_appraisal.append(
                f"the vendors disagree with each other by {spread:.0f}x "
                f"(${min(baselines):,} to ${max(baselines):,}), so their reconciled figure is not "
                f"a settled number either")
        else:
            against_comps.append(
                f"the vendor appraisals agree with each other to within {spread:.1f}x")
    elif len(baselines) == 1:
        against_appraisal.append("a single vendor with nothing to corroborate it")

    # --- verdict -------------------------------------------------------
    if len(against_comps) > len(against_appraisal):
        return "comps", against_comps
    if len(against_appraisal) > len(against_comps):
        return "appraisal", against_appraisal
    return None, against_comps + against_appraisal


def check_trademark_conflict(domain: str) -> dict:
    """Screen a domain for the conflicts that need no database.

    This was a stub returning `unknown` on every valuation the engine has
    ever produced, on the reasoning that a conflict is a legal determination
    and a plausible guess would be worse than silence. That reasoning still
    holds for the general case and is still not attempted.

    What it now does is the case a search would be wasted on: a household
    mark inside the name, whose owner enforces against domains as routine
    policy. nikeshoes.com does not need a lookup to be a problem.

    The vocabulary has no `clear` in it, deliberately. A domain that trips
    nothing has failed to trip an obvious wire and has not been searched.
    """
    from portfolio_agent.tools.trademark import _screen

    return _screen((domain or "").strip().lower(), searched=False)


def band_ratios(stats: "CompStats | None", confidence: float = 1.0) -> tuple[float, float]:
    """Low/high band ratios taken from the comps' own dispersion, widened as
    confidence falls.

    A flat ±35% band would claim the same confidence for a tight ten-sale
    cluster as for two sales an order of magnitude apart, so the base band
    comes from the distribution itself — p10 to p90, which is what a reader
    takes "low to high" to mean. The interquartile range this used to take
    contains only half the outcomes, so it was quietly promising a narrower
    thing than it was labelled. On top of that, a price reached from
    a structural cohort is a WIDER claim than one reached from direct sales
    of the same thing, and it has to look wider: the band is divided by the
    comp confidence, capped at MAX_TIER_SPREAD so it stays a range rather
    than a shrug.
    """
    if stats is not None and not isinstance(stats, CompStats):
        # A plain price list has a `.count` attribute too — it is a method,
        # so the duck-typing failure surfaces as an unrelated TypeError
        # several frames away. Name the real mistake here.
        raise TypeError(
            f"band_ratios takes CompStats, got {type(stats).__name__}; "
            "pass comp_stats(comps) rather than a price list")
    if stats is None or stats.count < 2 or stats.median <= 0:
        base_low, base_high = 0.70, 1.40
    else:
        low_limit, high_limit = config.TIER_BAND_LIMITS
        low_q, high_q = config.BAND_PERCENTILES
        low_point = stats.p10 if low_q == 0.10 else weighted_percentile(_as_pairs(stats), low_q)
        high_point = stats.p90 if high_q == 0.90 else weighted_percentile(_as_pairs(stats), high_q)
        base_low = max(low_limit, min(1.0, low_point / stats.median))
        base_high = min(high_limit, max(1.0, high_point / stats.median))

    confidence = max(0.05, min(1.0, confidence or 1.0))
    if confidence < 1.0:
        widen = min(1.0 / confidence, config.MAX_TIER_SPREAD)
        base_low = base_low / widen
        base_high = min(base_high * widen, config.MAX_TIER_SPREAD)

    return round(max(0.01, base_low), 4), round(max(1.0, base_high), 4)


def _tier(point: float, ratios: tuple[float, float], adjustments: Adjustments, refs: list[str],
          probability: float | None = None) -> PriceTier:
    point_int = max(0, round(point))
    return PriceTier(
        low=max(0, round(point_int * ratios[0])),
        point=point_int,
        high=max(point_int, round(point_int * ratios[1])),
        sale_probability=probability,
        inflationary_factors=list(adjustments.inflationary),
        deflationary_factors=list(adjustments.deflationary),
        evidence_refs=refs,
    )


# ── The four tiers ─────────────────────────────────────────────────────────


def auction_price(inputs: TierInputs) -> PriceTier | None:
    """Investor demand in a competitive auction. The anchor everything else
    hangs off: the median of venue-filtered comps, moved by trend velocity."""
    anchored, _ = anchor_comps(inputs)
    comps, venue_filtered, auction_share = auction_comps(anchored)
    prices = _prices(comps)
    stats = comp_stats(comps)
    if len(prices) < config.MIN_COMPS_FOR_TIERS or stats is None:
        return None

    anchor = stats.median
    adjustments = Adjustments()
    debug(log, "auction.anchor", anchor=round(anchor, 2), comps=len(prices),
          venue_filtered=venue_filtered)

    if not venue_filtered:
        # Priced, not merely mentioned. Brokered and retail sales clear above
        # auction, so an anchor including them is biased high in a known
        # direction — and a caveat with no multiplier behind it is a caveat
        # the price ignores. The haircut scales with how much of the set is
        # actually auction inventory.
        haircut = config.AUCTION_RETAIL_HAIRCUT + (1.0 - config.AUCTION_RETAIL_HAIRCUT) * auction_share
        # Said differently when the comps are acquisitions. "0% of these
        # are auction-venue sales" is true of every private acquisition ever
        # made, so as a finding it is a tautology — it tells the reader a
        # property of the category rather than anything about this domain.
        if auction_share == 0 and all(not c.get("marketplace") for c in comps):
            reason = (
                f"none of the {len(prices)} comps are marketplace sales at all — they are "
                f"privately negotiated acquisitions, which clear well above what a public "
                f"auction fetches, so the anchor is discounted toward auction reality"
            )
        else:
            reason = (
                f"only {auction_share:.0%} of the {len(prices)} comps are auction-venue sales; the "
                f"rest are brokered or retail and clear above auction, so the anchor is discounted "
                f"toward what an auction would actually fetch"
            )
        adjustments.apply(haircut, reason)

    velocity = float((inputs.trend or {}).get("trend_velocity") or 0.0)
    if velocity:
        factor = 1.0 + velocity * config.AUCTION_TREND_SENSITIVITY
        rules = (inputs.trend or {}).get("matched_rules") or []
        reason = rules[0] if rules else ("positive trend velocity" if velocity > 0 else "negative trend velocity")
        adjustments.apply(factor, f"{reason}, velocity {velocity:+.2f}")

    adjustments.apply(inputs.typo_deflator, "misspelling of a commercial term")

    return _tier(
        anchor * adjustments.multiplier,
        band_ratios(stats, inputs.comp_confidence),
        adjustments,
        inputs.refs("comps", "trends"),
    )


def floor_price(inputs: TierInputs, auction: PriceTier | None) -> PriceTier | None:
    """The minimum acceptable price for a quick sale, with the probability
    of actually getting it. Anchored to auction, discounted for speed, then
    moved by liquidity — and never below what the domain cost."""
    if auction is None:
        return None

    liquidity = liquidity_score(inputs)
    adjustments = Adjustments()

    low, high = config.FLOOR_LIQUIDITY_RANGE
    liquidity_factor = low + (high - low) * liquidity
    if liquidity >= 0.7:
        adjustments.apply(liquidity_factor, f"high liquidity signals (score {liquidity:.2f}) — clears near auction")
    elif liquidity <= 0.4:
        adjustments.apply(liquidity_factor, f"low liquidity signals (score {liquidity:.2f}) — needs a deeper discount")
    else:
        adjustments.apply(liquidity_factor, f"moderate liquidity (score {liquidity:.2f})")

    point = auction.point * config.FLOOR_DISCOUNT_FROM_AUCTION * adjustments.multiplier

    cost = inputs.acquisition_cost if inputs.acquisition_cost is not None else config.ACQUISITION_COST_PLACEHOLDER
    if point < cost:
        adjustments.apply(
            cost / max(point, 1),
            f"comps imply a floor below acquisition cost (${cost:,}) — raised to cost, which is the "
            f"real floor",
        )
        point = cost

    return _tier(
        point,
        band_ratios(comp_stats(anchor_comps(inputs)[0]), inputs.comp_confidence),
        adjustments,
        inputs.refs("comps", "base_appraisal"),
        probability=_floor_probability(inputs, liquidity),
    )


def _floor_probability(inputs: TierInputs, liquidity: float) -> float | None:
    """GoDaddy's wholesale sale probability, adjusted by liquidity.

    Returns None rather than a guess when there is no baseline appraisal:
    an unqualified "70% chance of sale" is exactly the kind of number a
    seller acts on and should not be invented.
    """
    appraisal = inputs.base_appraisal or {}
    base = appraisal.get("sales_probability_wholesale")
    if base is None:
        base = appraisal.get("sales_probability")
    if base is None:
        return None

    # Blended with our own reading rather than taken and scaled.
    #
    # The vendor number was previously used alone, stretched by liquidity.
    # That makes one appraisal the sole author of a figure a seller acts on,
    # and these models have been wrong by orders of magnitude on exactly the
    # domains where it matters — the same vendor put $25,000 on a name whose
    # comps support seven figures. Blending keeps the vendor's advantage (it
    # is trained on more transactions than we will ever see) without letting
    # a single bad reading carry the whole number.
    #
    # Our side of the blend is liquidity, which is what wholesale clearance
    # actually depends on: how fast the thing converts, and whether its
    # category has a standing bid at all.
    category_liquidity = config.CATEGORY_LIQUIDITY.get(
        inputs.category, config.CATEGORY_LIQUIDITY_DEFAULT)
    internal = category_liquidity * liquidity
    weight = config.PROVIDER_PROBABILITY_WEIGHT
    blended = weight * float(base) + (1.0 - weight) * internal

    # Scaled by comp confidence: a probability of clearing at a price we
    # inferred from a cohort is less certain than one at a price we watched
    # three near-identical domains actually fetch.
    blended *= max(0.05, min(1.0, inputs.comp_confidence or 1.0))
    return round(max(0.0, min(config.MAX_REPORTED_SALE_PROBABILITY, blended)), 4)


def end_user_price(inputs: TierInputs, auction: PriceTier | None) -> tuple[PriceTier | None, list[str]]:
    """Fair market value to a business that will develop the domain.

    Table 4.1.1's inflators (brand score, keyword volume, premium TLD,
    industry match) and deflators (phonetic issues, negative sentiment,
    trademark conflict) applied to the upper end of the comps distribution —
    an end user pays above investor prices, so the anchor is p75, not the
    median.

    Returns the tier and any warnings it raised; trademark screening is a
    stub, and a tier that quietly omits a named deflator has to say so.
    """
    anchored_comps = anchor_comps(inputs)[0]
    prices = _prices(anchored_comps)
    stats = comp_stats(anchored_comps)
    if len(prices) < config.MIN_COMPS_FOR_TIERS or auction is None or stats is None:
        return None, []

    anchor = max(stats.p75, auction.point)
    adjustments = Adjustments()
    warnings: list[str] = []

    # F20.1: Decouple brand quality from demand — gate the brand premium by
    # commercial intent. Brand score measures phonetic/structural quality
    # (isotope is pronounceable), but demand determines whether that quality
    # commands a premium (isotope is niche). High-brand, low-demand words get
    # a reduced premium; high-brand, high-demand words keep it.
    if inputs.brand_score is not None:
        low, high = config.END_USER_BRAND_RANGE
        brand_factor = low + (high - low) * (inputs.brand_score.total / 100.0)
        if not config.BRAND_DEMAND_GATE:
            adjustments.apply(brand_factor, f"brand score {inputs.brand_score.total:.0f}/100")
        else:
            intent = (inputs.intent or {}).get("intent") or classify_intent(inputs.sld)[0]
            # Commercial is the only high-demand intent.
            commercial_share = 1.0 if intent == "commercial" else 0.3
            category_demand = config.CATEGORY_LIQUIDITY.get(inputs.category, config.CATEGORY_LIQUIDITY_DEFAULT)
            demand_factor = commercial_share * 0.7 + category_demand * 0.3
            # High brand + high demand keeps the full premium (demand_factor
            # near 1.0); high brand + low demand keeps a fraction of it.
            gated = 1.0 + (brand_factor - 1.0) * demand_factor
            if demand_factor < 0.7:
                adjustments.apply(
                    gated,
                    f"brand score {inputs.brand_score.total:.0f}/100, gated by demand "
                    f"(intent: {intent}, category liquidity: {category_demand:.2f}) — "
                    f"high-brand but low-demand words earn a reduced premium",
                )
            else:
                adjustments.apply(gated, f"brand score {inputs.brand_score.total:.0f}/100")

    volume = _search_volume(inputs)
    if volume:
        low, high = config.END_USER_KEYWORD_RANGE
        # Volume without intent measures attention, not demand: 250,000
        # searches for "weather" bought the same premium as 250,000 for "car
        # insurance". The credit is discounted before it earns a premium,
        # rather than the tier being deflated after the fact, so the tier
        # reads as "fewer effective searches" and not "a penalty".
        # Resolved during evidence gathering, because resolving it may
        # need a network call and pricing stays pure. Falls back to the
        # lexicon so the engine still prices correctly when called directly.
        resolved = inputs.intent or {}
        intent = resolved.get("intent") or classify_intent(inputs.sld)[0]
        reasons = resolved.get("reasons") or classify_intent(inputs.sld)[1]
        credit = volume_credit(intent)
        effective = volume * credit
        ratio = min(1.0, effective / config.END_USER_KEYWORD_VOLUME_FOR_MAX)
        if credit < 1.0:
            adjustments.apply(
                low + (high - low) * ratio,
                f"{volume:,} monthly searches, discounted to {effective:,.0f} effective — "
                f"{reasons[0] if reasons else 'the keyword reads as informational'}, and an "
                f"audience that reads is not an audience that buys")
        else:
            adjustments.apply(low + (high - low) * ratio,
                              f"{volume:,} monthly searches for the keyword")

    tld_key = "com" if inputs.tld in PREMIUM_TLDS else "emerging" if inputs.tld in EMERGING_TLDS else "other"
    adjustments.apply(
        config.END_USER_TLD_PREMIUM[tld_key],
        {"com": ".com — the extension end users pay a premium for",
         "emerging": f".{inputs.tld} — priced at par with its own market",
         "other": f".{inputs.tld} — outside the extensions end users seek"}[tld_key],
    )

    band = (inputs.seo or {}).get("authority_band")
    if band:
        adjustments.apply(
            config.END_USER_SEO_MULTIPLIER.get(band, 1.0),
            f"existing SEO authority: {band} (DR {(inputs.seo or {}).get('domain_rating')})",
        )

    velocity = float((inputs.trend or {}).get("trend_velocity") or 0.0)
    if velocity:
        adjustments.apply(1.0 + velocity * config.END_USER_TREND_SENSITIVITY, f"market trend velocity {velocity:+.2f}")

    # Length against the extension's value window. A structural fact, and
    # the reason it was missing: brand scoring is skipped for keyword and
    # EMD domains — correctly, since pronounceability is a brandable concern
    # — and structural scoring went with it. Length is not a brandable
    # concern.
    length_factor, length_reason = length_adjustment(inputs)
    if length_reason:
        adjustments.apply(length_factor, length_reason)

    # A real offer above the floor is the only direct evidence in this whole
    # engine that a specific person would pay a specific amount for THIS
    # domain. It is not a comp — nobody paid it — but it is a stronger
    # demand signal than any inferred one, so it lifts the tier, capped:
    # one offer is one buyer, and a buyer who walked away proves nothing.
    demand = inputs.demand or {}
    offer = demand.get("best_offer")
    if offer and offer > 0:
        floor_estimate = auction.point * config.FLOOR_DISCOUNT_FROM_AUCTION
        if offer > floor_estimate:
            lift = min(config.INTERNAL_OFFER_MAX_LIFT, 1.0 + (offer / max(anchor, 1)) * 0.5)
            adjustments.apply(
                lift,
                f"a real offer of ${offer:,} was received — above the quick-exit floor, so somebody "
                f"has already put a number on this domain",
            )

    # Deflators named in Table 4.1.1 for this tier.
    if inputs.phonetic_score is not None and inputs.phonetic_score < 60:
        floor = config.END_USER_PHONETIC_FLOOR
        factor = floor + (1.0 - floor) * (inputs.phonetic_score / 60.0)
        adjustments.apply(factor, f"hard to pronounce (phonetic score {inputs.phonetic_score:.0f}/100)")

    if inputs.sentiment_score is not None and inputs.sentiment_score < 60:
        floor = config.END_USER_SENTIMENT_FLOOR
        factor = floor + (1.0 - floor) * (inputs.sentiment_score / 60.0)
        adjustments.apply(factor, f"negative keyword sentiment (score {inputs.sentiment_score:.0f}/100)")

    adjustments.apply(inputs.typo_deflator, "misspelling of a commercial term")

    # Table 4.1.1 lists trademark conflict as an End-User deflator, and for
    # every valuation until now it was named and never applied.
    trademark = check_trademark_conflict(inputs.domain)
    deflator = float(trademark.get("deflator") or 1.0)
    if deflator < 1.0:
        adjustments.apply(
            deflator,
            f"trademark risk ({trademark['status'].replace('_', ' ')}): {trademark['detail']}")
    if trademark["status"] in ("likely_conflict", "possible_conflict"):
        warnings.append(
            f"end_user: {trademark['detail']}. This is a screen against famous marks, NOT a "
            f"trademark search — clear it properly (USPTO/EUIPO/WIPO) before relying on the "
            f"End-User or higher tiers.")
    else:
        warnings.append(
            f"end_user: {trademark['detail']}. No trademark SEARCH has been run; only the "
            f"famous-mark screen. Table 4.1.1's deflator is applied only where that screen "
            f"fires, so an ordinary registered mark would pass through unnoticed.")
    
    # F20.1: TLD discount — apply per-TLD multiplier after all other adjustments.
    # A great word on .co is still a .co. isotope.co should land near its comp
    # median, not 4.7x above it. Unmeasured, and the ladder already prices
    # from same-extension comps, so this may count the extension twice —
    # off until a backtest says otherwise.
    tld_mult = get_tld_multiplier(inputs.tld) if config.TLD_TIER_DISCOUNT else 1.0
    if tld_mult != 1.0:
        adjustments.apply(
            tld_mult,
            f".{inputs.tld} TLD discount (×{tld_mult:.2f}) — extension-specific pricing adjustment"
        )

    return (
        _tier(anchor * adjustments.multiplier, band_ratios(stats, inputs.comp_confidence), adjustments,
              inputs.refs("comps", "brand_score", "seo", "trends", "keyword_demand")),
        warnings,
    )


def length_adjustment(inputs: TierInputs) -> tuple[float, str | None]:
    """How the name's length sits against where its extension's money is.

    Not a penalty for being long in the abstract. Each extension transacts
    within a window, and a name at or beyond that window's edge is competing
    for a far smaller pool of buyers — on .ai, $1.8M of recorded sales above
    ten characters against $45.2M at or below.
    """
    length = (inputs.structural or {}).get("char_length")
    if not length:
        return 1.0, None

    window = config.LENGTH_VALUE_WINDOW.get(inputs.tld, config.LENGTH_VALUE_WINDOW_DEFAULT)
    if length < window:
        return 1.0, None
    if length == window:
        return (
            config.LENGTH_BOUNDARY_FACTOR,
            f"{length} characters sits exactly on the boundary of .{inputs.tld}'s value window — "
            f"inside it, but at the edge, where the buyer pool thins sharply",
        )
    return (
        config.LENGTH_BEYOND_WINDOW_FACTOR,
        f"{length} characters is beyond .{inputs.tld}'s {window}-character value window, where "
        f"recorded sales fall away by more than an order of magnitude",
    )


def _search_volume(inputs: TierInputs) -> int | None:
    stats = inputs.keyword_stats or {}
    for key in ("global_monthly_searches", "local_monthly_searches"):
        value = stats.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return None


def list_bin_price(inputs: TierInputs, end_user: PriceTier | None) -> tuple[PriceTier | None, list[str]]:
    """A premium Buy-It-Now ask, with the probability of it being taken.

    Table 4.1.1's inflators are seller patience, scarcity of alternatives
    and a high-growth industry. Scarcity is measured from how widely the
    term is already registered — a keyword live in 340 TLDs has no scarcity
    to sell.
    """
    if end_user is None:
        return None, []

    adjustments = Adjustments()
    low, high = config.LIST_BIN_SCARCITY_RANGE
    scarcity = _scarcity_score(inputs)
    factor = low + (high - low) * scarcity
    adjustments.apply(factor, f"scarcity and industry growth (score {scarcity:.2f}) — premium BIN markup")

    listing_multiplier, listing_factor, listing_warning = listing_adjustment(inputs, end_user)
    if listing_factor:
        adjustments.apply(listing_multiplier, listing_factor)

    point = end_user.point * adjustments.multiplier

    # The ask is a ceiling — but only down to the end-user price, never
    # below it.
    #
    # A rational Buy-It-Now meets or undercuts the price the same domain is
    # visibly on sale at, because asking more than the open market is asking
    # is not a premium, it is a reason to click the other listing. That
    # holds while the ask sits above what transactions support.
    #
    # When the ask is BELOW the end-user price the two rules collide: the
    # ceiling says come down, the tier order says a BIN cannot sit under the
    # end-user price it is a premium on. Forcing the cap there and letting
    # the order guard raise it back produced a BIN identical to the
    # end-user price, which destroyed the information rather than resolving
    # it. So the cap stops at the end-user price and the disagreement is
    # recorded instead — an ask below the comps means either the comps are
    # optimistic or the domain is underpriced, and that is the owner's call.
    ask = (inputs.listings or {}).get("lowest_ask")
    listing_notes: list[str] = []
    if ask:
        if ask >= end_user.point and point > ask:
            point = adjustments.cap(
                point, float(ask),
                f"capped at the live marketplace ask of ${ask:,} — a BIN above the visible market "
                f"ask sells nothing",
            )
        elif ask < end_user.point:
            # Changes no price, so it is a warning rather than a factor.
            listing_notes = adjustments.note(
                f"the live ask of ${ask:,} is BELOW the comps-implied end-user price of "
                f"${end_user.point:,} — either these comparable sales are optimistic or the domain "
                "is currently underpriced. The BIN is not capped to a figure under its own anchor, "
                "so no tier was changed."
            )

    markup = (point / end_user.point) if end_user.point else 1.0

    return (
        _tier(
            point,
            band_ratios(comp_stats(anchor_comps(inputs)[0]), inputs.comp_confidence),
            adjustments,
            inputs.refs("comps", "keyword_demand", "trends", "listings"),
                probability=_scaled(sale_probability_at_markup(markup), inputs.comp_confidence),
        ),
        ([listing_warning] if listing_warning else []) + listing_notes,
    )


def ask_gap(inputs: TierInputs, end_user: PriceTier | None) -> tuple[float | None, str | None]:
    """How far a seller's ask sits from what transactions support.

    Informational only — no tier reads this beyond the BIN ceiling. It
    exists because the ratio is the most useful single number about a listed
    domain, and because "4.8x" tells a reader nothing while "ambitious"
    tells them whether to bother.
    """
    ask = (inputs.listings or {}).get("lowest_ask")
    if not ask or end_user is None or end_user.point <= 0:
        return None, None
    gap = round(ask / end_user.point, 2)
    for threshold, label in config.ASK_GAP_BUCKETS:
        if gap < threshold:
            return gap, label
    return gap, config.ASK_GAP_BUCKETS[-1][1]


def listing_adjustment(inputs: TierInputs, end_user: PriceTier) -> tuple[float, str | None, str | None]:
    """How a live asking price moves the BIN tier, and what to say about it.

    Returns (multiplier, factor, warning). Table 4.1.1 makes marketplace
    listings the List/BIN tier's primary adjustment source, and the logic is
    the seller's side of the ladder: a BIN price is an ask, so what other
    asks stand at is directly relevant to it — in a way it is NOT relevant
    to the floor, auction or end-user tiers, which are anchored on
    transactions.

    The pull is capped (config.LISTING_BIN_PULL) because an ask is not a
    transaction. consulting.ai carries an ask near $2.8M against 506
    searches in a year and one recorded sale in its class at $425. Letting a
    listing set the price outright would make the valuation a mirror of
    whatever the most optimistic seller typed.
    """
    data = inputs.listings or {}
    if not data.get("listed"):
        return 1.0, None, None

    ask = data.get("lowest_ask")
    if not ask or not end_user.point:
        return 1.0, None, None

    baseline = end_user.point
    ratio = ask / baseline
    pull = config.LISTING_BIN_PULL
    # Move a fraction of the way toward the ask, never all of it.
    multiplier = 1.0 + (ratio - 1.0) * pull

    live_auction = data.get("has_live_auction")
    kind = "a live auction bid" if live_auction else "an asking price"
    factor = (
        f"{kind} of ${ask:,} is on the open market, "
        f"{'above' if ratio > 1 else 'below'} the comps-implied end-user price of ${baseline:,}"
    )

    warning = None
    if max(ratio, 1 / ratio) >= config.LISTING_DISAGREEMENT_FACTOR:
        warning = (
            f"listing conflict: this domain is listed at ${ask:,}, which is "
            f"{max(ratio, 1 / ratio):.0f}x {'above' if ratio > 1 else 'below'} the comps-derived "
            f"end-user price of ${baseline:,}. An asking price is not a sale — premium listings sit "
            "unsold for years — so the BIN tier is pulled only partway toward it and the other tiers "
            "are untouched."
        )
    return multiplier, factor, warning


def _scarcity_score(inputs: TierInputs) -> float:
    """0-1. High means few alternatives to this exact name."""
    signals: list[float] = []

    tld_count = (inputs.keyword_stats or {}).get("exact_keyword_active_tld_count")
    if isinstance(tld_count, (int, float)):
        # Live in many TLDs means the term is available elsewhere cheaply.
        signals.append(1.0 if tld_count <= 20 else 0.6 if tld_count <= 100 else 0.25)

    length = len(inputs.sld)
    signals.append(1.0 if length <= 5 else 0.7 if length <= 9 else 0.35)

    velocity = float((inputs.trend or {}).get("trend_velocity") or 0.0)
    signals.append(max(0.0, min(1.0, 0.5 + velocity / 2)))

    if inputs.brand_score is not None:
        signals.append(inputs.brand_score.total / 100.0)

    return round(sum(signals) / len(signals), 4) if signals else 0.5


def _scaled(probability: float | None, confidence: float) -> float | None:
    """A stated probability is only as good as the price it is stated at."""
    if probability is None:
        return None
    return round(probability * max(0.05, min(1.0, confidence or 1.0)), 4)


def sale_probability_at_markup(markup: float) -> float:
    """Probability of a BIN sale at `markup` × the end-user price.

    Linear interpolation over the calibrated curve in config, clamped at
    both ends. The inverse relationship is the point: the whole reason to
    publish a BIN price alongside an end-user price is to make the trade
    between speed and money explicit.
    """
    curve = sorted(config.LIST_BIN_PROBABILITY_CURVE)
    if markup <= curve[0][0]:
        return round(curve[0][1], 4)
    if markup >= curve[-1][0]:
        return round(curve[-1][1], 4)
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= markup <= x1:
            span = x1 - x0
            ratio = 0.0 if span <= 0 else (markup - x0) / span
            return round(y0 + ratio * (y1 - y0), 4)
    return round(curve[-1][1], 4)


# ── Assembly ───────────────────────────────────────────────────────────────


def compute_market_tiers(inputs: TierInputs) -> tuple[TierSet, list[str]]:
    """The four market tiers plus any warnings they raised.

    The targeted-lead and strategic-M&A slots are left for
    pricing/targeted.py — they are conditional on an identified buyer, not
    on market data.
    """
    warnings: list[str] = []
    anchored, anchor_note = anchor_comps(inputs)
    prices = _prices(anchored)

    if anchor_note:
        info(log, "tiers.reanchored", cases=len(anchored), keyword_comps=len(inputs.comps))
        warnings.append(
            f"category-defining domain: {anchor_note}. This comp set is seventeen of the most "
            "famous domain sales ever made — survivorship-biased by construction, because "
            "unremarkable eight-figure sales are never reported — so it skews HIGH. Measured "
            "against known outcomes it lands within roughly 0.4x to 4x of the real price. Treat "
            "these tiers as an order of magnitude, not an estimate."
        )

    if len(prices) < config.MIN_COMPS_FOR_TIERS:
        warn(log, "tiers.insufficient_comps", comps=len(prices), required=config.MIN_COMPS_FOR_TIERS)
        for tier_name in ("floor", "auction", "end_user", "list_bin"):
            warnings.append(
                f"{tier_name}: only {len(prices)} comparable sale(s) found, "
                f"below the {config.MIN_COMPS_FOR_TIERS} needed for a distribution — left unfilled rather than guessed"
            )
        return TierSet(), warnings

    auction = auction_price(inputs)
    floor = floor_price(inputs, auction)
    end_user, end_user_warnings = end_user_price(inputs, auction)
    warnings.extend(end_user_warnings)
    list_bin, listing_warnings = list_bin_price(inputs, end_user)
    warnings.extend(listing_warnings)

    tiers = TierSet(floor=floor, auction=auction, end_user=end_user, list_bin=list_bin)

    # F17.2: Plausibility gate — check comps-derived end_user against sanity bounds.
    # Runs BEFORE tier order enforcement so implausible comps get flagged even
    # when they happen to sit in the right order. Either gate may lower
    # inputs.comp_confidence, which is what the output reports.
    if config.PLAUSIBILITY_GATE:
        plausibility_warning = check_plausibility(inputs, tiers)
        if plausibility_warning:
            warnings.append(plausibility_warning)

    # F18.1: Ask reconciliation — cross-check live asking price against comps.
    # The ask never enters comps (WS3 quarantine), but a live seller pricing
    # this exact domain is strong evidence and must not be silently dropped.
    if config.ASK_RECONCILIATION:
        # The signals dict is not read anywhere yet; only the warning reaches the report.
        _, ask_warning = check_ask_reconciliation(inputs, tiers)
        if ask_warning:
            warnings.append(ask_warning)

    warnings.extend(enforce_tier_order(tiers))

    trend_warning = (inputs.trend or {}).get("staleness_warning")
    if trend_warning:
        warnings.append(trend_warning)

    if inputs.comp_basis not in ("exact", ""):
        warnings.append(
            f"comp basis '{inputs.comp_basis}': no direct sales of this domain's own keyword on its own "
            f"extension were found, so the tiers rest on a broader comparison set (confidence "
            f"{inputs.comp_confidence:.2f}). The ranges above are widened to match."
        )

    conflict = appraisal_disagreement(inputs, tiers)
    if conflict:
        warn(log, "tiers.appraisal_conflict",
             reference=appraisal.value if (appraisal := inputs.independent_appraisal) else None,
             source=appraisal.source if appraisal else None,
             baselines=appraisal.all_baselines if appraisal else None,
             comps_end_user=tiers.end_user.point if tiers.end_user else None)
        warnings.append(conflict)

    return tiers, warnings


def check_ask_reconciliation(inputs: TierInputs, tiers: TierSet) -> tuple[dict | None, str | None]:
    """F18.1: Ask-vs-model check — reconcile live asking price with comps-derived tiers.
    
    When a live ask exists for the exact subject domain, compute the gap between
    it and the comps-derived end_user. If the gap exceeds thresholds, emit a
    warning and cap confidence — a live human pricing this exact domain today is
    strong real-world evidence, never silent.
    
    The ask NEVER enters the comps (WS3 quarantine holds). It is a cross-check,
    not an input.
    
    Returns (signals_dict, warning_str). Lowers inputs.comp_confidence when it
    caps, so the reported confidence matches the warning.
    """
    if tiers.end_user is None or tiers.end_user.point <= 0:
        return None, None
    
    # Extract live ask from listings
    listings = inputs.listings or {}
    ask = listings.get("lowest_ask")
    if not ask or ask <= 0:
        return None, None
    
    comps_end_user = tiers.end_user.point
    ask_ratio = comps_end_user / ask
    
    # Check thresholds
    gap_high = config.ASK_GAP_HIGH
    triggered = False
    verdict = "aligned"
    
    if ask < comps_end_user / gap_high:
        # Seller pricing far below model
        triggered = True
        verdict = "seller_below_model"
        reason = (
            f"live ask ${ask:,} is {ask_ratio:.1f}x below the comps-derived end_user "
            f"${comps_end_user:,}. A motivated seller pricing this exact domain today is "
            f"strong evidence the cohort overstates it; treat end_user as the optimistic bound "
            f"and the ask as the realistic floor of a negotiation."
        )
    elif ask > comps_end_user * gap_high:
        # Seller pricing far above model (conviction/moonshot ask)
        triggered = True
        verdict = "seller_above_model"
        reason = (
            f"live ask ${ask:,} is {1/ask_ratio:.1f}x above the comps-derived end_user "
            f"${comps_end_user:,}. A conviction ask reflects the seller's belief rather than "
            f"comparable transactions — premium listings sit unsold for years. The comps-anchored "
            f"tiers are untouched."
        )
    else:
        # Aligned: ask within threshold
        return {
            "ask": ask,
            "comps_end_user": comps_end_user,
            "ratio": round(ask_ratio, 2),
            "verdict": "aligned"
        }, None
    
    warning = None
    if triggered:
        # Cap confidence and widen end_user range when seller is below model
        if verdict == "seller_below_model":
            original_conf = inputs.comp_confidence
            # Already at or under the cap: nothing to move, but the conflict
            # is still reported.
            warning = f"ask conflict: {reason} Confidence is already {original_conf:.2f}."
            if original_conf > config.ASK_CONFLICT_CONF:
                # Force confidence down
                new_conf = config.ASK_CONFLICT_CONF
                widen_factor = original_conf / new_conf
                
                # Widen end_user range DOWN so ask falls within or near low bound
                if tiers.end_user:
                    old_low_ratio = tiers.end_user.low / tiers.end_user.point if tiers.end_user.point > 0 else 0.7
                    old_high_ratio = tiers.end_user.high / tiers.end_user.point if tiers.end_user.point > 0 else 1.4
                    # Widen low bound more aggressively to encompass ask
                    ask_low_ratio = ask / tiers.end_user.point if tiers.end_user.point > 0 else 0.3
                    new_low_ratio = min(old_low_ratio / widen_factor, ask_low_ratio * 0.9)
                    new_high_ratio = min(config.MAX_TIER_SPREAD, old_high_ratio * widen_factor)
                    tiers.end_user.low = max(0, round(tiers.end_user.point * new_low_ratio))
                    tiers.end_user.high = max(tiers.end_user.point, round(tiers.end_user.point * new_high_ratio))
                    # Scale sale probability
                    if tiers.end_user.sale_probability:
                        tiers.end_user.sale_probability = round(
                            tiers.end_user.sale_probability * new_conf / max(original_conf, 0.05), 4
                        )
                
                inputs.comp_confidence = new_conf
                warn(log, "tiers.ask_conflict",
                     ask=ask,
                     comps_end_user=comps_end_user,
                     ratio=ask_ratio,
                     verdict=verdict,
                     original_confidence=original_conf,
                     forced_confidence=new_conf)
                
                warning = (
                    f"ask conflict: {reason} Confidence capped at {new_conf:.2f} and end_user "
                    f"range widened down to encompass the ask."
                )
        else:
            # Seller above model: note it, don't move tiers
            info(log, "tiers.ask_above_model", ask=ask, comps_end_user=comps_end_user, ratio=1/ask_ratio)
            warning = f"ask observation: {reason}"
    
    signals = {
        "ask": ask,
        "comps_end_user": comps_end_user,
        "ratio": round(ask_ratio, 2),
        "verdict": verdict
    }
    
    return signals, warning


def check_plausibility(inputs: TierInputs, tiers: TierSet) -> str | None:
    """F17.2: Plausibility gate — catch comps-derived prices that are structurally wrong.
    
    When comps-derived end_user falls below the independent appraisal OR below
    the category floor, the comps are likely misidentified (substring junk,
    wrong quality band) rather than the market having collapsed.
    
    Returns a warning and forces inputs.comp_confidence down to IMPLAUSIBLE_CONF,
    widening ranges accordingly. Prevents "confident and wrong" (pay.ai $564 at conf 1.0).
    """
    if tiers.end_user is None or tiers.end_user.point <= 0:
        return None
    
    comps_end_user = tiers.end_user.point
    triggered = []
    
    # Floor check: comps-derived price below independent appraisal
    appraisal = inputs.independent_appraisal
    if appraisal and appraisal.value > 0:
        if comps_end_user < appraisal.value:
            ratio = appraisal.value / comps_end_user
            triggered.append(
                f"comps-derived end_user (${comps_end_user:,}) is {ratio:.1f}x below the "
                f"independent appraisal (${appraisal.value:,})"
            )
    
    # Category floor check
    category_floor = get_floor(inputs.category, inputs.tld)
    if comps_end_user < category_floor:
        ratio = category_floor / comps_end_user
        triggered.append(
            f"comps-derived end_user (${comps_end_user:,}) is {ratio:.1f}x below the "
            f"plausibility floor for a {inputs.category} .{inputs.tld} (${category_floor:,})"
        )
    
    if not triggered:
        return None

    reasons = "; ".join(triggered)
    # Cap confidence across ALL tiers (they're all derived from the same comps)
    original_conf = inputs.comp_confidence
    if original_conf <= config.IMPLAUSIBLE_CONF:
        return (
            f"plausibility gate: {reasons}. The comp set is likely misidentified, not the "
            f"market. Confidence is already {original_conf:.2f}."
        )
    # Force confidence down
    new_conf = config.IMPLAUSIBLE_CONF
    # Widen all tier ranges proportionally
    widen_factor = original_conf / new_conf
    for tier in [tiers.floor, tiers.auction, tiers.end_user, tiers.list_bin]:
        if tier and tier.point > 0:
            # Widen the range symmetrically
            old_low_ratio = tier.low / tier.point if tier.point > 0 else 0.7
            old_high_ratio = tier.high / tier.point if tier.point > 0 else 1.4
            new_low_ratio = max(0.01, old_low_ratio / widen_factor)
            new_high_ratio = min(config.MAX_TIER_SPREAD, old_high_ratio * widen_factor)
            tier.low = max(0, round(tier.point * new_low_ratio))
            tier.high = max(tier.point, round(tier.point * new_high_ratio))
            # Scale sale probability down
            if tier.sale_probability:
                tier.sale_probability = round(tier.sale_probability * new_conf / max(original_conf, 0.05), 4)
    
    inputs.comp_confidence = new_conf
    warn(log, "tiers.plausibility_gate_tripped",
         comps_end_user=comps_end_user,
         category=inputs.category,
         tld=inputs.tld,
         category_floor=category_floor,
         appraisal=appraisal.value if appraisal else None,
         original_confidence=original_conf,
         forced_confidence=new_conf,
         triggers=len(triggered))

    return (
        f"plausibility gate: {reasons}. The comp set is likely misidentified, not the "
        f"market. Confidence forced down from {original_conf:.2f} to {new_conf:.2f} and "
        f"ranges widened accordingly. Never silently emit an implausible number."
    )



def enforce_tier_order(tiers: TierSet) -> list[str]:
    """Clamp any tier that ended up below the one beneath it.

    Multiplicative adjustments can invert the chain — a badly deflated
    end-user price can land under an auction price that a hot trend
    inflated, and a modest funding multiplier can put a targeted lead price
    under a BIN ask that a scarcity premium lifted. Clamping keeps the
    output coherent; recording it keeps the clamp from hiding the fact that
    the adjustments disagreed.

    Called again after the conditional tiers are attached, since those are
    priced separately and can invert against the market tiers below them.
    """
    notes: list[str] = []
    previous: tuple[str, PriceTier] | None = None
    for name, tier in tiers.ordered_populated():
        if previous and tier.point < previous[1].point:
            warn(log, "tier.order_clamped", tier=name, computed=tier.point,
                 raised_to=previous[1].point, below=previous[0])
            notes.append(
                f"{name} (${tier.point:,}) computed below {previous[0]} (${previous[1].point:,}); "
                f"raised to match — the adjustments on these two tiers disagreed"
            )
            raised = previous[1].point
            ratio = raised / max(tier.point, 1)
            tier.point = raised
            tier.high = max(tier.high, raised)
            tier.low = min(tier.low, raised)
            tier.inflationary_factors.append(
                f"raised to {previous[0]} to keep the tier order coherent (×{ratio:.2f})"
            )
        previous = (name, tier)
    return notes
