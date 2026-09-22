"""
Offline regression eval for the tier engine.
Run with: uv run python -m portfolio_agent.evals.test_tiers

Two kinds of check, and both are needed:

  - **Exact outputs** from a fixed evidence bundle. These catch a changed
    constant or a reordered multiplier, which a range check would sail past.
    If one changes because you recalibrated something deliberately, update it
    here in the same commit and say why.
  - **Properties that must hold for every domain**, checked across the whole
    golden set through the real graph in mock mode: the tier ordering
    invariant, the typo collapse, and the rule that an unfilled tier is
    always explained.

No network, no keys, no model.
"""

import asyncio
import statistics
import sys
from datetime import date

from portfolio_agent import config
from portfolio_agent.evals.golden_set import GOLDEN_SET, TYPO_REGRESSION_PAIRS
from portfolio_agent.pricing import tier_engine as te
from portfolio_agent.schema.valuation_output import (
    TIER_ORDER,
    BrandScore,
    IndependentAppraisal,
    ValuationOutput,
)

# ── The fixture bundle ──────────────────────────────────────────────────────
# Eight sales across auction venues (GoDaddy, NameJet, DropCatch, Dynadot)
# and brokered ones (Sedo, Afternic), spread over 10 months and most of an
# order of magnitude — wide and right-skewed, like real comp sets.

BASE_COMPS = [
    {"domain": "alphaone.com", "price": 9000.0, "date": "2026-06-12", "marketplace": "GoDaddy"},
    {"domain": "betatwo.com", "price": 12500.0, "date": "2026-04-02", "marketplace": "NameJet"},
    {"domain": "gammathree.com", "price": 15000.0, "date": "2026-07-20", "marketplace": "DropCatch"},
    {"domain": "deltafour.com", "price": 18000.0, "date": "2026-02-11", "marketplace": "GoDaddy"},
    {"domain": "epsilon5.com", "price": 24000.0, "date": "2026-05-30", "marketplace": "Dynadot"},
    {"domain": "zetasix.com", "price": 31000.0, "date": "2026-01-15", "marketplace": "Sedo"},
    {"domain": "etaseven.com", "price": 46000.0, "date": "2026-03-08", "marketplace": "Afternic"},
    {"domain": "thetaeight.com", "price": 70000.0, "date": "2025-12-01", "marketplace": "NameJet"},
]

FIXTURE_TODAY = date(2026, 9, 21)

# (low, point, high, sale_probability)
EXPECTED_BASELINE = {
    "floor": (4059, 8482, 18660, 0.595),
    "auction": (10473, 18480, 40656, None),
    "end_user": (30244, 63192, 139022, None),
    "list_bin": (43551, 90996, 200191, 0.1592),
}

# Floor probability fell from 0.738 to 0.595 when the vendor's number
# stopped being the sole author of it. The old formula multiplied the
# vendor figure by up to 1.025, so it could publish a HIGHER confidence
# than the vendor itself held; the blend pulls a 72% wholesale claim on an
# illiquid "combination" domain toward what its category actually clears at.

# Points moved as well as bands. The anchors are weighted percentiles now,
# and midpoint interpolation does not agree with statistics.quantiles away
# from the median (p75 of BASE_COMPS is 38,500 rather than 34,750). That is
# a convention change forced by carrying weights, not an accuracy claim.

# The same bundle with a severe misspelling. Every tier scales by 0.02.
EXPECTED_WITH_TYPO = {
    "floor": (81, 170, 374, 0.595),
    "auction": (210, 370, 814, None),
    "end_user": (605, 1264, 2781, None),
    "list_bin": (871, 1820, 4004, 0.1592),
}

# The same comps with no brand score, no trend reading, no SEO and no keyword
# volume — the floor of what the engine can do on comps alone.
EXPECTED_BARE = {
    "floor": (3625, 7574, 16663, 0.595),
    "auction": (9351, 16500, 36300, None),
    "end_user": (21190, 44275, 97405, None),
    "list_bin": (29666, 61985, 136367, 0.172),
}

EXPECTED_LIQUIDITY = 0.55
EXPECTED_SCARCITY = 0.68
# Wider than the interquartile band this replaced, in both directions, and
# the high sits exactly on TIER_BAND_LIMITS because BASE_COMPS is genuinely
# dispersed — p90/median is 2.99 here. Across 100 random comp sets the clamp
# binds in 5, so the band still tracks dispersion rather than flattening.
EXPECTED_BANDS = (0.4786, 2.2)


def fixture_inputs(**overrides) -> te.TierInputs:
    kwargs = dict(
        domain="northwind.com",
        sld="northwind",
        tld="com",
        category="combination",
        comps=BASE_COMPS,
        brand_score=BrandScore(
            total=72.0,
            components={"structural": 60.0, "phonetic": 90.0, "memorability": 70.0, "sentiment": 70.0},
            weights={"structural": 0.3, "phonetic": 0.25, "memorability": 0.25, "sentiment": 0.2},
        ),
        trend={"trend_velocity": 0.4, "matched_rules": ["two-word .com brandable"], "staleness_warning": None},
        seo={"authority_band": "developing", "domain_rating": 22.0},
        base_appraisal={"sales_probability_wholesale": 0.72, "sales_probability": 0.4, "go_value": 9000},
        keyword_stats={"global_monthly_searches": 40000, "exact_keyword_active_tld_count": 45},
        phonetic_score=90.0,
        sentiment_score=70.0,
        typo_deflator=1.0,
        evidence_refs={
            "comps": ["ev4"], "trends": ["ev6"], "brand_score": ["ev3"],
            "base_appraisal": ["ev2"], "seo": ["ev5"], "keyword_demand": ["ev5"],
        },
        today=FIXTURE_TODAY,
    )
    kwargs.update(overrides)
    # The tests express an appraisal as a bare number because that is what
    # the case is about; the engine wants the reconciled object.
    raw = kwargs.get("independent_appraisal")
    if isinstance(raw, (int, float)):
        kwargs["independent_appraisal"] = IndependentAppraisal(
            value=int(raw), source="estibot", all_baselines={"estibot": int(raw)},
            method="single baseline")
    return te.TierInputs(**kwargs)


class Checks:
    def __init__(self):
        self.total = 0
        self.failures: list[str] = []

    def expect(self, condition, message: str):
        self.total += 1
        if not condition:
            self.failures.append(message)


def _compare(c: Checks, label: str, tiers, expected: dict):
    for name, (low, point, high, probability) in expected.items():
        tier = getattr(tiers, name)
        if tier is None:
            c.expect(False, f"{label}: tier '{name}' should be populated")
            continue
        actual = (tier.low, tier.point, tier.high, tier.sale_probability)
        c.expect(actual == (low, point, high, probability),
                 f"{label}: {name} = {actual}, expected {(low, point, high, probability)}")


# ── Exact-output checks ─────────────────────────────────────────────────────


def check_comp_weights_reach_the_price(c: Checks):
    """The regression this exists for: they did not.

    The comp ladder sets weights with care — an exact comp counts double, an
    own-book sale one and a half, a cross-band comp 0.15 — and the tier
    engine read the price and dropped the weight, so none of it reached a
    number. Swinging every weight from 0.01 to 9.0 moved no tier by a single
    dollar. The F7 comp-quality banding was live in the comp set and inert
    in the price.

    Asserted as a behaviour rather than a value so it cannot decay again.
    """
    cheap = [c_["domain"] for c_ in BASE_COMPS if c_["price"] <= 18_000]

    def weighted(heavy_if_cheap: bool):
        return [dict(comp, weight=(9.0 if (comp["domain"] in cheap) == heavy_if_cheap else 0.15))
                for comp in BASE_COMPS]

    cheap_heavy, _ = te.compute_market_tiers(fixture_inputs(comps=weighted(True)))
    dear_heavy, _ = te.compute_market_tiers(fixture_inputs(comps=weighted(False)))
    c.expect(cheap_heavy.auction.point < dear_heavy.auction.point,
             f"weighting the cheap comps heavily must lower the auction price: "
             f"{cheap_heavy.auction.point} vs {dear_heavy.auction.point}")
    c.expect(cheap_heavy.end_user.point < dear_heavy.end_user.point,
             f"weights must reach the end-user tier too: "
             f"{cheap_heavy.end_user.point} vs {dear_heavy.end_user.point}")

    # And weighting must be a no-op where there is no weighting to apply, so
    # that turning it on did not quietly reprice every uniform comp set.
    unweighted, _ = te.compute_market_tiers(fixture_inputs(comps=BASE_COMPS))
    uniform, _ = te.compute_market_tiers(
        fixture_inputs(comps=[dict(comp, weight=1.0) for comp in BASE_COMPS]))
    c.expect(unweighted.auction.point == uniform.auction.point,
             "an explicit uniform weight must price identically to no weight at all")

    # A weighted median of an evenly weighted set is the ordinary median.
    stats = te.comp_stats(BASE_COMPS)
    c.expect(stats.median == statistics.median(te._prices(BASE_COMPS)),
             f"weighted median {stats.median} must match the plain median on uniform weights")


def check_disagreement_names_the_outlier(c: Checks):
    """F14. Two numbers and a shrug is not advice.

    The conflict was detected and then handed over symmetrically, leaving
    the reader to adjudicate with less information than the engine had. The
    engine knows how its own comp set was built, and since F12 it knows
    whether the vendors agree with EACH OTHER — an appraisal whose two
    baselines are 40x apart is not one estimate to argue against.
    """
    strong_comps = [dict(comp) for comp in BASE_COMPS] * 3

    # Vendors agree, comps are a thin cohort -> the comps are the weaker side.
    _, warnings = te.compute_market_tiers(fixture_inputs(
        comp_basis="cohort", comp_confidence=0.45,
        independent_appraisal=IndependentAppraisal(
            value=3_000_000, source="median of 2 baselines",
            all_baselines={"estibot": 2_800_000, "godaddy_govalue": 3_200_000}, method="median")))
    note = next((w for w in warnings if "appraisal conflict" in w), "")
    c.expect("comp set is the weaker number" in note,
             f"a thin cohort against agreeing vendors must name the comps as weaker: {note[:160]}")
    c.expect("cohort" in note and "0.45" in note,
             "and must give its grounds rather than asserting a verdict")

    # Vendors 42x apart, comps exact and deep -> the appraisal is weaker.
    _, warnings = te.compute_market_tiers(fixture_inputs(
        comp_basis="exact", comp_confidence=0.95, comps=strong_comps,
        independent_appraisal=IndependentAppraisal(
            value=800, source="median of 2 baselines",
            all_baselines={"estibot": 120, "godaddy_govalue": 5_000}, method="median")))
    note = next((w for w in warnings if "appraisal conflict" in w), "")
    c.expect("appraisal is the weaker number" in note,
             f"vendors 42x apart against deep exact comps must name the appraisal: {note[:160]}")
    c.expect("disagree with each other" in note,
             "the vendors' disagreement with each other is the strongest available signal")

    # And the verdict must not argue against itself. Both directional
    # explanations are reasons to distrust the COMPS, so neither belongs on
    # a note that just called the comps the stronger side.
    c.expect("check the individual sales before relying on them" not in note,
             f"the note tells the reader to trust the comps and distrust them in one breath: {note}")
    c.expect("describe a different market" not in note,
             f"a comps-favouring verdict must not carry a comps-doubting explanation: {note}")

    # A disagreement inside the threshold stays silent whatever the grounds.
    _, warnings = te.compute_market_tiers(fixture_inputs(
        comp_basis="cohort", comp_confidence=0.45,
        independent_appraisal=IndependentAppraisal(
            value=70_000, source="estibot", all_baselines={"estibot": 70_000},
            method="single baseline")))
    c.expect(not any("appraisal conflict" in w for w in warnings),
             "ordinary disagreement must not produce an adjudication")


def check_reported_comps_can_produce_the_price(c: Checks):
    """The comp set in the report must be the one that set the tiers.

    hello.com listed seven keyword sales from $142 to $1,237 as its
    evidence and then an end-user price of $3,998,344. Both numbers were
    real: the ladder found those sales, and the tiers were re-anchored on
    comparable acquisitions that the report never showed. The reader was
    left holding a comp set that cannot produce the price beside it, which
    is worse than a wrong number because it looks like arithmetic.
    """
    was_mock = config.MOCK_MODE
    config.MOCK_MODE = True
    try:
        output = _value("hello.com")
    finally:
        config.MOCK_MODE = was_mock

    end_user = output.tiers.end_user
    c.expect(end_user is not None, "hello.com must price an end-user tier")
    c.expect(output.comp_basis.value == "category_defining",
             f"a re-anchored valuation must say so in its basis, got {output.comp_basis.value}")
    c.expect(output.comp_set, "a re-anchored valuation must still publish its comps")

    prices = [comp.price for comp in output.comp_set]
    if prices and end_user is not None:
        # Not an exact reconstruction — adjustments and premiums sit between
        # them — but the price must live inside the comps' own range rather
        # than three orders of magnitude above it.
        c.expect(min(prices) <= end_user.point <= max(prices) * 3,
                 f"end_user ${end_user.point:,} sits outside the range its own comps "
                 f"(${min(prices):,}-${max(prices):,}) could produce")
        c.expect(all(comp.source == "category_acquisitions" for comp in output.comp_set),
                 "the published comps must be the acquisitions, not the keyword sales set aside")
        c.expect(all(comp.adjustment < 1.0 and comp.adjustment_reason for comp in output.comp_set),
                 "each published comp must carry the discount that brought it down, and why")


def check_tiers_cite_what_priced_them(c: Checks):
    """No tier may cite a lookup that failed, or one it did not use.

    On hello.com every tier cited fetch_similar_sales as its comp evidence.
    That tool had crashed, and the tiers had been re-anchored on comparable
    acquisitions that appeared nowhere in the evidence list — so five prices
    named a failed lookup as their provenance and none named their actual
    source.
    """
    was_mock = config.MOCK_MODE
    config.MOCK_MODE = True
    try:
        output = _value("hello.com")
    finally:
        config.MOCK_MODE = was_mock

    by_id = {item.id: item for item in output.evidence}
    acquisitions = [item for item in output.evidence if item.tool == "category_acquisitions"]
    c.expect(acquisitions, "a re-anchored valuation must publish its acquisitions as evidence")

    for name in TIER_ORDER:
        tier = getattr(output.tiers, name)
        if tier is None:
            continue
        c.expect(tier.evidence_refs, f"{name} cites nothing at all")
        for ref in tier.evidence_refs:
            item = by_id.get(ref)
            c.expect(item is not None, f"{name} cites {ref}, which is in no evidence item")
            if item is not None:
                c.expect(item.ok,
                         f"{name} cites {ref} ({item.tool}), which FAILED — a crashed lookup is "
                         f"not evidence for a price")
        if acquisitions:
            c.expect(acquisitions[0].id in tier.evidence_refs,
                     f"{name} does not cite the acquisitions it was priced from")


def check_probability_is_blended_not_borrowed(c: Checks):
    """A vendor's probability must not be the sole author of ours.

    The published figure is one a seller acts on, and these models have been
    wrong by orders of magnitude on exactly the domains where it matters —
    the same vendor put $25,000 on hello.com against comps supporting seven
    figures. Blending keeps their advantage without letting one bad reading
    carry the number.
    """
    def published(vendor, category):
        i = fixture_inputs(category=category,
                           base_appraisal={"sales_probability_wholesale": vendor, "go_value": 9000})
        return te._floor_probability(i, te.liquidity_score(i))

    # A vendor number at either extreme must be pulled toward our reading.
    c.expect(published(0.02, "acronym") > 0.02 * 1.5,
             f"an implausibly low vendor figure on a liquid acronym must be tempered upward, "
             f"got {published(0.02, 'acronym')}")
    c.expect(published(0.95, "geo") < 0.95,
             f"a very high vendor figure on an illiquid geo must be tempered downward, "
             f"got {published(0.95, 'geo')}")

    # The category must matter, which is the point of having the table.
    c.expect(published(0.40, "acronym") > published(0.40, "geo"),
             "the same vendor figure must not imply the same odds for an acronym and a geo")

    # The vendor still leads — this is a blend, not a replacement.
    low, high = published(0.10, "generic"), published(0.90, "generic")
    c.expect(high - low > 0.5,
             f"the vendor must still dominate the blend: {low} vs {high}")

    # And no vendor figure still means no published probability.
    i = fixture_inputs(base_appraisal={"go_value": 9000})
    c.expect(te._floor_probability(i, te.liquidity_score(i)) is None,
             "with no vendor baseline the probability must stay None, not fall back to ours alone")

    # The liquidity table is configuration, not a literal in the pricing code.
    c.expect(set(config.CATEGORY_LIQUIDITY) >= {"acronym", "generic", "geo", "brandable"},
             "the category liquidity table must live in config")


def check_baseline(c: Checks):
    tiers, warnings = te.compute_market_tiers(fixture_inputs())
    _compare(c, "baseline", tiers, EXPECTED_BASELINE)

    c.expect(te.liquidity_score(fixture_inputs()) == EXPECTED_LIQUIDITY,
             f"liquidity = {te.liquidity_score(fixture_inputs())}, expected {EXPECTED_LIQUIDITY}")
    c.expect(te._scarcity_score(fixture_inputs()) == EXPECTED_SCARCITY,
             f"scarcity = {te._scarcity_score(fixture_inputs())}, expected {EXPECTED_SCARCITY}")
    c.expect(te.band_ratios(te.comp_stats(BASE_COMPS)) == EXPECTED_BANDS,
             f"band ratios = {te.band_ratios(te.comp_stats(BASE_COMPS))}, expected {EXPECTED_BANDS}")

    # Every adjustment that fired must be recorded with its multiplier.
    end_user = tiers.end_user
    recorded = " ".join(end_user.inflationary_factors + end_user.deflationary_factors)
    for expected_factor in ("brand score", "monthly searches", ".com", "SEO authority", "trend velocity"):
        c.expect(expected_factor in recorded,
                 f"end_user should record the '{expected_factor}' adjustment, got: {recorded}")
    c.expect(all("×1.00" not in f for f in end_user.inflationary_factors + end_user.deflationary_factors),
             "a no-op multiplier must not be recorded as a factor")

    c.expect(any("trademark" in w for w in warnings),
             "the trademark position must be stated, not silently skipped")
    # The screen may now return a verdict, but never a clearance: it checks
    # famous marks and has run no search, so a clean result means only that
    # the obvious wire was not tripped. The full contract is asserted in
    # evals/test_trademark.py.
    screened = te.check_trademark_conflict("northwind.com")
    c.expect(screened["status"] == "no_obvious_conflict",
             f"an ordinary name must pass the famous-mark screen, got {screened['status']}")
    c.expect(screened["searched"] is False,
             "the screen must not claim a search it did not run")
    c.expect(screened["deflator"] == 1.0,
             "and a clean screen must not deflate the tier")

    # Every tier must cite evidence it actually used.
    for name, tier in tiers.ordered_populated():
        c.expect(bool(tier.evidence_refs), f"tier '{name}' cites no evidence")


def check_typo_collapse(c: Checks):
    tiers, _ = te.compute_market_tiers(fixture_inputs(typo_deflator=0.02))
    _compare(c, "typo", tiers, EXPECTED_WITH_TYPO)

    baseline, _ = te.compute_market_tiers(fixture_inputs())
    for name in ("floor", "auction", "end_user", "list_bin"):
        clean = getattr(baseline, name).point
        deflated = getattr(tiers, name).point
        c.expect(deflated < clean * 0.05,
                 f"{name} should collapse to under 5% of its clean value: {deflated} vs {clean}")

    recorded = " ".join(tiers.auction.deflationary_factors)
    c.expect("misspelling" in recorded, f"the deflator must be recorded as a factor, got: {recorded}")


def check_bare_evidence(c: Checks):
    """Comps alone still produce all four tiers — the signals are
    adjustments, not prerequisites."""
    tiers, _ = te.compute_market_tiers(
        fixture_inputs(trend=None, brand_score=None, seo=None,
                       phonetic_score=None, sentiment_score=None, keyword_stats={})
    )
    _compare(c, "bare", tiers, EXPECTED_BARE)


def check_insufficient_comps(c: Checks):
    tiers, warnings = te.compute_market_tiers(fixture_inputs(comps=BASE_COMPS[:2]))
    c.expect(tiers.populated() == {}, f"two comps must not produce tiers, got {list(tiers.populated())}")
    for name in ("floor", "auction", "end_user", "list_bin"):
        c.expect(any(w.startswith(f"{name}:") for w in warnings),
                 f"an unfilled '{name}' tier must be explained in warnings")

    tiers, _ = te.compute_market_tiers(fixture_inputs(comps=[]))
    c.expect(tiers.populated() == {}, "no comps must not produce tiers")


def check_probability_curve(c: Checks):
    """The inverse relationship is the whole point of publishing a BIN price
    next to an end-user price."""
    markups = [1.0, 1.25, 1.5, 2.0, 3.0]
    probabilities = [te.sale_probability_at_markup(m) for m in markups]
    c.expect(probabilities == sorted(probabilities, reverse=True),
             f"sale probability must fall as markup rises, got {list(zip(markups, probabilities))}")
    c.expect(te.sale_probability_at_markup(0.5) == te.sale_probability_at_markup(1.0),
             "the curve must clamp below its first point")
    c.expect(te.sale_probability_at_markup(50.0) == te.sale_probability_at_markup(3.0),
             "the curve must clamp above its last point")
    c.expect(te.sale_probability_at_markup(1.4) == 0.172, "the curve must interpolate between points")

    # Only floor and list_bin carry a probability (APVA Table 4.1.1).
    tiers, _ = te.compute_market_tiers(fixture_inputs())
    c.expect(tiers.auction.sale_probability is None and tiers.end_user.sale_probability is None,
             "auction and end_user must not claim a sale probability")
    c.expect(tiers.floor.sale_probability is not None and tiers.list_bin.sale_probability is not None,
             "floor and list_bin must carry a sale probability")

    # Without a baseline appraisal there is no probability to report, and a
    # guessed one is exactly the number a seller would act on.
    no_appraisal, _ = te.compute_market_tiers(fixture_inputs(base_appraisal=None))
    c.expect(no_appraisal.floor.sale_probability is None,
             "floor probability must be None when no baseline appraisal exists")

    # No tier may advertise certainty. A liquidity adjustment on a high
    # baseline can arithmetically exceed 1.0, and "100% chance of sale" is a
    # promise rather than an estimate.
    certain = te.compute_market_tiers(
        fixture_inputs(base_appraisal={"sales_probability_wholesale": 1.0}, sld="abc", category="acronym")
    )[0]
    c.expect(certain.floor.sale_probability is not None
             and certain.floor.sale_probability <= config.MAX_REPORTED_SALE_PROBABILITY,
             f"floor probability must stay under {config.MAX_REPORTED_SALE_PROBABILITY}, "
             f"got {certain.floor.sale_probability}")


def check_signal_direction(c: Checks):
    """Each signal must move the price the way the spec says it does."""
    baseline, _ = te.compute_market_tiers(fixture_inputs())

    hot, _ = te.compute_market_tiers(fixture_inputs(trend={"trend_velocity": 0.9, "matched_rules": ["hot"]}))
    cold, _ = te.compute_market_tiers(fixture_inputs(trend={"trend_velocity": -0.9, "matched_rules": ["risk"]}))
    c.expect(hot.auction.point > baseline.auction.point > cold.auction.point,
             "trend velocity must move the auction price in its own direction")

    strong = BrandScore(total=95.0, components={"structural": 95.0}, weights={"structural": 1.0})
    weak = BrandScore(total=20.0, components={"structural": 20.0}, weights={"structural": 1.0})
    c.expect(te.compute_market_tiers(fixture_inputs(brand_score=strong))[0].end_user.point
             > te.compute_market_tiers(fixture_inputs(brand_score=weak))[0].end_user.point,
             "a higher brand score must lift the end-user price")

    c.expect(te.compute_market_tiers(fixture_inputs(tld="com"))[0].end_user.point
             > te.compute_market_tiers(fixture_inputs(tld="net"))[0].end_user.point,
             ".com must carry a premium over a non-premium TLD at the end-user tier")

    c.expect(te.compute_market_tiers(fixture_inputs(sentiment_score=10.0))[0].end_user.point
             < baseline.end_user.point,
             "negative keyword sentiment must deflate the end-user price")
    c.expect(te.compute_market_tiers(fixture_inputs(phonetic_score=20.0))[0].end_user.point
             < baseline.end_user.point,
             "poor pronounceability must deflate the end-user price")

    liquid = te.liquidity_score(fixture_inputs(sld="abc", category="acronym"))
    illiquid = te.liquidity_score(fixture_inputs(sld="averylongbrandablename", category="brandable"))
    c.expect(liquid > illiquid, f"a short acronym must be more liquid than a long brandable: {liquid} vs {illiquid}")


def check_appraisal_disagreement(c: Checks):
    """An independent appraisal that contradicts the comps must say so.

    The case this exists for is real and was found in production: a live
    valuation of insurance.com priced it at $8,970 from substring comps
    while the appraiser it had already queried said $36.49M, and the two
    never met. A silent wrong answer is worse than a loud uncertain one.
    """
    baseline, _ = te.compute_market_tiers(fixture_inputs())
    reference = baseline.end_user.point

    # Agreement, and near-misses on either side of the threshold, stay quiet.
    for appraisal in (reference, int(reference * 3), int(reference / 3), None, 0):
        tiers, warnings = te.compute_market_tiers(fixture_inputs(independent_appraisal=appraisal))
        c.expect(not any("appraisal conflict" in w for w in warnings),
                 f"an appraisal of {appraisal} is ordinary disagreement and must not warn")

    # An order of magnitude, in either direction, must.
    for appraisal, direction in ((reference * 4000, "far above"), (max(1, reference // 50), "far below")):
        tiers, warnings = te.compute_market_tiers(fixture_inputs(independent_appraisal=appraisal))
        conflict = [w for w in warnings if "appraisal conflict" in w]
        c.expect(len(conflict) == 1, f"an appraisal {direction} the comps must produce exactly one warning")
        if conflict:
            c.expect(direction in conflict[0], f"the warning must say which way the gap runs: {conflict[0][:80]}")
            c.expect(f"{appraisal:,}" in conflict[0] and f"{reference:,}" in conflict[0],
                     "the warning must quote both numbers so the reader can judge")

        # The whole point: it changes nothing.
        c.expect(tiers.end_user.point == reference,
                 "an appraisal conflict must NOT move a price — comps remain the anchor")
        c.expect(tiers.model_dump() == baseline.model_dump(),
                 "no tier may differ from the run without the appraisal")

    # It needs a tier to compare against.
    _, warnings = te.compute_market_tiers(
        fixture_inputs(comps=BASE_COMPS[:2], independent_appraisal=99_000_000)
    )
    c.expect(not any("appraisal conflict" in w for w in warnings),
             "with no tiers computed there is nothing to disagree with")


def check_order_enforced(c: Checks):
    """Multiplicative adjustments can invert the chain; the engine must
    clamp and say that it did."""
    inverted = fixture_inputs(
        trend={"trend_velocity": 1.0, "matched_rules": ["max hot"]},  # inflates auction
        brand_score=BrandScore(total=0.0, components={"structural": 0.0}, weights={"structural": 1.0}),
        sentiment_score=0.0,
        phonetic_score=0.0,
        tld="net",
        keyword_stats={},
        seo=None,
    )
    tiers, warnings = te.compute_market_tiers(inverted)
    c.expect(tiers.monotonicity_violations() == [],
             f"the engine must leave no inversions behind: {tiers.monotonicity_violations()}")
    c.expect(any("raised to match" in w for w in warnings),
             f"a clamped inversion must be recorded in warnings, got {warnings}")


# ── Property checks across the whole golden set ─────────────────────────────


def _value(domain: str) -> ValuationOutput:
    from portfolio_agent.nodes.classify import classify
    from portfolio_agent.nodes.gather_evidence import gather_evidence
    from portfolio_agent.nodes.synthesize import synthesize

    state = {"domain": domain}
    state.update(classify(state))
    state.update(asyncio.run(gather_evidence(state)))
    state.update(asyncio.run(synthesize(state)))
    return ValuationOutput.model_validate(state["valuation"])


# Domains whose real sale price is public, used to measure whether the
# category-killer re-anchoring is in the right universe. Not a precision
# test — seventeen press-reported sales cannot support one — but the
# difference between "$9,120" and "within an order of magnitude" is the
# whole value of the feature.
KNOWN_SALES = {
    "insurance.com": 35_600_000,
    "voice.com": 30_000_000,
    "hotels.com": 11_000_000,
    "diamond.com": 7_500_000,
    "beer.com": 7_000_000,
    "toys.com": 5_100_000,
    "fund.com": 9_999_950,
}

# Must NOT be re-anchored on eight-figure acquisitions. carinsurance.com is
# the important one: it passes is_category_defining as an exact category
# term, and re-anchoring it priced a low-six-figure EMD at $45.9M.
MUST_NOT_REANCHOR = ("carinsurance.com", "cheapinsurancequotes.com", "crusoe.com",
                     "zylantix.com", "austinplumbing.com")


def check_category_killer_anchoring(c: Checks):
    """A single-word .com must be priced against other single-word .com
    sales, not against names that merely contain the word."""
    from portfolio_agent.knowledge.ma_cases import comparable_cases
    from portfolio_agent.nodes.synthesize import _category_comps

    was_mock = config.MOCK_MODE
    config.MOCK_MODE = True
    try:
        for domain, real in KNOWN_SALES.items():
            output = _value(domain)
            end_user = output.tiers.end_user
            strategic = output.tiers.strategic_ma
            if end_user is None:
                c.expect(False, f"{domain}: no end-user price")
                continue

            # Compared against the STRATEGIC tier, because that is what
            # these sales were. Every price in KNOWN_SALES is a completed
            # acquisition in which the domain became a company's identity,
            # and this assertion used to hold end_user to them — which is
            # only satisfiable if end_user quietly carries the strategic
            # premium. It did, and hello.com came out at $18.1M as a result.
            c.expect(strategic is not None,
                     f"{domain} is a documented strategic acquisition and must price that tier")
            if strategic is not None:
                ratio = strategic.point / real
                c.expect(0.1 <= ratio <= 10,
                         f"{domain}: strategic ${strategic.point:,} against a real ${real:,} "
                         f"acquisition is {ratio:.2f}x — outside an order of magnitude")

                # The invariant that broke: an end-user price is a sale to a
                # business that will use the domain, not an acquisition of
                # the company it identifies. It must sit materially below.
                c.expect(end_user.point < strategic.point,
                         f"{domain}: end_user ${end_user.point:,} is not below strategic "
                         f"${strategic.point:,} — the strategic premium is being charged twice")

            # And end_user must still be in the right universe: a
            # category-defining .com is a seven-figure asset, not a $9,120 one.
            eu_ratio = end_user.point / real
            c.expect(0.02 <= eu_ratio <= 10,
                     f"{domain}: end_user ${end_user.point:,} against a real ${real:,} sale is "
                     f"{eu_ratio:.2f}x — outside any defensible relationship")
            c.expect(end_user.point > 1_000_000,
                     f"{domain}: a category killer priced at ${end_user.point:,} is being valued "
                     "off substring comps again")
            c.expect(any("category-defining domain" in w for w in output.warnings),
                     f"{domain}: re-anchoring must be disclosed in warnings")
            c.expect(any("skews HIGH" in w for w in output.warnings),
                     f"{domain}: the comp set's survivorship bias must be stated")

        for domain in MUST_NOT_REANCHOR:
            output = _value(domain)
            c.expect(not any("category-defining domain:" in w for w in output.warnings),
                     f"{domain} must NOT be re-anchored on eight-figure acquisitions")
            end_user = output.tiers.end_user
            c.expect(end_user is None or end_user.point < 5_000_000,
                     f"{domain}: priced at ${end_user.point:,} — the comp set leaked to the wrong shape")
    finally:
        config.MOCK_MODE = was_mock

    # A domain is never its own comparable.
    cases = comparable_cases("insurance", "com", limit=5, exclude_self=True)
    c.expect(all(case.sld != "insurance" for case in cases),
             "insurance.com must not be used as a comparable for insurance.com")

    # The anchor selection must not be pre-sorted by price, or the sample is
    # biased high before any arithmetic runs.
    #
    # The invariant, which always holds: within a single length bucket the
    # citation ordering is non-increasing in price. That is the bias.
    ranked = comparable_cases("diamond", "com", limit=8, exclude_self=True, by_price=True)
    for earlier, later in zip(ranked, ranked[1:]):
        if abs(len(earlier.sld) - 7) == abs(len(later.sld) - 7):
            c.expect(earlier.price >= later.price,
                     f"by_price should rank {earlier.sld} above {later.sld} within a length bucket")

    # And where the bias changes which cases are SELECTED rather than merely
    # their order, the unbiased set must come out lower. For 'insurance' the
    # price ranking swaps in privatejet ($30.2M) over slots ($5.5M).
    def average(by_price: bool) -> float:
        cases = comparable_cases("insurance", "com", limit=5, exclude_self=True, by_price=by_price)
        return sum(case.price for case in cases) / len(cases)

    c.expect(average(by_price=False) < average(by_price=True),
             f"the unbiased selection should average below the price-ranked one: "
             f"${average(False):,.0f} vs ${average(True):,.0f}")


def check_marketplace_listings(c: Checks):
    """An ask moves the BIN tier and nothing else.

    The rule this protects: a listing is what a seller wants, a comp is what
    a buyer paid. Only the BIN tier is itself an ask, so only the BIN tier
    may read one. If a listing ever moves the floor, auction or end-user
    price, an anchor built from transactions has been contaminated by one
    seller's optimism.
    """
    baseline, _ = te.compute_market_tiers(fixture_inputs())

    for ask, label in ((baseline.end_user.point * 40, "far above"),
                       (max(1, baseline.end_user.point // 40), "far below")):
        listed = fixture_inputs(listings={"listed": True, "lowest_ask": ask, "has_live_auction": False})
        tiers, warnings = te.compute_market_tiers(listed)

        for name in ("floor", "auction", "end_user"):
            c.expect(getattr(tiers, name).point == getattr(baseline, name).point,
                     f"a listing {label} must NOT move the {name} tier — those are anchored on sales")

        c.expect(tiers.list_bin.point != baseline.list_bin.point,
                 f"a listing {label} should move the BIN tier, which is itself an ask")
        c.expect(any("listing conflict" in w for w in warnings),
                 f"a listing {label} the comps must be reported")
        c.expect(tiers.monotonicity_violations() == [],
                 f"a listing {label} must not leave the tiers incoherent")

    # The pull is capped: the BIN tier must never simply become the ask.
    huge = baseline.end_user.point * 40
    tiers, _ = te.compute_market_tiers(
        fixture_inputs(listings={"listed": True, "lowest_ask": huge, "has_live_auction": False})
    )
    c.expect(tiers.list_bin.point < huge,
             f"BIN (${tiers.list_bin.point:,}) must stay below a ${huge:,} ask — an ask is not a sale")

    # No listing, no adjustment and no noise.
    unlisted, warnings = te.compute_market_tiers(
        fixture_inputs(listings={"listed": False, "lowest_ask": None})
    )
    c.expect(unlisted.list_bin.point == baseline.list_bin.point,
             "an unlisted domain must price exactly as if the source were absent")
    c.expect(not any("listing" in w for w in warnings),
             "an unlisted domain must not produce a listing warning")

    # A modest ask agrees rather than conflicts, and says so without alarm.
    close = int(baseline.end_user.point * 1.2)
    tiers, warnings = te.compute_market_tiers(
        fixture_inputs(listings={"listed": True, "lowest_ask": close, "has_live_auction": True})
    )
    c.expect(not any("listing conflict" in w for w in warnings),
             "an ask near the comps price is agreement, not conflict")
    recorded = " ".join(tiers.list_bin.inflationary_factors + tiers.list_bin.deflationary_factors)
    c.expect("live auction bid" in recorded,
             f"a live auction bid must be named as such, not called an asking price: {recorded!r}")


def check_ask_is_never_a_comp(c: Checks):
    """An ask and a sale must be different TYPES, not merely different fields.

    A convention that listings stay out of the comps distribution is a
    convention someone breaks. `Listing` is a plain dataclass and `CompRef`
    is a pydantic model with no shared base, so the separation survives a
    careless `comps.extend(...)` — it raises instead of silently pricing a
    domain off one seller's hope.
    """
    from portfolio_agent.schema.valuation_output import CompRef
    from portfolio_agent.tools.marketplace import Listing

    ask = Listing(domain="consulting.ai", marketplace="godaddy", kind="buy_now", price=2_827_928)

    c.expect(not isinstance(ask, CompRef), "a Listing must not be a CompRef")
    c.expect(not issubclass(type(ask), type(CompRef)) if isinstance(CompRef, type) else True,
             "a Listing must not share a base with CompRef")

    try:
        CompRef(**{"domain": ask.domain, "price": ask.price})
        constructed = True
    except Exception:
        constructed = False
    c.expect(not constructed,
             "a CompRef must not be constructible from a Listing's fields alone — it requires a "
             "`source`, which an ask has no honest value for")

    # And the price math refuses one outright rather than coercing it.
    try:
        te._prices([ask])
        coerced = True
    except (TypeError, AttributeError):
        coerced = False
    c.expect(not coerced, "the price math must reject a Listing rather than read a price off it")


def check_ask_ceiling_and_gap(c: Checks):
    """The ask caps the BIN, reports a gap, and moves nothing else."""
    baseline, _ = te.compute_market_tiers(fixture_inputs())
    end_user = baseline.end_user.point

    # Above the end-user price: the BIN is capped at the ask.
    modest = int(end_user * 1.1)
    tiers, _ = te.compute_market_tiers(
        fixture_inputs(listings={"listed": True, "lowest_ask": modest, "has_live_auction": False})
    )
    c.expect(tiers.list_bin.point <= modest,
             f"a BIN of ${tiers.list_bin.point:,} above the ${modest:,} ask would sell nothing")
    c.expect(any("capped at the live marketplace ask" in f for f in tiers.list_bin.deflationary_factors),
             "the cap must be recorded as a factor")

    # Far above: still capped, still coherent, gap reported.
    huge = int(end_user * 60)
    tiers, warnings = te.compute_market_tiers(
        fixture_inputs(listings={"listed": True, "lowest_ask": huge, "has_live_auction": False})
    )
    c.expect(tiers.list_bin.point < huge, "the BIN must stay well under a conviction-priced ask")
    c.expect(tiers.monotonicity_violations() == [], "an extreme ask must not break the tier order")

    # Below the end-user price: NOT capped under its own anchor, and said so.
    cheap = int(end_user * 0.5)
    tiers, warnings = te.compute_market_tiers(
        fixture_inputs(listings={"listed": True, "lowest_ask": cheap, "has_live_auction": False})
    )
    c.expect(tiers.list_bin.point >= tiers.end_user.point,
             "a BIN must never be capped below the end-user price it is a premium on")
    # It changed no price, so it is a warning rather than a factor — a
    # factor now has to state the multiplier it applied.
    c.expect(any("BELOW the comps-implied" in w for w in warnings),
             "an ask under the comps must be reported as the disagreement it is")
    c.expect(not any("BELOW the comps-implied" in f for f in tiers.list_bin.deflationary_factors),
             "an observation that moved no price must not masquerade as a factor")


    # The gap and its label.
    for multiple, expected in ((1.5, "aligned"), (8.0, "ambitious"), (60.0, "conviction")):
        inputs = fixture_inputs(
            listings={"listed": True, "lowest_ask": int(end_user * multiple), "has_live_auction": False}
        )
        tiers, _ = te.compute_market_tiers(inputs)
        gap, label = te.ask_gap(inputs, tiers.end_user)
        c.expect(gap is not None and abs(gap - multiple) < 0.6,
                 f"ask_gap at {multiple}x should report about {multiple}, got {gap}")
        c.expect(label and expected in label,
                 f"a {multiple}x gap should be labeled {expected!r}, got {label!r}")

    gap, label = te.ask_gap(fixture_inputs(), baseline.end_user)
    c.expect(gap is None and label is None, "no listing means no gap, not a gap of zero")

    # Nothing else moves. This is the rule the whole separation exists for.
    for multiple in (0.5, 8.0, 60.0):
        tiers, _ = te.compute_market_tiers(
            fixture_inputs(listings={"listed": True, "lowest_ask": int(end_user * multiple)})
        )
        for name in ("floor", "auction", "end_user"):
            c.expect(getattr(tiers, name).point == getattr(baseline, name).point,
                     f"an ask at {multiple}x moved the {name} tier — those are anchored on sales")


def check_every_factor_states_its_effect(c: Checks):
    """The invariant, across every tier the engine can produce.

    A caveat the price ignores is worse than no caveat: it reads as
    diligence. The schema enforces this on construction, so this check is
    really asking whether the engine can still build a full ladder now that
    it must justify every line it prints.
    """
    import re

    pattern = re.compile(r"\(×\d+(?:\.\d+)?\)$")
    scenarios = {
        "baseline": fixture_inputs(),
        "retail-heavy": fixture_inputs(comps=RETAIL_COMPS),
        "with an ask": fixture_inputs(listings={"listed": True, "lowest_ask": 9_000_000}),
        "long name": fixture_inputs(tld="ai", structural={"char_length": 14}),
        "misspelled": fixture_inputs(typo_deflator=0.02),
        "with demand": fixture_inputs(demand={"best_offer": 40_000, "inquiries_12m": 9}),
    }
    for label, inputs in scenarios.items():
        tiers, _ = te.compute_market_tiers(inputs)
        for name, tier in tiers.ordered_populated():
            for factor in tier.inflationary_factors + tier.deflationary_factors:
                c.expect(pattern.search(factor.strip()),
                         f"{label}/{name}: factor without an applied multiplier: {factor[:70]!r}")


def check_offer_is_a_signal_not_a_comp(c: Checks):
    """An offer lifts the end-user tier, and never enters the distribution."""
    baseline, _ = te.compute_market_tiers(fixture_inputs())
    floor_estimate = baseline.auction.point * config.FLOOR_DISCOUNT_FROM_AUCTION

    strong = te.compute_market_tiers(
        fixture_inputs(demand={"best_offer": int(floor_estimate * 4), "inquiries_12m": 12})
    )[0]
    c.expect(strong.end_user.point > baseline.end_user.point,
             "a real offer above the quick-exit floor must lift the end-user price")
    c.expect(any("real offer" in f for f in strong.end_user.inflationary_factors),
             f"the offer must be named as the reason, got {strong.end_user.inflationary_factors}")

    # Capped: one offer is one buyer.
    ratio = strong.end_user.point / baseline.end_user.point
    c.expect(ratio <= config.INTERNAL_OFFER_MAX_LIFT + 0.01,
             f"the offer lift must be capped at {config.INTERNAL_OFFER_MAX_LIFT}, got {ratio:.2f}x")

    # A lowball offer is not evidence of value above a quick exit.
    weak = te.compute_market_tiers(
        fixture_inputs(demand={"best_offer": int(floor_estimate * 0.2), "inquiries_12m": 3})
    )[0]
    c.expect(weak.end_user.point == baseline.end_user.point,
             "an offer below the quick-exit floor must not lift anything")

    # It moves the end-user tier and nothing anchored purely on transactions.
    for name in ("floor", "auction"):
        c.expect(getattr(strong, name).point == getattr(baseline, name).point,
                 f"an offer must not move the {name} tier")

    # And it never becomes a comp.
    c.expect(len(te._prices(fixture_inputs(demand={"best_offer": 999_999}).comps))
             == len(te._prices(baseline_comps := BASE_COMPS)),
             "demand signals must not enter the comps distribution")


RETAIL_COMPS = [dict(c, marketplace="Sedo") for c in BASE_COMPS]


def check_venue_mix_is_priced_not_just_noted(c: Checks):
    """A recorded factor must be an applied factor.

    The auction anchor used to note "these comps include brokered sales that
    clear above auction" and then not adjust for it — a caveat the price
    ignored, which is worse than no caveat because it reads as diligence.
    """
    mixed, _ = te.compute_market_tiers(fixture_inputs())
    retail, _ = te.compute_market_tiers(fixture_inputs(comps=RETAIL_COMPS))

    c.expect(retail.auction.point < mixed.auction.point,
             f"an all-brokered comp set must discount the auction anchor: "
             f"${retail.auction.point:,} vs ${mixed.auction.point:,}")

    recorded = " ".join(retail.auction.inflationary_factors + retail.auction.deflationary_factors)
    c.expect("auction-venue sales" in recorded,
             f"the venue mix must be named, got {recorded[:120]!r}")
    c.expect("×" in recorded,
             "the venue factor must carry a multiplier — a factor with no multiplier is a no-op")

    # Enough genuine auction sales and there is no haircut at all.
    auction_only = [dict(c_, marketplace="GoDaddy") for c_ in BASE_COMPS]
    clean, _ = te.compute_market_tiers(fixture_inputs(comps=auction_only))
    clean_factors = " ".join(clean.auction.inflationary_factors + clean.auction.deflationary_factors)
    c.expect("auction-venue sales" not in clean_factors,
             "an all-auction comp set needs no venue caveat and no haircut")

    # The haircut scales with the mix rather than switching on. Share is
    # only meaningful on the FALLBACK path: when the venue filter holds, the
    # returned set is auction sales by construction and the share is 1.0.
    filtered, held, share = te.auction_comps(BASE_COMPS)
    c.expect(held and share == 1.0,
             f"a set with enough auction sales filters cleanly, got held={held} share={share}")

    _, held, share = te.auction_comps(RETAIL_COMPS)
    c.expect(not held and share == 0.0,
             f"an all-brokered set falls back with no auction share, got held={held} share={share}")

    # Two auction sales is below the minimum, so it falls back — but the
    # haircut should be lighter than for a set with none.
    barely = [dict(c_, marketplace="GoDaddy" if i < 2 else "Sedo") for i, c_ in enumerate(BASE_COMPS)]
    _, held, partial = te.auction_comps(barely)
    c.expect(not held and 0 < partial < 1,
             f"a set below the filter minimum should report a partial share, got {partial}")

    lighter, _ = te.compute_market_tiers(fixture_inputs(comps=barely))
    c.expect(lighter.auction.point > retail.auction.point,
             f"a partly-auction set should be discounted less than an all-brokered one: "
             f"${lighter.auction.point:,} vs ${retail.auction.point:,}")


def check_length_value_window(c: Checks):
    """Length is structural and applies to every category, not just brandables.

    accounting.ai carried no length factor anywhere because brand scoring is
    skipped for keyword domains and structural scoring went with it — while
    .ai records $45.2M of sales at ten characters or fewer against $1.8M
    above.
    """
    short = fixture_inputs(tld="ai", structural={"char_length": 6})
    boundary = fixture_inputs(tld="ai", structural={"char_length": 10})
    beyond = fixture_inputs(tld="ai", structural={"char_length": 14})

    c.expect(te.length_adjustment(short) == (1.0, None),
             "a name inside the window takes no length discount")

    factor, reason = te.length_adjustment(boundary)
    c.expect(factor < 1.0 and reason and "boundary" in reason,
             f"a name ON the boundary must be discounted and say so, got {factor} {reason!r}")

    beyond_factor, beyond_reason = te.length_adjustment(beyond)
    c.expect(beyond_factor < factor,
             f"beyond the window must cost more than sitting on it: {beyond_factor} vs {factor}")
    c.expect(beyond_reason and "value window" in beyond_reason, "the reason must name the window")

    # The window differs by extension — .com sustains longer names.
    dotcom_factor, _ = te.length_adjustment(fixture_inputs(tld="com", structural={"char_length": 10}))
    c.expect(dotcom_factor == 1.0,
             "ten characters is comfortably inside .com's window and must not be penalised")

    # And it reaches the tier, for a category that runs no brand score.
    with_length, _ = te.compute_market_tiers(
        fixture_inputs(tld="ai", category="keyword", brand_score=None, structural={"char_length": 14})
    )
    without, _ = te.compute_market_tiers(
        fixture_inputs(tld="ai", category="keyword", brand_score=None, structural={"char_length": 6})
    )
    c.expect(with_length.end_user.point < without.end_user.point,
             "the length factor must reach the end-user tier without a brand score present")
    c.expect(any("character" in f for f in with_length.end_user.deflationary_factors),
             "the length discount must be recorded as a factor")

    c.expect(te.length_adjustment(fixture_inputs(structural=None)) == (1.0, None),
             "no structural data means no length claim, not a penalty")


def check_golden_set_invariants(c: Checks):
    was_mock = config.MOCK_MODE
    config.MOCK_MODE = True
    try:
        for case in GOLDEN_SET:
            output = _value(case["domain"])

            c.expect(output.tiers.monotonicity_violations() == [],
                     f"{case['domain']}: floor <= auction <= end_user <= list_bin violated: "
                     f"{output.tiers.monotonicity_violations()}")

            populated = output.tiers.populated()
            c.expect(len(populated) >= 4,
                     f"{case['domain']}: expected all four market tiers, got {sorted(populated)}")

            for name, tier in populated.items():
                c.expect(tier.low <= tier.point <= tier.high,
                         f"{case['domain']}: {name} band is out of order")
                c.expect(bool(tier.evidence_refs),
                         f"{case['domain']}: {name} cites no evidence")

            for name in TIER_ORDER:
                if getattr(output.tiers, name) is None:
                    c.expect(any(name in w for w in output.warnings),
                             f"{case['domain']}: unfilled tier '{name}' is not explained")

        # The APVA §4.4 requirement, end to end through the real graph.
        for misspelled, correct in TYPO_REGRESSION_PAIRS:
            bad = _value(misspelled)
            good = _value(correct)
            for name in ("floor", "auction", "end_user", "list_bin"):
                bad_tier, good_tier = getattr(bad.tiers, name), getattr(good.tiers, name)
                if bad_tier is None or good_tier is None:
                    continue
                c.expect(bad_tier.point < good_tier.point * 0.10,
                         f"{misspelled}'s {name} (${bad_tier.point:,}) should collapse against "
                         f"{correct}'s (${good_tier.point:,})")
    finally:
        config.MOCK_MODE = was_mock


def main() -> int:
    c = Checks()
    for check in (
        check_comp_weights_reach_the_price,
        check_probability_is_blended_not_borrowed,
        check_reported_comps_can_produce_the_price,
        check_tiers_cite_what_priced_them,
        check_disagreement_names_the_outlier,
        check_baseline,
        check_typo_collapse,
        check_bare_evidence,
        check_insufficient_comps,
        check_probability_curve,
        check_signal_direction,
        check_appraisal_disagreement,
        check_category_killer_anchoring,
        check_marketplace_listings,
        check_ask_is_never_a_comp,
        check_ask_ceiling_and_gap,
        check_offer_is_a_signal_not_a_comp,
        check_venue_mix_is_priced_not_just_noted,
        check_length_value_window,
        check_every_factor_states_its_effect,
        check_order_enforced,
        check_golden_set_invariants,
    ):
        check(c)

    if c.failures:
        print(f"FAILED {len(c.failures)}/{c.total} checks:")
        for failure in c.failures:
            print(f"  - {failure}")
        return 1
    print(f"PASSED {c.total}/{c.total} checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
