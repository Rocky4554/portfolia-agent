"""
Offline regression eval for the comp-widening ladder.
Run with: uv run python -m portfolio_agent.evals.test_comp_ladder

The failure this exists to prevent is the one consulting.ai exposed: one
comparable sale, so nothing priced at all. Refusing to extrapolate from a
single sale was right; treating "no answer" as the only alternative to "a
confident answer" was not.

What is asserted, in order of importance:

  - The ladder stops at the STRONGEST rung that has enough comps, never a
    weaker one it happens to reach first.
  - A stronger rung's comps survive into a weaker rung's set. An exact sale
    is never discarded for being lonely.
  - Confidence falls monotonically as the basis weakens, and the tier ranges
    widen to match — a cohort price must LOOK like a wider claim.
  - `none` still happens. A ladder that always finds something has stopped
    being evidence.
"""

import sys
from datetime import date

from portfolio_agent import config
from portfolio_agent.pricing import tier_engine as te
from portfolio_agent.pricing.comp_ladder import LadderInputs, build_comp_set
from portfolio_agent.schema.valuation_output import CompBasis

TODAY = date(2026, 9, 21)

# The real consulting.ai situation, captured from a live run: one .ai sale
# of a domain containing the keyword, at $425.
EXACT_ONE = [{"domain": "primeconsulting.ai", "price": 425, "date": "2026-04-30", "marketplace": "Sedo"}]

# The exact name, sold elsewhere. This is the evidence a human appraiser
# reaches for first when the domain's own extension has no history.
CROSS_TLD = [
    {"domain": "consulting.com", "price": 600_000, "date": "2025-06-01", "marketplace": "Afternic"},
    {"domain": "consulting.co", "price": 45_000, "date": "2025-11-02", "marketplace": "Sedo"},
]

# Single dictionary words on .ai — the class, not the name.
COHORT = [
    {"domain": "advisor.ai", "price": 88_000, "date": "2026-02-10", "marketplace": "GoDaddy"},
    {"domain": "counsel.ai", "price": 42_000, "date": "2025-12-05", "marketplace": "Sedo"},
    {"domain": "strategy.ai", "price": 610_000, "date": "2026-05-20", "marketplace": "Private"},
    {"domain": "finance.ai", "price": 155_000, "date": "2026-01-08", "marketplace": "GoDaddy"},
]

# The percentile shape the keyword rung synthesizes from. A single average
# is one comp and can never reach the minimum, which is why the rung was
# unreachable for any single-keyword domain until it learned to read a
# distribution.
KEYWORD_STATS = {"consulting": {"p25": 1_900, "median": 2_900, "p75": 5_400, "total_sales": 372}}


def inputs(**overrides) -> LadderInputs:
    kwargs = dict(
        domain="consulting.ai", sld="consulting", tld="ai", category="keyword",
        keywords=["consulting"], today=TODAY,
    )
    kwargs.update(overrides)
    return LadderInputs(**kwargs)


class Checks:
    def __init__(self):
        self.total = 0
        self.failures: list[str] = []

    def expect(self, condition, message: str):
        self.total += 1
        if not condition:
            self.failures.append(message)


def check_rung_selection(c: Checks):
    """The ladder stops at the strongest rung with enough comps."""
    # Domains that genuinely CONTAIN the keyword. The exact rung rejects
    # anything that does not — a semantic neighbour is a cohort member, and
    # counting it as exact is what pinned forge.ai at confidence 1.0 on nine
    # compound names.
    plenty = [dict(r, domain=f"consulting{i}.ai") for i, r in enumerate(EXACT_ONE * 4)]
    result = build_comp_set(inputs(keyword_sales=plenty))
    c.expect(result.basis is CompBasis.EXACT, f"four exact comps should stop at exact, got {result.basis}")
    c.expect(result.confidence == 1.0, f"exact basis should be full confidence, got {result.confidence}")

    result = build_comp_set(inputs(keyword_sales=EXACT_ONE, cross_tld_sales=CROSS_TLD))
    c.expect(result.basis is CompBasis.SLD_CROSS_TLD,
             f"one exact + two cross-TLD should stop at sld_cross_tld, got {result.basis}")

    result = build_comp_set(inputs(keyword_sales=EXACT_ONE, cohort_sales=COHORT))
    c.expect(result.basis is CompBasis.COHORT,
             f"one exact + a cohort should stop at cohort, got {result.basis}")

    result = build_comp_set(inputs(keyword_stats=KEYWORD_STATS))
    c.expect(result.basis is CompBasis.KEYWORD_ONLY or result.basis is CompBasis.NONE,
             f"aggregates alone reach keyword_only at best, got {result.basis}")

    # The rung is chosen by strength, not by order of arrival: a full cohort
    # must not win when cross-TLD already sufficed.
    result = build_comp_set(inputs(keyword_sales=EXACT_ONE, cross_tld_sales=CROSS_TLD, cohort_sales=COHORT))
    c.expect(result.basis is CompBasis.SLD_CROSS_TLD,
             f"cross-TLD is stronger than cohort and must win, got {result.basis}")


def check_rungs_are_cumulative(c: Checks):
    """A stronger rung's comps survive into a weaker rung's set."""
    result = build_comp_set(inputs(keyword_sales=EXACT_ONE, cohort_sales=COHORT))
    domains = {comp.domain for comp in result.comps}
    c.expect("primeconsulting.ai" in domains,
             "the lone exact comp must survive into the cohort set — it was not wrong, only lonely")

    exact = next(comp for comp in result.comps if comp.domain == "primeconsulting.ai")
    c.expect(exact.weight == config.COMP_WEIGHT_EXACT,
             f"an exact comp must outweigh a cohort member, got weight {exact.weight}")

    weighted = result.weighted_prices()
    c.expect(weighted.count(425) == round(config.COMP_WEIGHT_EXACT),
             f"the exact comp should appear {config.COMP_WEIGHT_EXACT}x in the weighted distribution")


def check_tld_adjustment(c: Checks):
    """Cross-TLD comps are adjusted per comp, and say so."""
    result = build_comp_set(inputs(keyword_sales=EXACT_ONE, cross_tld_sales=CROSS_TLD))
    dotcom = next(comp for comp in result.comps if comp.domain == "consulting.com")

    c.expect(dotcom.adjustment < 1.0,
             f".com→.ai must discount, got ×{dotcom.adjustment}")
    c.expect(dotcom.adjusted_price < dotcom.price,
             f"the adjusted price must be below the sale price: {dotcom.adjusted_price} vs {dotcom.price}")
    c.expect(dotcom.adjustment_reason and "transfer" in dotcom.adjustment_reason,
             f"the adjustment must state itself, got {dotcom.adjustment_reason!r}")
    c.expect("prior" in (dotcom.adjustment_reason or "") or "dual sales" in (dotcom.adjustment_reason or ""),
             "the reason must say whether the ratio was measured or assumed")

    c.expect(any("other extensions" in note for note in result.notes),
             f"a cross-TLD basis must explain itself, got {result.notes}")


def check_confidence_monotonicity(c: Checks):
    """A better rung never yields lower confidence."""
    ladder = [
        (build_comp_set(inputs(keyword_sales=[
            dict(r, domain=f"consulting{i}.ai") for i, r in enumerate(EXACT_ONE * 4)
        ])), "exact"),
        (build_comp_set(inputs(keyword_sales=EXACT_ONE, cross_tld_sales=CROSS_TLD)), "cross_tld"),
        (build_comp_set(inputs(keyword_sales=EXACT_ONE, cohort_sales=COHORT)), "cohort"),
        (build_comp_set(inputs(keyword_stats=KEYWORD_STATS)), "keyword_only"),
    ]
    confidences = [result.confidence for result, _ in ladder]
    c.expect(confidences == sorted(confidences, reverse=True),
             f"confidence must fall as the basis weakens, got {list(zip([l for _, l in ladder], confidences))}")


def check_ranges_widen(c: Checks):
    """A weaker basis must LOOK weaker: wider bands, lower probabilities."""
    stats = te.comp_stats([{"domain": f"d{i}.com", "price": p}
                           for i, p in enumerate((400.0, 9_000.0, 150_000.0))])
    widths = []
    for confidence in (1.0, 0.6, 0.45, 0.25):
        low, high = te.band_ratios(stats, confidence)
        widths.append(high / low)
    c.expect(widths == sorted(widths),
             f"bands must widen as confidence falls, got {[round(w, 1) for w in widths]}")
    c.expect(widths[-1] > widths[0] * 2,
             f"a keyword-only band should be markedly wider than an exact one: {widths[0]:.1f} vs {widths[-1]:.1f}")

    # And the stated probability of clearing scales with it.
    c.expect(te._scaled(0.40, 0.25) < te._scaled(0.40, 1.0),
             "a probability at an inferred price must be lower than one at an observed price")
    c.expect(te._scaled(None, 0.5) is None, "no probability stays no probability")


def check_none_is_still_reachable(c: Checks):
    """A ladder that always finds something has stopped being evidence."""
    result = build_comp_set(inputs())
    c.expect(result.basis is CompBasis.NONE, f"no data at all must give basis none, got {result.basis}")
    c.expect(result.confidence == 0.0, "an empty basis has no confidence")
    c.expect(result.comps == [], "an empty basis has no comps")
    c.expect(any("no comparable sales" in note for note in result.notes), "the failure must explain itself")

    # One exact comp is still not enough. That judgement was correct and
    # must survive the ladder being added.
    result = build_comp_set(inputs(keyword_sales=EXACT_ONE))
    c.expect(result.basis is CompBasis.NONE,
             f"one sale is an anecdote and must not price a ladder, got {result.basis}")


def check_stale_and_invalid_comps(c: Checks):
    """Old sales describe a market that no longer exists."""
    stale = [{"domain": f"consultingold{i}.ai", "price": 5_000, "date": "2019-01-01"} for i in range(4)]
    result = build_comp_set(inputs(keyword_sales=stale))
    c.expect(result.basis is CompBasis.NONE,
             f"sales from {config.COMP_MAX_AGE_MONTHS}+ months ago must not price a domain, got {result.basis}")

    undated = [{"domain": f"consultingu{i}.ai", "price": 5_000} for i in range(4)]
    c.expect(build_comp_set(inputs(keyword_sales=undated)).basis is CompBasis.EXACT,
             "an unknown sale date is not a stale one and must be kept")

    zeroed = [{"domain": f"consultingz{i}.ai", "price": 0, "date": "2026-01-01"} for i in range(4)]
    c.expect(build_comp_set(inputs(keyword_sales=zeroed)).basis is CompBasis.NONE,
             "a zero-priced row is not a sale")


def check_keyword_only_is_labeled(c: Checks):
    """The weakest rung must be unmistakable in the output."""
    result = build_comp_set(inputs(keyword_stats={
        "consulting": {"avg_price": 2_900, "total_sales": 372},
        "advisory": {"avg_price": 4_100, "total_sales": 90},
        "strategy": {"avg_price": 7_800, "total_sales": 210},
    }))
    c.expect(result.basis is CompBasis.KEYWORD_ONLY, f"three aggregates should reach keyword_only, got {result.basis}")
    c.expect(any("keyword aggregates only" in note for note in result.notes),
             f"the mandatory warning is missing, got {result.notes}")
    for comp in result.comps:
        c.expect("sales containing" in comp.domain,
                 f"a synthesized comp must not look like a domain that sold: {comp.domain!r}")
        c.expect(comp.adjustment_reason and "not a transaction" in comp.adjustment_reason,
                 f"a synthesized comp must say it is not a transaction: {comp.adjustment_reason!r}")


# Own-book sales, as a brokerage records them.
INTERNAL = [
    {"domain": "consultinghq.ai", "price": 9_859, "date": "2026-03-10", "marketplace": "internal"},
    {"domain": "consultingpro.ai", "price": 13_558, "date": "2026-04-11", "marketplace": "internal"},
]


def check_internal_comps(c: Checks):
    """Own-book sales are comps at every rung, and they lift confidence.

    The rung records how far the SEARCH had to widen. That is not the same
    question as how good the evidence is: a set that stopped at
    keyword_only but rests partly on sales we brokered ourselves is not
    resting on aggregates.
    """
    stats = {"consulting": {"p25": 195.0, "median": 296.5, "p75": 545.5, "total_sales": 372}}

    aggregates = build_comp_set(inputs(keyword_stats=stats))
    c.expect(aggregates.confidence == config.COMP_BASIS_CONFIDENCE["keyword_only"],
             f"aggregates alone should stay at keyword-only confidence, got {aggregates.confidence}")

    lifted = build_comp_set(inputs(keyword_stats=stats, internal_sales=INTERNAL))
    c.expect(lifted.confidence >= config.COMP_BASIS_CONFIDENCE["cohort"],
             f"two own-book sales should lift confidence to cohort grade, got {lifted.confidence}")
    c.expect(lifted.basis is CompBasis.KEYWORD_ONLY,
             "the BASIS must still record where the comps came from — the lift is confidence, not a "
             "relabelling of how hard the search was")
    c.expect(any("own brokerage book" in note for note in lifted.notes),
             f"the lift must explain itself, got {lifted.notes}")

    # One is an anecdote, same rule as everywhere else.
    single = build_comp_set(inputs(keyword_stats=stats, internal_sales=INTERNAL[:1]))
    c.expect(single.confidence == config.COMP_BASIS_CONFIDENCE["keyword_only"],
             f"one own-book sale must not lift confidence, got {single.confidence}")

    # Weighted between a cohort member and an exact comp.
    own = [comp for comp in lifted.comps if comp.source == "internal_comps"]
    c.expect(len(own) == 2, f"both own-book sales should be present, got {len(own)}")
    for comp in own:
        c.expect(comp.weight == config.COMP_WEIGHT_INTERNAL,
                 f"own-book weight should be {config.COMP_WEIGHT_INTERNAL}, got {comp.weight}")
        c.expect(comp.weight < config.COMP_WEIGHT_EXACT,
                 "own-book data is verified but single-venue — it must not outweigh an exact comp")
        c.expect(comp.venue == "internal", f"own-book comps must be marked, got venue={comp.venue!r}")

    # They reach the strongest rung too, not only the weak ones.
    exact = build_comp_set(inputs(keyword_sales=EXACT_ONE, internal_sales=INTERNAL))
    c.expect(exact.basis is CompBasis.EXACT,
             f"one exact sale plus two own-book sales is three same-TLD comps, got {exact.basis}")

    # Own-book sales are trusted without a substring check — the caller's
    # query selected them — while public-feed rows must contain the term.
    c.expect(all(c_.domain != "daft.ai" for c_ in exact.comps),
             "a semantic neighbour must not reach the exact rung")

    # And an offer is never one of them.
    c.expect(all("offer" not in (comp.adjustment_reason or "").lower() for comp in lifted.comps),
             "an offer must never enter the comp set — it is a price nobody paid")


def main() -> int:
    c = Checks()
    for check in (
        check_rung_selection,
        check_rungs_are_cumulative,
        check_tld_adjustment,
        check_confidence_monotonicity,
        check_ranges_widen,
        check_none_is_still_reachable,
        check_stale_and_invalid_comps,
        check_keyword_only_is_labeled,
        check_internal_comps,
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
