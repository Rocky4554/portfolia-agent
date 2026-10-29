"""
Offline checks for the F17–F20 corrections, each run with its flag on.
Run with: uv run python -m portfolio_agent.evals.test_gates

The flags default off, so the rest of the suite pins the engine as it was
before them. This file pins what each one does when a backtest turns it on.

What is asserted:

  - A tripped gate lowers the confidence the report shows, not only the
    ranges. The warning said "confidence forced down" while comp_confidence
    stayed at 1.0.
  - A gate that trips on an already-low confidence still says so.
  - The TLD discount lands once. list_bin is a markup on end_user, which is
    already discounted; applying it again pushed .co BINs under end_user.
  - An extension with no floor of its own gets the default, not the .com one.
  - Strict exact matching keeps plurals and drops longer names that merely
    contain the word.
"""

import sys
from contextlib import contextmanager

from portfolio_agent import config
from portfolio_agent.evals.test_comp_ladder import inputs as ladder_inputs
from portfolio_agent.evals.test_tiers import fixture_inputs
from portfolio_agent.knowledge.category_floors import DEFAULT_FLOOR, get_floor
from portfolio_agent.pricing import comp_ladder
from portfolio_agent.pricing import tier_engine as te


class Checks:
    def __init__(self):
        self.total = 0
        self.failures: list[str] = []

    def expect(self, condition, message: str):
        self.total += 1
        if not condition:
            self.failures.append(message)


@contextmanager
def flag(name: str, value: bool = True):
    before = getattr(config, name)
    setattr(config, name, value)
    try:
        yield
    finally:
        setattr(config, name, before)


def check_plausibility_caps_reported_confidence(c: Checks):
    with flag("PLAUSIBILITY_GATE"):
        tripped = fixture_inputs(independent_appraisal=10_000_000)
        _, warnings = te.compute_market_tiers(tripped)
        c.expect(any(w.startswith("plausibility gate") for w in warnings),
                 f"an end_user far under the appraisal must trip the gate, got {warnings}")
        c.expect(tripped.comp_confidence == config.IMPLAUSIBLE_CONF,
                 f"the cap must reach comp_confidence, got {tripped.comp_confidence}")

        low = fixture_inputs(independent_appraisal=10_000_000, comp_confidence=0.2)
        _, warnings = te.compute_market_tiers(low)
        c.expect(any(w.startswith("plausibility gate") for w in warnings),
                 "a gate that trips on an already-low confidence must still warn")
        c.expect(low.comp_confidence == 0.2, f"a lower confidence must not be raised, got {low.comp_confidence}")

    untouched = fixture_inputs(independent_appraisal=10_000_000)
    _, warnings = te.compute_market_tiers(untouched)
    c.expect(not any(w.startswith("plausibility gate") for w in warnings), "flag off, the gate must not run")
    c.expect(untouched.comp_confidence == 1.0, "flag off, confidence must not move")


def check_ask_conflict_caps_reported_confidence(c: Checks):
    with flag("ASK_RECONCILIATION"):
        cheap = fixture_inputs(listings={"lowest_ask": 100})
        tiers, warnings = te.compute_market_tiers(cheap)
        c.expect(any(w.startswith("ask conflict") for w in warnings),
                 f"an ask far under end_user must be reported, got {warnings}")
        c.expect(cheap.comp_confidence == config.ASK_CONFLICT_CONF,
                 f"the cap must reach comp_confidence, got {cheap.comp_confidence}")

        low = fixture_inputs(listings={"lowest_ask": 100}, comp_confidence=0.3)
        _, warnings = te.compute_market_tiers(low)
        c.expect(any(w.startswith("ask conflict") for w in warnings),
                 "an ask conflict on an already-low confidence must still warn")


def check_tld_discount_lands_once(c: Checks):
    off, _ = te.compute_market_tiers(fixture_inputs(tld="co", domain="northwind.co"))
    with flag("TLD_TIER_DISCOUNT"):
        on, _ = te.compute_market_tiers(fixture_inputs(tld="co", domain="northwind.co"))
    multiplier = on.end_user.point / off.end_user.point
    c.expect(abs(multiplier - 0.5) < 0.01, f".co end_user should be ×0.50, got ×{multiplier:.3f}")
    markup_off = off.list_bin.point / off.end_user.point
    markup_on = on.list_bin.point / on.end_user.point
    c.expect(abs(markup_on - markup_off) < 0.01,
             f"list_bin's markup on end_user must not change, got {markup_off:.3f} -> {markup_on:.3f}")
    c.expect(on.list_bin.point > on.end_user.point, "list_bin must stay above end_user")


def check_floor_fallback(c: Checks):
    c.expect(get_floor("generic", "com") == 5_000, "the listed .com floor must be used")
    c.expect(get_floor("generic", "xyz") == DEFAULT_FLOOR,
             f"an unlisted extension must get the default floor, got {get_floor('generic', 'xyz')}")


def check_strict_exact_match(c: Checks):
    sales = [
        {"domain": "pays.ai", "price": 40_000, "date": "2026-03-01"},
        {"domain": "paydayloans.ai", "price": 564, "date": "2026-03-01"},
    ]
    subject = ladder_inputs(domain="pay.ai", sld="pay", keywords=["pay"], keyword_sales=sales)

    def exact():
        return {comp.domain for comp in comp_ladder._step_exact(subject)}

    c.expect(exact() == {"pays.ai", "paydayloans.ai"}, "flag off, the substring rule must hold")
    with flag("STRICT_EXACT_MATCH"):
        c.expect(exact() == {"pays.ai"}, f"strict: a plural is exact and a longer name is not, got {exact()}")


def main() -> int:
    c = Checks()
    for check in (
        check_plausibility_caps_reported_confidence,
        check_ask_conflict_caps_reported_confidence,
        check_tld_discount_lands_once,
        check_floor_fallback,
        check_strict_exact_match,
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
