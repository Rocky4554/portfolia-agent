"""
Offline regression eval for the TLD lifecycle model and comp recency decay.
Run with: uv run python -m portfolio_agent.evals.test_tld_lifecycle

Two failures this exists to prevent.

EMERGING_TLDS was a flat, permanent set, so .xyz — which carried the NFT
cycle and fell with it — was treated exactly like .ai, which is mid-boom. A
model that calls both "emerging" forever keeps applying 2021's enthusiasm to
2026's prices.

And comp age reached only liquidity_score, never the distribution. A 2021
sale counted exactly as much as a 2026 one in the price it was helping to
set, on every extension.

What is asserted, in order of importance:

  - Recency decay runs in BOTH directions. In a falling market old comps
    overstate and must quieten; in a rising one they understate and the
    recent comps must dominate. One mechanism, correct either way — if it
    only worked downward it would be a pessimism knob, not a dating model.
  - An old sale is weaker evidence, never zero evidence. Letting the curve
    run to nothing would silently empty a thin comp set.
  - An unassessed extension is dated neutrally rather than guessed at.
  - The weighting is disclosed, not just applied: a price built from sales
    older than the extension's half-life says so.
"""

import sys
from datetime import date

from portfolio_agent import config
from portfolio_agent.knowledge.tld_lifecycle import (
    AS_OF, PHASES, is_assessed, phase_for, staleness_warning,
)
from portfolio_agent.pricing import tier_engine as te

TODAY = date(2026, 9, 21)


class Checks:
    def __init__(self):
        self.total = 0
        self.failures = []

    def expect(self, condition, message):
        self.total += 1
        if not condition:
            self.failures.append(message)


def comps(tld: str, rows):
    half_life = phase_for(tld).comp_half_life_days
    return [{"domain": f"d{i}.{tld}", "price": float(price), "age_days": age,
             "comp_half_life_days": half_life}
            for i, (price, age) in enumerate(rows)]


def check_phases_are_differentiated(c: Checks):
    """The whole point: these extensions are not in the same place."""
    c.expect(phase_for("ai").phase == "peak", "a .ai in 2026 is at peak, not merely 'emerging'")
    c.expect(phase_for("xyz").phase == "decline", ".xyz carried a cycle that has passed")
    c.expect(phase_for("com").phase == "mature", ".com is the benchmark")
    c.expect(phase_for("ai").comp_half_life_days < phase_for("com").comp_half_life_days,
             "a fast-moving extension must date its evidence faster than .com")
    for name in ("ai", "xyz", "com", "io"):
        c.expect(phase_for(name).phase in PHASES, f".{name} has a phase outside the vocabulary")


def check_unassessed_is_not_guessed(c: Checks):
    """Inventing a phase for an extension nobody looked at is the error this
    module exists to prevent."""
    c.expect(not is_assessed("wtf"), "an unlisted extension must not claim assessment")
    unknown = phase_for("wtf")
    c.expect(unknown.phase == "unassessed", f"got phase {unknown.phase!r} for an unlisted extension")
    c.expect(unknown.comp_half_life_days >= phase_for("com").comp_half_life_days,
             "an unassessed extension must not have its comps quietly discounted")


def check_decay_runs_both_ways(c: Checks):
    """If it only worked downward it would be a pessimism knob."""
    falling = [(200_000, 1400), (180_000, 1300), (160_000, 1200),
               (40_000, 60), (35_000, 90), (30_000, 120)]
    rising = [(30_000, 1400), (35_000, 1300), (40_000, 1200),
              (160_000, 60), (180_000, 90), (200_000, 120)]

    for label, rows, expect_lower in (("falling", falling, True), ("rising", rising, False)):
        undated = te.comp_stats([{k: v for k, v in row.items() if k != "comp_half_life_days"}
                                 for row in comps("ai", rows)])
        dated = te.comp_stats(comps("ai", rows))
        if expect_lower:
            c.expect(dated.median < undated.median * 0.75,
                     f"{label}: old expensive comps must quieten, {undated.median:,.0f} "
                     f"-> {dated.median:,.0f}")
        else:
            c.expect(dated.median > undated.median * 1.25,
                     f"{label}: recent expensive comps must lead, {undated.median:,.0f} "
                     f"-> {dated.median:,.0f}")


def check_extension_clock_differs(c: Checks):
    """The same sales, aged on two different clocks, are not the same evidence."""
    rows = [(200_000, 1400), (180_000, 1300), (30_000, 60), (35_000, 90)]
    fast = te.comp_stats(comps("ai", rows)).median
    slow = te.comp_stats(comps("com", rows)).median
    c.expect(fast < slow,
             f"old comps must weigh less on .ai's clock than .com's: {fast:,.0f} vs {slow:,.0f}")


def check_old_is_weak_not_worthless(c: Checks):
    for age in (2_000, 10_000, 100_000):
        weight = te.recency_weight(age, 365)
        c.expect(weight >= config.MIN_RECENCY_WEIGHT,
                 f"a {age}-day-old sale decayed to {weight}, below the floor")
        c.expect(weight > 0, "decay must never reach zero")
    c.expect(te.recency_weight(0, 365) == 1.0, "a sale today is undiscounted")
    c.expect(te.recency_weight(None, 365) == 1.0, "an undated sale is not penalised for being undated")
    c.expect(abs(te.recency_weight(365, 365) - 0.5) < 1e-9, "one half-life must halve the weight")


def check_staleness_is_reported(c: Checks):
    c.expect(staleness_warning(AS_OF) is None, "freshly reviewed data must not warn")
    late = date(AS_OF.year + 2, AS_OF.month, AS_OF.day)
    warning = staleness_warning(late)
    c.expect(warning is not None, "a two-year-old phase list must warn")
    c.expect("already left" in (warning or ""),
             f"the warning must say what the risk actually is, got {warning!r}")


def main() -> int:
    c = Checks()
    for fn in (check_phases_are_differentiated, check_unassessed_is_not_guessed,
               check_decay_runs_both_ways, check_extension_clock_differs,
               check_old_is_weak_not_worthless, check_staleness_is_reported):
        fn(c)
    if c.failures:
        print(f"FAILED {len(c.failures)}/{c.total} checks:")
        for failure in c.failures:
            print(f"  - {failure}")
        return 1
    print(f"PASSED {c.total}/{c.total} checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
