"""
Offline regression eval for trademark screening.
Run with: uv run python -m portfolio_agent.evals.test_trademark

Table 4.1.1 lists trademark conflict as an End-User deflator. It was named
on every valuation this engine has produced and applied on none of them —
the stub returned `unknown` and the warning told the reader to go and check
manually. Honest and useless in equal measure.

What is asserted, in order of importance:

  - "No obvious conflict" never reads as cleared. There is no `clear`
    status and the detail says outright that no search was run. A reader who
    takes a clean screen for a clean search is worse off than one who was
    told nothing.
  - The screen does not cry wolf. A false positive on ordinary vocabulary
    gets the whole check switched off, which costs more than the cases it
    catches: "metabolism" is not Meta and "visualise" is not Visa.
  - It catches what it exists for. nikeshoes.com must flag without a
    database lookup.
  - The deflator reaches the price. A named-and-never-applied deflator is
    what this replaces.
"""

import asyncio
import sys

from portfolio_agent.evals.test_tiers import fixture_inputs
from portfolio_agent.knowledge.famous_marks import ALSO_COMMON_WORDS, FAMOUS_MARKS
from portfolio_agent.pricing import tier_engine as te
from portfolio_agent.tools.trademark import CONFLICT_DEFLATORS, STATUSES, screen_trademarks


class Checks:
    def __init__(self):
        self.total = 0
        self.failures = []

    def expect(self, condition, message):
        self.total += 1
        if not condition:
            self.failures.append(message)


def screen(domain: str) -> dict:
    return asyncio.run(screen_trademarks(domain))["data"]


def check_clean_never_means_cleared(c: Checks):
    """The distinction the whole screen rests on."""
    c.expect("clear" not in STATUSES,
             f"there must be no status meaning 'cleared', got {STATUSES}")
    result = screen("northwind.com")
    c.expect(result["status"] == "no_obvious_conflict", "an ordinary name must pass the screen")
    c.expect("NOT a clearance" in result["detail"] or "not a clearance" in result["detail"].lower(),
             f"a clean screen must say it is not a clearance, got {result['detail']!r}")
    c.expect(result["searched"] is False,
             "nothing has been searched, and the field must say so")
    c.expect(result["deflator"] == 1.0, "a clean screen must not deflate")


def check_it_catches_what_it_exists_for(c: Checks):
    for domain in ("nikeshoes.com", "shopnike.com", "nike.com", "cocacola.com"):
        result = screen(domain)
        c.expect(result["status"] == "likely_conflict",
                 f"{domain} must flag, got {result['status']}")
        c.expect(result["deflator"] < 0.5, f"{domain} must deflate materially")
        c.expect(result["marks"], f"{domain} must name the mark it matched")


def check_it_does_not_cry_wolf(c: Checks):
    """A screen with false positives gets switched off, which costs more
    than the cases it catches."""
    for domain in ("metabolism.com", "visualise.com", "carinsurance.com", "forge.ai",
                   "hello.com", "consulting.ai", "shellfish.com"):
        result = screen(domain)
        c.expect(result["status"] == "no_obvious_conflict",
                 f"{domain} is ordinary vocabulary and must not flag: {result['marks']}")


def check_ambiguous_marks_are_hedged(c: Checks):
    """Famous marks that are also everyday words coexist with ordinary use:
    orange juice is a drink before it is a telecom."""
    for domain in ("orangejuice.com", "applepie.com", "targetpractice.com"):
        result = screen(domain)
        c.expect(result["status"] == "possible_conflict",
                 f"{domain} is ambiguous and must be hedged, got {result['status']}")
        c.expect("lawyer" in result["detail"],
                 "an ambiguous verdict must say who can actually decide it")

    # The mark standing alone is not ambiguous, whatever else the word means.
    for domain in ("orange.com", "apple.com"):
        c.expect(screen(domain)["status"] == "likely_conflict",
                 f"{domain} IS the mark and must not be hedged")


def check_wordlists_are_coherent(c: Checks):
    c.expect(ALSO_COMMON_WORDS <= FAMOUS_MARKS,
             f"these are in ALSO_COMMON_WORDS but not FAMOUS_MARKS, so nothing screens for "
             f"them: {sorted(ALSO_COMMON_WORDS - FAMOUS_MARKS)}")
    c.expect(all(m.islower() and m.isalpha() for m in FAMOUS_MARKS),
             "marks must be lowercase and alphabetic to match segmented tokens")
    c.expect(set(CONFLICT_DEFLATORS) == set(STATUSES),
             "every status needs a deflator and vice versa")
    c.expect(CONFLICT_DEFLATORS["unscreened"] == 1.0,
             "not screening something must never be treated as evidence against it")


def check_the_deflator_reaches_the_price(c: Checks):
    """A named-and-never-applied deflator is what this replaces."""
    flagged, _ = te.compute_market_tiers(fixture_inputs(domain="nikeshoes.com", sld="nikeshoes"))
    clean, _ = te.compute_market_tiers(fixture_inputs(domain="northwind.com", sld="northwind"))
    c.expect(flagged.end_user.point < clean.end_user.point * 0.5,
             f"a famous-mark conflict must move the end-user price: "
             f"{flagged.end_user.point} vs {clean.end_user.point}")

    factors = [f for f in flagged.end_user.deflationary_factors if "trademark" in f]
    c.expect(factors, "the deflator must appear as a factor on the tier")
    c.expect(factors and factors[0].rstrip().endswith(")") and "×" in factors[0],
             f"and must carry its multiplier, got {factors[0] if factors else None!r}")

    # Even a clean screen must leave the reader knowing what was not done.
    _, warnings = te.compute_market_tiers(fixture_inputs(domain="northwind.com", sld="northwind"))
    trademark_warnings = [w for w in warnings if "trademark" in w.lower()]
    c.expect(trademark_warnings, "a clean screen must still warn that no search was run")
    c.expect(any("SEARCH" in w or "search" in w for w in trademark_warnings),
             "and must name the difference between a screen and a search")


def main() -> int:
    c = Checks()
    for fn in (check_clean_never_means_cleared, check_it_catches_what_it_exists_for,
               check_it_does_not_cry_wolf, check_ambiguous_marks_are_hedged,
               check_wordlists_are_coherent, check_the_deflator_reaches_the_price):
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
