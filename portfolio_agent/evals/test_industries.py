"""
Offline regression eval for industry classification and cross-industry weighting.
Run with: uv run python -m portfolio_agent.evals.test_industries

The failure this exists to prevent: comp retrieval matches on shape, and
shape is not a market. A fifteen-letter two-word .com priced from SaaS
comps when it is a law firm's name is not a small error — the buyer, the
click economics and the customer lifetime value are all different.

What is asserted, in order of importance:

  - An unknown industry is NEVER treated as a mismatch. Most domains are
    coined names with no vertical at all, and penalising them on no
    evidence is the same mistake as scoring an unreadable sentiment 70.
  - A mixed comp set actually reprices. Down-weighting that does not reach
    the number is the bug F3 already found once.
  - When EVERY comp is cross-industry, the engine says so. Percentile
    weights are relative, so uniform down-weighting moves nothing and only
    a warning can carry the fact.
  - Substring matching stays out: "carbon" is not automotive and "lawn" is
    not legal.
"""

import sys
from types import SimpleNamespace

from portfolio_agent import config
from portfolio_agent.knowledge.industries import classify_industry, industry_relation
from portfolio_agent.pricing import tier_engine as te
from portfolio_agent.pricing.comp_ladder import CompSet, _apply_industry_weights
from portfolio_agent.schema.valuation_output import CompRef


def comp(domain: str, price: int, weight: float = 1.0) -> CompRef:
    return CompRef(domain=domain, price=price, date="2026-01-15", venue="GoDaddy",
                   source="comparables", weight=weight)


def subject(sld: str):
    return SimpleNamespace(sld=sld, tld="com", category="keyword")


class Checks:
    def __init__(self):
        self.total = 0
        self.failures = []

    def expect(self, condition, message):
        self.total += 1
        if not condition:
            self.failures.append(message)


def check_classification(c: Checks):
    for sld, expected in (
        ("divorcelawyer", "legal"), ("injuryattorney", "legal"),
        ("paydayloans", "finance"), ("accounting", "finance"),
        ("cloudstorage", "software"), ("quantum", "software"),
        ("bestcasino", "gambling"), ("plumbingpros", "home_services"),
        ("cryptowallet", "crypto"), ("jobboard", "recruiting"),
    ):
        actual = classify_industry(sld)
        c.expect(actual == expected, f"classify_industry({sld!r}) = {actual}, expected {expected}")


def check_head_final_compounds(c: Checks):
    """English compounds are head-final: car insurance is a kind of
    insurance, car loans a kind of loan. Taking the leftmost word would file
    both under automotive and price them off used-car comps."""
    for sld, expected in (("carinsurance", "insurance"), ("carloans", "finance"),
                          ("travelinsurance", "insurance"), ("healthinsurance", "insurance")):
        actual = classify_industry(sld)
        c.expect(actual == expected, f"classify_industry({sld!r}) = {actual}, expected {expected}")


def check_no_substring_matching(c: Checks):
    """Whole segmented words only."""
    c.expect(classify_industry("carbon") is None, "'carbon' must not read as automotive")
    c.expect(classify_industry("lawn") is None, "'lawn' must not read as legal")
    c.expect(classify_industry("zylantix") is None, "a coined name has no industry")
    c.expect(classify_industry("forge") is None, "an ordinary noun with no vertical has no industry")


def check_unknown_is_never_a_mismatch(c: Checks):
    """The asymmetry that keeps this honest."""
    c.expect(industry_relation("forge", "cloudstorage") == "unknown",
             "an unclassified SUBJECT must yield unknown, not a mismatch")
    c.expect(industry_relation("divorcelawyer", "zylantix") == "unknown",
             "an unclassified COMP must yield unknown, not a mismatch")
    c.expect(industry_relation("divorcelawyer", "injuryattorney") == "same", "same vertical")
    c.expect(industry_relation("divorcelawyer", "cloudstorage") == "different", "different vertical")

    # And the weighting honours it: nothing is touched when we cannot judge.
    unknown_subject = CompSet(comps=[comp("cloudstorage.com", 180_000), comp("divorcelawyer.com", 9_000)])
    _apply_industry_weights(unknown_subject, subject("zylantix"))
    c.expect(all(x.weight == 1.0 for x in unknown_subject.comps),
             "an unclassifiable subject must leave every comp weight alone")
    c.expect(not unknown_subject.notes, "and must not emit a note it cannot justify")

    unknown_comps = CompSet(comps=[comp("zylantix.com", 180_000), comp("brellium.com", 210_000)])
    _apply_industry_weights(unknown_comps, subject("divorcelawyer"))
    c.expect(all(x.weight == 1.0 for x in unknown_comps.comps),
             "unclassifiable comps must not be penalised for being unclassifiable")


def check_mixed_set_reprices(c: Checks):
    """The weight must reach the number, which is the bug F3 found once."""
    comps = [comp("divorcelawyer.com", 9_000), comp("injuryattorney.com", 12_000),
             comp("cloudstorage.com", 180_000), comp("devplatform.com", 210_000),
             comp("dataanalytics.com", 240_000)]
    result = CompSet(comps=comps)
    _apply_industry_weights(result, subject("malpracticelawyer"))

    legal = [x for x in result.comps if x.domain in ("divorcelawyer.com", "injuryattorney.com")]
    saas = [x for x in result.comps if x not in legal]
    c.expect(all(x.weight == 1.0 for x in legal), "same-industry comps keep full weight")
    c.expect(all(x.weight == config.CROSS_INDUSTRY_WEIGHT for x in saas),
             f"cross-industry comps must drop to {config.CROSS_INDUSTRY_WEIGHT}")

    plain = te.comp_stats([{"domain": x.domain, "price": x.price} for x in result.comps])
    weighted = te.comp_stats([{"domain": x.domain, "price": x.price, "weight": x.weight}
                              for x in result.comps])
    c.expect(weighted.median < plain.median / 2,
             f"a legal domain must not be priced off SaaS comps: median {plain.median:,.0f} "
             f"-> {weighted.median:,.0f}")
    c.expect(result.notes and "down-weighted" in result.notes[0],
             f"a mixed set must record what it did, got {result.notes}")


def check_all_cross_industry_warns(c: Checks):
    """The case weighting cannot fix.

    Percentile weights are relative. Down-weight every comp by the same
    factor and the distribution is identical, so the only honest response is
    to say the price rests on the wrong market.
    """
    comps = [comp("cloudstorage.com", 180_000), comp("devplatform.com", 210_000),
             comp("dataanalytics.com", 240_000)]
    result = CompSet(comps=comps)
    _apply_industry_weights(result, subject("divorcelawyer"))

    weighted = te.comp_stats([{"domain": x.domain, "price": x.price, "weight": x.weight}
                              for x in result.comps])
    plain = te.comp_stats([{"domain": x.domain, "price": x.price} for x in result.comps])
    c.expect(abs(weighted.median - plain.median) < 1,
             "uniform down-weighting must be a no-op — if it is not, this test is wrong")
    c.expect(result.notes, "an all-cross-industry set must produce a note")
    note = result.notes[0]
    c.expect("wrong industry" in note,
             f"the note must say the price rests on the wrong industry, got {note!r}")
    c.expect("shape comparison" in note,
             f"the note must say what the number still IS, got {note!r}")


def main() -> int:
    c = Checks()
    for fn in (check_classification, check_head_final_compounds, check_no_substring_matching,
               check_unknown_is_never_a_mismatch, check_mixed_set_reprices,
               check_all_cross_industry_warns):
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
