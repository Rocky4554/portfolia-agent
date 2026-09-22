"""
Offline regression eval for commercial-intent gating.
Run with: uv run python -m portfolio_agent.evals.test_intent

The failure this exists to prevent: the end-user tier paid for search volume
on a straight curve, so 250,000 searches for "weather" bought the same
premium as 250,000 for "car insurance". One audience is deciding whether to
take an umbrella; the other is about to buy a policy, which is why insurance
keywords carry some of the highest ad costs in existence. Volume without
intent measures attention, not demand.

What is asserted, in order of importance:

  - An unclassified keyword is never penalised. Most SLDs are a bare noun
    with no verb of purchase attached, and reading "we cannot tell" as
    "nobody is buying" would deflate most of a book on no evidence — the
    same asymmetry industries.py and sentiment.py hold.
  - Identical volume produces different prices for different intent, which
    is the whole point and is checked end to end rather than on the
    classifier alone.
  - The discount is disclosed with its reason, because an adjustment the
    report cannot account for is not an improvement over no adjustment.
  - "insurance quotes" stays commercial. The literary sense of "quotes"
    must not capture one of the most transactional searches there is.
"""

import sys

from portfolio_agent.evals.test_tiers import fixture_inputs
from portfolio_agent.knowledge.intent import VOLUME_CREDIT, classify_intent, volume_credit
from portfolio_agent.pricing import tier_engine as te

VOLUME = {"global_monthly_searches": 250_000, "exact_keyword_active_tld_count": 45}


class Checks:
    def __init__(self):
        self.total = 0
        self.failures = []

    def expect(self, condition, message):
        self.total += 1
        if not condition:
            self.failures.append(message)


def price_for(sld: str):
    tiers, _ = te.compute_market_tiers(
        fixture_inputs(sld=sld, domain=f"{sld}.com", keyword_stats=VOLUME))
    return tiers.end_user


def check_classification(c: Checks):
    for sld, expected in (
        ("carinsurancequotes", "commercial"),
        ("cheapflights", "commercial"),
        ("buycrypto", "commercial"),
        ("divorcelawyer", "commercial"),
        ("plumbingservices", "commercial"),
        ("weather", "informational"),
        ("recipes", "informational"),
        ("newsblog", "informational"),
        ("zylantix", "unclassified"),
        ("forge", "unclassified"),
    ):
        actual, _ = classify_intent(sld)
        c.expect(actual == expected, f"classify_intent({sld!r}) = {actual}, expected {expected}")


def check_quotes_stays_commercial(c: Checks):
    """'insurance quotes' is one of the most transactional searches there is,
    and must not be captured by the literary sense of the word."""
    intent, _ = classify_intent("insurancequotes")
    c.expect(intent == "commercial", f"'insurancequotes' read as {intent}")
    intent, _ = classify_intent("carinsurancequotes")
    c.expect(intent == "commercial", f"'carinsurancequotes' read as {intent}")


def check_unclassified_is_never_penalised(c: Checks):
    c.expect(volume_credit("unclassified") == 1.0,
             "an unreadable keyword must keep full volume credit")
    c.expect(volume_credit("commercial") == 1.0, "a commercial keyword is the baseline, not a bonus")
    c.expect(volume_credit("informational") < 1.0, "an informational keyword must be discounted")

    commercial = price_for("carinsurance")
    unknown = price_for("forge")
    c.expect(commercial.point == unknown.point,
             f"an unclassified keyword must price like a commercial one, "
             f"not below it: {unknown.point} vs {commercial.point}")


def check_intent_moves_the_price(c: Checks):
    """Checked through the tier, not on the classifier — a classification
    that does not reach a number is the recurring bug in this engine."""
    commercial = price_for("carinsurance")
    informational = price_for("weather")
    c.expect(informational.point < commercial.point,
             f"identical volume must not buy identical price: weather {informational.point} "
             f"vs carinsurance {commercial.point}")

    # No volume at all, so intent has nothing to discount and must do nothing.
    tiers, _ = te.compute_market_tiers(
        fixture_inputs(sld="weather", domain="weather.com", keyword_stats=None))
    c.expect(tiers.end_user is not None, "a domain with no keyword data must still price")


def check_discount_is_disclosed(c: Checks):
    tier = price_for("weather")
    factors = [f for f in tier.inflationary_factors + tier.deflationary_factors
               if "searches" in f]
    c.expect(factors, "the volume adjustment must appear as a factor")
    text = factors[0]
    c.expect("discounted" in text, f"the discount must be stated, got {text!r}")
    c.expect("reads" in text or "reader" in text,
             f"and must give its reason rather than asserting, got {text!r}")
    # The F2 invariant: every factor states its applied multiplier.
    c.expect(text.rstrip().endswith(")") and "×" in text,
             f"the factor must carry its multiplier, got {text!r}")


def main() -> int:
    c = Checks()
    for fn in (check_classification, check_quotes_stays_commercial,
               check_unclassified_is_never_penalised, check_intent_moves_the_price,
               check_discount_is_disclosed):
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
