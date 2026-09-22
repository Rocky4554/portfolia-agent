"""
The acceptance run for the thin-comps patch.
Run with: uv run python -m portfolio_agent.evals.test_e2e_consulting_ai

consulting.ai is the case the whole patch exists for. A live valuation of it
found one comparable sale at $425 and priced nothing at all — six empty
tiers and a rationale that then mischaracterised one of them.

This runs the real compiled graph against a consulting.ai-shaped market and
asserts the four things the patch promised:

    comp_basis != exact        the ladder widened rather than giving up
    four tiers populated       a thin market gets a wider answer, not none
    ask-aware List/BIN         a live ask caps it and is reported as an ask
    no unfilled tier narrated  the rationale cannot describe what it lacks

The thin market is forced by a fixture, because the mock layer is more
generous than reality: it synthesizes fourteen .ai comps for any keyword,
which would let this pass on evidence the real world does not supply.
"""

import asyncio
import sys
from datetime import date

from portfolio_agent import config
from portfolio_agent.nodes import rationale as rationale_node
from portfolio_agent.pricing.comp_ladder import LadderInputs, build_comp_set
from portfolio_agent.schema.rationale_view import build_rationale_view
from portfolio_agent.schema.valuation_output import TIER_ORDER, CompBasis

DOMAIN = "consulting.ai"
TODAY = date(2026, 9, 21)

# What the live market actually held: one sale, of a domain merely CONTAINING
# the keyword, for $425.
THE_ONE_SALE = [{"domain": "primeconsulting.ai", "price": 425, "date": "2026-04-30", "marketplace": "Sedo"}]

# Single dictionary words that really have sold on .ai — the cohort a human
# appraiser would reach for.
AI_COHORT = [
    {"domain": "wisdom.ai", "price": 750_000, "date": "2025-10-13", "marketplace": "Private"},
    {"domain": "cloud.ai", "price": 600_000, "date": "2025-07-02", "marketplace": "Private"},
    {"domain": "genesis.ai", "price": 400_000, "date": "2026-04-09", "marketplace": "Sedo"},
    {"domain": "law.ai", "price": 350_000, "date": "2025-08-16", "marketplace": "Afternic"},
    {"domain": "adapt.ai", "price": 300_000, "date": "2025-08-29", "marketplace": "Sedo"},
]

# GoDaddy's live ask, converted at roughly 85 INR/USD.
GODADDY_ASK = 2_827_928


class Checks:
    def __init__(self):
        self.total = 0
        self.failures: list[str] = []

    def expect(self, condition, message: str):
        self.total += 1
        if not condition:
            self.failures.append(message)


def thin_market_comp_set():
    """The ladder's verdict on the real consulting.ai market."""
    return build_comp_set(LadderInputs(
        domain=DOMAIN, sld="consulting", tld="ai", category="keyword", keywords=["consulting"],
        keyword_sales=THE_ONE_SALE, cohort_sales=AI_COHORT, today=TODAY,
    ))


def valuation_from(comp_set, *, listings=None):
    """Run synthesize over a state carrying this comp set."""
    from portfolio_agent.nodes.classify import classify
    from portfolio_agent.nodes.gather_evidence import gather_evidence
    from portfolio_agent.nodes.synthesize import build_valuation

    state = {"domain": DOMAIN}
    state.update(classify(state))
    state.update(asyncio.run(gather_evidence(state)))

    # Replace the generous fixture comps with the thin real market.
    state["comp_set"] = comp_set.as_dicts()
    state["comp_refs"] = [c.model_dump() for c in comp_set.comps]
    state["comp_basis"] = comp_set.basis.value
    state["comp_confidence"] = comp_set.confidence
    state["comp_notes"] = comp_set.notes
    if listings is not None:
        state["tool_results"]["get_marketplace_listings"] = {
            "source": "marketplace_listings", "ok": True, "mock": True,
            "data_age_days": None, "error": None, "data": listings,
        }
    return build_valuation(state)


def check_ladder_widened(c: Checks):
    """One sale must widen the search, not end it."""
    comp_set = thin_market_comp_set()

    c.expect(comp_set.basis is not CompBasis.EXACT,
             f"a single $425 sale must not read as an exact basis, got {comp_set.basis}")
    c.expect(comp_set.basis is not CompBasis.NONE,
             "the ladder must find a basis rather than give up — that was the original failure")
    c.expect(comp_set.confidence < 1.0,
             f"a widened basis must carry reduced confidence, got {comp_set.confidence}")
    c.expect(len(comp_set.comps) >= config.COMP_LADDER_MIN_COMPS,
             f"the widened set needs at least {config.COMP_LADDER_MIN_COMPS} comps, got {len(comp_set.comps)}")

    domains = {comp.domain for comp in comp_set.comps}
    c.expect("primeconsulting.ai" in domains,
             "the one real sale must survive the widening — it was not wrong, only lonely")
    c.expect("cloud.ai" in domains, "the cohort must actually be used")
    c.expect(any("cohort" in note or "extensions" in note for note in comp_set.notes),
             f"the widening must explain itself, got {comp_set.notes}")


def check_four_tiers_populated(c: Checks):
    """The acceptance target: a thin market gets a wider answer, not none."""
    output = valuation_from(thin_market_comp_set())

    for name in ("floor", "auction", "end_user", "list_bin"):
        tier = getattr(output.tiers, name)
        c.expect(tier is not None, f"{name} must be priced — the whole patch is about not leaving it empty")
        if tier:
            c.expect(tier.point > 0, f"{name} priced at {tier.point}")

    c.expect(output.comp_basis is not CompBasis.EXACT,
             f"the output must record the widened basis, got {output.comp_basis}")
    c.expect(output.tiers.monotonicity_violations() == [],
             f"the tier order must hold at a widened basis: {output.tiers.monotonicity_violations()}")

    # A widened basis must LOOK wider.
    end_user = output.tiers.end_user
    spread = end_user.high / max(end_user.low, 1)
    c.expect(spread > 2.5,
             f"a cohort-based price must carry a visibly wide range, got {spread:.1f}x")

    c.expect(any("comp basis" in w for w in output.warnings),
             "the widened basis must be disclosed in warnings")


def check_ask_aware_bin(c: Checks):
    """GoDaddy's ask caps the BIN, is labeled an ask, and moves nothing else."""
    comp_set = thin_market_comp_set()
    without = valuation_from(comp_set)
    with_ask = valuation_from(comp_set, listings={
        "listed": True, "lowest_ask": GODADDY_ASK, "highest_ask": GODADDY_ASK,
        "has_live_auction": False, "listings": [], "providers_checked": ["godaddy"],
        "providers_unavailable": {},
    })

    c.expect(with_ask.tiers.list_bin.point <= GODADDY_ASK,
             f"the BIN (${with_ask.tiers.list_bin.point:,}) must not exceed the ${GODADDY_ASK:,} ask")
    c.expect(with_ask.tiers.list_bin.point != without.tiers.list_bin.point,
             "a live ask must move the BIN tier")

    recorded = " ".join(with_ask.tiers.list_bin.inflationary_factors
                        + with_ask.tiers.list_bin.deflationary_factors)
    c.expect("ask" in recorded.lower(),
             f"the ask must be named as an ask in the BIN's factors, got {recorded[:120]!r}")

    for name in ("floor", "auction", "end_user"):
        c.expect(getattr(with_ask.tiers, name).point == getattr(without.tiers, name).point,
                 f"an ask must not move the {name} tier — that is anchored on transactions")

    c.expect(with_ask.signals.ask_gap and with_ask.signals.ask_gap > 1,
             f"the ask gap must be reported, got {with_ask.signals.ask_gap}")
    c.expect(with_ask.signals.ask_gap_label, "the gap needs a label a reader can act on")

    # No listing ever becomes a comp.
    c.expect(all(comp.venue != "godaddy" for comp in with_ask.comp_set),
             "a listing must never appear in the comp set")
    c.expect(all(comp.price != GODADDY_ASK for comp in with_ask.comp_set),
             "the ask price must not appear as a comparable sale")


def check_rationale_mentions_no_unfilled_tier(c: Checks):
    """The rationale cannot describe what it was not given."""
    output = valuation_from(thin_market_comp_set())
    view = build_rationale_view(output)

    unfilled = [name for name in TIER_ORDER if getattr(output.tiers, name) is None]
    c.expect(bool(unfilled), "this fixture should leave the conditional tiers unpriced")

    section_keys = {s.key for s in view.sections}
    for name in unfilled:
        c.expect(f"tier:{name}" not in section_keys,
                 f"{name} is unpriced and must have no section to be described in")
        c.expect(name not in view.tier_labels,
                 f"{name} must not be in tier_labels — nothing may reference it as a priced tier")

    # Every tier that DOES have a section is genuinely priced.
    for key in section_keys:
        if key.startswith("tier:"):
            c.expect(getattr(output.tiers, key.removeprefix("tier:")) is not None,
                     f"{key} has a section but no price")

    # The unpriced ones are still explained, from supplied text.
    not_priced = view.section("not_priced")
    c.expect(not_priced is not None, "unpriced valuations must be explained")
    if not_priced:
        rendered = str(not_priced.content)
        for name in unfilled:
            from portfolio_agent.schema.rationale_view import label

            head = label(name).split("—")[0].strip()
            c.expect(head in rendered, f"{name}'s absence must be explained, looking for {head!r}")

    # And the assembled narrative passes its own checks.
    assembled = rationale_node._assemble(
        [(s, rationale_node._section_template(s)) for s in view.sections]
    )
    for name in unfilled:
        from portfolio_agent.schema.rationale_view import label

        head = label(name).split("—")[0].strip().lower()
        tier_bodies = "\n".join(
            rationale_node._section_template(s) for s in view.sections if s.key.startswith("tier:")
        ).lower()
        c.expect(head not in tier_bodies,
                 f"no priced tier's section may name the unpriced {name}")


def main() -> int:
    c = Checks()
    was_mock = config.MOCK_MODE
    config.MOCK_MODE = True
    try:
        for check in (
            check_ladder_widened,
            check_four_tiers_populated,
            check_ask_aware_bin,
            check_rationale_mentions_no_unfilled_tier,
        ):
            check(c)
    finally:
        config.MOCK_MODE = was_mock

    if c.failures:
        print(f"FAILED {len(c.failures)}/{c.total} checks:")
        for failure in c.failures:
            print(f"  - {failure}")
        return 1
    print(f"PASSED {c.total}/{c.total} checks — consulting.ai prices end to end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
