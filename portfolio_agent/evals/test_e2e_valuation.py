"""
End-to-end eval: the real compiled graph, every golden domain, fully offline.
Run with: uv run python -m portfolio_agent.evals.test_e2e_valuation

This is the one eval that runs the thing a user would run —
`portfolio_agent.ainvoke({"domain": ...})` — rather than a node or a
function in isolation. Everything it asserts is a promise the system makes
to whoever consumes its output:

  - the output validates against the schema, for every category
  - the category routed to the right evidence plan, and every planned tool
    actually ran
  - the tier ordering invariant holds
  - every unfilled tier is explained, and every populated one cites evidence
  - the rationale contains no number the structured output does not assert
  - mock data is labeled as mock everywhere it appears
  - the whole thing finishes inside the latency budget

The rationale checks run against a stubbed model rather than a real one, so
this stays offline and deterministic. What is being tested is the validator
and the fallback path, which is where the correctness lives — whether a
particular model writes good prose is not something an eval can assert.
"""

import asyncio
import sys
import time

from portfolio_agent import config
from portfolio_agent.evals.golden_set import EXPECTED_CATEGORIES, GOLDEN_SET, TYPO_REGRESSION_PAIRS
from portfolio_agent.knowledge.category_playbook import ALL_TOOL_IDS, plan_for, plan_for_all
from portfolio_agent.tools.linguistics import classify_domain
from portfolio_agent.nodes import rationale as rationale_node
from portfolio_agent.schema.valuation_output import TIER_ORDER, EvidenceItem, ValuationOutput


class Checks:
    def __init__(self):
        self.total = 0
        self.failures: list[str] = []

    def expect(self, condition, message: str):
        self.total += 1
        if not condition:
            self.failures.append(message)


class StubModel:
    """A model that answers from a fixed script. Lets the e2e run exercise
    the rationale node's real code path with no key and no network."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        reply = self.replies[min(self.calls - 1, len(self.replies) - 1)]

        class Response:
            content = reply

        return Response()


def run_graph(domain: str, *, user_profile_hook: dict | None = None) -> tuple[ValuationOutput, float]:
    from portfolio_agent.nodes.graph import portfolio_agent

    payload: dict = {"domain": domain}
    if user_profile_hook is not None:
        payload["user_profile_hook"] = user_profile_hook

    started = time.perf_counter()
    result = asyncio.run(portfolio_agent.ainvoke(payload))
    elapsed = time.perf_counter() - started
    return ValuationOutput.model_validate(result["valuation"]), elapsed


def check_golden_set(c: Checks) -> float:
    """Schema, routing, tiers and evidence for every golden domain."""
    seen_categories = set()
    slowest = 0.0

    for case in GOLDEN_SET:
        domain = case["domain"]
        try:
            output, elapsed = run_graph(domain)
        except Exception as e:  # noqa: BLE001
            c.expect(False, f"{domain}: graph raised {type(e).__name__}: {e}")
            continue

        slowest = max(slowest, elapsed)
        seen_categories.add(output.category.value)

        c.expect(output.category.value == case["expected_category"],
                 f"{domain}: category {output.category.value!r}, expected {case['expected_category']!r}")
        c.expect(output.domain == domain, f"{domain}: output domain is {output.domain!r}")

        labels = [case["expected_category"]] + [
            name for name, _ in classify_domain(domain).secondary_categories
        ]
        planned = set(plan_for_all(labels).tool_ids())
        # Tool-sourced evidence only — the comp ladder emits its own item so
        # the rationale can cite the rung it priced from, but it is derived,
        # not planned.
        ran = {item.tool for item in output.evidence if item.tool in ALL_TOOL_IDS}
        c.expect(ran == planned, f"{domain}: ran {sorted(ran)}, plan says {sorted(planned)}")

        c.expect(output.tiers.monotonicity_violations() == [],
                 f"{domain}: tier ordering violated: {output.tiers.monotonicity_violations()}")

        populated = output.tiers.populated()
        c.expect(len(populated) >= 4, f"{domain}: expected the four market tiers, got {sorted(populated)}")
        for name, tier in populated.items():
            c.expect(tier.low <= tier.point <= tier.high, f"{domain}: {name} band out of order")
            c.expect(bool(tier.evidence_refs), f"{domain}: {name} cites no evidence")
            known = {item.id for item in output.evidence}
            c.expect(set(tier.evidence_refs) <= known, f"{domain}: {name} cites unknown evidence ids")

        for name in TIER_ORDER:
            if getattr(output.tiers, name) is None:
                c.expect(any(name in w for w in output.warnings),
                         f"{domain}: unfilled tier {name!r} is not explained in warnings")

        # Mock data must be labeled everywhere it surfaces, or a fixture
        # number can be read as an appraisal.
        c.expect(any(item.mock for item in output.evidence), f"{domain}: no evidence flagged mock in mock mode")
        c.expect(any("MOCK MODE" in w for w in output.warnings), f"{domain}: mock run not labeled in warnings")

        # The rationale is the templated one here (no model configured in
        # the e2e run) and must still satisfy the number rule.
        invented = rationale_node.unsupported_numbers(output.rationale, output)
        c.expect(not invented, f"{domain}: rationale cites unsupported numbers {invented}")

    c.expect(seen_categories == EXPECTED_CATEGORIES,
             f"golden set did not exercise every category; missing {sorted(EXPECTED_CATEGORIES - seen_categories)}")
    return slowest


def check_latency(c: Checks, slowest: float):
    c.expect(slowest <= config.LATENCY_BUDGET_S,
             f"slowest golden-set valuation took {slowest:.1f}s, over the {config.LATENCY_BUDGET_S:.0f}s budget")


def check_typo_collapse(c: Checks):
    """APVA §4.4, through the whole graph."""
    for misspelled, correct in TYPO_REGRESSION_PAIRS:
        bad, _ = run_graph(misspelled)
        good, _ = run_graph(correct)

        c.expect(bad.brand_score is None or bad.brand_score.typo_deflator < 1.0,
                 f"{misspelled}: no typo deflator recorded")
        for name in ("floor", "auction", "end_user", "list_bin"):
            bad_tier, good_tier = getattr(bad.tiers, name), getattr(good.tiers, name)
            if bad_tier is None or good_tier is None:
                continue
            c.expect(bad_tier.point < good_tier.point * 0.10,
                     f"{misspelled}: {name} ${bad_tier.point:,} should collapse against "
                     f"{correct}'s ${good_tier.point:,}")
        c.expect(any("misspelling" in w for w in bad.warnings),
                 f"{misspelled}: the deflation is not explained in warnings")


def check_conditional_tiers(c: Checks):
    """The two tiers that are supposed to be absent most of the time."""
    absent = present = 0
    for case in GOLDEN_SET:
        output, _ = run_graph(case["domain"])
        if output.tiers.targeted_lead is None:
            absent += 1
            c.expect(any("targeted_lead" in w for w in output.warnings),
                     f"{case['domain']}: missing targeted_lead is not explained")
        else:
            present += 1
            c.expect(output.tiers.targeted_lead.point >= output.tiers.end_user.point,
                     f"{case['domain']}: a targeted lead price below the end-user price is incoherent")
            c.expect(bool(output.tiers.targeted_lead.inflationary_factors),
                     f"{case['domain']}: targeted_lead records no reason for its premium")

        if output.tiers.strategic_ma is not None:
            c.expect(any("strategic_ma" in w for w in output.warnings),
                     f"{case['domain']}: a strategic M&A price must carry its uncertainty warning")
            # Citations are evidence, not adjustments — they move no price,
            # so they live in warnings now that a factor must state its
            # multiplier.
            c.expect(any("comparable acquisitions" in w for w in output.warnings),
                     f"{case['domain']}: strategic_ma must cite comparable acquisitions in warnings")

    c.expect(absent > 0, "no golden domain left targeted_lead unpriced — the 'no qualified lead' path is untested")
    c.expect(present > 0, "no golden domain produced a targeted_lead price — the priced path is untested")


def check_rationale_validator(c: Checks):
    """The number rule and the tier-scope rule, on the sectioned path."""
    from portfolio_agent.schema.rationale_view import build_rationale_view

    output, _ = run_graph("carinsurance.com")
    view = build_rationale_view(output)
    section = view.section("tier:floor")
    c.expect(section is not None, "the floor tier should have a section to test against")
    if section is None:
        return

    floor = output.tiers.floor
    supported = f"The floor price is {floor.point}, ranging from {floor.low} to {floor.high}."
    invented = "A comparable sold for 999123456, so we price this at 42424242."

    original = rationale_node.get_chat_model
    try:
        for label, replies, expect_fallback, expect_calls in (
            ("clean first draft", [supported], False, 1),
            ("recovers on retry", [invented, supported], False, 2),
            ("fails every attempt", [invented, invented], True, config.RATIONALE_MAX_ATTEMPTS),
        ):
            model = StubModel(replies)
            rationale_node.get_chat_model = lambda *a, m=model, **k: m
            text, warnings = asyncio.run(rationale_node.generate_section(model, section, view))

            # The template is the only thing that says "template" in its
            # warning; a regenerated section reports the retry instead.
            used_fallback = any("template" in w for w in warnings)
            c.expect(used_fallback == expect_fallback,
                     f"{label}: expected fallback={expect_fallback}, got {used_fallback}")
            c.expect(model.calls == expect_calls,
                     f"{label}: expected {expect_calls} model call(s), got {model.calls}")
            c.expect(not rationale_node.validate_section(text, section, view),
                     f"{label}: the returned section still fails its own validation")
            if expect_fallback:
                c.expect(any("template" in w for w in warnings),
                         f"{label}: a templated section must be disclosed in warnings")

        # A section must not reach for another tier's figures, even real ones.
        other = output.tiers.end_user
        if other:
            model = StubModel([f"The floor is {floor.point}, well under the end-user price of {other.point}."])
            rationale_node.get_chat_model = lambda *a, m=model, **k: m
            text, warnings = asyncio.run(rationale_node.generate_section(model, section, view))
            c.expect(bool(warnings), "citing another tier from a tier section must be rejected")

        # With no model, every section still renders from its template.
        rationale_node.get_chat_model = lambda *a, **k: None
        text, warnings = asyncio.run(rationale_node.generate_sectioned(output))
        c.expect(any("no LLM provider" in w for w in warnings),
                 "a missing model must be disclosed, not hidden")
        c.expect(all(s.heading in text for s in view.sections),
                 "every section must still appear when templated")
    finally:
        rationale_node.get_chat_model = original

    # The extractor itself.
    c.expect(rationale_node.extract_numbers("worth 1,250,000 at 0.85 confidence") == [1250000.0, 0.85],
             "number extraction must handle thousands separators and decimals")
    # Evidence ids and domain names must not be read as numerals, or every
    # correctly-cited rationale would be rejected for quoting "ev3".
    for text in ("ev3 cites cloud.ai", "see ev12 and p25", "hosted at 360.com"):
        c.expect(rationale_node.extract_numbers(text) == [],
                 f"{text!r} contains no claim about value, got {rationale_node.extract_numbers(text)}")
    c.expect(rationale_node.extract_numbers("priced at $57,037 (a 14% chance)") == [57037.0, 14.0],
             "currency and percentages must still be extracted")
    # A rounded magnitude must expand, or a model could write "$1.5M" for a
    # figure of 1,457,196 and pass on the bare 1.
    c.expect(rationale_node.extract_numbers("about $1.5M and 250k") == [1_500_000.0, 250_000.0],
             "magnitude suffixes must be expanded before checking")

    # A vendor float reaches the output at binary precision. A person writing
    # it rounds, and that is not an invention — a live run lost its narrative
    # twice over a model correctly rendering 0.008949999697506428 as 0.0089.
    noisy = ValuationOutput(
        domain="probe.com", category="generic", category_confidence=0.9,
        evidence=[EvidenceItem(
            id="ev1", tool="get_base_appraisal", source="s", timestamp="now",
            key_datapoints={"sales_probability": 0.008949999697506428})],
        warnings=[f"{name}: not priced" for name in TIER_ORDER],
    )
    for rendering in ("0.0089", "0.00895", "0.009"):
        c.expect(not rationale_node.unsupported_numbers(f"a probability of {rendering}", noisy),
                 f"{rendering} is a correct rendering of the stored figure and must be accepted")
    for invented in ("0.42", "0.75", "42424242"):
        c.expect(rationale_node.unsupported_numbers(f"a value of {invented}", noisy),
                 f"{invented} is absent from the output and must still be rejected")

    # And the evidence itself should not carry float noise in the first place.
    from portfolio_agent.nodes.gather_evidence import _round_floats

    c.expect(_round_floats({"p": 0.008949999697506428}) == {"p": 0.00895},
             "evidence floats must be rounded to a precision a human would write")
    c.expect(_round_floats({"nested": [{"p": 0.1234567891}]})["nested"][0]["p"] == 0.123457,
             "rounding must reach nested structures")


def check_provenance_guard(c: Checks):
    """The narrative must describe where its numbers came from correctly.

    The number validator cannot catch this — provenance is a word, not a
    figure — and a live run really did open with "This valuation is mock".
    Both directions matter: calling a live valuation fake destroys trust in
    numbers that are sound, and presenting fixture data as a real appraisal
    is how someone prices a domain off a hash of its own name.
    """
    mock_output, _ = run_graph("cloud.ai")  # mock mode is on for this eval
    c.expect(any(item.mock for item in mock_output.evidence),
             "this check needs a mock-sourced valuation to test against")

    c.expect(rationale_node.provenance_violations("These figures come from fixtures.", mock_output) == [],
             "disclosing fixture data on a mock valuation must pass")
    c.expect(rationale_node.provenance_violations("The end-user price is sound.", mock_output) != [],
             "a mock valuation whose narrative hides that must be rejected")

    # The live case, built by clearing the mock flags on the evidence.
    live_output = mock_output.model_copy(deep=True)
    for item in live_output.evidence:
        item.mock = False
    c.expect(rationale_node.provenance_violations("These figures come from live market services.", live_output) == [],
             "ordinary prose on a live valuation must pass")
    for claim in ("This valuation is mock.", "All figures come from fixtures.",
                  "These are simulated numbers.", "placeholder data only"):
        c.expect(rationale_node.provenance_violations(claim, live_output) != [],
                 f"a live valuation described as {claim!r} must be rejected")

    # And the guard drives the same retry-then-fall-back path as the number
    # check, now scoped to the section that failed.
    from portfolio_agent.schema.rationale_view import build_rationale_view

    mock_view = build_rationale_view(mock_output)
    opening = mock_view.section("classification")
    c.expect(opening is not None, "there should be a classification section to test provenance on")
    if opening is None:
        return

    c.expect(not rationale_node.section_provenance_problems(
        "These figures come from fixtures and describe no real market.", opening),
        "disclosing fixture data in the opening section must pass")
    c.expect(rationale_node.section_provenance_problems("This is a solid keyword domain.", opening),
             "a mock opening section that hides it must be rejected")

    live_view = build_rationale_view(live_output)
    live_opening = live_view.section("classification")
    c.expect(not rationale_node.section_provenance_problems(
        "These are live market figures.", live_opening),
        "ordinary prose on a live section must pass")
    for claim in ("This valuation is mock.", "All figures come from fixtures.",
                  "These are simulated numbers.", "placeholder data only"):
        c.expect(rationale_node.section_provenance_problems(claim, live_opening),
                 f"a live section described as {claim!r} must be rejected")

    original = rationale_node.get_chat_model
    try:
        model = StubModel(["These figures come from fixtures and are not real."])
        rationale_node.get_chat_model = lambda *a, m=model, **k: m
        text, warnings = asyncio.run(rationale_node.generate_section(model, live_opening, live_view))
        c.expect(bool(warnings), "a section with wrong provenance must be rejected and reported")
        c.expect(not rationale_node.section_provenance_problems(text, live_opening),
                 "the returned section must not still claim the wrong provenance")
    finally:
        rationale_node.get_chat_model = original


def check_user_profile_hook(c: Checks):
    """Accepted, echoed back untouched, and not acted on — personalization
    is explicitly out of scope for this build."""
    profile = {"risk_appetite": "Aggressive", "investment_strategy": "Quick Flip"}
    with_hook, _ = run_graph("cloud.ai", user_profile_hook=profile)
    without_hook, _ = run_graph("cloud.ai")

    c.expect(with_hook.user_profile_hook == profile,
             f"the user profile hook must be echoed back unchanged, got {with_hook.user_profile_hook}")
    c.expect(without_hook.user_profile_hook is None, "no hook supplied should stay None")
    c.expect(with_hook.tiers.model_dump() == without_hook.tiers.model_dump(),
             "the profile hook must not change any price in this build")


def main() -> int:
    c = Checks()
    was_mock = config.MOCK_MODE
    config.MOCK_MODE = True
    try:
        slowest = check_golden_set(c)
        check_latency(c, slowest)
        check_typo_collapse(c)
        check_conditional_tiers(c)
        check_rationale_validator(c)
        check_provenance_guard(c)
        check_user_profile_hook(c)
    finally:
        config.MOCK_MODE = was_mock

    if c.failures:
        print(f"FAILED {len(c.failures)}/{c.total} checks:")
        for failure in c.failures:
            print(f"  - {failure}")
        return 1
    print(f"PASSED {c.total}/{c.total} checks (slowest valuation {slowest:.2f}s, "
          f"budget {config.LATENCY_BUDGET_S:.0f}s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
