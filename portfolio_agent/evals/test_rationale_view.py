"""
Offline regression eval for the rationale view and sectioned generation.
Run with: uv run python -m portfolio_agent.evals.test_rationale_view

Three narratives have now been factually wrong while every number in them
was real. This file replays all three as fixtures, because they are one
failure class and a numeric validator cannot catch any of them:

  1. "the list_bin tier … suggests a range of $425" — list_bin was not
     priced. Every figure real, the claim false.
  2. "classified as an emerging market domain (emd)" — emd is Exact Match
     Domain. A model asked to expand an abbreviation expanded it wrongly.
  3. "the wholesale_value of $96 … is the only populated tier" — wholesale
     value is a GoValue field, not a tier of this engine.

The fix under test is not a better checker. It is that the model is never
handed material it can be wrong about: an unpriced tier is ABSENT rather
than marked, labels arrive pre-expanded, and third-party estimates arrive
under a heading that says what they are.
"""

import sys
from datetime import date

from portfolio_agent.nodes import rationale as rationale_node
from portfolio_agent.schema.rationale_view import Section, LABELS, build_rationale_view, label
from portfolio_agent.schema.valuation_output import (
    IndependentAppraisal,
    TIER_ORDER,
    CompBasis,
    CompRef,
    EvidenceItem,
    PriceTier,
    TierSet,
    ValuationOutput,
)


def output(**overrides) -> ValuationOutput:
    """The consulting.ai shape: some tiers priced, some not, a vendor
    appraisal present, one cheap comp."""
    kwargs = dict(
        domain="consulting.ai",
        category="emd",
        category_confidence=0.8,
        comp_basis=CompBasis.COHORT,
        comp_confidence=0.45,
        comp_set=[CompRef(domain="primeconsulting.ai", price=425, source="sales_history")],
        tiers=TierSet(
            floor=PriceTier(low=300, point=900, high=2_000),
            auction=PriceTier(low=800, point=1_900, high=4_100),
            end_user=PriceTier(low=3_000, point=9_100, high=20_000),
        ),
        evidence=[
            EvidenceItem(id="ev1", tool="get_base_appraisal", source="dvs_godaddy", timestamp="now",
                         key_datapoints={"go_value": 2_220, "wholesale_value": 96}),
            EvidenceItem(id="ev2", tool="get_domain_profile", source="dvs_estibot", timestamp="now",
                         key_datapoints={"estibot_value": 20_000}),
        ],
        independent_appraisal=IndependentAppraisal(
            value=11_110, source="median of 2 baselines",
            all_baselines={"godaddy_govalue": 2_220, "estibot": 20_000}, method="median"),
        warnings=[
            "list_bin: only 1 comparable sale found, below the 3 needed for a distribution",
            "targeted_lead: no clear, high-value end-user identified by the profiling tool",
            "strategic_ma: not a category-defining domain — .ai is not a .com",
        ],
    )
    kwargs.update(overrides)
    return ValuationOutput(**kwargs)


class Checks:
    def __init__(self):
        self.total = 0
        self.failures: list[str] = []

    def expect(self, condition, message: str):
        self.total += 1
        if not condition:
            self.failures.append(message)


def check_unpriced_tiers_are_absent(c: Checks):
    """Failure 1. A model cannot mischaracterise what it was never shown."""
    view = build_rationale_view(output())
    tier_sections = {s.key.removeprefix("tier:") for s in view.sections if s.key.startswith("tier:")}

    c.expect(tier_sections == {"floor", "auction", "end_user"},
             f"only priced tiers may have sections, got {sorted(tier_sections)}")
    for unpriced in ("list_bin", "targeted_lead", "strategic_ma"):
        c.expect(unpriced not in tier_sections, f"{unpriced} is not priced and must have no section")
        c.expect(unpriced not in view.tier_labels,
                 f"{unpriced} must not be in tier_labels — nothing may reference it as a tier")

    # Its absence is explained in its own section, from supplied text.
    not_priced = view.section("not_priced")
    c.expect(not_priced is not None, "unpriced valuations must be explained in their own section")
    rendered = str(not_priced.content)
    c.expect("Buy-It-Now" in rendered, "the BIN tier's absence must be stated in plain language")
    c.expect("below the 3 needed" in rendered,
             "the supplied reason must be carried verbatim so the model need not guess it")

    # The $425 comp exists, but nowhere near a tier the model could attach
    # it to — that is exactly how "list_bin suggests a range of $425" arose.
    bin_section = view.section("tier:list_bin")
    c.expect(bin_section is None, "there must be no BIN section to attach a figure to")


def check_labels_are_pre_expanded(c: Checks):
    """Failure 2. A model handed an expansion cannot expand it wrongly."""
    view = build_rationale_view(output())
    classification = str(view.section("classification").content)

    c.expect("exact match domain (EMD)" in classification,
             f"emd must arrive expanded, got {classification[:160]}")
    c.expect("emerging" not in classification.lower(),
             "nothing in the classification slice should suggest 'emerging'")

    # No raw machine name reaches the model anywhere in the view.
    for section in view.sections:
        for raw in ("emd", "list_bin", "end_user", "sld_cross_tld", "strategic_ma"):
            keys = str(list(section.content.keys()))
            c.expect(raw not in keys,
                     f"section {section.key!r} exposes the raw field name {raw!r} in its keys")

    c.expect(label("emd").startswith("exact match domain"), "LABELS must expand emd")
    c.expect(label("cohort").startswith("priced from the cohort"), "LABELS must expand the comp basis")
    for name in TIER_ORDER:
        c.expect(name in LABELS, f"every tier needs a label, {name} has none")


def check_third_party_is_not_a_tier(c: Checks):
    """Failure 3. A vendor estimate arrives labeled as one."""
    view = build_rationale_view(output())
    references = view.section("reference_points")

    c.expect(references is not None, "vendor appraisals must get their own section")
    rendered = str(references.content)
    c.expect("not one of this engine's price tiers" in rendered,
             "the section must say plainly that these are not tiers")
    c.expect("GoDaddy GoValue" in rendered and "2220" in rendered.replace(",", ""),
             f"the GoValue estimate must be present and attributed, got {rendered[:200]}")
    c.expect("Estibot" in rendered, "the Estibot estimate must be present and attributed")

    # And it is in no tier section, where it would read as a tier.
    for section in view.sections:
        if section.key.startswith("tier:"):
            c.expect("GoValue" not in str(section.content),
                     f"a vendor estimate must not appear inside {section.key}")


def check_floats_are_pre_rounded(c: Checks):
    """A model shown 0.008949999697506428 writes 0.0089, correctly, and was
    being rejected for it. Round before it is shown."""
    noisy = output(
        tiers=TierSet(floor=PriceTier(low=300, point=900, high=2_000, sale_probability=0.008949999697506428)),
        warnings=[f"{n}: not priced" for n in TIER_ORDER if n != "floor"],
    )
    view = build_rationale_view(noisy)
    rendered = str(view.section("tier:floor").content)
    c.expect("0.008949999697506428" not in rendered, "raw float precision must never reach the model")
    c.expect("0.0089" in rendered, f"the probability should arrive at 2 significant figures, got {rendered[:200]}")


def check_section_numeric_scope(c: Checks):
    """A section may state its own figures and no others.

    A whole-output allowlist let a tier's paragraph quote another tier's
    price — which is the cross-tier invention the slicing exists to stop.
    """
    view = build_rationale_view(output())
    floor = view.section("tier:floor")
    auction = view.section("tier:auction")

    allowed = rationale_node.section_allowed_numbers(floor)
    c.expect(900.0 in allowed, "a section must permit its own price")
    c.expect(1_900.0 not in allowed,
             "a section must NOT permit another tier's price, even though it is in the output")

    c.expect(not rationale_node.validate_section("The floor is 900, ranging 300 to 2000.", floor, view),
             "a section using only its own figures must pass")
    c.expect(rationale_node.validate_section("The floor is 900, below the auction at 1900.", floor, view),
             "quoting another tier's price must be rejected")
    c.expect(1_900.0 in rationale_node.section_allowed_numbers(auction),
             "the auction section may of course state the auction price")


def check_foreign_tier_mentions(c: Checks):
    """The lexical half. 'the BIN tier suggests a range of $425' contains
    only real figures and is still false."""
    view = build_rationale_view(output())
    floor = view.section("tier:floor")

    c.expect(not rationale_node.foreign_tier_mentions("The floor price is the quick-exit number.", floor, view),
             "a section naming its own tier must pass")
    c.expect(rationale_node.foreign_tier_mentions("This sits below the Auction price entirely.", floor, view),
             "naming another tier must be caught")
    c.expect(rationale_node.foreign_tier_mentions("The End-User price is higher.", floor, view),
             "naming the end-user tier from the floor section must be caught")

    # And the historical sentence itself, on a view where BIN is unpriced.
    evidence = view.section("evidence")
    c.expect(not rationale_node.foreign_tier_mentions(
        "the marketplace Buy-It-Now (BIN) price suggests a range of 425", evidence, view),
        "an unpriced tier is not in tier_labels, so this check cannot see it — "
        "which is why absence, not detection, is the real fix")
    c.expect(view.section("tier:list_bin") is None,
             "and the real protection is that no BIN section exists to write that sentence in")


def check_assembly_and_fallback(c: Checks):
    """Structure is fixed by code, and failure is section-scoped."""
    view = build_rationale_view(output())
    assembled = rationale_node._assemble([(s, f"body of {s.key}") for s in view.sections])
    for section in view.sections:
        c.expect(section.heading in assembled, f"the assembled narrative must carry {section.key}'s heading")
    c.expect(assembled.index("Classification") < assembled.index("Evidence"),
             "section order is fixed by code, not chosen by the model")

    template = rationale_node._section_template(view.section("tier:floor"))
    c.expect("900" in template, "a templated section must still state its figures")
    c.expect(not rationale_node.validate_section(template, view.section("tier:floor"), view),
             "the template must itself pass validation — a fallback that fails is not a fallback")

    # Verbose structures are summarized, not dumped.
    evidence_template = rationale_node._section_template(view.section("evidence"))
    c.expect(len(evidence_template) < 600,
             f"the evidence template must summarize, not dump ({len(evidence_template)} chars)")
    c.expect("recorded in the structured output" in evidence_template,
             "summarized fields should point at where the detail lives")


def check_spelled_numbers_are_validated(c: Checks):
    """The escape hatch: 26 was rejected as invented while twenty-six passed.

    The rationale's one hard rule is that every figure in the prose exists
    in the structured output. A model could satisfy it by spelling the
    figure out — "nine hundred thousand dollars" was invisible to the
    validator entirely.
    """
    from portfolio_agent.nodes.rationale import extract_number_words, extract_numbers

    for text, expected in (
        ("twenty-six comps", [26.0]),
        ("nine hundred thousand dollars", [900_000.0]),
        ("a hundred and twenty", [120.0]),
        ("two million", [2_000_000.0]),
        ("fifty thousand", [50_000.0]),
    ):
        actual = extract_number_words(text)
        c.expect(actual == expected, f"extract_number_words({text!r}) = {actual}, expected {expected}")

    # Ordinals name a position, not an amount. Parsing "twenty" out of
    # "twenty-fifth percentile" would reject a correct sentence.
    for text in ("the twenty-fifth percentile", "the seventy-fifth percentile",
                 "the second tier", "the third rung"):
        c.expect(extract_number_words(text) == [],
                 f"{text!r} is a position, not a quantity: got {extract_number_words(text)}")

    # And the two extractors agree, so spelling a figure is not a loophole.
    c.expect(extract_numbers("26 comps") == extract_numbers("twenty-six comps"),
             "a spelled number must validate identically to its digits")


def check_tier_relations_are_computed_and_linted(c: Checks):
    """F6. Six prices with nothing joining them.

    Every tier section is scoped to its own tier, which stops cross-tier
    invention and also stopped the report ever explaining why the end-user
    figure is several times the auction figure — the question a reader
    actually has. One section may compare, its comparisons are arithmetic
    rather than narration, and its claims are linted.
    """
    from portfolio_agent.nodes.rationale import foreign_tier_mentions, relational_problems
    from portfolio_agent.pricing.relations import magnitude_for, tier_relations

    # The magnitude word is a function of the ratio, not a judgement.
    for ratio, expected in ((1.02, "marginally"), (1.3, "moderately"),
                            (2.1, "substantially"), (9.0, "dramatically")):
        c.expect(magnitude_for(ratio) == expected,
                 f"magnitude_for({ratio}) = {magnitude_for(ratio)}, expected {expected}")

    view = build_rationale_view(output())
    section = view.section("tier_relations")
    c.expect(section is not None, "a priced ladder must get a relations section")

    # It is the one section allowed to name more than one tier.
    c.expect(foreign_tier_mentions("The Auction price and the Floor price.", section, view) == [],
             "the relations section must be exempt from the tier-scope rule")
    other = view.section("classification")
    c.expect(foreign_tier_mentions("The Auction price is high.", other, view) != [],
             "every other section must still be scoped to its own tier")

    steps = (section.content or {}).get("steps") or []
    c.expect(steps, "the relations section must carry computed steps")
    c.expect(all("multiple" in s and "magnitude" in s and "why" in s for s in steps),
             f"each step needs its multiple, its magnitude and its reason: {steps}")

    fixture = Section(key="tier_relations", heading="H", instruction="x", content={"steps": [
        {"from": "Floor price", "to": "Auction price", "multiple": 1.95,
         "magnitude": "substantially", "why": "w"}]})

    # Direction is a fact, and a false one is caught.
    c.expect(relational_problems("The Auction price is substantially above the Floor price.",
                                 fixture) == [],
             "a correct comparison must pass")
    c.expect(relational_problems("The Floor price is above the Auction price.", fixture),
             "a reversed comparison must be caught")
    c.expect(relational_problems("The Auction price is lower than the Floor price.", fixture),
             "a reversed comparison must be caught whichever way it is phrased")

    # Magnitude is supplied, so choosing a different one is invention.
    c.expect(relational_problems("The Auction price is dramatically above the Floor price.",
                                 fixture),
             "a magnitude belonging to no computed step must be caught")
    c.expect(relational_problems("The Auction price is well above the Floor price.", fixture) == [],
             "ordinary prose outside the closed vocabulary must not be policed")

    # A ratio written as a word carries no numeral, so every numeric check
    # is blind to it. A live run produced "doubling", "tripling" and "a
    # single multiple" — all roughly right, which is not the same as
    # verified.
    two_steps = Section(key="tier_relations", heading="H", instruction="x", content={"steps": [
        {"from": "Floor price", "to": "Auction price", "multiple": 1.95,
         "magnitude": "substantially", "why": "w"},
        {"from": "Auction price", "to": "End-User price", "multiple": 3.30,
         "magnitude": "dramatically", "why": "w"}]})
    for text in ("The step is a doubling.", "The step is a tripling.",
                 "roughly twice the floor price", "a single multiple above"):
        c.expect(relational_problems(text, two_steps) == [],
                 f"{text!r} fairly describes a computed step and must pass")
    for text in ("It quadruples.", "The jump is tenfold.", "This halves the price.",
                 "An order of magnitude higher."):
        c.expect(relational_problems(text, two_steps),
                 f"{text!r} claims a ratio no step supports and must be caught")

    # Adjacent ratio words must not overlap: any band loose enough to accept
    # "doubling" for 1.95x also accepts "quadrupling" for 3.3x, which is a
    # different claim about the same number. Nearest-word in log space needs
    # no threshold.
    from portfolio_agent.nodes.rationale import _nearest_ratio_word
    for multiple, expected in ((1.95, 2.0), (3.3, 3.0), (4.2, 4.0), (9.0, 10.0)):
        c.expect(_nearest_ratio_word(multiple) == expected,
                 f"_nearest_ratio_word({multiple}) = {_nearest_ratio_word(multiple)}, "
                 f"expected {expected}")

    # And the lint applies only where comparing is allowed.
    elsewhere = Section(key="tier:auction", heading="H", instruction="x", content={})
    c.expect(relational_problems("The Floor price is above the Auction price.", elsewhere) == [],
             "the relational lint must not fire on sections that carry no steps")


def main() -> int:
    c = Checks()
    for check in (
        check_spelled_numbers_are_validated,
        check_tier_relations_are_computed_and_linted,
        check_unpriced_tiers_are_absent,
        check_labels_are_pre_expanded,
        check_third_party_is_not_a_tier,
        check_floats_are_pre_rounded,
        check_section_numeric_scope,
        check_foreign_tier_mentions,
        check_assembly_and_fallback,
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
