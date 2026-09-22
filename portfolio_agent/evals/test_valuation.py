"""
Cheap regression eval — no API keys, no network, no LangGraph runtime needed.
Run with: python3 -m portfolio_agent.evals.test_valuation

Checks two things:
  1. Classification/typo-detection matches the golden set (pure logic).
  2. The tool-tier guardrail actually blocks a tool it shouldn't allow —
     a guardrail nobody tests is a guardrail nobody can trust.
"""

import asyncio
import sys

from portfolio_agent import config
from portfolio_agent.evals.golden_set import (
    CATEGORY_DEFINING_DOMAINS,
    COINED_DOMAINS,
    NOT_CATEGORY_DEFINING_DOMAINS,
    EXPECTED_CATEGORIES,
    GOLDEN_SET,
    REAL_WORD_DOMAINS,
)
from portfolio_agent.guardrails import ToolTierError, require_tier
from portfolio_agent.knowledge.category_playbook import (
    ALL_TOOL_IDS,
    PLAYBOOK,
    TOOL_BRAND_SCORE,
    TOOL_KEYWORD_SALES,
    TOOL_MARKET_TRENDS,
    TOOL_SEO_METRICS,
    TOOL_SIMILAR_SALES,
    plan_for,
    plan_for_all,
)
from portfolio_agent.schema.valuation_output import ValuationOutput
from portfolio_agent.tools.action_schema import ProposedAction, filter_valid_actions
from portfolio_agent.tools.linguistics import classify_domain


def run_classification_checks() -> list[str]:
    failures = []
    for case in GOLDEN_SET:
        result = classify_domain(case["domain"])
        if result.category != case["expected_category"]:
            failures.append(
                f"{case['domain']}: expected category '{case['expected_category']}', "
                f"got '{result.category}'"
            )
            continue
        expected_typo_of = case.get("expected_typo_of")
        if expected_typo_of and result.typo_of != expected_typo_of:
            failures.append(
                f"{case['domain']}: expected typo_of '{expected_typo_of}', got '{result.typo_of}'"
            )
    return failures


def run_guardrail_checks() -> list[str]:
    failures = []
    try:
        require_tier("some_mutating_tool", "mutating")
        failures.append(
            "guardrail did not raise: a 'mutating' tool was allowed under the "
            "'read_only' deployment tier"
        )
    except ToolTierError:
        pass  # expected — this is the guardrail working correctly.

    try:
        require_tier("classify_domain", "read_only")
    except ToolTierError:
        failures.append("guardrail incorrectly blocked a 'read_only' tool under a 'read_only' tier")

    return failures


def run_manager_action_validation_checks() -> list[str]:
    """A hallucinated id must never survive filter_valid_actions — this is
    the guardrail that stops the manager agent's plan from ever proposing an
    action against a domain/folder the caller didn't actually provide."""
    failures = []

    real_id = "11111111-1111-1111-1111-111111111111"
    fake_id = "99999999-9999-9999-9999-999999999999"

    actions = [
        ProposedAction(type="drop_domain", domain_id=real_id, reason="real"),
        ProposedAction(type="drop_domain", domain_id=fake_id, reason="hallucinated — must be dropped"),
        ProposedAction(type="create_folder", folder_name="New Folder", reason="no id needed, always kept"),
    ]
    result = filter_valid_actions(actions, valid_domain_ids={real_id}, valid_folder_ids=set())

    kept_domain_ids = {a.domain_id for a in result if a.domain_id}
    if fake_id in kept_domain_ids:
        failures.append("filter_valid_actions let a hallucinated domain_id survive")
    if real_id not in kept_domain_ids:
        failures.append("filter_valid_actions incorrectly dropped a real domain_id")
    if len(result) != 2:
        failures.append(f"expected 2 surviving actions (1 valid domain action + the id-less folder action), got {len(result)}")

    return failures


def run_dictionary_checks() -> list[str]:
    """A real word is not a coined name, not a typo, and not a compound.

    All three failed at once on voice.com: "voice" is absent from the
    400-word commercial lexicon, so it segmented as an unknown token, read as
    an invented name, and was disqualified from the category-defining test —
    on a domain that knowledge/ma_cases.py records selling for $30,000,000.
    """
    from portfolio_agent.knowledge.dictionary import is_common_noun, is_dictionary_word
    from portfolio_agent.knowledge.lexicon import segment
    from portfolio_agent.pricing.targeted import is_category_defining
    from portfolio_agent.pricing.tier_engine import TierInputs

    failures = []

    for domain in REAL_WORD_DOMAINS:
        sld = domain.split(".")[0]
        result = classify_domain(domain)

        if segment(sld) != [sld]:
            failures.append(f"{domain}: a real word must not be split, got {segment(sld)}")
        if result.typo_of is not None:
            failures.append(f"{domain}: a real word is not a misspelling, got typo_of={result.typo_of}")
        if "coined" in result.subcategory or "invented" in result.subcategory:
            failures.append(f"{domain}: read as coined — {result.subcategory!r}")
        if result.category not in ("generic", "emerging_tld", "keyword", "emd"):
            failures.append(f"{domain}: unexpected category {result.category!r} for a real word")

    for domain in COINED_DOMAINS:
        sld = domain.split(".")[0]
        if is_dictionary_word(sld):
            failures.append(f"{domain}: {sld!r} should not be a dictionary word")
        if classify_domain(domain).category != "brandable":
            failures.append(f"{domain}: a coined name must stay brandable, got {classify_domain(domain).category}")

    # A dictionary holds names as well as words, and only one of those can
    # name a category. Both directions are asserted because both errors are
    # expensive: a false negative costs a real tier, a false positive
    # invents one.
    for domain in CATEGORY_DEFINING_DOMAINS:
        sld = domain.split(".")[0]
        defining, reason = is_category_defining(
            TierInputs(domain=domain, sld=sld, tld="com", category="generic")
        )
        if not defining:
            failures.append(f"{domain} must qualify as category-defining — got: {reason}")

    for domain in NOT_CATEGORY_DEFINING_DOMAINS:
        sld = domain.split(".")[0]
        defining, reason = is_category_defining(
            TierInputs(domain=domain, sld=sld, tld="com", category="generic")
        )
        if defining:
            failures.append(
                f"{domain} is a proper noun and must NOT qualify as category-defining — got: {reason}"
            )
        elif "per million" not in reason:
            failures.append(f"{domain}: the rejection should state the frequency evidence, got: {reason}")

    if is_category_defining(
        TierInputs(domain="zylantix.com", sld="zylantix", tld="com", category="brandable")
    )[0]:
        failures.append("a coined name must not qualify as category-defining")

    # The separation the threshold rests on, asserted rather than assumed.
    for word in ("insurance", "voice", "coffee"):
        if not is_common_noun(word):
            failures.append(f"{word!r} should read as a common noun")
    for word in ("crusoe", "gatsby", "rembrandt"):
        if is_common_noun(word):
            failures.append(f"{word!r} is a proper noun and should not read as a common noun")
        if not is_dictionary_word(word):
            failures.append(f"{word!r} should still be a dictionary word — that is what makes it a trap")

    return failures


def run_routing_checks() -> list[str]:
    """Every category must route to the evidence plan its value drivers
    call for (APVA Table 3.2.1 / build spec §1.2). This is the check that
    stops the taxonomy from quietly degenerating into one generic plan."""
    failures = []

    required = {
        "brandable": {TOOL_SIMILAR_SALES, TOOL_MARKET_TRENDS, TOOL_BRAND_SCORE},
        "emerging_tld": {TOOL_SIMILAR_SALES, TOOL_MARKET_TRENDS, TOOL_BRAND_SCORE},
        "combination": {TOOL_BRAND_SCORE, TOOL_KEYWORD_SALES},
        "keyword": {TOOL_KEYWORD_SALES, TOOL_SEO_METRICS},
        "emd": {TOOL_KEYWORD_SALES, TOOL_SEO_METRICS},
        "generic": {TOOL_KEYWORD_SALES},
        "acronym": {TOOL_SIMILAR_SALES},
        "geo": {TOOL_KEYWORD_SALES, TOOL_SIMILAR_SALES},
    }
    for category, must_include in required.items():
        missing = must_include - set(plan_for(category).tool_ids())
        if missing:
            failures.append(f"category '{category}' plan is missing {sorted(missing)}")

    uncovered = EXPECTED_CATEGORIES - set(PLAYBOOK)
    if uncovered:
        failures.append(f"categories with no evidence plan: {sorted(uncovered)}")

    covered = {case["expected_category"] for case in GOLDEN_SET}
    if EXPECTED_CATEGORIES - covered:
        failures.append(f"golden set has no case for: {sorted(EXPECTED_CATEGORIES - covered)}")

    # Brand scoring is for the categories whose value IS the name (build spec
    # §2.2). Running it on an EMD would be wasted work and a misleading
    # component in the output.
    for category in ("generic", "keyword", "emd", "acronym", "geo"):
        if TOOL_BRAND_SCORE in plan_for(category).tool_ids():
            failures.append(f"category '{category}' should not run {TOOL_BRAND_SCORE}")

    return failures


def run_schema_checks() -> list[str]:
    """Every golden domain must produce a schema-valid ValuationOutput with
    the category it was routed as, running entirely offline in mock mode."""
    from portfolio_agent.nodes.classify import classify
    from portfolio_agent.nodes.gather_evidence import gather_evidence
    from portfolio_agent.nodes.synthesize import synthesize

    failures = []
    was_mock = config.MOCK_MODE
    config.MOCK_MODE = True
    try:
        for case in GOLDEN_SET:
            state = {"domain": case["domain"]}
            try:
                state.update(classify(state))
                state.update(asyncio.run(gather_evidence(state)))
                state.update(asyncio.run(synthesize(state)))
                output = ValuationOutput.model_validate(state["valuation"])
            except Exception as e:  # noqa: BLE001 — the eval's job is to report, not propagate
                failures.append(f"{case['domain']}: graph did not produce a valid output ({type(e).__name__}: {e})")
                continue

            if output.category.value != case["expected_category"]:
                failures.append(
                    f"{case['domain']}: output category '{output.category.value}' "
                    f"!= expected '{case['expected_category']}'"
                )
            # The plan is the union across every label the classifier
            # returned, not the primary's alone — a keyword domain on .ai
            # legitimately runs the emerging-TLD steps too.
            labels = [case["expected_category"]] + [
                name for name, _ in classify_domain(case["domain"]).secondary_categories
            ]
            expected_tools = set(plan_for_all(labels).tool_ids())
            # Only tool-sourced evidence. The comp ladder also emits an
            # evidence item so the rationale can cite which rung it priced
            # from, but it is a computation over tool results rather than a
            # tool, and it is in no plan.
            ran = {item.tool for item in output.evidence if item.tool in ALL_TOOL_IDS}
            if ran != expected_tools:
                failures.append(f"{case['domain']}: ran {sorted(ran)}, plan says {sorted(expected_tools)}")
            if not any(item.mock for item in output.evidence):
                failures.append(f"{case['domain']}: mock mode produced no evidence flagged mock=True")
            if not any("MOCK MODE" in w for w in output.warnings):
                failures.append(f"{case['domain']}: mock-mode run is not labeled in warnings")
    finally:
        config.MOCK_MODE = was_mock
    return failures


def main() -> int:
    failures = (
        run_classification_checks()
        + run_guardrail_checks()
        + run_manager_action_validation_checks()
        + run_routing_checks()
        + run_dictionary_checks()
        + run_schema_checks()
    )
    total_checks = len(GOLDEN_SET) * 2 + 2 + 3 + len(PLAYBOOK) + 3

    if failures:
        print(f"FAILED {len(failures)}/{total_checks} checks:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"PASSED {total_checks}/{total_checks} checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
