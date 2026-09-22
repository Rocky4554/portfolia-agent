"""
Offline regression eval for multi-label classification.
Run with: uv run python -m portfolio_agent.evals.test_multilabel

The failure this exists to prevent: the category selects the evidence plan,
and the classifier returned exactly one. So accounting.ai was a keyword
domain with no reading of its extension's trend cycle, and forge.ai was an
emerging-TLD domain with no keyword work — each gathering one market's
evidence and not the other's, and neither saying anything was missing.

What is asserted, in order of importance:

  - The union actually runs. A second label that adds no tools would be
    decoration.
  - The primary still leads. It drives the buyer persona, the plan order and
    anything downstream needing one answer, so multi-label must not become
    no-label.
  - A domain with nothing extra to say carries no secondary labels. Labels
    invented to look thorough are worse than none.
  - The labels reach the output and the rationale view, since evidence
    gathered for a reason the report never states is evidence the reader
    cannot check.
"""

import sys

from portfolio_agent.knowledge.category_playbook import PLAYBOOK, plan_for, plan_for_all
from portfolio_agent.schema.rationale_view import build_rationale_view
from portfolio_agent.tools.linguistics import classify_domain


class Checks:
    def __init__(self):
        self.total = 0
        self.failures = []

    def expect(self, condition, message):
        self.total += 1
        if not condition:
            self.failures.append(message)


def check_labels(c: Checks):
    for domain, primary, secondary in (
        ("accounting.ai", "keyword", ["emerging_tld"]),
        ("forge.ai", "emerging_tld", ["generic"]),
        ("quantum.io", "emerging_tld", ["generic"]),
        ("xyz.io", "acronym", ["emerging_tld"]),
        # A .com keyword domain has one story and must not be given two.
        ("carinsurance.com", "emd", []),
        ("propertymanagment.com", "emd", []),
    ):
        result = classify_domain(domain)
        names = [name for name, _ in result.secondary_categories]
        c.expect(result.category == primary,
                 f"{domain}: primary is {result.category}, expected {primary}")
        c.expect(names == secondary,
                 f"{domain}: secondary labels {names}, expected {secondary}")


def check_single_label_is_unchanged(c: Checks):
    """A .com with one story must run exactly the plan it always ran."""
    for category in PLAYBOOK:
        c.expect(plan_for_all([category]).tool_ids() == plan_for(category).tool_ids(),
                 f"a lone {category} label must not change its own plan")


def check_union_adds_real_tools(c: Checks):
    """The point of the whole change."""
    keyword_only = set(plan_for("keyword").tool_ids())
    merged = plan_for_all(["keyword", "emerging_tld"])
    added = [t for t in merged.tool_ids() if t not in keyword_only]
    c.expect(len(added) >= 3,
             f"keyword + emerging_tld must gather materially more, added only {added}")
    c.expect("scan_market_trends" in added,
             "a domain on a trend-sensitive extension must get a trend reading")

    # No tool runs twice, however many labels overlap.
    ids = plan_for_all(["geo", "keyword", "emerging_tld"]).tool_ids()
    c.expect(len(ids) == len(set(ids)), f"the merged plan repeats a tool: {ids}")


def check_primary_still_leads(c: Checks):
    """Multi-label must not become no-label."""
    merged = plan_for_all(["keyword", "emerging_tld"])
    primary = plan_for("keyword")
    c.expect(merged.category == primary.category, "the merged plan keeps the primary's category")
    c.expect(merged.buyer_persona == primary.buyer_persona,
             "the buyer persona comes from the primary label, not a blend")
    c.expect(merged.tool_ids()[:len(primary.steps)] == primary.tool_ids(),
             "the primary's steps must run first and in their own order")

    # Borrowed steps say why they are there.
    borrowed = [s for s in merged.steps if s.tool not in set(primary.tool_ids())]
    c.expect(all("also reads as" in s.reason for s in borrowed),
             f"a borrowed step must state which label brought it: {[s.reason for s in borrowed]}")


def check_labels_reach_the_report(c: Checks):
    """Evidence gathered for a reason the report never states cannot be
    checked by the reader."""
    from portfolio_agent.schema.valuation_output import (
        CompBasis, EvidenceItem, PriceTier, Signals, TierSet, ValuationOutput,
    )
    output = ValuationOutput(
        domain="accounting.ai", category="keyword", category_confidence=0.9,
        secondary_categories=[("emerging_tld", 0.85)],
        tiers=TierSet(auction=PriceTier(
            low=1, point=2, high=3, inflationary_factors=[], deflationary_factors=[],
            evidence_refs=["ev1"], comp_basis=CompBasis.COHORT)),
        comp_basis=CompBasis.COHORT, comp_confidence=0.45,
        evidence=[EvidenceItem(id="ev1", tool="t", source="s", timestamp="now", key_datapoints={})],
        signals=Signals(), rationale="x",
        warnings=["floor: not priced", "end_user: not priced", "list_bin: not priced",
                  "targeted_lead: not priced", "strategic_ma: not priced"],
    )
    c.expect(output.secondary_categories == [("emerging_tld", 0.85)],
             "the schema must carry secondary labels")

    view = build_rationale_view(output)
    section = view.section("classification")
    rendered = str(section.content)
    c.expect("also_reads_as" in rendered,
             f"the rationale view must expose the extra labels, got {rendered[:200]}")
    c.expect("emerging" in rendered.lower(),
             "and must name them in expanded form, not as a raw enum")


def main() -> int:
    c = Checks()
    for fn in (check_labels, check_single_label_is_unchanged, check_union_adds_real_tools,
               check_primary_still_leads, check_labels_reach_the_report):
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
