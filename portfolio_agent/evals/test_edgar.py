"""
Offline regression eval for the SEC EDGAR incumbent lookup.
Run with: uv run python -m portfolio_agent.evals.test_edgar

The lead type this exists for is invisible to everything else we gather. A
profitable incumbent does not appear in funding news because it has not
raised, which is precisely why it has cash: Hartford Insurance files $28bn
of revenue and would never surface in a search for recently funded startups.

What is asserted, in order of importance:

  - A candidate is not a lead. EDGAR does not report company websites, so
    the credibility gap the targeted tier prices cannot be measured, and
    calling these leads would price an imagined problem.
  - Name matching respects word boundaries. A substring search for "forge"
    returns Forgent Power Solutions, a different word and a different
    company, and a lead list built from coincidental letter runs is worse
    than an empty one.
  - SIC maps to the same verticals as industries.py, and the ranges do not
    trip over each other — scheduled passenger air sits inside a transport
    block that is otherwise freight.
  - No contact address means unavailable, not anonymous. SEC returns 403
    without one, and the tool must say so rather than retry into a wall.
"""

import sys

from portfolio_agent import config
from portfolio_agent.tools import edgar


class Checks:
    def __init__(self):
        self.total = 0
        self.failures = []

    def expect(self, condition, message):
        self.total += 1
        if not condition:
            self.failures.append(message)


def check_sic_maps_to_our_verticals(c: Checks):
    from portfolio_agent.knowledge.industries import INDUSTRIES

    for sic, expected in ((6331, "insurance"), (7372, "software"), (8111, "legal"),
                          (6022, "finance"), (5812, "food"), (4213, "logistics"),
                          (4512, "travel"), (7011, "travel"), (6531, "real_estate")):
        actual = edgar.industry_for_sic(sic)
        c.expect(actual == expected, f"SIC {sic} mapped to {actual}, expected {expected}")

    # Passenger air nests inside a freight block; order must not file
    # airlines under haulage.
    c.expect(edgar.industry_for_sic(4512) != edgar.industry_for_sic(4513),
             "scheduled passenger air and air freight must not share a vertical")

    # Every vertical this emits must exist in the industry taxonomy, or a
    # filter against it silently matches nothing.
    for _, industry in edgar._SIC_RANGES:
        c.expect(industry in INDUSTRIES, f"SIC maps to {industry!r}, absent from industries.py")

    for junk in (None, "", "abc", -1):
        c.expect(edgar.industry_for_sic(junk) is None, f"{junk!r} must not map to a vertical")


def check_no_contact_means_unavailable(c: Checks):
    """SEC returns 403 without a contact address, and there is deliberately
    no default — an address belongs to whoever runs this, and baking one in
    would send a person's email to a third party on every future operator's
    behalf."""
    was_mock, was_ua = config.MOCK_MODE, config.SEC_USER_AGENT
    config.MOCK_MODE, config.SEC_USER_AGENT = False, ""
    try:
        import asyncio
        result = asyncio.run(edgar.find_public_incumbents("insurance"))
    finally:
        config.MOCK_MODE, config.SEC_USER_AGENT = was_mock, was_ua
    c.expect(not result["ok"], "no contact address must report unavailable")
    c.expect("SEC_USER_AGENT" in (result.get("error") or ""),
             f"and must name what to set, got {result.get('error')!r}")
    c.expect("403" in (result.get("error") or "") or "requires" in (result.get("error") or ""),
             "and must say why it cannot proceed")


def check_candidates_are_not_leads(c: Checks):
    """The honesty that keeps the highest tier from being invented."""
    import asyncio

    was_mock = config.MOCK_MODE
    config.MOCK_MODE = True
    try:
        result = asyncio.run(edgar.find_public_incumbents("insurance", industry="insurance"))
    finally:
        config.MOCK_MODE = was_mock

    data = result["data"] or {}
    c.expect("candidates" in data and "leads" not in data,
             f"EDGAR returns candidates, never leads: keys {sorted(data)}")
    for candidate in data.get("candidates", []):
        c.expect(candidate.get("current_domain") is None,
                 "EDGAR cannot know the domain a company is on and must not claim one")
        c.expect(candidate.get("lead_type") == "incumbent",
                 f"a public filer is an incumbent, got {candidate.get('lead_type')}")
    c.expect("not" in (data.get("note") or "").lower(),
             "the payload must say these are not qualified leads")


def check_word_boundaries(c: Checks):
    """'forge' must not match 'Forgent'."""
    import re

    pattern = re.compile(r"(?<![a-z])forge(?![a-z])", re.IGNORECASE)
    c.expect(pattern.search("Forge Group, Inc."), "the real match must survive")
    c.expect(not pattern.search("Forgent Power Solutions, Inc."),
             "a different word must not match")
    c.expect(not pattern.search("MindForged Inc"), "nor a suffixed one")
    # A compound ending in the keyword is NOT matched, and that is the
    # deliberate trade: the same permissiveness that admits MindForge for
    # "forge" admits Build-A-Bear Workshop for "hop", and an invented lead
    # prices the highest tier in the system.
    c.expect(not pattern.search("MindForge Inc."),
             "a compound is not matched — the cost of keeping short keywords honest")
    hop = re.compile(r"(?<![a-z])hop(?![a-z])", re.IGNORECASE)
    c.expect(not hop.search("BUILD-A-BEAR WORKSHOP INC"),
             "'hop' must not match 'Workshop' — this was a real false lead")
    c.expect(hop.search("Hop Industries Corp"), "but a company actually called Hop still matches")


def main() -> int:
    c = Checks()
    for fn in (check_sic_maps_to_our_verticals, check_no_contact_means_unavailable,
               check_candidates_are_not_leads, check_word_boundaries):
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
