"""
Guard against the bug this engine keeps producing.
Run with: uv run python -m portfolio_agent.evals.test_no_orphaned_work

Five times in one day, work reached a data structure and stopped short of
the output:

  - comp weights, computed by the ladder and dropped by `_prices()`, so the
    quality banding was live in the comp set and inert in the price;
  - `comp_notes`, written into state since the ladder was built and read by
    nothing, so every caveat it raised about its own comps died there;
  - `weighted_prices()`, called only by its own test;
  - `secondary_categories`, which could not exist at all;
  - incumbent leads, dropped in three separate places and rejected by the
    tier they exist to price.

Every one passed its own tests, because the tests checked the producer. The
engine is deliberately layered, which is what lets a layer be correct in
isolation and inert in place — so the layering needs a check that looks
across it rather than within it.

Two checks here, and neither asserts a value:

  1. No state key is written and never read. A new orphan fails the build.
     Known-dead keys must be listed below WITH a reason, so adding one is a
     decision somebody made rather than an accident.
  2. Every tool the evidence plan runs contributes something to the output.
     A tool whose result is gathered and then summarised away is the
     `incumbents` case exactly.
"""

import asyncio
import re
import sys
from pathlib import Path

from portfolio_agent import config

# State keys that are genuinely written and never read, each with the reason
# it is tolerated. Empty is the goal. Anything NOT listed here that turns up
# orphaned fails this test, which is the point: dead state has to be
# declared, not discovered a year later.
KNOWN_ORPHANS = {
    "estimated_value_low": "superseded by TierSet; kept for one release so downstream readers "
                           "of the old shape do not break",
    "estimated_value_high": "as above",
    "typo_distance": "the deflator uses typo_severity; the raw distance is carried for "
                     "diagnostics and read only in logs",
    "structural_notes": "surfaced through the domain-profile tool's own datapoints rather than "
                        "from state",
    "rationale": "written by the rationale node onto the ValuationOutput, not consumed from "
                 "state by another node",
}


class Checks:
    def __init__(self):
        self.total = 0
        self.failures = []

    def expect(self, condition, message):
        self.total += 1
        if not condition:
            self.failures.append(message)


def _package_sources() -> dict[Path, str]:
    root = Path(__file__).resolve().parent.parent
    return {p: p.read_text(encoding="utf-8", errors="replace")
            for p in root.rglob("*.py") if "evals" not in str(p)}


def check_no_new_orphaned_state(c: Checks):
    root = Path(__file__).resolve().parent.parent
    declared = re.findall(r"^\s{4}([a-z_][a-z0-9_]*)\s*:", (root / "state.py").read_text(), re.M)
    c.expect(len(declared) > 10, f"state.py parsed as {len(declared)} keys — the scan is broken")

    sources = {p: t for p, t in _package_sources().items() if p.name != "state.py"}
    orphans = []
    for key in set(declared):
        pattern = re.compile(rf"""\.get\(["']{key}["']|\[["']{key}["']\]""")
        if not any(pattern.search(text) for text in sources.values()):
            orphans.append(key)

    new = sorted(set(orphans) - set(KNOWN_ORPHANS))
    c.expect(not new,
             f"state keys written and never read, and not declared dead: {new}. Either wire them "
             f"into the output or add them to KNOWN_ORPHANS with a reason — work that reaches a "
             f"data structure and stops is the bug this file exists to catch")

    # And the allowlist must not rot: a key listed as dead that IS read
    # should be removed from it, or the list stops meaning anything.
    stale = sorted(set(KNOWN_ORPHANS) - set(orphans))
    c.expect(not stale,
             f"listed as dead but actually read: {stale} — remove them from KNOWN_ORPHANS")


def check_every_planned_tool_reaches_the_output(c: Checks):
    """A tool gathered and then summarised away is the incumbents case."""
    from portfolio_agent.nodes.graph import portfolio_agent

    was_mock = config.MOCK_MODE
    config.MOCK_MODE = True
    try:
        state = asyncio.run(portfolio_agent.ainvoke(
            {"domain": "carinsurancequotes.com", "mock_llm": True}))
    finally:
        config.MOCK_MODE = was_mock

    # The graph returns the valuation as a dict on some paths and as the
    # model on others; both are legitimate and this check cares about
    # neither.
    valuation = state.get("valuation")
    c.expect(valuation is not None, "the graph must produce a valuation")
    if valuation is None:
        return
    evidence = valuation["evidence"] if isinstance(valuation, dict) else valuation.evidence

    def field(item, name):
        return item[name] if isinstance(item, dict) else getattr(item, name)

    by_tool = {field(item, "tool"): item for item in evidence}
    ran = [name for name in (state.get("tool_results") or {})]
    c.expect(ran, "the run must have executed tools")

    for tool in ran:
        item = by_tool.get(tool)
        c.expect(item is not None,
                 f"{tool} ran but appears in no evidence item — its result cannot be cited")
        if item is not None and field(item, "ok"):
            c.expect(field(item, "key_datapoints"),
                     f"{tool} succeeded but contributed no datapoints — it was gathered and "
                     f"then summarised away, which is how three real buyers worth $58bn "
                     f"became invisible")


def check_comp_weights_still_reach_the_price(c: Checks):
    """The first instance, pinned here too so the guard file names it.

    The full behavioural assertion lives in test_tiers.py; this is a
    cross-layer smoke check that the wiring exists at all.
    """
    import inspect

    from portfolio_agent.pricing import tier_engine

    source = inspect.getsource(tier_engine._pairs)
    c.expect('comp.get("weight")' in source,
             "the tier engine must read each comp's weight, not just its price")
    c.expect("weighted_percentile" in inspect.getsource(tier_engine.comp_stats),
             "and must carry those weights into the distribution")


def main() -> int:
    c = Checks()
    for fn in (check_no_new_orphaned_state, check_every_planned_tool_reaches_the_output,
               check_comp_weights_still_reach_the_price):
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
