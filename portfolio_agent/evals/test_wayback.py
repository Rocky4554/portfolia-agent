"""
Offline regression eval for archive history and historical listings.
Run with: uv run python -m portfolio_agent.evals.test_wayback

The historical listing is the one signal here nobody is charging for. A
domain that carried an Afternic lander at $12,000 in 2021 and serves a
working site today tells you two things no current-state check can: the
owner has been willing to sell, and roughly where they started.

What is asserted, in order of importance:

  - A historical ask never reaches the comp set. Nobody paid it, and it is
    not even a current ask — the seller may have moved or given up since.
    It is the weakest price evidence in the system and is named so that
    nothing downstream can mistake it for a sale.
  - Finding no lander is reported as a SAMPLE, not a search. The archive
    does not capture every day and a lander can come and go between
    captures, so "we looked at four snapshots" must never read as "this was
    never for sale".
  - A throttled or slow archive is a failed lookup, never an empty history.
    A rate limit must not masquerade as evidence a domain was never
    developed.
  - It stays off by default. The CDX endpoint measured ~35s per call,
    against a 60s valuation budget.
"""

import asyncio
import sys
from datetime import date

from portfolio_agent import config
from portfolio_agent.tools import wayback


class Checks:
    def __init__(self):
        self.total = 0
        self.failures = []

    def expect(self, condition, message):
        self.total += 1
        if not condition:
            self.failures.append(message)


def check_off_by_default(c: Checks):
    was_mock, was_enabled = config.MOCK_MODE, config.ENABLE_WAYBACK
    config.MOCK_MODE, config.ENABLE_WAYBACK = False, False
    try:
        for call in (wayback.get_site_history("example.com"),
                     wayback.find_historical_listings("example.com")):
            result = asyncio.run(call)
            c.expect(not result["ok"], "archive lookups must be disabled by default")
            c.expect("ENABLE_WAYBACK" in (result.get("error") or ""),
                     f"and must name the flag, got {result.get('error')!r}")
    finally:
        config.MOCK_MODE, config.ENABLE_WAYBACK = was_mock, was_enabled


def check_historical_asks_are_not_comps(c: Checks):
    was_mock = config.MOCK_MODE
    config.MOCK_MODE = True
    try:
        # A fixture that definitely has a listing in it.
        data = next(
            d for d in (asyncio.run(wayback.find_historical_listings(f"d{i}.com"))["data"]
                        for i in range(12))
            if d["ever_listed"])
    finally:
        config.MOCK_MODE = was_mock

    c.expect("historical_asks" in data and "comps" not in data,
             f"the payload must expose asks and no comps, got keys {sorted(data)}")
    for ask in data["historical_asks"]:
        c.expect("asking_price" in ask and "price" not in ask,
                 f"the figure must be named asking_price and nothing else: {sorted(ask)}")
    c.expect("not sales" in data["note"],
             f"the note must say these were never paid, got {data['note'][:80]!r}")
    c.expect("must not reach the comp set" in data["note"],
             "and must say where they may not go")


def check_nothing_found_is_a_sample_not_a_search(c: Checks):
    was_mock = config.MOCK_MODE
    config.MOCK_MODE = True
    try:
        data = next(
            d for d in (asyncio.run(wayback.find_historical_listings(f"e{i}.com"))["data"]
                        for i in range(12))
            if not d["ever_listed"])
    finally:
        config.MOCK_MODE = was_mock
    c.expect("sample, not a search" in data["note"],
             f"an empty result must not read as 'never for sale', got {data['note'][:90]!r}")
    c.expect(data["lowest_historical_ask"] is None and data["highest_historical_ask"] is None,
             "no asks means no bounds, not zero")


def check_timeline_and_dormancy(c: Checks):
    from datetime import timedelta

    today = date.today()
    fresh = wayback._result("x.com", stamps=[today - timedelta(days=4000), today - timedelta(days=5)])
    c.expect(fresh["dormant"] is False, "a site captured last week is not dormant")
    c.expect(fresh["years_archived"] > 10, f"span computed as {fresh['years_archived']}")

    stale = wayback._result("y.com", stamps=[today - timedelta(days=4000),
                                             today - timedelta(days=1500)])
    c.expect(stale["dormant"] is True, "a site last captured four years ago is dormant")

    never = wayback._result("z.com", stamps=[])
    c.expect(never["dormant"] is None,
             "never captured is not the same as dormant, and must not claim to be")
    c.expect(never["snapshot_count"] == 0 and never["note"],
             "and must explain what an empty archive usually means")


def check_samples_span_the_timeline(c: Checks):
    """Recent captures of a domain all look alike; the lander years are
    usually in the middle."""
    picks = wayback._spread_samples(["19990101000000", "20260101000000"], 4)
    c.expect(len(picks) <= 4, f"must not exceed the sample budget, got {len(picks)}")
    years = [int(p[:4]) for p in picks]
    c.expect(years == sorted(years), f"samples must be ordered, got {years}")
    c.expect(max(years) - min(years) > 10,
             f"samples must span the timeline rather than cluster, got {years}")
    c.expect(wayback._spread_samples([], 4) == [], "no stamps means no samples")
    c.expect(len(wayback._spread_samples(["20200101000000"], 4)) == 1,
             "a single capture yields a single sample")


def check_original_bytes_are_requested(c: Checks):
    """The archive rewrites pages and injects a toolbar. Classifying that
    instead of the original would mean matching lander markup that the
    archive, not the seller, produced."""
    import inspect

    source = inspect.getsource(wayback._fetch_snapshot)
    c.expect("id_" in source, "the snapshot fetch must ask for original bytes via id_")
    c.expect("rewritten" in source or "original" in source,
             "and must record why, since the flag is not self-explaining")


def main() -> int:
    c = Checks()
    for fn in (check_off_by_default, check_historical_asks_are_not_comps,
               check_nothing_found_is_a_sample_not_a_search, check_timeline_and_dormancy,
               check_samples_span_the_timeline, check_original_bytes_are_requested):
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
