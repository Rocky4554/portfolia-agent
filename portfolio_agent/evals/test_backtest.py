"""
Offline checks for the backtest holdout.
Run with: uv run python -m portfolio_agent.evals.test_backtest

What is asserted:

  - A sale of the domain being graded cannot enter the ladder, nor can any
    sale dated on or after the holdout, nor an undated row.
  - The same rows are kept when there is no holdout, so a live valuation
    still sees a prior sale of the name.
  - A brokerage CSV used as the book drops the subject and anything not yet
    closed on the morning of the sale.
  - The summary reports predicted/actual, so a 3x-high slice is visible.
"""

import sys
import tempfile
from datetime import date
from pathlib import Path

from portfolio_agent.evals.backtest import Score, load_cases, price_ratio, summarize, within_factor
from portfolio_agent.evals.test_comp_ladder import inputs
from portfolio_agent.pricing.comp_ladder import build_comp_set
from portfolio_agent.schema.valuation_output import CompBasis
from portfolio_agent.tools.internal_comps import load_book

AS_OF = date(2026, 6, 1)

ANSWER = [{"domain": "consulting.ai", "price": 50_000, "date": "2026-06-01", "marketplace": "NameBio"}]
COHORT = [
    {"domain": "advisor.ai", "price": 88_000, "date": "2026-02-10", "marketplace": "GoDaddy"},
    {"domain": "counsel.ai", "price": 42_000, "date": "2026-08-01", "marketplace": "Sedo"},
    {"domain": "strategy.ai", "price": 61_000, "date": "2026-01-20", "marketplace": "Private"},
    {"domain": "undated.ai", "price": 10_000, "marketplace": "Sedo"},
]


class Checks:
    def __init__(self):
        self.total = 0
        self.failures: list[str] = []

    def expect(self, condition, message: str):
        self.total += 1
        if not condition:
            self.failures.append(message)


def _names(as_of: date | None) -> set[str]:
    result = build_comp_set(inputs(
        keyword_sales=ANSWER, cohort_sales=COHORT, as_of=as_of,
        today=as_of or date(2026, 9, 21),
    ))
    return {comp.domain for comp in result.comps}


def check_holdout_hides_the_answer(c: Checks):
    hidden = _names(AS_OF)
    c.expect("consulting.ai" not in hidden, f"the sale being graded leaked: {sorted(hidden)}")
    c.expect("counsel.ai" not in hidden, f"a sale after the cutoff leaked: {sorted(hidden)}")
    c.expect("undated.ai" not in hidden, f"an undated sale leaked: {sorted(hidden)}")
    c.expect("advisor.ai" in hidden and "strategy.ai" in hidden,
             f"sales from before the cutoff should remain, got {sorted(hidden)}")

    visible = _names(None)
    c.expect("consulting.ai" in visible, "without a holdout, a prior sale of the name is a real comp")
    c.expect("undated.ai" in visible, "without a holdout, an undated sale is still kept")


def check_holdout_skips_undated_aggregates(c: Checks):
    from portfolio_agent.evals.test_comp_ladder import KEYWORD_STATS

    held = build_comp_set(inputs(keyword_stats=KEYWORD_STATS, as_of=AS_OF))
    c.expect(held.basis is CompBasis.NONE, f"undated aggregates must not price a holdout, got {held.basis}")
    c.expect(not held.comps, f"holdout comps should be empty, got {[comp.domain for comp in held.comps]}")
    c.expect(any("holdout" in note for note in held.notes), f"the holdout must be stated, got {held.notes}")


def check_book_holdout(c: Checks):
    body = """domain,actual_price,sale_date,sale_tier
consulting.ai,"$50,000",2026-06-01,end_user
advisor.ai,88000,2026-02-10,end_user
counsel.ai,42000,2026-08-01,auction
# a comment the loader must skip
strategy.ai,61000,2026-01-20,end_user
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "book.csv"
        path.write_text(body, encoding="utf-8")
        visible = load_book(path, before="2026-06-01", exclude_domain="consulting.ai")
        names = {row["domain"] for row in visible}
        c.expect(names == {"advisor.ai", "strategy.ai"}, f"book holdout kept {sorted(names)}")
        c.expect(load_cases(path)[0].actual_price == 50_000, "prices with $ and commas must parse")
        c.expect(load_cases(path)[0].sale_tier == "end_user", "sale_tier must parse")


def check_summary_reports_bias(c: Checks):
    rows = [
        Score("a.com", "com", "generic", "end_user", 100, 300, 100, 500, 3.0, "cohort", 0.45, False, ""),
        Score("b.com", "com", "generic", "end_user", 100, 250, 80, 400, 2.5, "cohort", 0.45, False, ""),
        Score("c.ai", "ai", "keyword", "end_user", 100, 100, 50, 200, 1.0, "cohort", 0.45, False, ""),
        Score("d.ai", "ai", "keyword", "end_user", 100, None, None, None, None, "none", 0.0, False, "unpriced"),
    ]
    text = summarize(rows)
    c.expect("median=2.50x" in text or "median=2.5x" in text, f"overall median missing:\n{text}")
    c.expect("systematically high" in text, f"a 2.5x median must be called high:\n{text}")
    c.expect("unpriced or failed: 1" in text, f"the unpriced row must be counted:\n{text}")
    c.expect(price_ratio(300, 100) == 3.0, "ratio is predicted / actual")
    c.expect(within_factor(3.0, 3) and not within_factor(3.0, 2), "3.0x is inside 3x and outside 2x")
    c.expect("generic" in text and ".ai" in text, f"slices missing:\n{text}")


def main() -> int:
    c = Checks()
    for check in (
        check_holdout_hides_the_answer,
        check_holdout_skips_undated_aggregates,
        check_book_holdout,
        check_summary_reports_bias,
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
