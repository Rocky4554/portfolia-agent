"""
Score the engine against sales it is not allowed to see.

    uv run python -m portfolio_agent.evals.backtest portfolio_agent/evals/backtest_sales.example.csv --dry-run
    uv run python -m portfolio_agent.evals.backtest sales.csv --book sales.csv --out backtest_results.csv

A backtest is a measurement, not another review. Each row is a domain that
already sold. The graph is run as of the morning of that sale: the domain
itself is removed from the comp set, every comp dated on or after the sale
is removed, and undated keyword aggregates are not used. The tier named in
the row (the way the domain actually sold) is then compared to the price
that was paid.

predicted / actual is the ratio. A median near 1.0 means no systematic
bias. A median of 3 means the engine is consistently about 3x high in that
slice. Landing inside 2–3x on most rows is a strong result in this market;
the same name clears at very different prices depending on the buyer.

The book and the ground-truth file may be the same CSV. On each row the
subject is held out, and the other rows dated before that sale become
internal comps. Point --book at Namekart's closed book when you have it.
Without --book the public feeds still run, with the same date cutoff.
"""

import argparse
import asyncio
import csv
import statistics
import sys
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

from portfolio_agent import config
from portfolio_agent.logging_setup import setup_logging
from portfolio_agent.schema.valuation_output import TIER_ORDER

_TIERS = set(TIER_ORDER)
_WITHIN = (2.0, 3.0)


@dataclass
class Case:
    domain: str
    actual_price: float
    sale_date: date
    sale_tier: str


@dataclass
class Score:
    domain: str
    tld: str
    category: str
    sale_tier: str
    actual: int
    predicted: int | None
    low: int | None
    high: int | None
    ratio: float | None
    comp_basis: str
    comp_confidence: float | None
    leaked: bool
    error: str


def parse_money(value) -> float:
    if value is None or value == "":
        return 0.0
    text = str(value).replace("$", "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def load_cases(path: str | Path) -> list[Case]:
    """Read the ground-truth file. Blank lines and # comments are ignored."""
    text_path = Path(path)
    with text_path.open(newline="", encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise SystemExit(f"{text_path}: no sales. Add rows under the header in the example file.")
    cases = []
    for index, row in enumerate(csv.DictReader(lines), start=2):
        domain = str(row.get("domain") or "").strip().lower()
        tier = str(row.get("sale_tier") or "end_user").strip().lower()
        sold = str(row.get("sale_date") or row.get("date") or "")[:10]
        price = parse_money(row.get("actual_price") or row.get("price") or row.get("sale_price"))
        if not domain or price <= 0 or tier not in _TIERS:
            raise SystemExit(
                f"{text_path}:{index}: need domain, a positive actual_price, and sale_tier "
                f"one of {', '.join(TIER_ORDER)}"
            )
        try:
            sale_date = date.fromisoformat(sold)
        except ValueError:
            raise SystemExit(f"{text_path}:{index}: sale_date must be YYYY-MM-DD, got {sold!r}")
        cases.append(Case(domain=domain, actual_price=price, sale_date=sale_date, sale_tier=tier))
    if not cases:
        raise SystemExit(f"{text_path}: header only. Add one row per closed sale.")
    return cases


def price_ratio(predicted: float | None, actual: float) -> float | None:
    if predicted is None or predicted <= 0 or actual <= 0:
        return None
    return predicted / actual


def within_factor(ratio: float | None, factor: float) -> bool:
    """True when the prediction is within `factor` times the sale, either way."""
    if ratio is None or ratio <= 0:
        return False
    return (1.0 / factor) <= ratio <= factor


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _slice_line(label: str, rows: list[Score]) -> str:
    scored = [r for r in rows if r.ratio is not None]
    if not scored:
        return f"  {label:<28} n=0"
    ratios = [r.ratio for r in scored if r.ratio is not None]
    median = statistics.median(ratios)
    inside_2 = sum(1 for r in ratios if within_factor(r, 2))
    inside_3 = sum(1 for r in ratios if within_factor(r, 3))
    return (
        f"  {label:<28} n={len(scored):<3} median={median:.2f}x"
        f"  p25={_percentile(ratios, 0.25):.2f}x  p75={_percentile(ratios, 0.75):.2f}x"
        f"  within 2x={inside_2}/{len(scored)}  within 3x={inside_3}/{len(scored)}"
    )


def summarize(rows: list[Score]) -> str:
    """Median ratio and spread, overall and by category and extension.

    Ratio is predicted / actual. Above 1 the engine is high; below 1 it is low.
    """
    scored = [r for r in rows if r.ratio is not None]
    missed = [r for r in rows if r.ratio is None]
    leaked = [r for r in rows if r.leaked]
    lines = ["", "  backtest", "  " + "-" * 72]
    lines.append(_slice_line("all", rows))
    if scored:
        median = statistics.median([r.ratio for r in scored if r.ratio is not None])
        if median > 1.25:
            bias = f"median {median:.2f}x means the engine is systematically high"
        elif median < 0.8:
            bias = f"median {median:.2f}x means the engine is systematically low"
        else:
            bias = f"median {median:.2f}x — no large systematic bias"
        lines.append(f"  {bias}")
    if missed:
        lines.append(f"  unpriced or failed: {len(missed)} ({', '.join(r.domain for r in missed[:8])})")
    if leaked:
        lines.append(
            "  LEAK: the held-out domain still appeared in the comp set for "
            + ", ".join(r.domain for r in leaked)
        )
    lines.append("")
    lines.append("  by category")
    categories = sorted({r.category or "unknown" for r in rows})
    lines.extend(_slice_line(name, [r for r in rows if (r.category or "unknown") == name]) for name in categories)
    lines.append("")
    lines.append("  by extension")
    tlds = sorted({r.tld or "?" for r in rows})
    lines.extend(_slice_line(f".{name}", [r for r in rows if (r.tld or "?") == name]) for name in tlds)
    lines.append("")
    lines.append("  by sale tier")
    for tier in TIER_ORDER:
        group = [r for r in rows if r.sale_tier == tier]
        if group:
            lines.append(_slice_line(tier, group))
    lines.append("")
    return "\n".join(lines)


def write_results(path: str | Path, rows: list[Score]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(Score.__dataclass_fields__)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


async def score_case(case: Case) -> Score:
    from portfolio_agent.nodes.graph import portfolio_agent
    from portfolio_agent.schema.valuation_output import ValuationOutput

    tld = case.domain.rpartition(".")[2]
    try:
        result = await portfolio_agent.ainvoke({
            "domain": case.domain,
            "comp_as_of": case.sale_date.isoformat(),
            "skip_rationale": True,
        })
        output = ValuationOutput.model_validate(result["valuation"])
    except Exception as exc:  # noqa: BLE001 — one bad domain must not abort the table
        return Score(
            domain=case.domain, tld=tld, category="", sale_tier=case.sale_tier,
            actual=int(case.actual_price), predicted=None, low=None, high=None, ratio=None,
            comp_basis="", comp_confidence=None, leaked=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    tier = getattr(output.tiers, case.sale_tier)
    predicted = tier.point if tier is not None else None
    leaked = any(str(comp.domain).lower() == case.domain for comp in output.comp_set)
    return Score(
        domain=case.domain,
        tld=tld,
        category=output.category.value,
        sale_tier=case.sale_tier,
        actual=int(case.actual_price),
        predicted=predicted,
        low=tier.low if tier is not None else None,
        high=tier.high if tier is not None else None,
        ratio=None if predicted is None else round(price_ratio(predicted, case.actual_price) or 0, 4),
        comp_basis=output.comp_basis.value,
        comp_confidence=output.comp_confidence,
        leaked=leaked,
        error="held-out domain was still a comp" if leaked else "",
    )


def _dry_run(cases: list[Case], book: str | None) -> str:
    from portfolio_agent.tools.internal_comps import load_book

    lines = [f"  {len(cases)} sales. No valuation will run.", ""]
    for case in cases:
        visible = "public feeds only, cutoff applied inside the ladder"
        if book:
            comps = load_book(book, before=case.sale_date, exclude_domain=case.domain)
            visible = f"{len(comps)} book comps dated before {case.sale_date.isoformat()}"
        lines.append(
            f"  {case.domain:<28} {case.sale_tier:<14} ${case.actual_price:,.0f}"
            f"  as of {case.sale_date.isoformat()}  {visible}"
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m portfolio_agent.evals.backtest",
        description="Compare the engine's tier to closed sales it was not allowed to see.",
    )
    parser.add_argument("sales", help="CSV of domain, actual_price, sale_date, sale_tier")
    parser.add_argument("--book", default=None, help="CSV of closed sales to use as internal comps (may be the same file)")
    parser.add_argument("--out", default="backtest_results.csv", help="where to write one row per domain")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N rows")
    parser.add_argument("--dry-run", action="store_true", help="check the file and print the holdout, then stop")
    parser.add_argument("--log-level", default="WARNING")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level, force=True)
    cases = load_cases(args.sales)
    if args.limit > 0:
        cases = cases[: args.limit]

    previous_book = config.INTERNAL_COMPS_CSV
    if args.book:
        config.INTERNAL_COMPS_CSV = args.book

    if args.dry_run:
        print(_dry_run(cases, args.book))
        config.INTERNAL_COMPS_CSV = previous_book
        return 0

    async def _run() -> list[Score]:
        rows = []
        for index, case in enumerate(cases, start=1):
            print(f"  [{index}/{len(cases)}] {case.domain}", file=sys.stderr)
            rows.append(await score_case(case))
        return rows

    try:
        rows = asyncio.run(_run())
    finally:
        config.INTERNAL_COMPS_CSV = previous_book

    write_results(args.out, rows)
    print(summarize(rows))
    print(f"  wrote {len(rows)} rows to {args.out}")
    print("")
    return 1 if any(r.leaked for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
