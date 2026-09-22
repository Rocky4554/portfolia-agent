"""
Command-line entry point for a single valuation.

    uv run python -m portfolio_agent.cli cloud.ai
    uv run python -m portfolio_agent.cli cloud.ai --mock --json

Exists so that running the agent is one command rather than an inline Python
blob. It is a thin wrapper: it parses arguments, configures logging, invokes
the compiled graph and renders the result. No valuation logic lives here —
if something is wrong with a number, the cause is in the graph, not in this
file.
"""

import argparse
import asyncio
import json
import sys

from portfolio_agent import config
from portfolio_agent.logging_setup import setup_logging
from portfolio_agent.schema.valuation_output import TIER_ORDER, ValuationOutput

# What each tier means, one line each, so the output explains itself to
# someone who has not read APVA Table 4.1.1.
_TIER_BLURB = {
    "floor": "quick wholesale exit",
    "auction": "competitive investor auction",
    "end_user": "direct sale to a business that will develop it",
    "list_bin": "premium Buy-It-Now ask",
    "targeted_lead": "one identified, funded buyer",
    "strategic_ma": "key asset in an acquisition",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m portfolio_agent.cli",
        description="Value a domain with the APVA valuation graph.",
    )
    parser.add_argument("domain", help="the domain to value, e.g. cloud.ai")
    parser.add_argument(
        "--mock", action="store_true",
        help="answer every external service (and the LLM) from deterministic fixtures — no keys, no network",
    )
    parser.add_argument(
        "--mock-llm", action="store_true",
        help="with --mock, still call the real model; for iterating on the rationale prompt",
    )
    parser.add_argument("--json", action="store_true", help="print the raw ValuationOutput as JSON")
    parser.add_argument(
        "--summary", action="store_true",
        help="print only the one-screen JSON digest (implied at the end of a normal run)")
    parser.add_argument(
        "--no-summary", action="store_true",
        help="suppress the digest printed after the full report")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO (default), WARNING, ERROR")
    parser.add_argument("--log-format", default=None, choices=["text", "json"])
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="suppress step logs; print only the result",
    )
    return parser


def _money(value: int | None) -> str:
    return f"${value:,}" if value is not None else "—"


def render(output: ValuationOutput) -> str:
    lines = [
        "",
        f"  {output.domain}",
        f"  {output.category.value} — confidence {output.category_confidence}",
    ]
    if output.brand_score:
        components = "  ".join(f"{k} {v:g}" for k, v in output.brand_score.components.items())
        lines.append(f"  brand score {output.brand_score.total:g}/100   ({components})")
        if output.brand_score.typo_deflator < 1.0:
            lines.append(f"  TYPO DEFLATOR {output.brand_score.typo_deflator:g} applied to every tier")
    lines.append("")

    lines.append(f"  {'tier':<14} {'point':>12} {'low':>12} {'high':>12}  {'P(sale)':>8}  what it means")
    lines.append(f"  {'-' * 14} {'-' * 12} {'-' * 12} {'-' * 12}  {'-' * 8}  {'-' * 40}")
    for name in TIER_ORDER:
        tier = getattr(output.tiers, name)
        if tier is None:
            lines.append(f"  {name:<14} {'not priced':>12} {'':>12} {'':>12}  {'':>8}  {_TIER_BLURB[name]}")
            continue
        probability = f"{tier.sale_probability:.0%}" if tier.sale_probability is not None else ""
        lines.append(
            f"  {name:<14} {_money(tier.point):>12} {_money(tier.low):>12} {_money(tier.high):>12}"
            f"  {probability:>8}  {_TIER_BLURB[name]}"
        )

    # The factors are the audit trail; without them a price is just a number.
    for name, tier in output.tiers.ordered_populated():
        factors = [f"+ {f}" for f in tier.inflationary_factors] + [f"- {f}" for f in tier.deflationary_factors]
        if factors:
            lines.append("")
            lines.append(f"  {name}  (evidence: {', '.join(tier.evidence_refs) or 'none'})")
            lines.extend(f"      {factor}" for factor in factors)

    lines.append("")
    lines.append("  evidence")
    for item in output.evidence:
        status = "ok" if item.ok else f"FAILED — {item.error}"
        mock = " [fixture]" if item.mock else ""
        lines.append(f"      {item.id:<5} {item.tool:<28} {status}{mock}")

    if output.warnings:
        lines.append("")
        lines.append("  warnings")
        lines.extend(f"      - {warning}" for warning in output.warnings)

    lines.append("")
    lines.append("  rationale")
    lines.append("")
    for paragraph in output.rationale.split("\n"):
        lines.append(f"      {paragraph}")
    lines.append("")
    return "\n".join(lines)


async def value(domain: str) -> ValuationOutput:
    # Imported here, not at module scope: importing the graph configures
    # logging, and the CLI wants to set the level first.
    from portfolio_agent.nodes.graph import portfolio_agent

    result = await portfolio_agent.ainvoke({"domain": domain})
    return ValuationOutput.model_validate(result["valuation"])


def main(argv: list[str] | None = None) -> int:
    from portfolio_agent.schema.summary import summarize

    args = build_parser().parse_args(argv)

    if args.mock:
        config.MOCK_MODE = True
        config.MOCK_ALLOW_LLM = bool(args.mock_llm)

    setup_logging(
        level="ERROR" if args.quiet else args.log_level,
        fmt=args.log_format,
        force=True,
    )

    try:
        output = asyncio.run(value(args.domain))
    except Exception as e:  # noqa: BLE001 — a CLI reports failures, it does not traceback at users
        print(f"valuation failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(output.model_dump(mode="json"), indent=2))
    elif args.summary:
        print(json.dumps(summarize(output), indent=2))
    else:
        print(render(output))
        if not args.no_summary:
            # The full report is for auditing one number; this is for
            # scanning twenty. Printed after, not instead — the digest
            # compresses the report rather than replacing it.
            print("\n  summary")
            print("\n".join("      " + line
                             for line in json.dumps(summarize(output), indent=2).splitlines()))

    # Exit 0 even with warnings: warnings are the normal, correct output of a
    # valuation that could not price every tier, not a failure.
    return 0


if __name__ == "__main__":
    sys.exit(main())
