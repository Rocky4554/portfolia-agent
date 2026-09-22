"""
Measure TLD transfer ratios from names that sold on more than one extension.

    uv run python scripts/derive_tld_ratios.py --min-samples 8

Queries the Namebio sales mirror for each SLD that appears on two
extensions within the window, takes the median price ratio per pair, and
prints a `TLD_TRANSFER` block ready to paste into
`knowledge/tld_transfer.py`.

Prints rather than writes. A ratio table is a market claim that someone
should read before it starts moving prices, and a script that silently
rewrites the file it derives from is a script whose output nobody reviews.
"""

import argparse
import asyncio
import statistics
import sys
from collections import defaultdict
from datetime import date

from portfolio_agent.logging_setup import get_logger, info, setup_logging
from portfolio_agent.tools import sales_history

log = get_logger(__name__)


async def collect(keywords: list[str], tlds: tuple[str, ...], window_months: int) -> dict:
    """SLD → {tld: [prices]} for every sale the mirror can find."""
    by_sld: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for keyword in keywords:
        result = await sales_history.get_keyword_sales(keyword, min_price=100)
        if not result["ok"]:
            continue
        for sale in (result["data"] or {}).get("comps") or []:
            domain = str(sale.get("domain") or "").lower()
            sld, _, tld = domain.partition(".")
            if sld and tld in tlds and sale.get("price"):
                by_sld[sld][tld].append(float(sale["price"]))
    return by_sld


def ratios(by_sld: dict, min_samples: int) -> dict:
    pairs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for sld, by_tld in by_sld.items():
        sold_on = sorted(by_tld)
        for source in sold_on:
            for target in sold_on:
                if source == target:
                    continue
                source_price = statistics.median(by_tld[source])
                target_price = statistics.median(by_tld[target])
                if source_price > 0:
                    pairs[(source, target)].append(target_price / source_price)
    return {
        pair: (round(statistics.median(values), 3), len(values))
        for pair, values in pairs.items()
        if len(values) >= min_samples
    }


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--window-months", type=int, default=60)
    parser.add_argument("--keywords", default="cloud,data,pay,health,ai,app,shop,care,work,home")
    args = parser.parse_args(argv)

    setup_logging()
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    tlds = ("com", "co", "io", "net", "ai", "org")

    by_sld = await collect(keywords, tlds, args.window_months)
    dual = {sld: t for sld, t in by_sld.items() if len(t) > 1}
    info(log, "tld_ratios.collected", slds=len(by_sld), sold_on_multiple=len(dual))

    measured = ratios(dual, args.min_samples)
    if not measured:
        print(f"No pair reached {args.min_samples} samples. The priors in "
              f"knowledge/tld_transfer.py remain the best available answer.", file=sys.stderr)
        return 1

    print(f"# Derived {date.today().isoformat()} from {len(dual)} SLDs sold on multiple extensions.")
    print("# Paste into knowledge/tld_transfer.py, replacing the matching priors.")
    for (source, target), (multiplier, samples) in sorted(measured.items()):
        print(f'    ("{source}", "{target}"): TransferRatio('
              f'{multiplier}, "derived", samples={samples}),')
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
