"""
The collector. This is what the cron job runs.

    uv run python -m portfolio_agent.funding.pipeline
    uv run python -m portfolio_agent.funding.pipeline --resolve-domains --report funding.md

    # crontab: every morning at 07:00
    0 7 * * * cd /path/to/portfolio-agent && uv run python -m portfolio_agent.funding.pipeline \\
              --resolve-domains --report reports/funding-$(date +\\%F).md >> logs/funding.log 2>&1

Runs entirely outside the valuation graph, on purpose. A valuation must not
depend on a news site being reachable; it reads whatever the collector has
already stored.

Never raises. A feed being down, a database being unreachable or an article
failing to load each cost the run one piece of data, and the run continues
and says what it lost.
"""

import argparse
import asyncio
import sys
from datetime import date

from portfolio_agent import config
from portfolio_agent.funding import feeds, report, store
from portfolio_agent.funding.extract import FundingEvent, extract
from portfolio_agent.logging_setup import get_logger, info, log_step, run_context, setup_logging, warn

log = get_logger(__name__)

# Article pages are fetched one at a time when resolving company domains.
# Politeness, not performance: this is somebody else's server, and a handful
# of funding stories a day does not justify hammering it.
_RESOLVE_CONCURRENCY = 3


async def collect(*, resolve_domains: bool = False, feed_urls=None) -> list[FundingEvent]:
    """Fetch, filter and extract. No storage, so it is testable on its own."""
    items = await feeds.fetch_all(feed_urls)

    events: list[FundingEvent] = []
    for item in items:
        event = extract(item)
        if event:
            events.append(event)

    info(log, "funding.extracted", articles=len(items), events=len(events),
         rejected=len(items) - len(events))

    if resolve_domains and events:
        await _resolve_all(events)

    return events


async def _resolve_all(events: list[FundingEvent]) -> None:
    """Fill in each company's own domain from its article.

    The field the whole pipeline turns on: without it a funded company is
    just a company, and no credibility gap can be measured.
    """
    semaphore = asyncio.Semaphore(_RESOLVE_CONCURRENCY)

    async def one(event: FundingEvent) -> None:
        async with semaphore:
            event.company_domain = await feeds.resolve_company_url(event.article_url, event.company)
            if not event.company_domain:
                event.notes.append("company domain could not be resolved from the article")

    with log_step(log, "funding.resolve_domains", events=len(events)) as step:
        await asyncio.gather(*(one(e) for e in events))
        step.set(resolved=sum(1 for e in events if e.company_domain))


async def run(args: argparse.Namespace) -> int:
    with run_context("funding-collector"):
        events = await collect(resolve_domains=args.resolve_domains)

        backend = store.open_store()
        result = backend.upsert_many(events)
        info(log, "funding.stored", backend=result.backend,
             written=result.written, updated=result.skipped)

        since = store.days_ago(args.since_days)
        rows = backend.recent(since=since, limit=args.limit)
        text = report.build(rows, since=since, today=date.today())

        if args.report:
            from pathlib import Path

            path = Path(args.report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            info(log, "funding.report_written", path=str(path), events=len(rows))
            print(f"report written to {path} ({len(rows)} events)")
        else:
            print(text)

        if not events:
            warn(log, "funding.no_events",
                 detail="no company raises found this run — normal on a quiet day, "
                        "suspicious if it persists")
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m portfolio_agent.funding.pipeline",
        description="Collect funding announcements from news feeds and report on them.",
    )
    parser.add_argument(
        "--resolve-domains", action="store_true",
        help="fetch each article to find the company's own domain — slower, and required "
             "before any lead can be scored",
    )
    parser.add_argument("--report", metavar="PATH", help="write the report here instead of stdout")
    parser.add_argument("--since-days", type=int, default=30, help="report window in days (default 30)")
    parser.add_argument("--limit", type=int, default=200, help="maximum events in the report")
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--quiet", "-q", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level="ERROR" if args.quiet else args.log_level, force=True)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # noqa: BLE001 — a collector reports, it does not traceback at cron
        print(f"funding collector failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
