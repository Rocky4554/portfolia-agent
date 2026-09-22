"""
Offline regression eval for the funding collector.
Run with: uv run python -m portfolio_agent.evals.test_funding

No network. The feed fixtures below are real TechCrunch headlines captured
on 2026-09-21, including the ones that broke the first three versions of the
extractor — those are the point of the file, not incidental.

What is being protected: a funding amount drives a 1.15x-2.20x multiplier on
the Targeted Lead tier, so a misparsed figure is a misparsed price. And a
wrongly resolved company domain is worse than none at all, because it prices
a domain against a company that was never involved.
"""

import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from portfolio_agent.funding import feeds, report, store
from portfolio_agent.funding.extract import (
    FundingEvent,
    detect_sector,
    extract,
    is_investor_name,
    parse_amount,
    parse_round,
)


@dataclass
class FakeItem:
    title: str
    summary: str = ""
    link: str = "https://techcrunch.com/x"
    published: datetime | None = None
    source: str = "fixture"


# Real headlines. The comment on each says what it is there to catch.
HEADLINES = [
    # Ordinary raises — must extract.
    ("Crusoe raises $3.9B to build massive data centers and small modular 'AI factories'",
     "Crusoe", 3_900_000_000, "ai"),
    ("Evvy raises $40M as it seeks to further women's health research with vaginal microbiome data",
     "Evvy", 40_000_000, "healthtech"),
    # An investor endorsement leads the headline, and an adverb sits between
    # subject and verb. Produced "Khosla-backed Mazama Energy just".
    ("Khosla-backed Mazama Energy just raised $135M to drill deeper into superhot-rock geothermal",
     "Mazama Energy", 135_000_000, "climate"),
    # Descriptor prefix.
    ("AI startup Acme raises $12 million in seed funding", "Acme", 12_000_000, "ai"),
    ("Exclusive: Fintech startup Payflow lands $25M Series A", "Payflow", 25_000_000, "fintech"),
]

# Must all be rejected, each for a different reason.
REJECTED = [
    ("6 days left to save up to $200 to TechCrunch Disrupt 2026", "conference promo, no raise verb"),
    ("Prices go up in 7 days. Get your Disrupt ticket now.", "no amount, no verb"),
    ("Family offices are clamoring for AI investments", "analysis piece"),
    ("How Bain Capital Ventures plans to deploy its fresh $1.6B fund", "a fund, not a company raise"),
    ("Sequoia raises $2B fund to back early-stage founders", "investor raising a fund"),
    ("Andreessen Horowitz Ventures raises $900M", "investor name"),
    # Produced "that builds other startups" as a company name.
    ("A startup that builds other startups raised $100M and is all-in on physical AI",
     "headline never names the company"),
    ("Acme raises a round to expand into Europe", "no amount stated — guessing one invents the tier"),
    ("Tiny Co raises $500 in pocket change", "implausibly small"),
]

FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item>
    <title>Crusoe raises $3.9B to build data centers</title>
    <link>https://techcrunch.com/2026/09/17/crusoe/</link>
    <description>&lt;p&gt;The AI infrastructure company said...&lt;/p&gt;</description>
    <pubDate>Thu, 17 Sep 2026 14:00:00 +0000</pubDate>
    <category>Venture</category><category>AI</category>
  </item>
  <item>
    <title>7 days left to get your Disrupt ticket</title>
    <link>https://techcrunch.com/2026/09/19/disrupt/</link>
    <description>Save $200.</description>
    <pubDate>Sat, 19 Sep 2026 09:00:00 +0000</pubDate>
  </item>
</channel></rss>"""

# Hosts as they appear on a real TechCrunch article, in page order. The
# publisher's own newsletter appears long before the company — which is
# exactly how the first version returned strictlyvc.com for every story.
ARTICLE_HOSTS = ["strictlyvc.com", "techcrunch.com", "twitter.com", "crusoe.ai", "example-partner.com"]


class Checks:
    def __init__(self):
        self.total = 0
        self.failures: list[str] = []

    def expect(self, condition, message: str):
        self.total += 1
        if not condition:
            self.failures.append(message)


def check_amounts(c: Checks):
    for text, expected in (
        ("raises $3.9B", 3_900_000_000), ("raises $40M", 40_000_000),
        ("lands $12 million", 12_000_000), ("nets $1.5 billion", 1_500_000_000),
        ("raises $250,000", 250_000), ("secures $2.75M", 2_750_000),
    ):
        amount, _ = parse_amount(text)
        c.expect(amount == expected, f"parse_amount({text!r}) = {amount}, expected {expected}")

    c.expect(parse_amount("raises €25M")[1] == "€",
             "a non-USD round must keep its currency, not be read as dollars")
    c.expect(parse_amount("no money here")[0] is None, "text with no amount must yield None")
    c.expect(parse_amount("raises $500")[0] is None, "an implausibly small figure must be rejected")

    c.expect(parse_round("raises $40M Series B") == "Series B", "Series B should be detected")
    c.expect(parse_round("in a seed round") == "Seed", "seed should be detected")
    c.expect(parse_round("raises $40M") is None, "no round stated means None, not a guess")

    c.expect(detect_sector("an AI agents platform") == "ai", "AI sector should be detected")
    c.expect(detect_sector("a knitting supplies shop") is None, "an unknown sector must be None")


def check_extraction(c: Checks):
    for title, company, amount, sector in HEADLINES:
        event = extract(FakeItem(title=title))
        if event is None:
            c.expect(False, f"should have extracted a raise from: {title[:70]}")
            continue
        c.expect(event.company == company, f"company = {event.company!r}, expected {company!r}")
        c.expect(event.amount == amount, f"{company}: amount = {event.amount}, expected {amount}")
        c.expect(event.sector == sector, f"{company}: sector = {event.sector!r}, expected {sector!r}")

    for title, reason in REJECTED:
        event = extract(FakeItem(title=title))
        c.expect(event is None,
                 f"should have rejected ({reason}): {title[:60]} — got {event.company if event else ''!r}")

    for name in ("Bain Capital Ventures", "Sequoia Capital", "Index Ventures", "Accel Partners"):
        c.expect(is_investor_name(name), f"{name} should be recognized as an investor")
    for name in ("Crusoe", "Evvy", "Capital One", "Mazama Energy"):
        c.expect(not is_investor_name(name), f"{name} should NOT be recognized as an investor")


def check_dedupe_key(c: Checks):
    """The same raise reported twice must not become two leads, or double a
    company's funding total."""
    when = datetime(2026, 9, 17, tzinfo=timezone.utc)
    first = extract(FakeItem(title="Crusoe raises $3.9B for data centers", published=when))
    second = extract(FakeItem(title="Crusoe raises $3.9B for data centers",
                              published=when, link="https://other.example/x"))
    c.expect(first.key == second.key, "the same raise from two URLs must share a key")

    different = extract(FakeItem(title="Evvy raises $40M for research", published=when))
    c.expect(first.key != different.key, "different companies must not collide")


def check_feed_parsing(c: Checks):
    import xml.etree.ElementTree as ET

    root = ET.fromstring(FEED_XML)
    items = [feeds._to_item(node, "fixture") for node in root.findall(".//item")]
    items = [i for i in items if i]
    c.expect(len(items) == 2, f"expected 2 items from the fixture feed, got {len(items)}")

    first = items[0]
    c.expect(first.title == "Crusoe raises $3.9B to build data centers", f"title: {first.title!r}")
    c.expect("<p>" not in first.summary and "The AI infrastructure company" in first.summary,
             f"HTML must be stripped from the summary, got {first.summary!r}")
    c.expect(first.published and first.published.date() == date(2026, 9, 17),
             f"pubDate should parse, got {first.published}")
    c.expect("AI" in first.categories, f"categories should parse, got {first.categories}")

    extracted = [e for e in (extract(i) for i in items) if e]
    c.expect(len(extracted) == 1, f"only the raise should extract, got {len(extracted)}")


def check_company_host_ranking(c: Checks):
    """The publisher's own links come first on the page; the company's may be
    anywhere. Ranking by name is what makes this work."""
    c.expect(feeds.pick_company_host(ARTICLE_HOSTS, "Crusoe") == "crusoe.ai",
             f"should pick crusoe.ai, got {feeds.pick_company_host(ARTICLE_HOSTS, 'Crusoe')}")
    c.expect(feeds.pick_company_host(["strictlyvc.com", "mazamaenergy.com"], "Mazama Energy")
             == "mazamaenergy.com", "a two-word name should match its joined domain")
    c.expect(feeds.pick_company_host(["strictlyvc.com", "example.com"], "Crusoe") is None,
             "no name match must return None, never the publisher's link as a guess")
    c.expect(feeds.pick_company_host([], "Crusoe") is None, "no candidates means None")


def check_store_roundtrip(c: Checks):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "events.jsonl"
        backend = store.JsonlStore(str(path))

        events = [
            FundingEvent(company="Crusoe", amount=3_900_000_000, currency="$", round_type=None,
                         announced_on=date(2026, 9, 17), sector="ai",
                         article_title="Crusoe raises $3.9B", article_url="https://tc/1",
                         source="fixture", company_domain="crusoe.ai"),
            FundingEvent(company="Evvy", amount=40_000_000, currency="$", round_type="Series B",
                         announced_on=date(2026, 9, 15), sector="healthtech",
                         article_title="Evvy raises $40M", article_url="https://tc/2",
                         source="fixture", company_domain="evvy.com"),
        ]

        first = backend.upsert_many(events)
        c.expect(first.written == 2 and first.skipped == 0, f"first write: {first}")

        # Idempotency. A cron job re-reading yesterday's feed is the normal
        # case, not an error case.
        second = backend.upsert_many(events)
        c.expect(second.written == 0 and second.skipped == 2, f"re-write should skip both: {second}")
        c.expect(len(backend.recent(limit=100)) == 2, "re-writing must not duplicate rows")

        c.expect(len(backend.search(min_amount=100_000_000)) == 1, "min_amount should filter")
        c.expect(backend.search(keywords=["crusoe"])[0]["company"] == "Crusoe", "keyword search failed")
        c.expect(backend.search(keywords=["healthtech"])[0]["company"] == "Evvy", "sector search failed")
        c.expect(backend.search(keywords=["nothing-like-this"]) == [], "a non-match must return nothing")

        ordered = backend.search(min_amount=0)
        c.expect([r["company"] for r in ordered] == ["Crusoe", "Evvy"],
                 f"search should order by amount descending, got {[r['company'] for r in ordered]}")


def check_report(c: Checks):
    strong, reasons = report.domain_weakness("evvy.com")
    c.expect(strong == 0 and not reasons, f"a clean .com must score 0, got {strong}")

    emerging, reasons = report.domain_weakness("crusoe.ai")
    c.expect(emerging > 0 and any(".ai" in r for r in reasons),
             f"a non-.com must score above 0 and say why, got {emerging} {reasons}")

    hyphen, _ = report.domain_weakness("get-my-startup-now.net")
    c.expect(hyphen > emerging, "a hyphenated non-.com must score worse than a plain .ai")

    prefixed, reasons = report.domain_weakness("getvoice.com")
    c.expect(prefixed > 0 and any("prefix" in r for r in reasons),
             f"a 'get' prefix is the clearest lead signal there is, got {prefixed} {reasons}")

    unknown, reasons = report.domain_weakness(None)
    c.expect(unknown == 0 and not reasons,
             "an unresolved domain must score 0 with no reasons — 'we don't know' is not 'they're fine'")

    rows = [
        {"company": "Crusoe", "amount": 3_900_000_000, "sector": "ai", "round_type": None,
         "announced_on": "2026-09-17", "company_domain": "crusoe.ai"},
        {"company": "Evvy", "amount": 40_000_000, "sector": "healthtech", "round_type": "Series B",
         "announced_on": "2026-09-15", "company_domain": "evvy.com"},
    ]
    text = report.build(rows, since=date(2026, 8, 1), today=date(2026, 9, 21))
    c.expect("$3,940,000,000" in text, "the report must total the capital raised")
    c.expect("Crusoe" in text and "crusoe.ai" in text, "a lead must appear with its current domain")
    c.expect("| Evvy |" not in text.split("## By sector")[0],
             "a company already on a strong .com is not a domain lead")
    c.expect("not independently confirmed" in text,
             "reported figures must carry their caveat into the report")

    empty = report.build([], since=date(2026, 8, 1), today=date(2026, 9, 21))
    c.expect("No funding events" in empty and "not a failure" in empty,
             "an empty window must explain itself rather than look broken")


def main() -> int:
    c = Checks()
    for check in (
        check_amounts,
        check_extraction,
        check_dedupe_key,
        check_feed_parsing,
        check_company_host_ranking,
        check_store_roundtrip,
        check_report,
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
