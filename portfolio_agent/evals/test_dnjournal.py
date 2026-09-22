"""
Offline regression eval for the DNJournal sales parser.
Run with: uv run python -m portfolio_agent.evals.test_dnjournal

This is the start of the one comp source that is ours. Every other one here
wraps a database somebody else owns and can withdraw; a store assembled from
the public record grows with time and survives a revoked key.

The fixture is synthetic on purpose. A scraper test that fetches the live
page tests the network, and the page changes weekly — so the contract is
pinned against markup shaped like the real thing, including the rows that
must NOT parse. Storing several megabytes of someone else's page in this
repo to test a regex would also be a poor answer to a question nobody asked.

What is asserted, in order of importance:

  - Prose does not become a sale. The report's narrative tables share the
    markup of its sales table, and a positional parser reads paragraphs as
    transactions — a fabricated comp is far worse than a missed one.
  - An empty parse is a parser failure, never an empty week. The report has
    never had no sales in it, so reporting zero would be a false fact.
  - These sales are weighted BELOW our own book. They are reported by the
    venue that made them: a real transaction and an interested witness.
  - robots.txt is obeyed, and a 5xx on it means disallow, per RFC 9309.
"""

import sys

from portfolio_agent import config
from portfolio_agent.tools.dnjournal import SOURCE, ReportedSale, parse_sales

# Shaped like the real report: a numbered sales table, and prose laid out in
# the same table markup, which is what a positional parser trips over.
FIXTURE = """
<html><body>
<table>
  <tr><td>In our report this week we look at how the market has moved since
      August, and what $250,000 for a single .ai name says about it.</td>
      <td>By Ron Jackson</td><td></td></tr>
  <tr><td>1.</td><td>Orchestra.ai</td><td>$250,000</td><td>DomainBooth</td></tr>
  <tr><td>2.</td><td>Plateful.com</td><td>$87,500</td><td>Snagged.com</td></tr>
  <tr><td>3.</td><td>Little.co</td><td>$83,945</td><td>Atom.com</td></tr>
  <tr><td>4.</td><td>KLO.com</td><td>$39,500</td><td></td></tr>
  <tr><td>5.</td><td>Broken.com</td><td>price on request</td><td>Sedo</td></tr>
  <tr><td>6.</td><td>not a domain at all</td><td>$10,000</td><td>Sedo</td></tr>
  <tr><td>7.</td><td>Zero.com</td><td>$0</td><td>Sedo</td></tr>
  <tr><td>Vertex.io</td><td>$12,000</td><td>Sedo</td></tr>
</table>
</body></html>
"""


class Checks:
    def __init__(self):
        self.total = 0
        self.failures = []

    def expect(self, condition, message):
        self.total += 1
        if not condition:
            self.failures.append(message)


def check_sales_parse(c: Checks):
    sales = parse_sales(FIXTURE)
    by_domain = {s.domain: s for s in sales}

    c.expect("Orchestra.ai" in by_domain, f"the top sale must parse, got {sorted(by_domain)}")
    top = by_domain.get("Orchestra.ai")
    if top:
        c.expect(top.price == 250_000, f"price parsed as {top.price}")
        c.expect(top.venue == "DomainBooth", f"venue parsed as {top.venue!r}")
        c.expect(top.rank == 1, f"rank parsed as {top.rank}")

    c.expect(by_domain.get("Little.co") and by_domain["Little.co"].price == 83_945,
             "a price with a comma inside must parse whole, not truncate at the comma")

    # A row with no rank column is still a sale.
    c.expect("Vertex.io" in by_domain, "an unranked row must still parse as a sale")
    if "Vertex.io" in by_domain:
        c.expect(by_domain["Vertex.io"].rank is None, "and must not invent a rank")

    # A missing venue is missing, not empty string.
    c.expect(by_domain.get("KLO.com") and by_domain["KLO.com"].venue is None,
             "an absent venue must be None rather than ''")


def check_prose_does_not_become_a_sale(c: Checks):
    """The failure that would poison the comp store.

    The report's narrative sits in the same table markup as its sales, and
    the paragraph in the fixture contains '$250,000'. A fabricated comp is
    far worse than a missed one: it enters the distribution and prices
    other people's domains.
    """
    sales = parse_sales(FIXTURE)
    domains = {s.domain for s in sales}
    c.expect(not any("report" in d.lower() or " " in d for d in domains),
             f"prose parsed as a sale: {sorted(domains)}")
    c.expect("not a domain at all" not in domains, "a non-domain must not become a comp")
    c.expect(all(s.price > 0 for s in sales), "a zero price is not a sale")
    c.expect("Broken.com" not in domains, "a row with no parseable price must be skipped")
    c.expect(len(sales) == 5, f"exactly the five real sales must parse, got {len(sales)}: "
                              f"{sorted(domains)}")


def check_empty_parse_is_a_failure_not_a_week(c: Checks):
    """The report has never had no sales in it."""
    c.expect(parse_sales("<html><body><p>nothing here</p></body></html>") == [],
             "a page with no table must yield no sales")
    c.expect(parse_sales("") == [], "empty input must not raise")
    c.expect(parse_sales(None) == [], "None must not raise")
    # The tool turns that emptiness into an error rather than an empty week;
    # asserted here as the contract the caller depends on.
    import inspect

    from portfolio_agent.tools import dnjournal
    source = inspect.getsource(dnjournal.fetch_reported_sales)
    c.expect("parsed no sales" in source and "error_result" in source,
             "an unparseable report must return an error, never an empty sales list")


def check_reported_sales_weigh_less_than_our_own(c: Checks):
    """Reported by the venue that made the sale: a real transaction and an
    interested witness."""
    comp = ReportedSale("example.com", 10_000, "Sedo", 1).as_comp()
    c.expect(comp["source"] == SOURCE, f"provenance must be tagged, got {comp['source']}")
    c.expect(comp["weight"] == config.COMP_WEIGHT_REPORTED, "weight must come from config")
    c.expect(config.COMP_WEIGHT_REPORTED < config.COMP_WEIGHT_INTERNAL,
             f"a reported sale must weigh less than a verified own-book sale: "
             f"{config.COMP_WEIGHT_REPORTED} vs {config.COMP_WEIGHT_INTERNAL}")
    c.expect(config.COMP_WEIGHT_REPORTED > 1.0,
             "but more than an inference, since a transaction did occur")
    c.expect(comp["date"] is None,
             "the report gives a week, not a per-sale date, and must not invent one")


def check_robots_is_obeyed(c: Checks):
    """A 5xx on robots.txt means assume disallow (RFC 9309) — which is not a
    hypothetical here: the apex host serves exactly that."""
    import inspect

    from portfolio_agent.tools import dnjournal
    source = inspect.getsource(dnjournal.robots_allow)
    c.expect("status_code >= 500" in source, "a 5xx on robots.txt must be handled explicitly")
    c.expect("9309" in source, "and the reason must be recorded, not just the behaviour")
    c.expect("Protego" in source, "robots parsing must use a real parser, not a hand-rolled one")
    c.expect(dnjournal.BASE.startswith("https://www."),
             "the www host is used deliberately: the apex serves a 500 for robots.txt")


def main() -> int:
    c = Checks()
    for fn in (check_sales_parse, check_prose_does_not_become_a_sale,
               check_empty_parse_is_a_failure_not_a_week,
               check_reported_sales_weigh_less_than_our_own, check_robots_is_obeyed):
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
