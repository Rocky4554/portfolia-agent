"""
fetch_reported_sales — the weekly public record of domain sales.

**Why this is the most valuable thing left to build.** Every other comp
source here is a wrapper over a database somebody else owns and can
withdraw. A store of sales assembled from the public record is the one input
that is ours, grows with time, and does not disappear when a key is revoked.
It is also the only way the comp ladder ever gets deeper than the rungs it
has: consulting.ai was unpriceable on one sale, and the fix then was to
widen the basis rather than to find more sales.

**What these sales ARE, which matters for how they weigh.** DNJournal's
weekly report is a compilation of sales REPORTED to it by marketplaces and
brokers. They are not independently audited, the venues choose what to
report, and a venue has an obvious interest in reporting its wins. So they
are real transactions with a real selection bias — closer to evidence than
to an appraisal, and further from it than our own book. They are tagged
`dnjournal_reported` and carry a weight below verified internal sales for
exactly that reason.

**On crawling politely.** robots.txt is fetched and obeyed before anything
is requested, via Crawlee's protego parser rather than a hand-rolled one.
The apex host serves a 500 for robots.txt while www serves it correctly, and
RFC 9309 says a 5xx on robots.txt should be read as a complete disallow --
so the www host is used deliberately, not incidentally. One request at a
time, an identifying User-Agent, and a real delay between pages: this is a
small publication and hammering it would be both rude and the fastest way
to get blocked.

**Terms of service are a question this module cannot answer.** robots.txt
carries no directives at all -- twenty-four lines, every one a comment --
and its Content Signals preamble sets no signal, which by its own wording
"neither grants nor restricts permission". That is silence, not consent. The
site's terms, and any database right in the compilation, are a commercial
decision for whoever runs this, and CRAWL_ARCHIVE is off by default so that
decision is made deliberately rather than by importing a module.
"""

import re
from dataclasses import dataclass
from datetime import date

from portfolio_agent import config
from portfolio_agent.logging_setup import get_logger, info, warn
from portfolio_agent.tools.common import ToolResult, error_result, mock_result, ok_result

log = get_logger(__name__)

SOURCE = "dnjournal_reported"

# The www host, deliberately: the apex serves a 500 for robots.txt, and a
# 5xx on robots.txt means "assume disallowed" under RFC 9309.
BASE = "https://www.dnjournal.com"
WEEKLY = f"{BASE}/domainsales.htm"
ARCHIVE = f"{BASE}/archive.htm"

_TIMEOUT_S = 45.0  # the weekly report is a ~5MB page
_RANK = re.compile(r"^\d+\.?$")
_PRICE = re.compile(r"\$\s*([0-9][0-9,]*)")
_DOMAIN = re.compile(r"^[a-z0-9-]+(?:\.[a-z0-9-]+)+$", re.IGNORECASE)


@dataclass(frozen=True)
class ReportedSale:
    domain: str
    price: int
    venue: str | None
    rank: int | None

    def as_comp(self) -> dict:
        return {
            "domain": self.domain.lower(),
            "price": float(self.price),
            "date": None,  # the report gives a week, not a per-sale date
            "marketplace": self.venue,
            "source": SOURCE,
            # Below a verified own-book sale and above a cohort inference.
            # Reported by the venue that made the sale, which is a real
            # transaction and an interested witness.
            "weight": config.COMP_WEIGHT_REPORTED,
        }


def parse_sales(html: str) -> list[ReportedSale]:
    """Sale rows out of a weekly report.

    A pure function over the HTML so the parser is testable against a saved
    page — which is the only honest way to test a scraper, since the live
    page changes weekly and a test that fetches it tests the network.
    """
    from selectolax.parser import HTMLParser

    sales: list[ReportedSale] = []
    for row in HTMLParser(html or "").css("tr"):
        cells = [c.text(strip=True) for c in row.css("td")]
        if len(cells) < 3:
            continue
        # Layout: rank, domain, price, venue. Matched on shape rather than
        # position, because the report's prose tables share the same markup
        # and a positional parser reads paragraphs as sales.
        rank_cell, rest = (cells[0], cells[1:]) if _RANK.match(cells[0]) else (None, cells)
        if len(rest) < 2:
            continue
        domain, price_cell = rest[0], rest[1]
        if not _DOMAIN.match(domain):
            continue
        match = _PRICE.search(price_cell)
        if not match:
            continue
        try:
            price = int(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if price <= 0:
            continue
        sales.append(ReportedSale(
            domain=domain,
            price=price,
            venue=(rest[2].strip() or None) if len(rest) > 2 else None,
            rank=int(rank_cell.rstrip(".")) if rank_cell else None,
        ))
    return sales


async def fetch_reported_sales() -> ToolResult:
    """This week's reported sales.

    Returns `{sales: [...], comps: [...], source_url, reported_not_audited}`.
    """
    if config.MOCK_MODE:
        return mock_result(SOURCE, _mock_sales())

    allowed, reason = await robots_allow(WEEKLY)
    if not allowed:
        warn(log, "dnjournal.robots_disallow", url=WEEKLY, reason=reason)
        return error_result(SOURCE, f"robots.txt does not permit this fetch: {reason}")

    html, error = await _fetch(WEEKLY)
    if error:
        return error_result(SOURCE, error)

    sales = parse_sales(html)
    if not sales:
        # A layout change is a parser failure, not an empty week. The
        # report has never had no sales in it.
        return error_result(
            SOURCE, "fetched the report but parsed no sales — the page layout has probably "
                    "changed, and reporting an empty week would be a false fact")
    info(log, "dnjournal.parsed", sales=len(sales), top=sales[0].domain if sales else None)
    return ok_result(SOURCE, {
        "sales": [s.__dict__ for s in sales],
        "comps": [s.as_comp() for s in sales],
        "source_url": WEEKLY,
        "reported_not_audited": True,
        "note": "sales reported to DNJournal by marketplaces and brokers, not independently "
                "audited — venues choose what to report and have an interest in reporting wins",
    })


async def robots_allow(url: str) -> tuple[bool, str]:
    """Whether robots.txt permits fetching `url`.

    Uses protego, the parser Crawlee ships, rather than a hand-rolled one:
    robots.txt has more edge cases than it looks and getting them wrong
    errs against the site owner.
    """
    import httpx
    from protego import Protego

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(f"{BASE}/robots.txt",
                                        headers={"User-Agent": config.CRAWLER_USER_AGENT})
    except Exception as exc:  # noqa: BLE001
        return False, f"could not fetch robots.txt ({type(exc).__name__}) — treated as disallow"
    if response.status_code >= 500:
        # RFC 9309: unavailability means assume complete disallow.
        return False, f"robots.txt returned HTTP {response.status_code} — RFC 9309 says assume disallow"
    if response.status_code == 404:
        return True, "no robots.txt"
    parser = Protego.parse(response.text)
    if parser.can_fetch(url, config.CRAWLER_USER_AGENT):
        return True, "permitted by robots.txt"
    return False, "disallowed by robots.txt"


async def _fetch(url: str) -> tuple[str, str | None]:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S, follow_redirects=True) as client:
            response = await client.get(url, headers={"User-Agent": config.CRAWLER_USER_AGENT})
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}: {exc}"
    if response.status_code != 200:
        return "", f"HTTP {response.status_code}"
    return response.text, None


def _mock_sales() -> dict:
    from portfolio_agent.tools.mocks import _seed, _spread

    seed = _seed("dnjournal", "weekly")
    sales = [
        ReportedSale(domain=d, price=int(_spread(seed, i, 2_000, 250_000)),
                     venue=v, rank=i + 1)
        for i, (d, v) in enumerate([
            ("orchestra.ai", "DomainBooth"), ("plateful.com", "Snagged.com"),
            ("little.co", "Atom.com"), ("paragon.xyz", "Efty.com"),
            ("vertex.io", "Sedo"), ("harbour.com", "Afternic"),
        ])
    ]
    return {
        "sales": [s.__dict__ for s in sales],
        "comps": [s.as_comp() for s in sales],
        "source_url": WEEKLY,
        "reported_not_audited": True,
        "note": "fixture data",
    }
