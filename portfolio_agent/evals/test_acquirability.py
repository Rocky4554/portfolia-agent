"""
Offline regression eval for the sale-lander detector and the acquirability field.
Run with: uv run python -m portfolio_agent.evals.test_acquirability

The failure this exists to prevent is a confident price on a domain nobody
can buy. Every tier in this engine answers "what is it worth"; none of them
answered "can you have it", so a live company's homepage and an abandoned
registration produced identically authoritative numbers.

What is asserted, in order of importance:

  - A working site is NEVER reported as for sale. This is the expensive
    direction to get wrong: telling a user a running business's domain is on
    the market is worse than saying nothing, so a page that talks about
    domain sales but is plainly a real site stays `developed`.
  - A sale lander IS caught, with its venue, and its price is carried as an
    ASK that never reaches the comps.
  - The zero-credential limit is stated rather than papered over. Without
    WHOIS we cannot distinguish unregistered from registered-with-no-DNS, so
    the word "unregistered" must appear in no verdict.
  - Every status that means "not currently purchasable" produces a warning.
    A quiet field nobody reads would not have prevented the failure.
"""

import asyncio
import sys

from portfolio_agent.nodes.synthesize import _acquirability
from portfolio_agent.schema.valuation_output import Acquirability
from portfolio_agent.tools import lander
from portfolio_agent.tools.lander import classify_page

# ── Fixture pages ──────────────────────────────────────────────────────

AFTERNIC_LANDER = """<html><head><title>quantum.io is for sale</title></head>
<body><script src="https://afternic.com/js/lander.js"></script>
<h1>quantum.io</h1><p>This domain is for sale.</p>
<p>Buy now: $56,000 &nbsp; or 24 monthly payments of $2,333</p>
<form><input name="offer"></form></body></html>"""

# No venue, thin, plainly a sale page.
BARE_LANDER = """<html><head><title>forge.ai</title></head><body>
<h1>forge.ai</h1><p>This domain name is for sale. Make an offer.</p>
<a href="mailto:owner@example.com">Contact</a></body></html>"""

# The false positive that matters: a real publication that writes ABOUT
# domain sales. Every keyword a naive matcher looks for is present.
DOMAIN_NEWS_SITE = """<html><head><title>DNJournal - Domain Sales News</title></head><body>
<nav>Home | Weekly Sales | Archives | About | Advertise | Subscribe</nav>
<h1>This week's biggest domain sales</h1>
<p>Buy this domain, purchase this domain, this domain is for sale — the phrases
every lander uses. We track them all.</p>
""" + "<p>A long article about the aftermarket. " * 400 + """</p>
<footer>Copyright 2026. Contact us. Privacy policy. Terms of service.</footer>
</body></html>"""

PARKED_PAGE = """<html><head><title>example.com</title></head><body>
<h2>Related Searches</h2><ul><li>Insurance quotes</li><li>Car loans</li></ul>
<p>Sponsored listings</p></body></html>"""

LIVE_BUSINESS = """<html><head><title>Forge — Build faster</title></head><body>
<nav>Product Pricing Docs Careers Login</nav>
<h1>Ship your infrastructure in minutes</h1>
""" + "<p>We help engineering teams deploy reliably at scale. " * 300 + """</p>
<footer>Forge Inc. San Francisco. Status. Security. SOC 2.</footer></body></html>"""

EMPTY_PAGE = "<html><head><title></title></head><body></body></html>"


class Checks:
    def __init__(self):
        self.total = 0
        self.failures = []

    def expect(self, condition, message):
        self.total += 1
        if not condition:
            self.failures.append(message)


def check_lander_is_caught(c: Checks):
    v = classify_page("quantum.io", AFTERNIC_LANDER, "https://quantum.io/", 200)
    c.expect(v["status"] == "listed_for_sale", f"an Afternic lander must be caught, got {v['status']}")
    c.expect(v["venue"] == "Afternic", f"the venue must be named, got {v['venue']}")
    c.expect(v["confidence"] == "high", "a venue-identified lander is a high-confidence verdict")
    # The buy-now price, not the instalment.
    c.expect(v["asking_price"] == 56_000,
             f"the ask must be the buy-now price, not the instalment, got {v['asking_price']}")

    # A redirect onto the marketplace itself is conclusive without any prose.
    v = classify_page("quantum.io", "<html><body>Sedo</body></html>",
                      "https://sedo.com/search/details/?domain=quantum.io", 200)
    c.expect(v["status"] == "listed_for_sale" and v["venue"] == "Sedo",
             f"a redirect to a marketplace must be conclusive, got {v['status']}/{v['venue']}")


def check_bare_lander_is_hedged(c: Checks):
    """A sale with no venue is still a sale, but we know less about it."""
    v = classify_page("forge.ai", BARE_LANDER, "https://forge.ai/", 200)
    c.expect(v["status"] == "listed_for_sale", f"a bare lander is still for sale, got {v['status']}")
    c.expect(v["venue"] is None, "no venue may be invented for a bare lander")
    c.expect(v["confidence"] == "medium",
             f"a venueless sale page must not claim high confidence, got {v['confidence']}")


def check_real_site_is_never_for_sale(c: Checks):
    """The expensive direction. Two pages that a keyword matcher gets wrong."""
    v = classify_page("dnjournal.com", DOMAIN_NEWS_SITE, "https://dnjournal.com/", 200)
    c.expect(v["status"] == "developed",
             f"a publication about domain sales must not be listed as for sale, got {v['status']}")
    c.expect(v["confidence"] == "low",
             "a page carrying sale wording is a low-confidence 'developed', and must say so")
    c.expect("human" in v["detail"].lower(),
             "the ambiguous case must invite a human glance rather than assert")

    v = classify_page("forge.ai", LIVE_BUSINESS, "https://forge.ai/", 200)
    c.expect(v["status"] == "developed", f"a live business must read as developed, got {v['status']}")
    c.expect(v["confidence"] == "high", "an unambiguous live site is a confident verdict")
    c.expect(v["asking_price"] is None, "a developed site has no asking price to report")


def check_parked_is_not_for_sale(c: Checks):
    """Parked means held, not offered. Collapsing the two would invent an
    availability that does not exist."""
    v = classify_page("example.com", PARKED_PAGE, "https://example.com/", 200)
    c.expect(v["status"] == "parked", f"a PPC page is parked, not for sale, got {v['status']}")
    c.expect(v["asking_price"] is None, "a parked page carries no ask")
    c.expect("approaching the owner" in v["detail"],
             "the parked verdict must say how acquisition would actually work")


def check_inactive_and_errors(c: Checks):
    v = classify_page("example.com", EMPTY_PAGE, "https://example.com/", 200)
    c.expect(v["status"] == "inactive", f"an empty page is inactive, got {v['status']}")
    v = classify_page("example.com", LIVE_BUSINESS, "https://example.com/", 503)
    c.expect(v["status"] == "inactive", f"a 5xx is not a developed site, got {v['status']}")


def check_no_verdict_claims_unregistered(c: Checks):
    """The zero-credential limit, asserted rather than trusted.

    getaddrinfo cannot tell NXDOMAIN from a registered domain publishing no
    A record. cloud.ai is the live example: it does not resolve and is
    certainly registered. Any verdict using the word would be a fact we did
    not check.
    """
    pages = (AFTERNIC_LANDER, BARE_LANDER, DOMAIN_NEWS_SITE, PARKED_PAGE, LIVE_BUSINESS, EMPTY_PAGE)
    for page in pages:
        v = classify_page("example.com", page, "https://example.com/", 200)
        c.expect("unregistered" not in v["detail"].lower(),
                 f"a verdict claims 'unregistered' without a WHOIS lookup: {v['detail'][:80]}")

    status = asyncio.run(lander.check_domain_status("this-domain-does-not-exist-xyzq.invalid"))
    data = status["data"]
    c.expect(data["status"] == "no_dns", f"a non-resolving domain is no_dns, got {data['status']}")
    c.expect("either" in data["detail"] and "WHOIS" in data["detail"],
             "the no_dns verdict must name both possibilities it cannot separate")
    c.expect(data["resolves"] is False, "no_dns must record that resolution failed")


def check_every_unavailable_status_warns(c: Checks):
    """A field nobody reads would not have prevented the failure."""
    cases = {
        "developed": ("not for sale", True),
        "parked": ("approaching the owner", True),
        "inactive": ("approaching the owner", True),
        "listed_for_sale": ("ASK, not a sale", True),
        "no_dns": ("unverified", True),
        "unknown": ("unverified", True),
    }
    for status, (phrase, must_warn) in cases.items():
        results = {"check_domain_status": {"ok": True, "data": {
            "status": status, "venue": "Sedo" if status == "listed_for_sale" else None,
            "asking_price": 56_000 if status == "listed_for_sale" else None,
            "confidence": "high", "detail": "fixture", "signals": [],
        }}}
        field, warnings = _acquirability(results)
        c.expect(isinstance(field, Acquirability), f"{status} must produce an Acquirability")
        c.expect(field.status == status, f"{status} round-trips into the field")
        c.expect(len(warnings) == must_warn,
                 f"{status} must produce exactly one warning, got {len(warnings)}")
        c.expect(any(phrase in w for w in warnings),
                 f"the {status} warning must say {phrase!r}, got {warnings}")

    # on_the_market is tri-state: unknown is not the same answer as no.
    for status, expected in (("listed_for_sale", True), ("developed", False), ("no_dns", None)):
        results = {"check_domain_status": {"ok": True, "data": {"status": status, "detail": "", "signals": []}}}
        field, _ = _acquirability(results)
        c.expect(field.on_the_market is expected,
                 f"on_the_market for {status} must be {expected}, got {field.on_the_market}")

    # No check run at all is not a verdict.
    field, warnings = _acquirability({})
    c.expect(field is None and not warnings, "an absent check must produce no field and no warning")


def check_ask_never_becomes_a_comp(c: Checks):
    """marketplace.py's doctrine, enforced on this path too."""
    v = classify_page("quantum.io", AFTERNIC_LANDER, "https://quantum.io/", 200)
    c.expect("asking_price" in v and "price" not in v,
             f"the ask must be named asking_price and nothing else, got keys {sorted(v)}")
    field = Acquirability(status="listed_for_sale", asking_price=56_000, venue="Afternic")
    c.expect("ASK" in _acquirability({"check_domain_status": {"ok": True, "data": {
        "status": "listed_for_sale", "asking_price": 56_000, "venue": "Afternic",
        "detail": "", "signals": []}}})[1][0],
        "the listed warning must mark the number as an ask")
    c.expect(field.headline() == "listed for sale on Afternic asking $56,000",
             f"the headline must carry venue and ask, got {field.headline()!r}")


def check_mock_mode_covers_every_status(c: Checks):
    """Mock mode must be able to produce the case that changes the report.
    A fixture that never returns `developed` would let the embarrassing path
    go untested forever."""
    from portfolio_agent import config
    original = config.MOCK_MODE
    config.MOCK_MODE = True
    try:
        seen = set()
        for i in range(400):
            result = asyncio.run(lander.check_domain_status(f"fixture{i}.com"))
            seen.add(result["data"]["status"])
            c.expect(result["mock"] is True, "mock mode must flag its results as mock")
        for status in ("listed_for_sale", "parked", "developed", "inactive", "no_dns"):
            c.expect(status in seen, f"mock mode never produces {status}")
    finally:
        config.MOCK_MODE = original


def check_registration_resolves_what_dns_cannot(c: Checks):
    """The limit check_domain_status documented as unfixable, fixed.

    getaddrinfo returns the same failure for a domain nobody has registered
    and one registered since 1997 that publishes no DNS, so the verdict was
    `no_dns` and named both possibilities. cloud.ai is the live case: it
    does not resolve and has been registered since 2017. RDAP answers it
    outright, for free, with no key.
    """
    def acquirability(status, registration):
        return _acquirability({
            "check_domain_status": {"ok": True, "data": {
                "status": status, "detail": "d", "signals": [], "confidence": "medium"}},
            "get_registration": {"ok": True, "data": registration},
        })

    held, warnings = acquirability("no_dns", {"registered": True, "registered_on": "1997-04-30",
                                              "age_years": 29.4, "days_to_expiry": 221,
                                              "registrar": "Squarespace Domains II LLC"})
    c.expect(held.status == "inactive",
             f"a registered domain with no DNS is held, not unknown: got {held.status}")
    c.expect("owner decision" in held.detail,
             f"and must say what it means, got {held.detail!r}")
    c.expect(held.age_years == 29.4 and held.registrar,
             "registration facts must reach the field")

    free, warnings = acquirability("no_dns", {"registered": False})
    c.expect(free.status == "unregistered",
             f"a domain with no registry record is unregistered: got {free.status}")
    c.expect(free.on_the_market is True,
             "an unregistered domain is available in the only sense that matters")
    c.expect(any("registration cost" in w for w in warnings),
             f"and the warning must say the tiers are not an asking price: {warnings}")

    # Neither answer is invented when the lookup did not run.
    unknown, _ = acquirability("no_dns", {})
    c.expect(unknown.status == "no_dns",
             f"without a registration lookup the hedge must stand: got {unknown.status}")
    c.expect(unknown.registered is None, "and it must not claim a registration either way")


def check_a_dropping_registration_is_surfaced(c: Checks):
    """The one moment a portfolio buyer most wants to hear about, and it is
    invisible in everything else gathered."""
    field, warnings = _acquirability({
        "check_domain_status": {"ok": True, "data": {
            "status": "parked", "detail": "d", "signals": [], "confidence": "medium"}},
        "get_registration": {"ok": True, "data": {
            "registered": True, "dropping": True, "statuses": ["redemptionperiod"],
            "days_to_expiry": -12}},
    })
    c.expect(field.dropping, "a dropping registration must be recorded on the field")
    c.expect(any("redemption" in w for w in warnings),
             f"and must produce a warning naming it: {warnings}")
    c.expect(any("caught at the drop" in w for w in warnings),
             "the warning must say how acquisition would actually work here")


def main() -> int:
    c = Checks()
    for fn in (check_registration_resolves_what_dns_cannot,
               check_a_dropping_registration_is_surfaced,
               check_lander_is_caught, check_bare_lander_is_hedged,
               check_real_site_is_never_for_sale, check_parked_is_not_for_sale,
               check_inactive_and_errors, check_no_verdict_claims_unregistered,
               check_every_unavailable_status_warns, check_ask_never_becomes_a_comp,
               check_mock_mode_covers_every_status):
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
