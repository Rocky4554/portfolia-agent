"""
get_marketplace_listings — APVA Table 4.1.1's "Marketplace Listings", the
one adjustment source in that table which was never wired.

**The distinction this module exists to preserve.** A listing is what a
seller ASKS. A comp is what a buyer PAID. They are not the same fact and
they must never be averaged together. consulting.ai is listed at roughly
$2.8M; the only recorded sale of a consulting `.ai` domain is $425. Feeding
that ask into a comps distribution would not improve the valuation, it would
poison the anchor with one seller's hope. So listings come back in their own
structure, the tier engine takes them as a separate input, and nothing here
can reach `TierInputs.comps`.

**No single source exists.** Each marketplace controls its own inventory and
its own API access, so providers are registered independently and any subset
may be missing. A provider with no credentials reports itself unavailable
rather than failing the lookup — a valuation with two of three marketplaces
is still a valuation.

**Provider status.** The Name.ai and GoDaddy Auctions adapters are written
to their documented request shapes and have NOT been exercised against the
live services — neither credential is configured here. Their normalization
is the part most likely to need correcting once a real response is seen,
which is why every adapter returns through `_normalize` rather than shaping
its own output.
"""

from dataclasses import dataclass, field
from datetime import date, datetime

from portfolio_agent import config
from portfolio_agent.logging_setup import debug, get_logger, info, warn
from portfolio_agent.tools.common import ToolResult, error_result, mock_result, ok_result, request_json

log = get_logger(__name__)

SOURCE = "marketplace_listings"

_TIMEOUT_S = 25.0

# What a listing IS, which determines how much it can be trusted as a price
# signal. A live auction with bids is a market clearing in real time; a
# buy-now ask is one person's number.
LISTING_KINDS = ("buy_now", "auction", "make_offer", "lease")


@dataclass
class Listing:
    """One live listing, normalized across marketplaces."""

    domain: str
    marketplace: str
    kind: str
    price: int | None = None  # buy-now / ask, in USD
    current_bid: int | None = None  # auctions only
    currency: str = "USD"
    ends_at: str | None = None
    listed_on: str | None = None
    url: str | None = None
    note: str | None = None

    @property
    def headline_price(self) -> int | None:
        """The number a buyer would act on. An auction's live bid is a real
        market position; a buy-now price is an ask. Bid first when both
        exist, because someone has actually committed to it."""
        return self.current_bid or self.price

    def as_row(self) -> dict:
        return {
            "domain": self.domain,
            "marketplace": self.marketplace,
            "kind": self.kind,
            "price": self.price,
            "current_bid": self.current_bid,
            "currency": self.currency,
            "ends_at": self.ends_at,
            "listed_on": self.listed_on,
            "url": self.url,
            "note": self.note,
        }


@dataclass
class ProviderResult:
    provider: str
    available: bool
    listings: list[Listing] = field(default_factory=list)
    reason: str | None = None


# ── Providers ──────────────────────────────────────────────────────────────


async def _nameai(domain: str) -> ProviderResult:
    """Name.ai's own marketplace, via its public v1 API.

    Rewritten against the actual service rather than a documented shape.
    Every detail of the previous version was wrong: it called
    `/v1/marketplace/listings` (the route is `/api/v1/domains`), sent a
    Bearer token (it wants `x-api-key` and `x-api-secret`, two headers), and
    read `buyNowPrice` and `floorPrice` (the fields are `bin_price` and
    `floor_price`). This is what the module's own header warned about —
    adapters written to a request shape and never exercised — and it is
    worth recording that the warning was accurate.

    `/api/v1/domains` is a marketplace LISTING, so everything it returns is
    an ask. The search filter is a substring match, so an exact match on the
    domain has to be checked here rather than assumed.
    """
    if not (config.NAMEAI_API_KEY and config.NAMEAI_API_SECRET):
        missing = " and ".join(
            name for name, value in (("NAMEAI_API_KEY", config.NAMEAI_API_KEY),
                                     ("NAMEAI_API_SECRET", config.NAMEAI_API_SECRET))
            if not value)
        return ProviderResult(
            "name.ai", False,
            reason=f"no {missing} — name.ai's v1 API needs both an api key and a secret",
        )

    sld = domain.split(".")[0]
    body, error = await request_json(
        "GET",
        f"{config.NAMEAI_API_BASE_URL}/api/v1/domains",
        params={"search": sld, "limit": 50},
        headers={"x-api-key": config.NAMEAI_API_KEY,
                 "x-api-secret": config.NAMEAI_API_SECRET,
                 "Accept": "application/json"},
        timeout=_TIMEOUT_S,
    )
    if error:
        return ProviderResult("name.ai", False, reason=error)

    listings = []
    for row in _rows_from(body, ("domains",)):
        listed = str(row.get("domain") or "").lower()
        # `search` matches a substring, so "hop" returns hopscotch.com too.
        # A listing for a different domain is not a price for this one.
        if listed != domain.lower():
            continue
        floor = _money(row, "floor_price")
        ask = _money(row, "bin_price")
        # What the OWNER receives, which is a different number from what the
        # buyer pays and is the one a seller actually decides on.
        #
        # The commission is deducted from the sale price, not added to it:
        # orders.js states `seller_payout = sale_price - marketplace_commission`.
        # So a $10,000 BIN nets the seller $9,000 at a 10% rate — the listing
        # is not marked up by the fee, it is reduced by it.
        #
        # The v1 API does not return the rate, so this uses the configured
        # default and says so. A net figure computed from a guessed rate
        # would be worse than none, which is why `net_rate_assumed` travels
        # with it.
        net = round(ask * (1 - config.NAMEAI_COMMISSION_PCT / 100.0)) if ask else None
        listings.append(Listing(
            domain=listed,
            marketplace="name.ai",
            # A floor price means offers are accepted, which is a different
            # kind of ask from a fixed buy-now.
            kind="buy_now" if row.get("bin_price") else "make_offer",
            price=_money(row, "bin_price"),
            currency="USD",
            listed_on=_as_date(row.get("listed_at")),
            url=row.get("lander_url"),
            note="; ".join(part for part in (
                f"offers accepted from {floor:,.0f}" if floor else None,
                (f"seller nets ~{net:,} after {config.NAMEAI_COMMISSION_PCT:.0f}% commission "
                 f"(rate assumed, not returned by the API)") if net else None,
            ) if part) or None,
        ))
    return ProviderResult("name.ai", True, listings)


async def _godaddy_auctions(domain: str) -> ProviderResult:
    """GoDaddy Auctions aftermarket listings.

    POST /v1/customers/{customerId}/aftermarket/listings/available, which
    takes a list of domains and reports which have live listings, with the
    current bid and close time where there is an auction running.
    """
    if not (config.GODADDY_API_KEY and config.GODADDY_API_SECRET):
        return ProviderResult(
            "godaddy", False,
            reason="no GODADDY_API_KEY / GODADDY_API_SECRET — aftermarket listings unavailable",
        )

    body, error = await request_json(
        "POST",
        f"{config.GODADDY_API_BASE_URL}/v1/customers/{config.GODADDY_CUSTOMER_ID}"
        f"/aftermarket/listings/available",
        json={"domains": [domain]},
        headers={
            "Authorization": f"sso-key {config.GODADDY_API_KEY}:{config.GODADDY_API_SECRET}",
            "Accept": "application/json",
        },
        timeout=_TIMEOUT_S,
    )
    if error:
        return ProviderResult("godaddy", False, reason=error)

    rows = _rows_from(body, ("listings", "results", "data", "domains"))
    listings = [
        Listing(
            domain=str(row.get("domain") or domain).lower(),
            marketplace="godaddy",
            kind=_kind_of(row, default="auction" if row.get("auctionEndTime") or row.get("currentBid") else "buy_now"),
            price=_money(row, "buyNowPrice", "price", "listPrice"),
            current_bid=_money(row, "currentBid", "bidPrice", "highBid"),
            currency=str(row.get("currency") or "USD"),
            ends_at=_as_date(row.get("auctionEndTime") or row.get("endTime")),
            url=row.get("url"),
        )
        for row in rows
        if row.get("available") is not False
    ]
    return ProviderResult("godaddy", True, listings)


async def _dynadot(domain: str) -> ProviderResult:
    """Dynadot's aftermarket.

    A genuinely different inventory from GoDaddy's, not a fallback for it.
    Dynadot carries millions of listings and does not carry consulting.ai,
    which GoDaddy lists at roughly $2.8M — each marketplace knows only its
    own book, which is the whole reason this module is a registry.

    A domain absent from Dynadot's book is not an error and not evidence
    that it is unlisted. It returns available with no listings, and the
    caller reports "listed here: no" rather than "unlisted".
    """
    if not config.DYNADOT_API_KEY:
        return ProviderResult("dynadot", False, reason="no DYNADOT_API_KEY — aftermarket listings unavailable")

    path = config.DYNADOT_LISTING_PATH.format(domain=domain)
    body, error = await request_json(
        "GET",
        f"{config.DYNADOT_BASE_URL}{path}",
        params={"currency": "USD"},
        headers={"Authorization": f"Bearer {config.DYNADOT_API_KEY}", "Accept": "application/json"},
        timeout=_TIMEOUT_S,
    )
    if error:
        # "Can not find the <domain>" is Dynadot's answer for a domain it
        # does not list, and that is information rather than a failure.
        if "404" in error or "not find" in error.lower():
            return ProviderResult("dynadot", True, [])
        return ProviderResult("dynadot", False, reason=error)

    rows = _rows_from(body, ("listing_item", "listing_item_list", "listings", "data"))
    listings = [
        Listing(
            domain=str(row.get("domain") or domain).lower(),
            marketplace="dynadot",
            # BIN, BIN_MAKE_OFFER, and auction types.
            kind=_kind_of(row, default="buy_now"),
            price=_money(row, "price", "buyNowPrice", "bin_price"),
            current_bid=_money(row, "current_bid", "currentBid", "high_bid"),
            currency="USD",
            listed_on=_as_date(row.get("registration_date")),
            url=f"https://www.dynadot.com/market/listing/{row.get('listing_id')}"
            if row.get("listing_id") else None,
            note=_dynadot_note(row),
        )
        for row in rows
    ]
    return ProviderResult("dynadot", True, listings)


def _dynadot_note(row: dict) -> str | None:
    """The fields Dynadot carries that other marketplaces do not.

    Inbound links and domain age are demand and provenance signals, not
    price signals — kept as a note rather than promoted into the listing, so
    nothing downstream can mistake them for part of the ask.
    """
    parts = []
    if row.get("in_bound_links"):
        parts.append(f"{row['in_bound_links']:,} inbound links")
    if row.get("age"):
        parts.append(f"{row['age']} years old")
    if str(row.get("type") or "").upper().endswith("MAKE_OFFER"):
        parts.append("offers accepted below the listed price")
    return "; ".join(parts) or None


async def get_dynadot_appraisal(domain: str) -> ToolResult:
    """Dynadot's automated appraisal — a second opinion, never an anchor.

    Kept apart from the listings call because an appraisal and an ask are
    different facts, the same separation this module enforces everywhere.
    It feeds the appraisal cross-check, where disagreement is reported
    rather than resolved. On consulting.ai the field currently reads:
    Estibot $20,000, GoDaddy GoValue $2,220, Dynadot $63,961, a live GoDaddy
    ask of ~$2,827,928. A 1,275x spread across four professional sources is
    itself the finding.
    """
    domain = (domain or "").strip().lower()
    if not domain:
        return error_result("dynadot_appraisal", "domain is empty")
    if config.MOCK_MODE:
        from portfolio_agent.tools.mocks import anchor_value

        return mock_result("dynadot_appraisal", {"domain": domain, "appraisal": anchor_value(domain)})
    if not config.DYNADOT_API_KEY:
        return error_result("dynadot_appraisal", "no DYNADOT_API_KEY configured")

    path = config.DYNADOT_APPRAISAL_PATH.format(domain=domain)
    body, error = await request_json(
        "GET",
        f"{config.DYNADOT_BASE_URL}{path}",
        headers={"Authorization": f"Bearer {config.DYNADOT_API_KEY}", "Accept": "application/json"},
        timeout=_TIMEOUT_S,
    )
    if error:
        return error_result("dynadot_appraisal", error)

    raw = body if isinstance(body, dict) else {}
    # Reported as "$63961" — a string with a currency symbol.
    value = raw.get("appraisal_price") or raw.get("appraisal") or raw.get("price")
    parsed = _parse_money_string(value)
    if parsed is None:
        return error_result("dynadot_appraisal", "unexpected response shape")
    return ok_result("dynadot_appraisal", {"domain": domain, "appraisal": parsed})


def _parse_money_string(value) -> int | None:
    """'$63961' or 63961 or '63,961.00' → 63961."""
    if isinstance(value, (int, float)):
        return int(value) if value > 0 else None
    if not isinstance(value, str):
        return None
    digits = "".join(ch for ch in value if ch.isdigit() or ch == ".")
    try:
        parsed = float(digits)
    except ValueError:
        return None
    return int(parsed) if parsed > 0 else None


# Registered providers, tried in order. Adding a marketplace is one entry.
PROVIDERS = (_nameai, _godaddy_auctions, _dynadot)


# ── Entry point ────────────────────────────────────────────────────────────


async def get_marketplace_listings(domain: str) -> ToolResult:
    """Live listings for `domain` across every configured marketplace.

    Always succeeds when at least one provider answered, even with no
    listings: "nobody is selling this" is a finding, not a failure. Fails
    only when no provider could be reached at all, because then we know
    nothing either way and must not imply the domain is unlisted.
    """
    domain = (domain or "").strip().lower()
    if not domain:
        return error_result(SOURCE, "domain is empty")

    if config.MOCK_MODE:
        return mock_result(SOURCE, _mock_payload(domain))

    results = [await provider(domain) for provider in PROVIDERS]
    reachable = [r for r in results if r.available]

    if not reachable:
        reasons = "; ".join(f"{r.provider}: {r.reason}" for r in results)
        return error_result(SOURCE, f"no marketplace is configured — {reasons}")

    listings = [listing for result in reachable for listing in result.listings]
    for result in results:
        if not result.available:
            debug(log, "marketplace.provider_skipped", provider=result.provider, reason=result.reason)
    info(log, "marketplace.checked", domain=domain,
         providers_reachable=len(reachable), providers_total=len(results), listings=len(listings))

    return ok_result(SOURCE, _payload(domain, listings, results))


def _payload(domain: str, listings: list[Listing], results: list[ProviderResult]) -> dict:
    priced = [l for l in listings if l.headline_price]
    return {
        "domain": domain,
        "listed": bool(listings),
        "listings": [l.as_row() for l in listings],
        "lowest_ask": min((l.headline_price for l in priced), default=None),
        "highest_ask": max((l.headline_price for l in priced), default=None),
        "has_live_auction": any(l.kind == "auction" and l.current_bid for l in listings),
        "providers_checked": [r.provider for r in results if r.available],
        "providers_unavailable": {r.provider: r.reason for r in results if not r.available},
        # Carried into every consumer, because the whole risk with this
        # source is that an ask gets read as a valuation.
        "caveat": (
            "These are ASKING prices, not sales. A listing is what a seller wants; only a "
            "recorded sale is what a buyer paid. Premium listings routinely sit unsold for years."
        ),
    }


def _mock_payload(domain: str) -> dict:
    """A deterministic listing, present for roughly half of domains.

    Half, deliberately: the unlisted path is the common one and has to be
    exercised by the evals as much as the listed one.
    """
    from portfolio_agent.tools.mocks import _seed, _spread, anchor_value

    seed = _seed(domain, "marketplace")
    if seed % 2:
        return _payload(domain, [], [ProviderResult("name.ai", True), ProviderResult("godaddy", True)])

    # Asks sit above what the thing actually trades for. That is the point
    # of the source and the fixture should show it.
    ask = max(500, int(anchor_value(domain) * _spread(seed, 0, 1.4, 4.0)))
    listings = [
        Listing(domain=domain, marketplace="name.ai", kind="buy_now", price=ask,
                listed_on="2026-03-01", url=f"https://name.ai/domain/{domain}",
                note="fixture listing — not a real offer")
    ]
    if seed % 4 == 0:
        listings.append(
            Listing(domain=domain, marketplace="godaddy", kind="auction",
                    current_bid=int(ask * 0.35), ends_at="2026-10-01",
                    note="fixture listing — not a real offer")
        )
    return _payload(domain, listings, [ProviderResult("name.ai", True), ProviderResult("godaddy", True)])


# ── Normalization helpers ──────────────────────────────────────────────────


def _rows_from(body, keys: tuple[str, ...]) -> list[dict]:
    """Rows out of a response whose envelope shape we cannot rely on.

    Written defensively on purpose: neither adapter has been run against its
    live service, so the container key is the most likely thing to be wrong
    and the least interesting thing to fail on.
    """
    if isinstance(body, list):
        return [r for r in body if isinstance(r, dict)]
    if not isinstance(body, dict):
        return []
    for key in keys:
        value = body.get(key)
        if isinstance(value, list):
            return [r for r in value if isinstance(r, dict)]
    return [body] if body.get("domain") else []


def _money(row: dict, *keys: str) -> int | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, dict):  # {"value": 25000, "currency": "USD"}
            value = value.get("value") or value.get("amount")
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return None


def _kind_of(row: dict, default: str) -> str:
    raw = str(row.get("kind") or row.get("type") or row.get("listingType") or "").lower()
    for kind in LISTING_KINDS:
        if kind.replace("_", "") in raw.replace("-", "").replace("_", "").replace(" ", ""):
            return kind
    return default


def _as_date(value) -> str | None:
    if not value:
        return None
    text = str(value)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return text[:10] if len(text) >= 10 else None
