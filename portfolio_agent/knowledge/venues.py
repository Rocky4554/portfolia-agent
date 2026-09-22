"""
What kind of market a sale happened in.

A price means different things depending on where it cleared. An expiry
auction is investors bidding against each other with no end user present; a
brokered sale is an end user who was found, courted and negotiated with.
The second clears well above the first for the same asset, so an "auction
price" built from brokered sales is not a broader estimate — it is a
different number wearing the wrong label.

Venue strings arrive from NameBio and from our own feeds in whatever casing
and spelling the source used, so matching is normalized and substring-based.
Anything unrecognized is `unknown` rather than guessed: an unclassified
venue should dilute a claim about venue mix, not silently join whichever
class happens to be convenient.
"""

AUCTION = "auction"
MARKETPLACE_RETAIL = "marketplace_retail"
BROKERED = "brokered"
INTERNAL = "internal"
UNKNOWN = "unknown"

# Substring → class. Ordered most specific first; the first hit wins.
_VENUE_CLASSES: tuple[tuple[str, str], ...] = (
    # Competitive auctions: investors bidding, usually on expiry inventory.
    ("godaddy auction", AUCTION),
    ("namejet", AUCTION),
    ("dropcatch", AUCTION),
    ("snapnames", AUCTION),
    ("park.io", AUCTION),
    ("dynadot", AUCTION),
    ("sav.com", AUCTION),
    ("pheenix", AUCTION),
    ("catched", AUCTION),
    # Retail marketplaces: listed at an asking price, bought by whoever
    # arrives. Above auction, below a brokered end-user sale.
    ("afternic", MARKETPLACE_RETAIL),
    ("godaddy", MARKETPLACE_RETAIL),  # after the auction entry above
    ("sedo", MARKETPLACE_RETAIL),
    ("dan.com", MARKETPLACE_RETAIL),
    ("dan ", MARKETPLACE_RETAIL),
    ("atom", MARKETPLACE_RETAIL),
    ("squadhelp", MARKETPLACE_RETAIL),
    ("brandbucket", MARKETPLACE_RETAIL),
    ("efty", MARKETPLACE_RETAIL),
    ("undeveloped", MARKETPLACE_RETAIL),
    ("namecheap", MARKETPLACE_RETAIL),
    # Negotiated. The top of the range for the same asset.
    ("private", BROKERED),
    ("broker", BROKERED),
    ("escrow", BROKERED),
    ("media options", BROKERED),
    ("mediaoptions", BROKERED),
    ("vpn.com", BROKERED),
    ("sudos", BROKERED),
    # Our own book.
    ("internal", INTERNAL),
)

# Classes that clear ABOVE an auction for the same asset, and therefore bias
# an auction anchor upward when they are in it.
ABOVE_AUCTION = frozenset({MARKETPLACE_RETAIL, BROKERED})


def classify_venue(venue: str | None) -> str:
    """The market class a venue string belongs to."""
    name = (venue or "").strip().lower()
    if not name:
        return UNKNOWN
    for needle, venue_class in _VENUE_CLASSES:
        if needle in name:
            return venue_class
    return UNKNOWN


def retail_share(comps: list[dict]) -> float:
    """The weighted share of comps that clear above auction.

    Weighted, because the comp set is already weighted — an exact comp
    counts double and an own-book sale one and a half — and a venue mix
    computed on raw counts would disagree with the distribution the price is
    actually built from.

    `unknown` venues count toward neither side but toward the total, so an
    unclassified set dilutes the claim rather than asserting one.
    """
    total = sum(float(c.get("weight") or 1.0) for c in comps)
    if total <= 0:
        return 0.0
    above = sum(
        float(c.get("weight") or 1.0)
        for c in comps
        if classify_venue(c.get("marketplace") or c.get("platform") or c.get("venue")) in ABOVE_AUCTION
    )
    return round(above / total, 4)
