"""
Comparable domain acquisitions, for the Strategic M&A tier (APVA Table
4.1.1, "M&A Case Studies").

A static, curated dataset. It exists because the top tier has no comps in
the ordinary sense — there are only a few dozen transactions of this kind
ever, they are years apart, and no sales feed covers them. Citing a named
case is the only honest way to justify a number at this level.

**On the figures.** These are prices as publicly reported in the domain
industry press at the time of sale. Several were disclosed by one party
only, some were later disputed, and a few bundle a domain with other assets.
They are a reference frame for reasoning about magnitude, not confirmed
transaction records, and `DISCLAIMER` travels with them into any output that
cites one. Verify a specific case before putting it in front of a client.
"""

from dataclasses import dataclass

DISCLAIMER = (
    "Figures are as publicly reported in the domain industry press; several were disclosed by one "
    "party only and some bundle other assets. Treat them as a reference frame for magnitude, not as "
    "confirmed transaction records."
)


@dataclass(frozen=True)
class MACase:
    domain: str
    price: int
    year: int
    note: str

    @property
    def sld(self) -> str:
        return self.domain.split(".")[0]

    @property
    def tld(self) -> str:
        return self.domain.split(".")[-1]


# Ordered by price. Every entry is a single dictionary word or a two/three
# character string on .com — the shape that defines the tier.
CASES: tuple[MACase, ...] = (
    MACase("insurance.com", 35_600_000, 2010, "single-word category killer in the highest-CPC vertical"),
    MACase("vacationrentals.com", 35_000_000, 2007, "exact category term; bought defensively to protect search position"),
    MACase("privatejet.com", 30_180_000, 2012, "exact category term in a high-ticket niche"),
    MACase("voice.com", 30_000_000, 2019, "single dictionary word acquired for a product launch"),
    MACase("internet.com", 18_000_000, 2009, "single dictionary word defining the medium itself"),
    MACase("360.com", 17_000_000, 2015, "three-character numeric matching the acquirer's own brand"),
    MACase("sex.com", 13_000_000, 2010, "single word with enormous direct type-in traffic"),
    MACase("hotels.com", 11_000_000, 2001, "exact category term; became the acquirer's primary brand"),
    MACase("fund.com", 9_999_950, 2008, "single dictionary word in finance"),
    MACase("porn.com", 9_500_000, 2007, "single word with direct type-in traffic"),
    MACase("fb.com", 8_500_000, 2010, "two-letter matching the acquirer's existing abbreviation"),
    MACase("we.com", 8_000_000, 2015, "two-letter dictionary word"),
    MACase("diamond.com", 7_500_000, 2006, "single word defining a retail category"),
    MACase("beer.com", 7_000_000, 2004, "single word defining a consumer category"),
    MACase("z.com", 6_800_000, 2014, "single character, the scarcest class of domain"),
    MACase("slots.com", 5_500_000, 2010, "exact category term in gaming"),
    MACase("toys.com", 5_100_000, 2009, "single word defining a retail category"),
)


def as_comps(cases: list[MACase]) -> list[dict]:
    """M&A cases in the shape the tier engine reads comparable sales in.

    `marketplace` is None because these were negotiated acquisitions, not
    venue sales — which is also why the auction tier's venue filter finds
    nothing to filter on and correctly says so.
    """
    return [
        {
            "domain": case.domain,
            "price": float(case.price),
            "date": f"{case.year}-01-01",
            "marketplace": None,
            "note": case.note,
        }
        for case in cases
    ]


def comparable_cases(
    sld: str, tld: str, limit: int = 3, exclude_self: bool = False, by_price: bool = True
) -> list[MACase]:
    """The cases most structurally like this domain.

    Similarity is structural, not semantic — length class and TLD — because
    that is what these transactions actually have in common.

    Note what this dataset IS: seventeen of the most famous domain sales
    ever made. It is survivorship-biased by construction, because unremarkable
    eight-figure sales do not get reported. Any average taken over it skews
    high, and a caller using it as an anchor has to say so. Pairing a
    one-word .com with other one-word .coms is a defensible comparison;
    pairing it by industry would suggest a precision seventeen data points
    cannot support.
    """
    target_length = len(sld)
    pool = [case for case in CASES if case.tld == tld] or list(CASES)
    if exclude_self:
        # A domain cannot be its own comparable. insurance.com is in this
        # dataset, and anchoring its valuation on its own sale price would
        # be circular rather than comparative.
        pool = [case for case in pool if case.sld != (sld or "").lower()]

    # `by_price` picks the most notable case at each length, which is what
    # you want when CITING one. It is exactly wrong when the selection is
    # going to be read as a distribution: taking the priciest case in every
    # length bucket biases the sample high before any arithmetic runs. An
    # anchor asks for by_price=False and gets a length-ordered slice with
    # its natural price spread intact.
    if by_price:
        ordering = lambda case: (abs(len(case.sld) - target_length), -case.price)
    else:
        ordering = lambda case: (abs(len(case.sld) - target_length), case.domain)
    return sorted(pool, key=ordering)[:limit]


def median_price(cases: list[MACase]) -> int | None:
    if not cases:
        return None
    prices = sorted(case.price for case in cases)
    middle = len(prices) // 2
    if len(prices) % 2:
        return prices[middle]
    return (prices[middle - 1] + prices[middle]) // 2
