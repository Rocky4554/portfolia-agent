"""
What the same name is worth on a different extension.

The comp ladder's second rung searches for this exact SLD sold on other
TLDs, which only helps if those sales can be made comparable. `cars.co.uk`
selling for $X says something about `cars.com`, but not that they are worth
the same.

Two kinds of entry, and the difference is visible in the output:

  - **derived**: a median price ratio measured from names that actually sold
    on both extensions. Written by `scripts/derive_tld_ratios.py`, which
    records the sample count alongside it.
  - **prior**: a hand-set band used where too few names have sold on both
    sides to measure anything. Every prior carries a comment explaining what
    it rests on, because an unexplained prior is indistinguishable from a
    guess six months later.

Ratios are directional and NOT symmetric-by-assumption. com→ai and ai→com
are stored separately, because the discount for moving off .com is not the
reciprocal of the premium for moving onto it: .com carries incumbency that a
newer extension cannot buy back.
"""

from dataclasses import dataclass
from datetime import date

# When the derived entries were last measured. A ratio is a market fact and
# market facts rot.
AS_OF = date(2026, 9, 21)


@dataclass(frozen=True)
class TransferRatio:
    multiplier: float
    basis: str  # "derived" | "prior"
    samples: int = 0
    note: str = ""

    def describe(self, from_tld: str, to_tld: str) -> str:
        if self.basis == "derived":
            return f".{from_tld}→.{to_tld} transfer ×{self.multiplier:.2f} (median of {self.samples} dual sales)"
        return f".{from_tld}→.{to_tld} transfer ×{self.multiplier:.2f} (prior: {self.note})"


# (from_tld, to_tld) → ratio. Seeded entirely with priors; running
# scripts/derive_tld_ratios.py replaces the pairs it can measure.
#
# The priors below encode three observations that hold across aftermarket
# reporting and are not controversial among domain investors:
#   1. .com commands a large premium over every alternative for the same name.
#   2. .ai currently trades closest to .com of any alternative, and in some
#      AI-adjacent names above it — hence a band rather than a point.
#   3. Legacy alternatives (.net, .org) trade well below .com and below the
#      currently-hot extensions.
TLD_TRANSFER: dict[tuple[str, str], TransferRatio] = {
    ("com", "ai"): TransferRatio(0.25, "prior", note="com→ai band 0.15-0.35; midpoint pending measurement"),
    ("ai", "com"): TransferRatio(3.20, "prior", note="moving ONTO .com buys incumbency; not 1/0.25"),
    ("com", "io"): TransferRatio(0.15, "prior", note="established tech extension, below .ai's current cycle"),
    ("io", "com"): TransferRatio(4.00, "prior", note="see ai→com; .com premium is not a reciprocal"),
    ("com", "co"): TransferRatio(0.12, "prior", note="persistent .com confusion caps .co independently"),
    ("co", "com"): TransferRatio(5.00, "prior", note="see ai→com"),
    ("com", "net"): TransferRatio(0.10, "prior", note="legacy alternative; thin end-user demand"),
    ("net", "com"): TransferRatio(6.00, "prior", note="see ai→com"),
    ("com", "org"): TransferRatio(0.10, "prior", note="non-profit association narrows the buyer pool"),
    ("org", "com"): TransferRatio(6.00, "prior", note="see ai→com"),
    ("ai", "io"): TransferRatio(0.55, "prior", note="both tech extensions; .ai in the stronger cycle"),
    ("io", "ai"): TransferRatio(1.60, "prior", note="inverse of ai→io, both weak priors"),
    # Into .ai from the other alternatives. Needed because .ai is the most
    # common cross-TLD TARGET right now, and without these every such comp
    # fell back to the flagged conservative default.
    ("co", "ai"): TransferRatio(1.80, "prior", note="co→ai; .ai in the stronger cycle, both non-.com"),
    ("net", "ai"): TransferRatio(2.20, "prior", note="legacy alternative into the current cycle"),
    ("org", "ai"): TransferRatio(2.20, "prior", note="see net→ai"),
    ("ai", "co"): TransferRatio(0.55, "prior", note="inverse of co→ai, weak prior"),
    ("ai", "net"): TransferRatio(0.45, "prior", note="inverse of net→ai, weak prior"),
    ("ai", "org"): TransferRatio(0.45, "prior", note="see ai→net"),
    ("co", "io"): TransferRatio(1.20, "prior", note="comparable alternatives, .io slightly ahead"),
    ("io", "co"): TransferRatio(0.80, "prior", note="inverse of co→io"),
    ("net", "org"): TransferRatio(1.00, "prior", note="comparable legacy alternatives"),
    ("org", "net"): TransferRatio(1.00, "prior", note="comparable legacy alternatives"),
}

# Applied when a pair is absent entirely. Deliberately pessimistic and
# deliberately flagged: an unmeasured, unpriored pair should contribute to a
# valuation weakly and visibly.
UNKNOWN_PAIR = TransferRatio(0.20, "prior", note="no ratio for this pair — conservative default")


def transfer_ratio(from_tld: str, to_tld: str) -> TransferRatio:
    """The multiplier taking a sale price on `from_tld` to `to_tld`."""
    from_tld = (from_tld or "").lstrip(".").lower()
    to_tld = (to_tld or "").lstrip(".").lower()
    if from_tld == to_tld:
        return TransferRatio(1.0, "derived", samples=0, note="same extension")
    return TLD_TRANSFER.get((from_tld, to_tld), UNKNOWN_PAIR)


def refresh(measured: dict[tuple[str, str], TransferRatio], as_of: date | None = None) -> None:
    """Replace measured pairs, leaving priors for everything unmeasured.

    Merges rather than replaces, unlike market_trends.refresh: a prior that
    has not been superseded by measurement is still the best available
    answer, whereas a trend that has dropped off the list has genuinely
    stopped being a trend.
    """
    global AS_OF
    TLD_TRANSFER.update(measured)
    if as_of:
        AS_OF = as_of
