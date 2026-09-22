"""
Where each extension sits in its own life cycle — APVA §3.3 read as a curve
rather than a flag.

**The flaw this replaces.** `EMERGING_TLDS` is a flat set, so `.xyz` and
`.ai` were treated identically and permanently. One of those peaked with the
NFT cycle and fell; the other is in the middle of the largest extension boom
since `.io`. A model that calls both "emerging" forever will keep applying
2021's enthusiasm to 2026's prices, which is the first-generation failure
APVA exists to fix.

**What the phase is FOR: dating the evidence.** An extension's phase decides
how quickly its past sales stop describing its present market. A 2021 `.io`
sale is weak evidence about `.io` in 2026 and strong evidence about `.com`,
because `.com` barely moves and `.io` had a cycle in between. So the phase
sets a half-life on comps, and the decay runs in both directions: in a
declining extension old comps overstate today's value, in a rising one they
understate it. One mechanism, correct either way.

**This is curated judgement with a date on it**, exactly like
`market_trends.py`, and it is perishable in the same way. `AS_OF` is not
decoration — a phase list older than the phase it describes is worse than
none, because it is confidently wrong. `refresh()` is the hook for replacing
it from a live source without a deploy.
"""

from dataclasses import dataclass
from datetime import date

# When these phases were last reviewed against the market. Update this in the
# same commit as any edit below.
AS_OF = date(2026, 9, 1)

# Past this, the lifecycle model reports itself stale rather than being
# silently trusted. Extensions move more slowly than categories, so this is
# longer than market_trends' quarter.
STALE_AFTER_DAYS = 270

# Ordered from youngest to oldest so callers can compare positions.
PHASES = ("emerging", "growth", "peak", "mature", "decline", "dormant")


@dataclass(frozen=True)
class TldPhase:
    tld: str
    phase: str
    # How long a sale in this extension stays good evidence. Short where the
    # market is moving fast in either direction; long where it barely moves.
    comp_half_life_days: int
    note: str


# A deliberately small list. An extension not named here gets the neutral
# default rather than a guess, because inventing a phase for an extension
# nobody has looked at is the error this module exists to prevent.
_PHASES: dict[str, TldPhase] = {
    "com": TldPhase("com", "mature", 1460,
                    "the benchmark extension; its prices move with the economy, not with fashion"),
    "ai": TldPhase("ai", "peak", 365,
                   "in the largest extension boom since .io; sales more than a year old "
                   "describe a materially cheaper market than today's"),
    "io": TldPhase("io", "mature", 900,
                   "long-established startup extension, past its own boom and largely settled, "
                   "with some attention lost to .ai"),
    "co": TldPhase("co", "mature", 1095, "settled general-purpose alternative to .com"),
    "app": TldPhase("app", "mature", 1095, "stable, with a defined use and little speculation"),
    "dev": TldPhase("dev", "mature", 1095, "stable developer extension"),
    "xyz": TldPhase("xyz", "decline", 540,
                    "carried the NFT and web3 cycle and fell with it; older sales reflect a "
                    "speculative peak that has not returned"),
    "tech": TldPhase("tech", "decline", 540,
                     "broad extension that has not held its early pricing"),
    "cloud": TldPhase("cloud", "mature", 1095, "niche but stable, tied to infrastructure branding"),
    "gg": TldPhase("gg", "growth", 540, "gaming-specific and still gaining adoption"),
    "sh": TldPhase("sh", "growth", 720, "small developer-adjacent niche"),
}

# What an unknown extension gets. Neutral phase, and a half-life long enough
# that the recency decay does almost nothing — an extension we have not
# assessed should not have its comps quietly discounted.
_DEFAULT = TldPhase("", "unassessed", 1460,
                    "no lifecycle assessment for this extension; comps are dated neutrally")


def phase_for(tld: str) -> TldPhase:
    tld = (tld or "").strip().lower().lstrip(".")
    known = _PHASES.get(tld)
    return known if known else TldPhase(tld, _DEFAULT.phase, _DEFAULT.comp_half_life_days, _DEFAULT.note)


def is_assessed(tld: str) -> bool:
    return (tld or "").strip().lower().lstrip(".") in _PHASES


def staleness_warning(today: date | None = None) -> str | None:
    """Whether the phase list is older than the phases it describes."""
    age = ((today or date.today()) - AS_OF).days
    if age <= STALE_AFTER_DAYS:
        return None
    return (
        f"TLD lifecycle data is {age} days old (reviewed {AS_OF.isoformat()}); an extension's "
        f"phase can change inside that window, so the comp dating below may be applying a "
        f"position the extension has already left"
    )


def refresh(phases: dict[str, TldPhase], as_of: date) -> None:
    """Replace the phase table from a live source. See market_trends.refresh."""
    global AS_OF
    _PHASES.clear()
    _PHASES.update(phases)
    AS_OF = as_of
