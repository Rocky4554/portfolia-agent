"""
Hot and high-risk category lists — APVA §3.3's Market Trend Model as data.

The spec's examples are explicit: positive modifiers for "single-word .ai
domains or premium two-word .com brandables sought by funded startups",
negative ones for "domains tied to past hype cycles (e.g. NFT-related .xyz
domains) or containing obsolete technological terms".

Two things this file is careful about:

**It has an as-of date.** A trend list is a perishable good. A stale one
does not fail loudly — it silently keeps applying last year's enthusiasm to
this year's prices, which is precisely the first-generation failure mode APVA
exists to fix. `staleness_warning()` makes the age visible in the output, so
a reader can discount a trend adjustment that is older than the trend.

**It is data, not code.** `refresh()` is the hook a scheduled job calls to
replace these lists from a live source (DNJournal, forum scrapes, the trends
tool's own findings). Nothing else in the system imports the literals
directly — everything goes through the lookup functions, so a refreshed list
takes effect without a deploy.
"""

from dataclasses import dataclass, field
from datetime import date

# When the lists below were last reviewed against the market. Update this in
# the same commit as any edit to them.
AS_OF = date(2026, 9, 1)

# Past this, trend adjustments are reported as stale rather than silently
# trusted. One quarter is roughly how long a domain-market trend holds.
STALE_AFTER_DAYS = 90


@dataclass
class TrendSignal:
    """What the trend layer concluded about one domain."""

    hot_category: bool = False
    risk_category: bool = False
    # -1 (a deflating hype cycle) to +1 (peak demand). The tier engine reads
    # this as a modifier, never as a price.
    trend_velocity: float = 0.0
    matched_rules: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    stale: bool = False


# ── Hot categories ─────────────────────────────────────────────────────────
# (name, velocity, predicate description). Velocity is deliberately well
# short of 1.0: even a peak market is a modifier on comps, not a replacement
# for them.
HOT_TLDS = {
    "ai": 0.75,  # the current cycle — AI startups, single words at a premium
    "io": 0.25,  # long-established tech default, still liquid
}

HOT_KEYWORDS = {
    "ai": 0.6, "agent": 0.5, "agents": 0.5, "llm": 0.5, "gpt": 0.4,
    "robotics": 0.4, "quantum": 0.35, "biotech": 0.3, "genomics": 0.3,
    "climate": 0.3, "solar": 0.3, "energy": 0.25, "payments": 0.3,
    "fintech": 0.3, "security": 0.3, "cyber": 0.35, "cloud": 0.25,
    "data": 0.2, "health": 0.2, "wellness": 0.2,
}

# ── High-risk categories ───────────────────────────────────────────────────
# Past hype cycles and obsolete technology terms. Negative velocity.
RISK_KEYWORDS = {
    "nft": -0.8, "nfts": -0.8, "metaverse": -0.7, "dao": -0.6,
    "ico": -0.7, "defi": -0.4, "token": -0.3, "mining": -0.3,
    "blockchain": -0.3, "crypto": -0.2,
    # Obsolete technology.
    "flash": -0.6, "wap": -0.7, "fax": -0.6, "dvd": -0.6, "cdrom": -0.8,
    "blackberry": -0.7, "myspace": -0.8, "web20": -0.7, "dialup": -0.8,
}

# A TLD that carried a hype cycle and is now priced against that memory.
RISK_TLDS = {"xyz": -0.3}

# Combinations the spec names explicitly. Checked before the per-token rules
# because they describe a whole domain, not a word in it.
HOT_COMBINATIONS = [
    ("single-word .ai", 0.85, lambda category, tld, words: tld == "ai" and len(words) == 1),
    ("two-word .com brandable", 0.4, lambda category, tld, words: tld == "com" and category == "combination"),
]

RISK_COMBINATIONS = [
    ("NFT-era .xyz", -0.85, lambda category, tld, words: tld == "xyz" and any(w in RISK_KEYWORDS for w in words)),
]


def staleness_warning(today: date | None = None) -> str | None:
    """A warning when the trend lists are older than the trends they
    describe, or None when they are current."""
    age = ((today or date.today()) - AS_OF).days
    if age <= STALE_AFTER_DAYS:
        return None
    return (
        f"market trend data is {age} days old (as of {AS_OF.isoformat()}, stale after {STALE_AFTER_DAYS}); "
        "trend adjustments below should be discounted accordingly"
    )


def evaluate(category: str, tld: str, words: list[str], today: date | None = None) -> TrendSignal:
    """Score a domain against the trend lists.

    Velocity is the single strongest matching rule rather than a sum of
    them: a .ai domain about AI is one trend, not two, and adding them would
    let a domain accumulate its way past any cap.
    """
    signal = TrendSignal(stale=staleness_warning(today) is not None)
    tld = (tld or "").lstrip(".").lower()
    words = [w.lower() for w in words or []]

    candidates: list[tuple[str, float]] = []

    for name, velocity, predicate in HOT_COMBINATIONS + RISK_COMBINATIONS:
        if predicate(category, tld, words):
            candidates.append((name, velocity))

    if tld in HOT_TLDS:
        candidates.append((f".{tld} is a hot TLD", HOT_TLDS[tld]))
    if tld in RISK_TLDS:
        candidates.append((f".{tld} carries hype-cycle risk", RISK_TLDS[tld]))

    for word in words:
        if word in HOT_KEYWORDS:
            candidates.append((f"'{word}' is in a hot category", HOT_KEYWORDS[word]))
        if word in RISK_KEYWORDS:
            candidates.append((f"'{word}' is tied to a past hype cycle or obsolete technology", RISK_KEYWORDS[word]))

    if not candidates:
        return signal

    # Risk dominates: a domain that is both trendy and tied to a collapsed
    # hype cycle is a risk, because the downside is what determines whether
    # it sells at all.
    negatives = [c for c in candidates if c[1] < 0]
    chosen = min(negatives, key=lambda c: c[1]) if negatives else max(candidates, key=lambda c: c[1])

    signal.trend_velocity = round(max(-1.0, min(1.0, chosen[1])), 3)
    signal.matched_rules = [name for name, _ in candidates]
    signal.hot_category = signal.trend_velocity > 0
    signal.risk_category = signal.trend_velocity < 0
    return signal


def refresh(
    *,
    hot_tlds: dict | None = None,
    hot_keywords: dict | None = None,
    risk_keywords: dict | None = None,
    risk_tlds: dict | None = None,
    as_of: date | None = None,
) -> None:
    """Replace the trend lists at runtime.

    The hook a scheduled refresh job calls after scraping DNJournal or the
    forums. Replaces rather than merges, so a term that stops being hot
    actually stops being hot — merging would make this file monotonically
    accumulate every trend that ever existed.
    """
    global HOT_TLDS, HOT_KEYWORDS, RISK_KEYWORDS, RISK_TLDS, AS_OF
    if hot_tlds is not None:
        HOT_TLDS = dict(hot_tlds)
    if hot_keywords is not None:
        HOT_KEYWORDS = dict(hot_keywords)
    if risk_keywords is not None:
        RISK_KEYWORDS = dict(risk_keywords)
    if risk_tlds is not None:
        RISK_TLDS = dict(risk_tlds)
    if as_of is not None:
        AS_OF = as_of
