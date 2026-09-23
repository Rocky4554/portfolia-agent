"""
Category-specific plausibility floors for the F17 comp-plausibility gate.

Each (category, TLD) pair has a known floor below which a comp-derived price
is implausible — not merely low, but structurally wrong. These are hand-set
priors derived from observed market minimums, not from theory.

When comps-derived end_user falls below its category floor, the comps are
likely misidentified (substring junk labeled as exact matches, wrong quality
band, etc.) rather than the market having collapsed. The plausibility gate
trips and confidence is capped.

Floors are conservative — set below the typical range, not at the median —
because the gate's job is to catch catastrophic failures (pay.ai at $564),
not to second-guess every low comp set.

Refreshable from sales_history analysis as the market evolves.
"""

# (category, tld) → floor in dollars
# Premium single-word names on premium/emerging TLDs
CATEGORY_FLOORS = {
    # Premium TLDs: .com, .ai, .io
    ("generic", "com"): 5_000,      # Single dictionary word .com
    ("generic", "ai"): 15_000,      # Premium single-word .ai (post-boom floor)
    ("generic", "io"): 8_000,       # Premium single-word .io
    ("generic", "co"): 3_000,       # Single-word .co (discounted TLD)
    
    ("brandable", "com"): 3_000,
    ("brandable", "ai"): 8_000,
    ("brandable", "io"): 4_000,
    ("brandable", "co"): 2_000,
    
    ("keyword", "com"): 2_000,
    ("keyword", "ai"): 5_000,
    ("keyword", "io"): 3_000,
    ("keyword", "co"): 1_500,
    
    ("emd", "com"): 1_500,
    ("emd", "ai"): 3_000,
    ("emd", "io"): 2_000,
    ("emd", "co"): 1_000,
    
    # Emerging TLD category (startup-brandable floor)
    ("emerging_tld", "ai"): 10_000,
    ("emerging_tld", "io"): 5_000,
    ("emerging_tld", "co"): 2_500,
    
    # Acronyms (premium short names)
    ("acronym", "com"): 8_000,
    ("acronym", "ai"): 12_000,
    ("acronym", "io"): 6_000,
    ("acronym", "co"): 3_000,
    
    # Geo domains (local business floor)
    ("geo", "com"): 1_000,
    ("geo", "co"): 500,
    ("geo", "us"): 800,
    
    # Combination (brand + keyword)
    ("combination", "com"): 2_500,
    ("combination", "ai"): 6_000,
    ("combination", "io"): 3_500,
    ("combination", "co"): 1_800,
}

# Default floor for categories/TLDs not explicitly listed
DEFAULT_FLOOR = 500


def get_floor(category: str, tld: str) -> int:
    """Return the plausibility floor for this (category, tld) pair.
    
    Conservative floor: set below typical range to catch catastrophic failures
    without false-positives on legitimate low comps.
    """
    category = (category or "").lower().strip()
    tld = (tld or "").lower().strip()
    
    # Try exact match first
    key = (category, tld)
    if key in CATEGORY_FLOORS:
        return CATEGORY_FLOORS[key]
    
    # Fall back to category with wildcard TLD
    for cat, t in CATEGORY_FLOORS:
        if cat == category:
            return CATEGORY_FLOORS[(cat, t)]
    
    return DEFAULT_FLOOR


def get_ceiling(category: str, tld: str) -> int | None:
    """Return the plausibility ceiling for trophy inflation detection.
    
    Optional symmetric high check: if comps_end_user > ceiling × K, the comp
    set may be inflated by trophy sales. Returns None when no ceiling applies.
    
    Currently unused (F17.2 implements floor check only), but kept for future
    trophy guard expansion.
    """
    # Reserved for future trophy inflation guard
    # Example: ("generic", "ai") → 5_000_000 (above this, check for trophies)
    return None
