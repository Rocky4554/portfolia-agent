"""
Per-TLD value multipliers for the F20 TLD discount fix.

A great word on .co is still a .co. Brand score measures phonetic/structural
quality, which is TLD-agnostic, but end-user prices are not: .com commands a
premium, emerging TLDs trade at par with their own markets, and discounted
TLDs (.co, .biz, etc.) sell below both.

Applying the same brand premium regardless of TLD caused isotope.co to land
4.7x above its comp median — it's a high-brand word (pronounceable, clean)
on a discounted extension.

These multipliers apply to end_user and list_bin tiers, after all other
adjustments. They are multiplicative with the existing END_USER_TLD_PREMIUM
in config (which handles .com vs emerging vs other), adding finer-grained
per-TLD control.
"""

# TLD → multiplier on end_user/list_bin tiers
# 1.0 = par (no adjustment), < 1.0 = discount, > 1.0 = premium
TLD_TIER_MULTIPLIERS = {
    # Premium TLDs (end users seek these)
    "com": 1.00,  # Baseline; END_USER_TLD_PREMIUM already applies 1.15x
    "net": 0.85,
    "org": 0.80,
    
    # Emerging/startup TLDs (trade at par within their markets)
    "ai": 0.90,   # Post-boom pricing, premium within emerging class
    "io": 0.75,   # Tech/startup TLD, below .ai
    "app": 0.70,
    "dev": 0.70,
    
    # Discounted ccTLDs and gTLDs
    "co": 0.50,   # isotope.co should land near comp median, not 4.7x above
    "me": 0.45,
    "ly": 0.40,
    "cc": 0.35,
    "biz": 0.35,
    "info": 0.30,
    "us": 0.55,   # ccTLD with moderate US business use
    
    # Geographic ccTLDs (variable by market)
    "uk": 0.65,
    "de": 0.60,
    "fr": 0.55,
    "ca": 0.55,
    "au": 0.60,
    
    # New gTLDs (brand-specific or category)
    "xyz": 0.40,
    "online": 0.35,
    "site": 0.30,
    "tech": 0.50,
    "store": 0.45,
    "shop": 0.45,
    
    # Add more as needed
}

# Default multiplier for unlisted TLDs
DEFAULT_TLD_MULTIPLIER = 0.40  # Conservative floor


def get_tld_multiplier(tld: str) -> float:
    """Return the value multiplier for this TLD.
    
    Multiplies end_user and list_bin tiers to account for TLD-specific pricing.
    A high-brand word on a discounted TLD should not inflate above its cohort.
    """
    tld = (tld or "").lower().strip()
    return TLD_TIER_MULTIPLIERS.get(tld, DEFAULT_TLD_MULTIPLIER)
