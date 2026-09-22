"""
Marks famous enough that any domain containing them is a legal problem.

**What this is FOR.** A valuation that prices nikeshoes.com at $50,000 and
says nothing about Nike is not merely incomplete, it is the kind of output
that gets someone sued. That class of conflict does not need a trademark
search to spot — the marks are household words, and their owners enforce
against domains as a matter of routine policy.

**What this is NOT.** It is not a trademark search and cannot become one. A
conflict is a legal determination turning on class, jurisdiction, use in
commerce and prior registrations, none of which a wordlist knows. A domain
absent from this file has NOT been cleared; it has only failed to trip the
obvious wire. Every consumer has to carry that distinction or the screen
becomes worse than no screen — a reader who sees "no conflict found" and
reads it as "safe" is worse off than one who sees "not screened".

**Why famous marks specifically.** Ordinary marks coexist: `delta.com` is
an airline and Delta Faucet sells taps, both legitimately, because classes
differ. Famous marks are the documented exception — they are protected
across classes precisely because their fame means any use trades on them.
So a wordlist can be right about these and could not be right about the
rest.

Curated, non-exhaustive, and dated like every other knowledge file here.
"""

from datetime import date

AS_OF = date(2026, 9, 1)

# Grouped only for readability. Every entry is a mark whose owner has a
# documented history of domain enforcement.
FAMOUS_MARKS: frozenset[str] = frozenset({
    # Technology
    "google", "youtube", "android", "chrome", "gmail", "apple", "iphone",
    "ipad", "macbook", "microsoft", "windows", "xbox", "azure", "amazon",
    "aws", "kindle", "meta", "facebook", "instagram", "whatsapp", "netflix",
    "spotify", "tiktok", "twitter", "linkedin", "snapchat", "pinterest",
    "reddit", "oracle", "salesforce", "adobe", "photoshop", "nvidia",
    "intel", "cisco", "ibm", "samsung", "huawei", "xiaomi", "sony",
    "playstation", "nintendo", "tesla", "spacex", "openai", "chatgpt",
    "anthropic", "uber", "lyft", "airbnb", "paypal", "stripe", "shopify",
    "dropbox", "slack", "zoom", "github", "gitlab", "atlassian", "jira",
    # Retail, food and drink
    "walmart", "costco", "target", "ikea", "nike", "adidas", "puma",
    "reebok", "gucci", "prada", "chanel", "hermes", "rolex", "cartier",
    "louisvuitton", "zara", "uniqlo", "levis", "cocacola", "pepsi",
    "sprite", "fanta", "nestle", "starbucks", "mcdonalds", "burgerking",
    "kfc", "subway", "dominos", "pizzahut", "heineken", "budweiser",
    "guinness", "redbull", "gatorade", "oreo", "kelloggs", "danone",
    # Finance, motoring, media, pharma
    "visa", "mastercard", "amex", "americanexpress", "goldmansachs",
    "morganstanley", "jpmorgan", "blackrock", "vanguard", "fidelity",
    "toyota", "honda", "nissan", "bmw", "mercedes", "audi", "volkswagen",
    "porsche", "ferrari", "lamborghini", "ford", "chevrolet", "hyundai",
    "disney", "pixar", "marvel", "hbo", "espn", "nbc", "cnn", "bbc",
    "nfl", "nba", "fifa", "olympics", "pfizer", "moderna", "novartis",
    "astrazeneca", "johnsonandjohnson", "bayer", "roche", "viagra",
    # Famous marks that are also everyday English. Listed here AND in
    # ALSO_COMMON_WORDS, which only ever narrows this set — an entry in one
    # and not the other would be a mark nothing screens for.
    "orange", "shell",
})

# Marks that are also ordinary English words. Present in the set above only
# where the famous sense dominates; listed here so a caller can reason about
# the difference rather than treating every hit alike.
ALSO_COMMON_WORDS: frozenset[str] = frozenset({
    "apple", "target", "amazon", "meta", "oracle", "chrome", "subway",
    "visa", "windows", "android", "sprite", "puma", "shell", "orange",
})
