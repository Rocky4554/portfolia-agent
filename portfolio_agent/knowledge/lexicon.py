"""
The word lists classification runs on, kept as data so they can be extended
without touching the rules that read them.

Deliberately curated rather than a full English dictionary. Segmentation
needs a list where every entry is a word someone would plausibly build a
commercial domain out of: a general dictionary turns "zylantix" into
"zy|lan|tix" and every coined brandable reads as a keyword domain. The full
dictionary is still used, but only for spell-checking (scoring/spelling.py),
where its breadth is an asset instead of a liability.
"""

# Head nouns — the thing being sold. A domain whose last token is one of
# these is describing a product or service, which is what separates a
# keyword domain from a coined one.
COMMERCIAL_HEADS = {
    "insurance", "management", "marketing", "software", "solutions", "services",
    "systems", "consulting", "logistics", "analytics", "security", "hosting",
    "payments", "banking", "lending", "loans", "mortgage", "finance", "capital",
    "investments", "trading", "crypto", "wallet", "exchange", "ventures",
    "partners", "advisors", "agency", "studio", "labs", "works", "group",
    "network", "media", "news", "reviews", "quotes", "deals", "coupons",
    "rentals", "realty", "estate", "property", "properties", "homes", "houses",
    "apartments", "hotels", "flights", "travel", "tours", "cruise", "cruises",
    "tickets", "events", "weddings", "photography", "design", "graphics",
    "printing", "furniture", "mattress", "clothing", "shoes", "jewelry",
    "watches", "fitness", "gym", "yoga", "wellness", "health", "medical",
    "dental", "dentist", "clinic", "pharmacy", "therapy", "care", "nursing",
    "legal", "lawyer", "lawyers", "attorney", "attorneys", "law", "tax",
    "accounting", "payroll", "recruiting", "jobs", "careers", "staffing",
    "education", "learning", "training", "courses", "tutoring", "school",
    "academy", "university", "energy", "solar", "electric", "plumbing", "hvac",
    "roofing", "construction", "contractors", "landscaping", "cleaning",
    "moving", "storage", "shipping", "delivery", "freight", "auto", "autos",
    "cars", "trucks", "motors", "parts", "repair", "towing", "pets", "vet",
    "veterinary", "grooming", "food", "recipes", "catering", "restaurant",
    "coffee", "wine", "beer", "brewing", "farm", "garden", "tools", "supply",
    "supplies", "equipment", "machinery", "manufacturing", "wholesale",
    "market", "marketplace", "shop", "store", "outlet", "cloud", "data",
    "hub", "platform", "app", "apps", "tech", "digital", "web", "site",
    "hosting", "domains", "hosting", "servers", "robotics", "drones",
    "biotech", "genomics", "pharma", "chemicals", "mining", "metals",
    # Singular forms of the plurals above. Segmentation is exact-match, so a
    # missing singular silently turns "carinsurance" into an unknown token
    # plus a keyword, i.e. a Combination instead of an EMD.
    "car", "truck", "motor", "part", "job", "career", "home", "house",
    "apartment", "hotel", "flight", "tour", "ticket", "event", "wedding",
    "loan", "deal", "coupon", "review", "quote", "rental", "course", "pet",
    "tool", "supply", "watch", "shoe", "doctor", "nurse", "bank", "payment",
}

# Modifiers — words that qualify a head noun. On their own they're not a
# product, which is why they never trigger the generic (single-word) rule.
COMMERCIAL_MODIFIERS = {
    "best", "top", "cheap", "cheapest", "affordable", "budget", "discount",
    "low", "lowest", "free", "instant", "fast", "quick", "easy", "simple",
    "smart", "pro", "premium", "elite", "expert", "trusted", "reliable",
    "online", "local", "national", "global", "international", "worldwide",
    "my", "your", "our", "the", "get", "buy", "find", "compare", "shop",
    "hire", "rent", "sell", "new", "modern", "advanced", "next", "first",
    "prime", "direct", "express", "rapid", "super", "ultra", "mega", "all",
    "open", "all",
}

# Evocative rather than commercial-intent. "zen" doesn't describe a product,
# so zenbusiness.com is a brand fused to a keyword (Combination, the case
# APVA §3.2 works through) rather than a search phrase (EMD).
BRAND_MODIFIERS = {
    "true", "real", "pure", "bright", "clear", "bold", "blue", "red", "green",
    "gold", "silver", "iron", "stone", "river", "summit", "peak", "north",
    "south", "east", "west", "zen", "one", "go", "up", "now", "nova", "lumen",
    "apex", "vertex", "orbit", "atlas", "echo", "ember", "onyx", "quill",
    "sage", "vista", "haven", "forge", "arc", "aura", "halo", "nimbus",
}

# Single common nouns that stand alone as a category — the "Generic" row of
# APVA Table 3.2.1 (scarcity, trust, type-in traffic).
GENERIC_SINGLE_WORDS = {
    "insurance", "loans", "mortgage", "hotels", "flights", "cars", "jobs",
    "shoes", "furniture", "coffee", "wine", "beer", "pets", "law", "tax",
    "health", "fitness", "travel", "cloud", "data", "energy", "solar",
    "crypto", "wallet", "bank", "credit", "money", "invest", "trading",
    "shopping", "deals", "news", "weather", "sports", "games", "music",
    "movies", "books", "art", "photos", "video", "chat", "mail", "search",
    "business", "market", "store", "shop", "home", "auto", "food", "medicine",
    # Demonstrated categories that sit below the frequency floor — slots.com
    # sold for $5,500,000. Curated entries pass on sight, which is what the
    # curated list is for.
    "slots", "casino", "poker", "bingo",
    "education", "science", "space", "water", "air", "fire", "earth",
}

# Geo tokens — a location component moves a domain into the Geo row of
# Table 3.2.1 regardless of what it's paired with.
GEO_TOKENS = {
    "newyork", "nyc", "losangeles", "chicago", "houston", "phoenix",
    "philadelphia", "sanantonio", "sandiego", "dallas", "austin", "boston",
    "seattle", "denver", "miami", "atlanta", "vegas", "lasvegas", "portland",
    "nashville", "detroit", "orlando", "tampa", "charlotte", "sacramento",
    "london", "manchester", "birmingham", "paris", "berlin", "munich",
    "madrid", "barcelona", "rome", "milan", "amsterdam", "dublin", "lisbon",
    "zurich", "geneva", "vienna", "prague", "warsaw", "stockholm", "oslo",
    "toronto", "vancouver", "montreal", "calgary", "sydney", "melbourne",
    "brisbane", "auckland", "tokyo", "osaka", "seoul", "beijing", "shanghai",
    "singapore", "hongkong", "mumbai", "delhi", "bangalore", "chennai",
    "dubai", "abudhabi", "doha", "riyadh", "cairo", "lagos", "nairobi",
    "johannesburg", "capetown", "saopaulo", "riodejaneiro", "buenosaires",
    "mexicocity", "texas", "california", "florida", "arizona", "nevada",
    "colorado", "georgia", "virginia", "carolina", "ohio", "michigan",
    "oregon", "utah", "idaho", "montana", "alaska", "hawaii", "canada",
    "mexico", "brazil", "france", "germany", "spain", "italy", "japan",
    "china", "india", "australia", "africa", "europe", "asia", "america",
}

# TLDs the spec calls trend-sensitive (Table 3.2.1, "Emerging TLDs").
EMERGING_TLDS = {"ai", "io", "xyz", "app", "dev", "co", "gg", "sh", "tech", "cloud"}

# .com is the premium the End-User tier pays for (Table 4.1.1).
PREMIUM_TLDS = {"com"}

# Everything segmentation is allowed to split on.
SEGMENTABLE = COMMERCIAL_HEADS | COMMERCIAL_MODIFIERS | BRAND_MODIFIERS | GENERIC_SINGLE_WORDS | GEO_TOKENS

_MIN_TOKEN_LEN = 2


def segment(sld: str) -> list[str]:
    """Split an SLD into known words, longest-match-first, left to right.

    Unrecognized runs come back as single tokens, so the caller can tell
    "two known words" (keyword domain) from "a known word plus something
    invented" (combination) from "nothing recognized" (brandable).

    A whole SLD that is itself a real English word is never split. Greedy
    matching against a small commercial list happily mangles real words —
    "isotope" becomes iso|top|e because "top" is in the list, and "apple"
    becomes app|le — which then reads as a multi-word domain built from
    commercial terms. The dictionary check is what stops a premium
    single-word domain being classified as a compound.

    >>> segment("cheapinsurancequotes")
    ['cheap', 'insurance', 'quotes']
    >>> segment("isotope")
    ['isotope']
    >>> segment("zylantix")
    ['zylantix']
    """
    from portfolio_agent.knowledge.dictionary import is_dictionary_word

    sld = "".join(ch for ch in sld.lower() if ch.isalnum())
    if is_dictionary_word(sld):
        return [sld]
    tokens: list[str] = []
    unknown = ""
    i = 0
    while i < len(sld):
        match = _longest_word_at(sld, i)
        if match:
            if unknown:
                tokens.append(unknown)
                unknown = ""
            tokens.append(match)
            i += len(match)
        else:
            unknown += sld[i]
            i += 1
    if unknown:
        tokens.append(unknown)
    return tokens


def _longest_word_at(sld: str, start: int) -> str | None:
    # Longest first so "insurance" wins over a shorter prefix that would
    # strand "urance" as an unknown token.
    for end in range(len(sld), start + _MIN_TOKEN_LEN - 1, -1):
        candidate = sld[start:end]
        if candidate in SEGMENTABLE:
            return candidate
    return None


def is_known_word(token: str) -> bool:
    return token in SEGMENTABLE


def normalize_tld(tld: str | None) -> str | None:
    if not tld:
        return None
    tld = tld.strip().lower().lstrip(".")
    return tld or None
