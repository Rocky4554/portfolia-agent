"""
Industry classification for a domain, and the comparison rule that uses it.

**Why the tier engine needs this.** Comp retrieval matches on shape — same
length, same TLD, same structure — and shape is not a market. A fifteen-
letter two-word .com sells for wildly different money depending on whether
the words are "cloud storage" or "divorce lawyer", because the buyer, the
click economics and the lifetime value of a customer are different. Pricing
a law-firm domain from SaaS comps is not a small error; it is an answer
about the wrong industry.

**Curated, like every other wordlist here.** There is no free API that maps
a second-level domain to an industry, and inferring one from a live page
would only work for developed domains, which are the minority. So this is a
lexicon, with the same discipline as `lexicon.py` and `sentiment.py`: it
covers the verticals where domain economics actually diverge, and it returns
None rather than a guess everywhere else.

**The rule is deliberately asymmetric.** A penalty is applied only when both
sides are KNOWN and they differ. An unclassified comp is not evidence of a
mismatch — most domains are coined names with no industry at all — and
treating "we don't know" as "it's different" would quietly down-weight most
of a comp set on no evidence. That is the same mistake as scoring an
unreadable sentiment 70.
"""

from portfolio_agent.knowledge.lexicon import segment

# Verticals where the price of a name genuinely diverges, keyed to the terms
# that actually appear in domains. Kept to whole segmented words: substring
# matching would read "car" out of "carbon" and "law" out of "lawn".
INDUSTRIES: dict[str, frozenset[str]] = {
    "insurance": frozenset({
        "insurance", "insure", "insurer", "insured", "underwriting", "actuary",
        "coverage", "policy", "premium", "claim", "claims", "deductible",
    }),
    "finance": frozenset({
        "loan", "loans", "lending", "lender", "mortgage", "refinance", "credit",
        "bank", "banking", "capital", "invest", "investing", "investment",
        "fund", "funds", "equity", "wealth", "finance", "financial", "payday",
        "payment", "payments", "payroll", "accounting", "accountant", "tax",
        "taxes", "bookkeeping", "audit", "invoice", "billing",
    }),
    "legal": frozenset({
        "law", "lawyer", "lawyers", "attorney", "attorneys", "legal", "counsel",
        "litigation", "paralegal", "solicitor", "barrister", "divorce",
        "injury", "malpractice", "defense", "patent", "trademark", "notary",
    }),
    "health": frozenset({
        "health", "healthcare", "medical", "medicine", "doctor", "doctors",
        "clinic", "hospital", "dental", "dentist", "pharma", "pharmacy",
        "therapy", "therapist", "surgery", "surgeon", "patient", "care",
        "nursing", "wellness", "mental", "rehab", "diagnostic",
    }),
    "real_estate": frozenset({
        "realty", "realtor", "estate", "property", "properties", "homes",
        "housing", "apartment", "apartments", "rental", "rentals", "lease",
        "landlord", "tenant", "condo", "mortgagebroker", "appraisal",
    }),
    "travel": frozenset({
        "travel", "trip", "trips", "tour", "tours", "tourism", "hotel",
        "hotels", "flight", "flights", "airfare", "vacation", "holiday",
        "cruise", "resort", "booking", "hostel", "itinerary", "safari",
    }),
    "education": frozenset({
        "education", "school", "schools", "college", "university", "academy",
        "tutor", "tutoring", "course", "courses", "learn", "learning",
        "training", "degree", "student", "students", "campus", "exam",
    }),
    "software": frozenset({
        "software", "saas", "cloud", "api", "apis", "app", "apps", "platform",
        "devops", "server", "hosting", "database", "code", "coding", "dev",
        "developer", "stack", "deploy", "kubernetes", "container", "compute",
        "analytics", "dashboard", "integration", "automation", "ai", "ml",
        "data", "agent", "agents", "llm", "neural", "quantum",
    }),
    "crypto": frozenset({
        "crypto", "bitcoin", "ethereum", "blockchain", "token", "tokens",
        "wallet", "defi", "nft", "mining", "miner", "staking", "ledger",
        "coin", "coins", "web3", "dao", "exchange",
    }),
    "ecommerce": frozenset({
        "shop", "shopping", "store", "stores", "buy", "deal", "deals",
        "discount", "coupon", "coupons", "cart", "checkout", "marketplace",
        "retail", "wholesale", "dropship", "merch", "outlet",
    }),
    "marketing": frozenset({
        "marketing", "advertising", "ads", "adtech", "seo", "ppc", "brand",
        "branding", "agency", "campaign", "leads", "leadgen", "funnel",
        "affiliate", "influencer", "content", "copywriting",
    }),
    "gambling": frozenset({
        "casino", "poker", "bet", "bets", "betting", "bingo", "slots",
        "lottery", "wager", "odds", "sportsbook", "blackjack", "roulette",
    }),
    "automotive": frozenset({
        "auto", "car", "cars", "vehicle", "vehicles", "truck", "trucks",
        "motor", "motors", "dealership", "tires", "garage", "mechanic",
        "ev", "automotive",
    }),
    "food": frozenset({
        "food", "foods", "restaurant", "restaurants", "recipe", "recipes",
        "kitchen", "cafe", "coffee", "bakery", "catering", "menu", "dining",
        "grocery", "meal", "meals", "chef", "brewery", "wine",
    }),
    "fitness": frozenset({
        "fitness", "gym", "workout", "training", "yoga", "pilates", "nutrition",
        "diet", "supplement", "supplements", "protein", "bodybuilding",
    }),
    "home_services": frozenset({
        "plumbing", "plumber", "hvac", "roofing", "roofer", "electrician",
        "contractor", "remodeling", "landscaping", "cleaning", "pest",
        "handyman", "flooring", "painting", "locksmith", "moving",
    }),
    "recruiting": frozenset({
        "jobs", "job", "career", "careers", "hiring", "recruit", "recruiting",
        "recruiter", "staffing", "talent", "resume", "employment", "hr",
    }),
    "logistics": frozenset({
        "shipping", "freight", "logistics", "courier", "delivery", "cargo",
        "warehouse", "fulfillment", "supply", "trucking", "customs",
    }),
}

# Built once. A term appearing in two verticals is ambiguous and belongs to
# neither — "training" is fitness and education, "payment" is finance and
# ecommerce — so it is dropped rather than arbitrarily assigned.
_TERM_TO_INDUSTRY: dict[str, str] = {}
_AMBIGUOUS: set[str] = set()
for _industry, _terms in INDUSTRIES.items():
    for _term in _terms:
        if _term in _TERM_TO_INDUSTRY and _TERM_TO_INDUSTRY[_term] != _industry:
            _AMBIGUOUS.add(_term)
        _TERM_TO_INDUSTRY[_term] = _industry
for _term in _AMBIGUOUS:
    _TERM_TO_INDUSTRY.pop(_term, None)


def classify_industry(sld: str) -> str | None:
    """The vertical this name sells into, or None where it does not say.

    None is the common answer and the correct one: a coined brandable has no
    industry until someone builds a company on it. Callers must treat None
    as absence of evidence, never as a mismatch.
    """
    sld = (sld or "").strip().lower()
    if not sld:
        return None
    hits: dict[str, int] = {}
    last_seen: dict[str, int] = {}
    for position, token in enumerate(segment(sld)):
        industry = _TERM_TO_INDUSTRY.get(token)
        if industry:
            hits[industry] = hits.get(industry, 0) + 1
            last_seen[industry] = position
    if not hits:
        return None
    # A name spanning two verticals resolves to the one with more evidence.
    # On a tie the rightmost word wins, because English compounds are
    # head-final: "car insurance" is a kind of insurance, "car loans" a kind
    # of loan. Taking the leftmost would file both under automotive.
    ranked = sorted(hits.items(), key=lambda kv: (-kv[1], -last_seen[kv[0]]))
    return ranked[0][0]


def industry_relation(subject_sld: str, comp_sld: str) -> str:
    """`same`, `different`, or `unknown` — and `unknown` is never a penalty."""
    subject = classify_industry(subject_sld)
    if subject is None:
        return "unknown"
    comp = classify_industry(comp_sld)
    if comp is None:
        return "unknown"
    return "same" if comp == subject else "different"
