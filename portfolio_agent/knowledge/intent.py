"""
Whether a keyword's searches are people about to spend money.

**The flaw this addresses.** The end-user tier reads monthly search volume
and pays for it on a straight curve, so 250,000 searches for "weather" earns
the same premium as 250,000 for "car insurance". Those are not the same
asset. One audience is checking whether to take an umbrella; the other is
about to buy a policy, which is why insurance keywords carry some of the
highest advertising costs in existence and weather keywords do not. Volume
without intent measures attention, not demand.

**Why this is a lexicon and not a search call.** The obvious implementation
queries a search API and reads the result mix — product pages and ads mean
commercial, encyclopaedia entries mean informational. That works, costs a
credit per valuation, cannot be tested offline, and fails closed when the
provider rate-limits. The signal it recovers is mostly already present in
the words themselves: nobody searches "cheap car insurance quotes" to read
about actuarial science. So the lexicon carries the common cases
deterministically, and a search-backed reading can refine it later without
changing anything downstream.

**The rule is asymmetric, as in `industries.py`.** A keyword with no intent
marker either way is `unclassified`, and unclassified never costs a domain
anything. Most SLDs are a bare noun with no verb of purchase attached, and
treating "we cannot tell" as "nobody is buying" would deflate most of the
book on no evidence.
"""

from portfolio_agent.knowledge.industries import classify_industry
from portfolio_agent.knowledge.lexicon import segment

# Words that appear when somebody is preparing to transact. Two families,
# because they behave differently: a purchase verb is decisive on its own,
# while a comparison term only signals intent next to something to buy.
_TRANSACTIONAL = frozenset({
    "buy", "purchase", "order", "shop", "price", "prices", "pricing", "cost",
    "costs", "quote", "quotes", "deal", "deals", "discount", "coupon",
    "coupons", "sale", "cheap", "cheapest", "affordable", "budget", "rates",
    "rate", "hire", "rent", "rental", "lease", "book", "booking", "subscribe",
    "trial", "demo", "free", "estimate", "estimates", "insurance", "loan",
    "loans", "mortgage", "refinance",
})

# Commercial only in company: "best" alone is nothing, "best insurance" is a
# buyer three clicks from a form.
_COMPARISON = frozenset({
    "best", "top", "review", "reviews", "compare", "comparison", "vs",
    "alternative", "alternatives", "rated", "recommended", "cheapest",
})

# Someone who wants to buy a service names the service provider.
_SERVICE = frozenset({
    "service", "services", "company", "companies", "agency", "agencies",
    "provider", "providers", "firm", "consultant", "consulting", "expert",
    "experts", "specialist", "contractor", "software", "platform", "tool",
    "tools", "app", "solution", "solutions", "near", "local",
})

# Words that mark a reader rather than a buyer. Strong signals: a domain
# built on these sells advertising, not a product, and the end-user tier is
# the wrong frame for it entirely.
_INFORMATIONAL = frozenset({
    "what", "why", "how", "when", "who", "which", "meaning", "definition",
    "define", "history", "wiki", "wikipedia", "facts", "fact", "examples",
    "example", "guide", "tutorial", "explained", "explain", "about",
    "encyclopedia", "dictionary", "quotations", "trivia",
    # NB "quotes" is deliberately NOT here: "insurance quotes" is one of the
    # most commercial searches there is, and the literary sense loses.
    "news", "blog", "forum", "wikihow", "symptoms", "causes", "statistics",
    "chart", "charts", "map", "maps", "weather", "calendar", "time",
    "lyrics", "recipe", "recipes", "biography", "timeline", "origin",
})

INTENTS = ("commercial", "informational", "unclassified")

# Multipliers on the SEARCH VOLUME credit, not on the price. An
# informational keyword's searches are real; they are just worth less per
# search to somebody selling something, so the volume is discounted before
# it earns a premium rather than the tier being deflated afterwards.
VOLUME_CREDIT = {
    "commercial": 1.0,
    "informational": 0.25,
    "unclassified": 1.0,
}


def classify_intent(sld: str) -> tuple[str, list[str]]:
    """`(intent, reasons)` for a second-level domain.

    Reasons are returned so the tier can say why it discounted, rather than
    applying an adjustment the report cannot account for.
    """
    tokens = segment((sld or "").strip().lower())
    if not tokens:
        return "unclassified", []

    transactional = [t for t in tokens if t in _TRANSACTIONAL]
    comparison = [t for t in tokens if t in _COMPARISON]
    service = [t for t in tokens if t in _SERVICE]
    informational = [t for t in tokens if t in _INFORMATIONAL]

    # Informational markers are checked first and win outright. "best weather
    # app" is a real product search, but "weather" sitting alone is the
    # commoner case by far, and over-pricing an ad-supported domain as though
    # it had end users is the expensive direction.
    if informational and not transactional:
        return "informational", [f"'{t}' marks a reader rather than a buyer" for t in informational]

    if transactional:
        return "commercial", [f"'{t}' is a purchase term" for t in transactional]

    # A comparison or service word needs something to be about. Paired with a
    # recognised commercial vertical it is a buyer; on its own it is noise.
    industry = classify_industry(sld)
    if (comparison or service) and industry:
        marker = (comparison + service)[0]
        return "commercial", [f"'{marker}' alongside a {industry} term is a buying signal"]

    if industry:
        return "commercial", [f"names a {industry} market, where searches carry commercial intent"]

    return "unclassified", []


def volume_credit(intent: str) -> float:
    return VOLUME_CREDIT.get(intent, 1.0)


async def resolve_intent(sld: str) -> dict:
    """The lexicon, and a live SERP only where the lexicon is silent.

    The lexicon answers most names for nothing, so search runs only for the
    ones it returns `unclassified` for — which is the difference between
    spending a credit per valuation and spending one occasionally.

    What a SERP measures is the right thing to measure. "Is this word's
    demand commercial" is answered by what actually ranks for it — product
    and comparison pages, or encyclopaedia entries — and not by a trend
    index, which counts attention without saying what kind. A failed or
    throttled lookup stays `unclassified`, which costs the domain nothing,
    so a rate limit can never read as "nobody is buying".
    """
    intent, reasons = classify_intent(sld)
    if intent != "unclassified":
        return {"intent": intent, "reasons": reasons, "source": "lexicon"}

    from portfolio_agent.tools.search import serp_composition

    result = await serp_composition(sld)
    if not result["ok"]:
        return {"intent": "unclassified", "reasons": [],
                "source": f"lexicon silent; SERP unavailable ({result.get('error')})"}
    data = result["data"] or {}
    serp_intent = data.get("intent") or "unclassified"
    if serp_intent == "unclassified":
        return {"intent": "unclassified", "reasons": [], "source": "lexicon silent; SERP mixed"}
    share = data.get("commercial_share")
    return {
        "intent": serp_intent,
        "reasons": [f"{share:.0%} of the results for this term are commercial pages rather than "
                    f"reference or editorial ones" if isinstance(share, float) else "SERP composition"],
        "source": "serp" + (" (cached)" if data.get("cached") else ""),
    }
