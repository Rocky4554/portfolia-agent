"""
Category → evidence plan. APVA Table 3.2.1 as executable routing.

This is the table that makes the agent's analytical path depend on what the
domain *is*. A brandable gets a brand score and semantic comps because it has
no keyword to search on; an EMD gets keyword sales and SEO metrics because
its value is measurable search ROI. Running one generic plan for everything
is the first-generation behaviour APVA exists to replace.

Kept as data, separately from the code that executes it (nodes/
gather_evidence.py), so adding a tool to a category is a one-line edit here
and never a change to graph wiring.
"""

from dataclasses import dataclass, field

# Stable tool ids. gather_evidence.py maps each to a callable; a plan naming
# an id with no callable fails the playbook's own self-check at import, not
# silently mid-run.
TOOL_DOMAIN_PROFILE = "get_domain_profile"
TOOL_BASE_APPRAISAL = "get_base_appraisal"
TOOL_KEYWORD_DEMAND = "get_keyword_demand"
TOOL_KEYWORD_SALES = "get_keyword_sales"
TOOL_SIMILAR_SALES = "fetch_similar_sales"
TOOL_BRAND_SCORE = "calculate_brand_score"
TOOL_MARKET_TRENDS = "scan_market_trends"
TOOL_SEO_METRICS = "get_seo_and_traffic_metrics"
TOOL_END_USERS = "find_potential_end_users"
TOOL_MARKETPLACE = "get_marketplace_listings"
TOOL_INTERNAL_COMPS = "fetch_internal_comps"
TOOL_DOMAIN_STATUS = "check_domain_status"
TOOL_REGISTRATION = "get_registration"
TOOL_INTENT = "resolve_intent"
TOOL_REGISTRY_CHECK = "check_availability"

ALL_TOOL_IDS = {
    TOOL_DOMAIN_PROFILE,
    TOOL_BASE_APPRAISAL,
    TOOL_KEYWORD_DEMAND,
    TOOL_KEYWORD_SALES,
    TOOL_SIMILAR_SALES,
    TOOL_BRAND_SCORE,
    TOOL_MARKET_TRENDS,
    TOOL_SEO_METRICS,
    TOOL_END_USERS,
    TOOL_MARKETPLACE,
    TOOL_INTERNAL_COMPS,
    TOOL_DOMAIN_STATUS,
    TOOL_REGISTRATION,
    TOOL_INTENT,
    TOOL_REGISTRY_CHECK,
}


@dataclass(frozen=True)
class EvidenceStep:
    tool: str
    reason: str  # why THIS category needs it — ends up in the rationale
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvidencePlan:
    category: str
    steps: tuple[EvidenceStep, ...]
    buyer_persona: str  # Table 3.2.1, "Typical Buyer Persona"
    holding_strategy: str  # Table 3.2.1, "Ideal Holding Strategy"
    red_flags: tuple[str, ...]  # Table 3.2.1, "Red Flags"

    def tool_ids(self) -> list[str]:
        return [step.tool for step in self.steps]


# Run for every category. The baseline appraisal and the domain profile feed
# tiers that every domain has (the floor's sale probability comes off
# GoValue's curve), so routing them per-category would only mean writing them
# into all eight plans.
_COMMON_PRELUDE = (
    EvidenceStep(TOOL_DOMAIN_PROFILE, "structure, keyword economics and trademark risk for any domain"),
    EvidenceStep(TOOL_BASE_APPRAISAL, "baseline ML appraisal and the price→sale-probability curve"),
    # Table 4.1.1 names marketplace listings as the List/BIN tier's primary
    # adjustment source, and it applies to every category — any domain can
    # be listed for sale.
    EvidenceStep(TOOL_MARKETPLACE, "live asking prices — what the market is being offered today"),
    # Own-book sales are the most trustworthy comps available and the only
    # ones nobody else has, so every category gets them.
    EvidenceStep(TOOL_INTERNAL_COMPS, "verified sales from our own brokerage book"),
    # Whether the domain can be bought at all. Universal for the same reason
    # as listings, and needed even when every other signal is strong: a
    # confident price on an unavailable domain is the most embarrassing
    # output this engine can produce.
    EvidenceStep(TOOL_DOMAIN_STATUS, "what is served at the domain today, and whether it is for sale"),
    # The registration behind the page. Free, keyless, and the only thing
    # that can tell a domain nobody has ever registered from one held since
    # 1997 that simply publishes no DNS.
    EvidenceStep(TOOL_REGISTRATION, "who holds the registration, since when, and until when"),
    # Whether the keyword's demand is people about to spend money. Costs
    # nothing for the names the lexicon knows, which is most of them.
    EvidenceStep(TOOL_INTENT, "whether this term's search demand is commercial or merely curious"),
    # A registry answer on availability, and the registration price when the
    # name is free — the one number the acquirability warning could not put
    # on "can be had at registration cost".
    EvidenceStep(TOOL_REGISTRY_CHECK, "registry availability, and what it costs to register if free"),
)


PLAYBOOK: dict[str, EvidencePlan] = {
    # Scarcity, trust, type-in traffic. Value is anchored to comps for the
    # word itself, so demand data matters more than brand analysis.
    "generic": EvidencePlan(
        category="generic",
        steps=_COMMON_PRELUDE
        + (
            EvidenceStep(TOOL_KEYWORD_SALES, "comps for the word itself set the anchor", {"same_tld": True}),
            EvidenceStep(TOOL_KEYWORD_DEMAND, "how widely the term is registered — the scarcity signal"),
            EvidenceStep(TOOL_SIMILAR_SALES, "semantic comps as a cross-check on the keyword comps"),
            EvidenceStep(TOOL_END_USERS, "a category term is the class of domain a corporate buyer acquires outright"),
        ),
        buyer_persona="Corporates, lead-gen companies",
        holding_strategy="Long-Term Hold",
        red_flags=("unrealistic pricing", "weak industry"),
    ),
    # No keyword to search on, so comps must be semantic and the score has to
    # carry the valuation (APVA §4.3).
    "brandable": EvidencePlan(
        category="brandable",
        steps=_COMMON_PRELUDE
        + (
            EvidenceStep(TOOL_BRAND_SCORE, "a coined name has no keyword value — brand potential IS the value"),
            EvidenceStep(
                TOOL_SIMILAR_SALES,
                "semantic comps: short pronounceable names in the same TLD",
                {"same_tld": True, "short_pronounceable": True},
            ),
            EvidenceStep(TOOL_MARKET_TRENDS, "VC-backed startup naming trends drive this category"),
            EvidenceStep(TOOL_END_USERS, "the buyer is a funded startup that needs a name, not an investor"),
        ),
        buyer_persona="VC-backed startups",
        holding_strategy="Quick Flip / Medium Hold",
        red_flags=("hard to spell", "low recall", "weak brand pull"),
    ),
    # Measurable SEO ROI. Search volume and keyword sales, not brand feel.
    "keyword": EvidencePlan(
        category="keyword",
        steps=_COMMON_PRELUDE
        + (
            EvidenceStep(TOOL_KEYWORD_SALES, "aggregate sales for the keyword set the range", {"same_tld": True}),
            EvidenceStep(TOOL_KEYWORD_DEMAND, "registration depth across TLDs shows demand for the term"),
            EvidenceStep(TOOL_SEO_METRICS, "buyers price this category on measurable search ROI"),
        ),
        buyer_persona="SEO marketers, niche players",
        holding_strategy="Price for SEO ROI",
        red_flags=("low search volume",),
    ),
    # Same drivers as keyword, heavier on SEO: an EMD is bought for the
    # traffic the phrase already carries.
    "emd": EvidencePlan(
        category="emd",
        steps=_COMMON_PRELUDE
        + (
            EvidenceStep(TOOL_KEYWORD_SALES, "sales of the exact phrase are the closest possible comp", {"same_tld": True}),
            EvidenceStep(TOOL_KEYWORD_DEMAND, "how contested the exact phrase is"),
            EvidenceStep(TOOL_SEO_METRICS, "direct traffic and user intent are the whole thesis"),
            EvidenceStep(TOOL_END_USERS, "lead-gen and marketplace buyers pay above investor prices"),
        ),
        buyer_persona="Lead-gen, marketplaces",
        holding_strategy="Bundle with SEO profile",
        red_flags=("spam history", "low search volume"),
    ),
    # Rarity and liquidity. Comps must be length-matched — a 3-letter .com
    # and a 4-letter .com are different markets, not neighbours.
    "acronym": EvidencePlan(
        category="acronym",
        steps=_COMMON_PRELUDE
        + (
            EvidenceStep(
                TOOL_SIMILAR_SALES,
                "comps restricted to same-length acronyms — length IS the market",
                {"same_length": True, "same_tld": True},
            ),
            EvidenceStep(TOOL_KEYWORD_DEMAND, "how many TLDs the string is already taken in — rarity"),
        ),
        buyer_persona="Corporates, Chinese buyers",
        holding_strategy="Long-Term Hold",
        red_flags=("meaningless string", "low liquidity"),
    ),
    # Local lead-gen. Keyword stats have to be scoped to the location or the
    # national keyword's volume floods the estimate.
    "geo": EvidencePlan(
        category="geo",
        steps=_COMMON_PRELUDE
        + (
            EvidenceStep(
                TOOL_KEYWORD_SALES,
                "keyword stats scoped to the location term, not the national keyword",
                {"scope_to_geo": True, "same_tld": True},
            ),
            EvidenceStep(TOOL_SIMILAR_SALES, "comps from the same local-services market"),
            EvidenceStep(TOOL_END_USERS, "the buyer is a local business or franchise, not an investor"),
        ),
        buyer_persona="Local businesses, franchises",
        holding_strategy="Medium Hold / Sell Locally",
        red_flags=("low-demand market", "too long-tail"),
    ),
    # Trend alignment and scarcity in a hot niche — and exposure to the hype
    # cycle deflating, which is why trends is not optional here.
    "emerging_tld": EvidencePlan(
        category="emerging_tld",
        steps=_COMMON_PRELUDE
        + (
            EvidenceStep(TOOL_BRAND_SCORE, "these sell to startups on brand feel, not search volume"),
            # Sales of this exact term on this extension. Absent until
            # forge.ai showed what that costs: with no keyword search, the
            # only comps available were semantic neighbours, and the ladder
            # then read those as exact matches.
            EvidenceStep(TOOL_KEYWORD_SALES, "sales of this exact term on this extension", {"same_tld": True}),
            EvidenceStep(
                TOOL_SIMILAR_SALES,
                "comps scoped to the same TLD — .ai prices do not transfer from .com",
                {"same_tld": True},
            ),
            EvidenceStep(TOOL_MARKET_TRENDS, "the whole category is a trend position; velocity is the risk"),
            EvidenceStep(TOOL_END_USERS, "AI/tech startups are the buyer; funding rounds drive the bid"),
        ),
        buyer_persona="AI/tech startups",
        holding_strategy="Quick Flip",
        red_flags=("hype cycle exposure", "high renewal costs"),
    ),
    # Brand fused to a keyword (ZenBusiness.com). Needs both halves of the
    # analysis — the brand score AND the keyword comps.
    "combination": EvidencePlan(
        category="combination",
        steps=_COMMON_PRELUDE
        + (
            EvidenceStep(TOOL_BRAND_SCORE, "the brand half of the name carries most of the premium"),
            EvidenceStep(TOOL_KEYWORD_SALES, "the keyword half anchors it to real sales", {"same_tld": True}),
            EvidenceStep(TOOL_SIMILAR_SALES, "two-word .com brands are their own comp set"),
            EvidenceStep(TOOL_MARKET_TRENDS, "funded-startup naming trends move this category"),
            EvidenceStep(TOOL_END_USERS, "a brand+keyword name sells to the company it describes"),
        ),
        buyer_persona="VC-backed startups, growth companies",
        holding_strategy="Medium Hold",
        red_flags=("weak brand pull", "generic keyword half"),
    ),
}


def plan_for(category: str) -> EvidencePlan:
    """The evidence plan for a category, falling back to generic's.

    Falls back rather than raising: an unrecognized category should still
    produce a valuation with a warning, not a crashed graph.
    """
    return PLAYBOOK.get(category, PLAYBOOK["generic"])


def plan_for_all(categories: list[str]) -> EvidencePlan:
    """One plan covering every label the domain carries.

    The category selects the evidence, so a single label meant gathering one
    market's evidence and not the other's: accounting.ai got keyword
    economics and no TLD trend reading, while forge.ai got the trend reading
    and no keyword work. Neither was wrong about what it did gather; both
    were missing half the picture.

    The primary label's plan leads, so its ordering and its buyer persona
    still shape the report, and each additional label contributes only the
    steps the primary did not already run. A step's `reason` is the primary's
    where they overlap, because that is the framing the rationale will use.
    """
    if not categories:
        return PLAYBOOK["generic"]
    primary = plan_for(categories[0])
    steps = list(primary.steps)
    seen = {step.tool for step in steps}
    extra_flags: list[str] = []
    for name in categories[1:]:
        other = plan_for(name)
        for step in other.steps:
            if step.tool not in seen:
                seen.add(step.tool)
                steps.append(EvidenceStep(
                    step.tool, f"{step.reason} (because it also reads as {name})", step.params))
        extra_flags.extend(f for f in other.red_flags if f not in primary.red_flags)
    return EvidencePlan(
        category=primary.category,
        steps=tuple(steps),
        buyer_persona=primary.buyer_persona,
        holding_strategy=primary.holding_strategy,
        red_flags=primary.red_flags + tuple(extra_flags),
    )


def categories_using(tool_id: str) -> list[str]:
    """Which categories invoke a tool — used by the evals to assert routing."""
    return sorted(name for name, plan in PLAYBOOK.items() if tool_id in plan.tool_ids())


# Self-check at import: a typo in a tool id above would otherwise surface as
# a silently skipped step in the middle of a live valuation.
for _name, _plan in PLAYBOOK.items():
    _unknown = set(_plan.tool_ids()) - ALL_TOOL_IDS
    if _unknown:
        raise ValueError(f"category_playbook['{_name}'] names unknown tool ids: {sorted(_unknown)}")
