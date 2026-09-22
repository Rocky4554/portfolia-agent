"""
How the tiers relate to each other, computed rather than described.

**What was missing.** Each tier section states its own figure and its own
adjustments, and the scope rules forbid it from naming any other tier — for
good reason, since a tier paragraph quoting another tier's price is the
cross-tier invention the section slicing exists to prevent. The effect was
a report of facts sitting side by side with nothing joining them: six
prices, no account of why the end-user figure is four times the auction
figure, which is the question a reader actually has.

**So the relationships are arithmetic, not narration.** The ratios and their
magnitude words are computed here, and the narrative is handed the finished
comparison. A model asked to compare two numbers will sometimes call 1.08x
"far above"; a model handed "marginally above (1.08x)" cannot.

**The direction is the part worth linting.** Magnitude is a judgement with a
defensible range, but direction is a fact: a sentence saying the floor
exceeds the auction price when it does not is simply wrong, and that is
checkable without guessing what the model meant.
"""

from dataclasses import dataclass

from portfolio_agent.schema.valuation_output import TIER_ORDER, TierSet

# Ratio thresholds and the word each one earns. Fixed bands rather than a
# model's adjective, so the same gap is described the same way every run.
_MAGNITUDES = (
    (1.15, "marginally"),
    (1.60, "moderately"),
    (3.00, "substantially"),
    (float("inf"), "dramatically"),
)

# Why each step up the ladder exists. The tier engine knows this; the
# rationale had no way to say it.
_REASONS = {
    ("floor", "auction"): "an auction adds competition that a wholesale exit does not have",
    ("auction", "end_user"): (
        "an end user buys the name for its own sake rather than to resell it, and pays for "
        "the credibility rather than the inventory"
    ),
    ("end_user", "list_bin"): (
        "a buy-it-now price is an ask carrying a premium for immediacy, and is expected to sit "
        "above what a negotiated end-user sale clears at"
    ),
    ("list_bin", "targeted_lead"): (
        "a named, funded buyer with a credibility gap to close is worth more than an anonymous "
        "listing, because the domain solves a specific problem for them"
    ),
    ("targeted_lead", "strategic_ma"): (
        "a category-defining name bought to become a company's identity is priced against "
        "acquisition comparables, not against domain sales"
    ),
}


@dataclass(frozen=True)
class Relation:
    lower: str
    upper: str
    ratio: float
    magnitude: str
    reason: str

    def sentence(self) -> str:
        return (f"the {self.upper} price is {self.magnitude} above the {self.lower} price "
                f"({self.ratio:.2f}x) because {self.reason}")


def magnitude_for(ratio: float) -> str:
    """The word this ratio earns, and only this word."""
    for threshold, word in _MAGNITUDES:
        if ratio < threshold:
            return word
    return _MAGNITUDES[-1][1]


def tier_relations(tiers: TierSet) -> list[Relation]:
    """Each adjacent pair of populated tiers, in ladder order.

    Adjacent only. Comparing every tier with every other would produce
    fifteen relationships for six tiers, most of them restating the others,
    and a reader does not need the BIN price expressed as a multiple of the
    floor to understand the ladder.
    """
    populated = [(name, getattr(tiers, name)) for name in TIER_ORDER if getattr(tiers, name)]
    relations: list[Relation] = []
    for (lower_name, lower), (upper_name, upper) in zip(populated, populated[1:]):
        if lower.point <= 0:
            continue
        ratio = upper.point / lower.point
        relations.append(Relation(
            lower=lower_name,
            upper=upper_name,
            ratio=round(ratio, 2),
            magnitude=magnitude_for(ratio),
            # Adjacent in the ladder but not adjacent in the table when a
            # tier between them went unpriced, so the stock reason may not
            # apply and is omitted rather than misapplied.
            reason=_REASONS.get((lower_name, upper_name),
                                "the tiers between these two were not priced, so this step spans "
                                "more than one change in buyer"),
        ))
    return relations
