"""
Keyword sentiment — APVA §4.3's fourth brand-score component.

The spec routes sentiment through `scan_market_trends`, which is an LLM- and
search-backed tool. That is the right long-term source and it is wired in
Phase 3. It is the wrong source for a *score*: the brand score has to be
byte-reproducible so it can be regression-tested and so two runs of the same
valuation never disagree, and a search-backed tool is neither deterministic
nor offline.

So sentiment here is lexicon-based (VADER), which is a fixed table lookup:
same words in, same number out, forever. The trends tool's sentiment reading
is carried as separate evidence and used by the tier engine as an
*adjustment*, where non-determinism is acceptable because it is attributed
and explained. Keeping them apart is the same split as everywhere else here:
deterministic math in the score, probabilistic judgement in the narrative.
"""

from dataclasses import dataclass, field
from functools import lru_cache

from portfolio_agent.knowledge.lexicon import is_known_word, segment

# Segmentation only splits on the commercial lexicon, so a sentiment-bearing
# word that isn't in it ("scam", "payday") never becomes its own token. These
# are appended to the segmentable set for sentiment purposes only — widening
# the shared lexicon would change classification, which is not this module's
# business.

# Terms VADER has no opinion on but domain buyers certainly do. VADER is
# trained on social media, so it scores "scam" correctly and "foreclosure"
# not at all.
_INDUSTRY_SENTIMENT = {
    # Negative associations that suppress end-user demand.
    "scam": -0.9, "fraud": -0.9, "debt": -0.4, "bankruptcy": -0.5,
    "foreclosure": -0.5, "lawsuit": -0.4, "divorce": -0.3, "casino": -0.3,
    "gambling": -0.3, "payday": -0.4, "porn": -0.9, "adult": -0.3,
    "toxic": -0.6, "plague": -0.7, "junk": -0.5, "cheapest": -0.2,
    "spam": -0.8, "virus": -0.6, "malware": -0.8, "scandal": -0.6,
    # Positive associations that lift it.
    "premium": 0.4, "trusted": 0.5, "secure": 0.4, "expert": 0.3,
    "wellness": 0.3, "growth": 0.3, "smart": 0.3, "prime": 0.3,
    # Commercially positive technology connotation. Curated rather than
    # measured, for the same reason the rest of this file is: a general
    # sentiment lexicon is trained on social media and has no opinion on the
    # vocabulary domains are built from. Each of these is a term the market
    # currently pays a premium to be associated with.
    "quantum": 0.4, "cloud": 0.3, "neural": 0.35, "genome": 0.3,
    "fusion": 0.35, "orbit": 0.25, "vertex": 0.2, "apex": 0.3,
    "forge": 0.25, "atlas": 0.25, "nova": 0.3, "catalyst": 0.3,
    "momentum": 0.3, "velocity": 0.25, "pioneer": 0.35, "frontier": 0.3,
    "summit": 0.3, "beacon": 0.3, "compass": 0.25, "anchor": 0.2,
    "wisdom": 0.45, "clarity": 0.4, "insight": 0.4, "signal": 0.2,
}

# Where a genuinely neutral word lands. Reached only when we HAVE signal and
# that signal is neutral — never as a stand-in for having no signal, which is
# what it used to be.
_NEUTRAL_SCORE = 70.0


@dataclass
class SentimentScore:
    """A sentiment reading, or an honest absence of one.

    `available` is the important field. VADER is a social-media lexicon and
    has no opinion on most domain words — quantum, forge, cloud and plague
    all score exactly 0.0 — so a score derived from silence is not a neutral
    reading, it is no reading. Returning 70 anyway made three unrelated
    words tie at 70 and quietly dragged every brand score toward the middle.
    """

    sld: str
    score: float | None
    compound: float | None
    terms: list[str]
    available: bool = True
    notes: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def _sentiment_tokens(sld: str) -> list[str]:
    """Segment, then rescue any sentiment-bearing word the commercial
    lexicon swallowed into an unknown token."""
    tokens: list[str] = []
    for token in segment(sld):
        if is_known_word(token):
            tokens.append(token)
            continue
        remainder = token
        for term in sorted(_INDUSTRY_SENTIMENT, key=len, reverse=True):
            if term in remainder:
                tokens.append(term)
                remainder = remainder.replace(term, " ")
    return tokens


def calculate_sentiment(sld: str) -> SentimentScore:
    """0-100 sentiment for a domain's component words.

    Scored on the words the domain is actually built from, not the raw
    string: VADER reads "propertymanagement" as an unknown token and
    "property management" as English.
    """
    # Tokens the sentiment layer recognizes: the commercial lexicon plus
    # this module's own vocabulary. "scam" is not a word anyone builds a
    # lexicon entry for, but it is exactly the word that has to be caught.
    tokens = _sentiment_tokens(sld)
    # The whole SLD is read too. A word absent from the commercial lexicon
    # can still carry connotation, and dropping it was why "quantum" and
    # "plague" both came back as neutral.
    candidates = tokens or [sld.lower()]

    industry = [(t, _INDUSTRY_SENTIMENT[t]) for t in candidates if t in _INDUSTRY_SENTIMENT]
    compound = _analyzer().polarity_scores(" ".join(candidates))["compound"]

    if not industry and compound == 0.0:
        # No signal from either source. Say so, rather than inventing a
        # neutral that will be averaged into the brand score as if measured.
        return SentimentScore(
            sld=sld, score=None, compound=None, terms=tokens, available=False,
            notes=[
                "no sentiment signal: this term carries no connotation in the sentiment lexicon "
                "and none in the commercial wordlist, so the component is excluded rather than "
                "scored neutral"
            ],
        )
    if industry:
        # The strongest industry association dominates: one toxic word in a
        # domain is not averaged away by neutral ones.
        strongest = min(industry, key=lambda pair: pair[1]) if min(v for _, v in industry) < 0 else max(
            industry, key=lambda pair: pair[1]
        )
        compound = strongest[1] if abs(strongest[1]) > abs(compound) else compound

    notes = []
    if compound <= -0.2:
        notes.append(f"negative keyword sentiment ({compound:+.2f}) — suppresses end-user demand")
    elif compound >= 0.2:
        notes.append(f"positive keyword sentiment ({compound:+.2f})")

    # -1..1 → 0..100, with 0 landing on the neutral score rather than 50.
    if compound >= 0:
        score = _NEUTRAL_SCORE + compound * (100.0 - _NEUTRAL_SCORE)
    else:
        score = _NEUTRAL_SCORE + compound * _NEUTRAL_SCORE

    return SentimentScore(
        sld=sld,
        score=round(max(0.0, min(100.0, score)), 2),
        compound=round(compound, 4),
        terms=candidates,
        available=True,
        notes=notes,
    )
