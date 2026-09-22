"""
Simplicity and brevity — the first component of APVA §4.3's brand score,
"inversely proportional to character length and word count".

Length is the dominant term because it is the one thing every domain buyer
agrees on: a 5-character name is scarce, a 20-character one is a sentence.
Hyphens and digits are scored separately and harshly, because they break a
name in the one place brandability is actually tested — someone saying it out
loud and someone else typing it.
"""

from dataclasses import dataclass, field

from portfolio_agent.knowledge.lexicon import EMERGING_TLDS, PREMIUM_TLDS, segment
from portfolio_agent.tools.linguistics import split_domain

# Where the length curve sits. At or below IDEAL a name scores full marks;
# past MAX it bottoms out. Between them the score falls linearly.
_IDEAL_LENGTH = 6
_MAX_LENGTH = 18

# Word-count penalty per word beyond the first. Two-word brandables are
# normal and barely penalized; four-word domains are descriptions.
_WORD_PENALTY = 12

_HYPHEN_PENALTY = 30  # unspeakable: "my-domain-name" cannot be said aloud
_DIGIT_PENALTY = 22  # ambiguous: "4" vs "four" splits the type-in traffic


@dataclass
class StructuralMetrics:
    domain: str
    sld: str
    tld: str
    char_length: int
    word_count: int
    hyphens: int
    digits: int
    is_premium_tld: bool
    is_emerging_tld: bool
    score: float = 0.0
    notes: list[str] = field(default_factory=list)


def get_structural_metrics(domain: str) -> StructuralMetrics:
    """Structure of a domain plus a 0-100 simplicity score."""
    sld, tld = split_domain(domain)
    words = segment(sld)
    hyphens = sld.count("-")
    digits = sum(1 for ch in sld if ch.isdigit())

    metrics = StructuralMetrics(
        domain=f"{sld}.{tld}",
        sld=sld,
        tld=tld,
        char_length=len(sld),
        word_count=max(1, len(words)),
        hyphens=hyphens,
        digits=digits,
        is_premium_tld=tld in PREMIUM_TLDS,
        is_emerging_tld=tld in EMERGING_TLDS,
    )

    score = _length_score(metrics.char_length)
    if metrics.char_length <= _IDEAL_LENGTH:
        metrics.notes.append(f"{metrics.char_length} characters — short enough to be scarce")
    elif metrics.char_length >= _MAX_LENGTH:
        metrics.notes.append(f"{metrics.char_length} characters — long enough to read as a description")

    extra_words = metrics.word_count - 1
    if extra_words > 0:
        score -= extra_words * _WORD_PENALTY
        metrics.notes.append(f"{metrics.word_count} words ({' + '.join(words)})")

    if hyphens:
        score -= hyphens * _HYPHEN_PENALTY
        metrics.notes.append(f"{hyphens} hyphen(s) — cannot be said aloud unambiguously")
    if digits:
        score -= _DIGIT_PENALTY
        metrics.notes.append(f"{digits} digit(s) — splits type-in traffic between the numeral and the word")

    metrics.score = round(max(0.0, min(100.0, score)), 2)
    return metrics


def _length_score(length: int) -> float:
    """100 at or below the ideal length, falling linearly to 0 at the max."""
    if length <= _IDEAL_LENGTH:
        return 100.0
    if length >= _MAX_LENGTH:
        return 0.0
    span = _MAX_LENGTH - _IDEAL_LENGTH
    return 100.0 * (1 - (length - _IDEAL_LENGTH) / span)
