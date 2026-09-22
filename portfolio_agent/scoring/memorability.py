"""
Recall — APVA §4.3's third brand-score component, "proxied by analyzing
branded search volume and other recall metrics".

Branded search volume is the honest measure and we do not have it: a domain
being valued has usually never been a brand, so there is nothing to measure.
Rather than invent a number, this computes a proxy from the things that
*predict* recall and takes the real signal as an optional input for when it
exists (an aftermarket domain with an existing business behind it, or a
Phase-3 SEO/trends lookup).

The proxy rests on three observations that hold across brand-naming research
and, more usefully here, across what domainers actually pay for:

  - Short and sayable names are remembered. That is already measured, so
    structural and phonetic scores carry most of the weight.
  - A real word is remembered far better than a coined string, because it is
    already in the listener's head — "cloud" needs no learning, "quorvexis"
    does.
  - Repeated or rhyming sound patterns aid recall, which is why so many
    brandables double a consonant or repeat a vowel.
"""

from dataclasses import dataclass, field

from portfolio_agent.knowledge.lexicon import is_known_word, segment
from portfolio_agent.scoring.phonetic import PhoneticScore
from portfolio_agent.scoring.structural import StructuralMetrics

# The proxy's internal mix. Structure and sound dominate because they are
# measured rather than inferred.
_STRUCTURAL_SHARE = 0.45
_PHONETIC_SHARE = 0.35
_LEXICAL_SHARE = 0.20

_DICTIONARY_WORD_BONUS = 100.0  # every token is a real word
_PARTIAL_WORD_BONUS = 60.0  # some tokens are real words
_COINED_BASELINE = 25.0  # nothing recognizable — all recall must be bought

# Branded search volume, when we have it, is worth more than the whole
# proxy: it is evidence that people already remember the name.
_REAL_SIGNAL_WEIGHT = 0.6
_VOLUME_FOR_FULL_MARKS = 50_000  # monthly branded searches


@dataclass
class MemorabilityScore:
    sld: str
    score: float
    lexical_component: float
    used_branded_search_volume: bool
    notes: list[str] = field(default_factory=list)


def calculate_memorability(
    structural: StructuralMetrics,
    phonetic: PhoneticScore,
    branded_search_volume: int | None = None,
) -> MemorabilityScore:
    """0-100 recall proxy.

    `branded_search_volume` is the real signal and overrides most of the
    proxy when supplied. It is optional because it is almost never available
    at valuation time — see the module docstring.
    """
    tokens = segment(structural.sld)
    known = [t for t in tokens if is_known_word(t)]
    notes: list[str] = []

    if tokens and len(known) == len(tokens):
        lexical = _DICTIONARY_WORD_BONUS
        notes.append(f"built entirely from real words ({' + '.join(tokens)}) — nothing to teach a listener")
    elif known:
        lexical = _PARTIAL_WORD_BONUS
        notes.append(f"partly real words ({' + '.join(known)}) — anchors recall to something familiar")
    else:
        lexical = _COINED_BASELINE
        notes.append("coined string — recall has to be bought with marketing spend")

    if _has_sound_repetition(structural.sld):
        lexical = min(100.0, lexical + 10.0)
        notes.append("repeated sound pattern aids recall")

    proxy = (
        _STRUCTURAL_SHARE * structural.score
        + _PHONETIC_SHARE * phonetic.score
        + _LEXICAL_SHARE * lexical
    )

    used_real_signal = branded_search_volume is not None
    if used_real_signal:
        measured = min(100.0, 100.0 * (branded_search_volume or 0) / _VOLUME_FOR_FULL_MARKS)
        score = _REAL_SIGNAL_WEIGHT * measured + (1 - _REAL_SIGNAL_WEIGHT) * proxy
        notes.append(f"{branded_search_volume:,} monthly branded searches — measured recall, not a proxy")
    else:
        score = proxy

    return MemorabilityScore(
        sld=structural.sld,
        score=round(max(0.0, min(100.0, score)), 2),
        lexical_component=round(lexical, 2),
        used_branded_search_volume=used_real_signal,
        notes=notes,
    )


def _has_sound_repetition(sld: str) -> bool:
    """A doubled letter or a repeated two-character pattern."""
    clean = "".join(ch for ch in sld.lower() if ch.isalpha())
    if any(a == b for a, b in zip(clean, clean[1:])):
        return True
    pairs = [clean[i : i + 2] for i in range(len(clean) - 1)]
    return len(pairs) != len(set(pairs))
