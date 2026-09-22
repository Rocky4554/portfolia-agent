"""
Misspelling detection and the valuation deflator it triggers — the
check_spelling_and_typos tool from APVA §4.3, and the failure mode the whole
product is pitched against: first-generation models "often penalize valuable
domains containing common but commercially understood misspellings", and
symmetrically, they price propertymanagment.com like propertymanagement.com.

This extends tools/linguistics.check_typo rather than replacing it. The two
have different jobs and must not be merged:

  - linguistics.check_typo runs against the curated commercial lexicon and
    exists to keep segmentation and classification honest. It is cheap and
    it runs on every domain.
  - this module runs against a full 160k-word dictionary (pyspellchecker)
    and produces the severity grade and the multiplicative deflator. It is
    the authority on whether a domain is a typo and by how much that costs.

The one rule that keeps this from destroying legitimate coined brandables:
**a misspelling only counts when the word it misspells is commercially
valuable.** canva.com is one edit from "canvas" and one of the most valuable
brandables ever sold; propertymanagment.com is one edit from "management"
and nearly worthless. The difference is not the edit distance — it is
whether the correctly spelled term is one people search for and pay for. So
the corrected word must appear in the commercial lexicon before any
deflation is applied.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import jellyfish

from portfolio_agent.config import TYPO_DEFLATORS, TYPO_MAX_DISTANCE
from portfolio_agent.knowledge.dictionary import is_dictionary_word, spell_checker
from portfolio_agent.knowledge.lexicon import (
    BRAND_MODIFIERS,
    COMMERCIAL_HEADS,
    COMMERCIAL_MODIFIERS,
    GENERIC_SINGLE_WORDS,
    GEO_TOKENS,
    is_known_word,
    segment,
)
from portfolio_agent.tools.linguistics import split_domain

# Below this, an edit-distance-1 neighbour is coincidence, not a typo.
_MIN_TOKEN_LEN = 4
# Above this, nothing is a plausible single misspelled word — it is a
# concatenation, and its parts are checked individually.
_MAX_TOKEN_LEN = 20
# Beyond this length, only single-edit corrections are considered.
# pyspellchecker generates edit-distance-2 candidates by brute force, which
# is O((26·n)²) and takes over a second on a 20-character string — long
# enough to dominate the whole valuation's latency. Two-edit typos in long
# words are also rare: the longer the word, the more obviously wrong two
# errors look, and the less type-in traffic the domain leaks.
_SINGLE_EDIT_ABOVE = 8

# Misspelling one of these costs real money — they carry search volume and
# type-in traffic that the correctly spelled domain collects instead.
_HIGH_VALUE_TERMS = COMMERCIAL_HEADS | GENERIC_SINGLE_WORDS
# These are real words too, but cheap ones. Getting them wrong is a blemish,
# not a collapse.
_LOW_VALUE_TERMS = COMMERCIAL_MODIFIERS | GEO_TOKENS | BRAND_MODIFIERS


@lru_cache(maxsize=2048)
def _cached_report(domain: str) -> "SpellingReport":
    return _check_spelling(domain)


@dataclass
class SpellingReport:
    domain: str
    sld: str
    tokens: list[str]
    is_typo: bool
    misspelled_token: str | None
    corrected_word: str | None
    corrected_form: str | None  # the correctly spelled sibling DOMAIN
    distance: int | None
    severity: str | None  # "severe" | "moderate" | "mild" | None
    deflator: float  # multiplicative, applied to the whole valuation
    notes: list[str] = field(default_factory=list)


def _checker_for(token: str):
    """Single-edit correction for long tokens; see _SINGLE_EDIT_ABOVE."""
    return spell_checker(1 if len(token) > _SINGLE_EDIT_ABOVE else 2)


def _best_correction(token: str) -> tuple[str | None, int | None]:
    """The nearest dictionary word to `token`, and the distance to it.

    Two deliberate choices here:

    Distance is Damerau-Levenshtein, not plain Levenshtein, because it
    counts a transposition as one edit. Transposed letters are among the
    most common real typos — "insruance", "travle" — and plain Levenshtein
    scores them 2, which would grade the single most frequent class of
    domain typo as merely "moderate".

    Ranking is by (distance, -frequency). pyspellchecker's own
    `correction()` ranks purely by corpus frequency, which happily returns a
    distance-2 common word over a distance-1 rare one; closest-reading-first
    is what a human seeing the typo would assume was meant.
    """
    checker = _checker_for(token)
    candidates = checker.candidates(token)
    if not candidates:
        return None, None
    ranked = sorted(
        candidates,
        key=lambda word: (jellyfish.damerau_levenshtein_distance(token, word), -checker.word_frequency[word]),
    )
    best = ranked[0]
    if best == token:
        return None, None
    return best, jellyfish.damerau_levenshtein_distance(token, best)


def _severity_for(corrected: str, distance: int) -> str | None:
    """How badly a misspelling costs, or None when it costs nothing.

    Returns None for a word outside the commercial lexicon: a coined name
    that happens to sit one edit from an ordinary English word is a brand,
    not an error.
    """
    if corrected in _HIGH_VALUE_TERMS:
        return "severe" if distance == 1 else "moderate"
    if corrected in _LOW_VALUE_TERMS:
        return "mild"
    return None


def check_spelling(domain: str) -> SpellingReport:
    """Whether `domain` misspells a commercially valuable term, and the
    deflator that follows from it.

    Cached: the graph asks for the same domain's spelling from the classify
    node, the brand score and the tier inputs, and candidate generation is
    the most expensive pure computation in the system.
    """
    return _cached_report((domain or "").strip().lower())


def _check_spelling(domain: str) -> SpellingReport:
    sld, tld = split_domain(domain)
    tokens = segment(sld)
    report = SpellingReport(
        domain=f"{sld}.{tld}",
        sld=sld,
        tokens=tokens,
        is_typo=False,
        misspelled_token=None,
        corrected_word=None,
        corrected_form=None,
        distance=None,
        severity=None,
        deflator=1.0,
    )

    # The whole SLD is checked alongside its segments. Segmentation is
    # greedy and exact, so "hotells" splits into a known "hotel" plus a
    # two-character remainder that is too short to test — the misspelling
    # only shows up when the string is read as one word.
    candidates_to_test = list(dict.fromkeys(tokens + [sld]))

    best: tuple[str, str, int, str] | None = None  # (token, corrected, distance, severity)
    for token in candidates_to_test:
        if not (_MIN_TOKEN_LEN <= len(token) <= _MAX_TOKEN_LEN):
            continue
        if is_known_word(token) or is_dictionary_word(token):
            continue
        corrected, distance = _best_correction(token)
        if not corrected or distance is None or distance > TYPO_MAX_DISTANCE:
            continue
        severity = _severity_for(corrected, distance)
        if severity is None:
            report.notes.append(
                f"'{token}' resembles '{corrected}', but '{corrected}' is not a commercially "
                "valuable term — read as a coined name, not a misspelling"
            )
            continue
        # Worst severity wins: one severe misspelling defines the domain.
        rank = {"severe": 0, "moderate": 1, "mild": 2}
        if best is None or rank[severity] < rank[best[3]]:
            best = (token, corrected, distance, severity)

    if best is None:
        return report

    token, corrected, distance, severity = best
    report.is_typo = True
    report.misspelled_token = token
    report.corrected_word = corrected
    report.corrected_form = f"{sld.replace(token, corrected)}.{tld}"
    report.distance = distance
    report.severity = severity
    report.deflator = TYPO_DEFLATORS[severity]
    report.notes.append(
        f"'{token}' is {distance} edit(s) from '{corrected}' — a {severity} misspelling of a commercial term. "
        f"The correctly spelled {report.corrected_form} collects the search and type-in traffic this domain "
        f"cannot, so the valuation is deflated to {report.deflator:.0%} of its comps-implied level."
    )
    return report
