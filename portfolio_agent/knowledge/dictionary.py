"""
The full English dictionary, as a shared service.

Separate from `lexicon.py`, which is a small curated list of *commercial*
words used for segmentation, and separate from both callers, which need it
for different jobs and must not import each other:

  - `tools/linguistics.py` asks "is this a real word, or a coined name?" —
    the difference between voice.com and zylantix.com.
  - `scoring/spelling.py` asks "what real word did this nearly spell?"

Putting it here rather than in either caller is what keeps
linguistics → spelling → linguistics from becoming an import cycle.

Loaded lazily and cached, because pyspellchecker reads a 160k-word
dictionary at construction and plenty of callers never need it.
"""

from functools import lru_cache

# Below this, a "word" is more likely an acronym or a fragment than a word.
MIN_WORD_LENGTH = 3


@lru_cache(maxsize=4)
def spell_checker(distance: int = 2):
    """A checker at a given edit distance. Cached per distance.

    Distance matters for cost, not just behaviour: pyspellchecker generates
    edit-distance-2 candidates by brute force, which is O((26·n)²) and takes
    over a second on a 20-character string.
    """
    from spellchecker import SpellChecker

    return SpellChecker(distance=distance)


@lru_cache(maxsize=8192)
def is_dictionary_word(token: str) -> bool:
    """Whether `token` is a real English word.

    This is what separates a premium single-word domain from a coined
    brandable — and getting it wrong is expensive in both directions.
    voice.com sold for $30M; calling it "a coined or invented name" routes it
    to the wrong evidence plan and disqualifies it from the tiers that price
    category-defining names.
    """
    token = (token or "").strip().lower()
    if len(token) < MIN_WORD_LENGTH or not token.isalpha():
        return False
    return bool(spell_checker(2).known([token]))


@lru_cache(maxsize=8192)
def frequency_per_million(token: str) -> float:
    """How often `token` appears in the reference corpus, per million words.

    The only signal available here that separates a common noun from a
    proper noun without a part-of-speech tagger. "insurance" appears 29
    times per million; "crusoe" appears 0.04 times. Both are dictionary
    words, and only one of them names a category.
    """
    token = (token or "").strip().lower()
    if not token:
        return 0.0
    checker = spell_checker(2)
    total = checker.word_frequency.total_words or 1
    return checker.word_frequency[token] / total * 1_000_000


def is_common_noun(token: str, min_frequency: float | None = None) -> bool:
    """Whether `token` is an ordinary word in broad use, rather than a name.

    A frequency floor, not a part-of-speech tag, and it is worth being clear
    about what that does and does not buy. Measured on 37 category terms
    against 22 proper nouns, a floor of 2.0 per million keeps 36 of the
    category terms and admits none of the proper nouns — the two
    populations barely overlap, because a name is rare in general prose
    however famous the thing it names.

    The error it still makes is rejecting a genuine but uncommon category
    term ("slots", at 0.91). That direction is the right one to fail in: a
    rejected term produces an honestly unpriced tier with a stated reason,
    while an accepted proper noun produces a confident price built on the
    claim that Robinson Crusoe names a product category.
    """
    from portfolio_agent import config

    if not is_dictionary_word(token):
        return False
    floor = config.CATEGORY_TERM_MIN_FREQUENCY_PPM if min_frequency is None else min_frequency
    return frequency_per_million(token) >= floor
