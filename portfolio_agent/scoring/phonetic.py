"""
Pronounceability — APVA §4.3's second brand-score component, "derived from
calculate_phonetic_score, which uses standard algorithms like Soundex and
Metaphone".

What those algorithms actually give us is a phonetic *encoding*, not a score,
so the scoring is built on top of them in three parts:

  1. Ambiguity. Double Metaphone returns a second encoding when a string has
     more than one defensible pronunciation ("schwartz" → XRTS / XFRTS). Two
     readings means two spellings a listener might type — the radio test
     fails, and the name leaks traffic to whichever one they guess.
  2. Consonant clusters. The single best structural predictor of a name being
     hard to say. Compare "zylantix" with "zqxvlth".
  3. Syllable rhythm. Two or three syllables is the brandable sweet spot;
     one is often too thin to be memorable and five is a sentence.

Soundex is kept as a stability check rather than a scorer: it collapses far
more aggressively than Metaphone, so a name whose Soundex code is very short
relative to its length has lost most of its distinguishing sounds.
"""

from dataclasses import dataclass, field

import jellyfish
from metaphone import doublemetaphone

_VOWELS = set("aeiouy")

# A run of this many consonants starts costing points. Three is common in
# English ("str", "nch"); four is where it gets hard.
_CLUSTER_THRESHOLD = 3
_CLUSTER_PENALTY = 18

# Letter pairs that spell ONE sound. Counting them as two overstated the
# difficulty of most of the English language: "chrome" was charged for a
# three-consonant run that is really /kr/, and "twitch" for /tʃ/.
_DIGRAPHS = ("sch", "tch", "ch", "sh", "th", "ph", "wh", "gh", "ck", "ng", "qu")

# Consonant clusters English can START a syllable with. Not a style
# judgement — a name is hard to say when its sounds cannot be assembled into
# syllables, and these are the assemblies that work.
_ONSETS = frozenset({
    "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s",
    "t", "v", "w", "x", "y", "z",
    "bl", "br", "cl", "cr", "dr", "dw", "fl", "fr", "gl", "gr", "kl", "kr",
    "pl", "pr", "sc", "sk", "sl", "sm", "sn", "sp", "st", "sw", "tr", "tw",
    "vr", "wr", "kn", "gn", "ps", "pn", "sf", "shr", "thr", "chr", "phr",
    # Digraph onsets. Omitting these made "tch" unsayable, which would
    # surprise anyone who has said "kitchen".
    "ch", "sh", "th", "wh", "ph", "qu", "gh", "rh",
    "scr", "spl", "spr", "squ", "str", "sch", "sph", "sthr",
})

# Clusters English can END a syllable with.
_CODAS = frozenset({
    "", "b", "c", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v",
    "x", "z", "ch", "sh", "th", "ng", "ck", "ph", "gh",
    "ct", "ft", "ld", "lf", "lk", "lm", "lp", "ls", "lt", "lth", "mp", "nd",
    "nk", "nt", "nch", "pt", "rb", "rd", "rk", "rl", "rm", "rn", "rp", "rs",
    "rt", "rth", "sk", "sp", "st", "nce", "mpt", "nct", "rst", "rch", "lch",
    "ns", "ms", "ds", "ts", "ks", "ps", "bs", "gs", "rg", "rv", "lv", "lb",
    # Spellings where letters outnumber sounds at the end of a syllable:
    # "bright" ends in /t/, "match" in /tʃ/. Omitting them charged
    # brightside for a four-letter run that is two sounds.
    "ght", "ghts", "tch", "dge", "lves", "rves", "nds", "nts", "rds", "rts",
    "lds", "lts", "mps", "nks", "sts", "sks", "cks", "ngs", "ths", "shes",
})


def _collapse_digraphs(run: str) -> str:
    """Rewrite letter pairs that spell one sound as a single placeholder, so
    a run is measured in sounds rather than in characters."""
    out = run
    for digraph in _DIGRAPHS:  # longest first, so "tch" beats "ch"
        out = out.replace(digraph, "\0")
    return out


def is_pronounceable_run(run: str) -> bool:
    """Whether a consonant run can be said by an English speaker.

    Two ways it can be. It is a cluster a syllable may begin with ("str" in
    strava), or it straddles a syllable boundary and splits into an ending
    plus a beginning ("netflix" is net + flix, never a "tfl" cluster). Only
    a run that can be neither is genuinely hard, which is what separates
    zqxvlth from chrome.
    """
    run = (run or "").lower()
    if len(run) <= 2:
        return True
    if run in _ONSETS:
        return True
    # Every way of cutting it into a coda and an onset.
    return any(run[:i] in _CODAS and run[i:] in _ONSETS for i in range(1, len(run)))

_AMBIGUITY_PENALTY = 15  # a second Double Metaphone reading
_NO_VOWEL_PENALTY = 45  # unpronounceable as written

# Syllable counts that read as brand names rather than as words or sentences.
_IDEAL_SYLLABLES = (2, 3)
_SYLLABLE_PENALTY = 10


@dataclass
class PhoneticScore:
    sld: str
    soundex: str
    metaphone_primary: str
    metaphone_alternate: str
    syllables: int
    longest_consonant_run: int
    vowel_ratio: float
    # False where the name has nothing to pronounce. 163.ai scored 0/100
    # with the note "no pronounceable characters" — the scorer knew it had
    # nothing to measure and published a zero anyway, which the tier engine
    # then read as "hard to say" and deflated 20% for. A numeric domain is
    # not a badly-chosen word; it is a different kind of asset, and the
    # honest reading is that this axis does not apply.
    available: bool = True
    score: float = 0.0
    notes: list[str] = field(default_factory=list)


def estimate_syllables(sld: str) -> int:
    """Vowel-group count, the standard cheap English syllable proxy.

    A trailing silent "e" is not a syllable ("brellium" vs "brelle"), and
    every word has at least one.
    """
    word = "".join(ch for ch in sld.lower() if ch.isalpha())
    if not word:
        return 0
    count = 0
    previous_was_vowel = False
    for char in word:
        is_vowel = char in _VOWELS
        if is_vowel and not previous_was_vowel:
            count += 1
        previous_was_vowel = is_vowel
    if word.endswith("e") and count > 1 and not word.endswith(("le", "ee", "ye")):
        count -= 1
    return max(1, count)


def longest_consonant_run(sld: str) -> int:
    return len(longest_consonant_substring(sld))


def longest_consonant_substring(sld: str) -> str:
    """The run itself, not just its length — the letters are what decide
    whether it is sayable."""
    longest = current = ""
    for char in (sld or "").lower():
        if char.isalpha() and char not in _VOWELS:
            current += char
            if len(current) > len(longest):
                longest = current
        else:
            current = ""
    return longest


def calculate_phonetic_score(sld: str) -> PhoneticScore:
    """0-100 pronounceability for a second-level domain."""
    clean = "".join(ch for ch in (sld or "").lower() if ch.isalpha())
    if not clean:
        return PhoneticScore(
            sld=sld, available=False, soundex="", metaphone_primary="",
            metaphone_alternate="", syllables=0, longest_consonant_run=0,
            vowel_ratio=0.0, score=0.0,
            notes=["no letters to pronounce — this name is not a word and cannot be scored "
                   "on how it sounds, so the component is excluded rather than failed"],
        )

    primary, alternate = doublemetaphone(clean)
    result = PhoneticScore(
        sld=sld,
        soundex=jellyfish.soundex(clean),
        metaphone_primary=primary,
        metaphone_alternate=alternate,
        syllables=estimate_syllables(clean),
        longest_consonant_run=longest_consonant_run(clean),
        vowel_ratio=round(sum(1 for c in clean if c in _VOWELS) / len(clean), 3),
    )

    score = 100.0

    if alternate:
        score -= _AMBIGUITY_PENALTY
        result.notes.append(
            f"two defensible pronunciations ({primary} / {alternate}) — listeners will type two different spellings"
        )

    if result.longest_consonant_run >= _CLUSTER_THRESHOLD:
        run = longest_consonant_substring(result.sld)
        sounds = len(_collapse_digraphs(run))
        if sounds >= _CLUSTER_THRESHOLD and not is_pronounceable_run(run):
            excess = sounds - _CLUSTER_THRESHOLD + 1
            score -= excess * _CLUSTER_PENALTY
            result.notes.append(f"consonant run of {result.longest_consonant_run} — hard to say")

    if result.vowel_ratio == 0:
        score -= _NO_VOWEL_PENALTY
        result.notes.append("no vowels")

    if result.syllables < _IDEAL_SYLLABLES[0]:
        score -= _SYLLABLE_PENALTY
        result.notes.append(f"{result.syllables} syllable — thin for a brand name")
    elif result.syllables > _IDEAL_SYLLABLES[1]:
        score -= (result.syllables - _IDEAL_SYLLABLES[1]) * _SYLLABLE_PENALTY
        result.notes.append(f"{result.syllables} syllables — long to say")

    # Soundex collapses to at most 4 characters. A long name whose Soundex is
    # padded with zeros has few distinct consonant sounds left — it will be
    # confused with its neighbours.
    if len(clean) >= 6 and result.soundex.endswith("00"):
        score -= 8
        result.notes.append(f"few distinct sounds (Soundex {result.soundex}) — easily confused with similar names")

    result.score = round(max(0.0, min(100.0, score)), 2)
    return result
