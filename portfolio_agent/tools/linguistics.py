"""
Pure-logic domain classification — no external vendor, so this runs with zero
API keys and zero network. Two jobs:

  1. Classify the domain into the APVA Table 3.2.1 taxonomy (generic /
     brandable / keyword / EMD / acronym / geo / emerging-TLD / combination),
     with a confidence, because the category decides which evidence plan the
     graph runs (knowledge/category_playbook.py).
  2. Catch common misspellings (the propertymanagment.com problem from APVA
     §4.3) so nothing downstream prices a typo like its correctly spelled
     sibling.

Typo detection rides on segmentation rather than scanning the whole SLD: a
domain that segments into a known word plus a near-miss token ("property" +
"managment") *is* the failure case, and finding it that way is both exact and
cheap. The wordlist lives in knowledge/lexicon.py; scoring/spelling.py layers
a full dictionary and a severity grade on top of what's here.
"""

from dataclasses import dataclass, field

from portfolio_agent.knowledge.dictionary import is_dictionary_word
from portfolio_agent.knowledge.lexicon import (
    BRAND_MODIFIERS,
    COMMERCIAL_HEADS,
    COMMERCIAL_MODIFIERS,
    EMERGING_TLDS,
    GENERIC_SINGLE_WORDS,
    GEO_TOKENS,
    SEGMENTABLE,
    is_known_word,
    segment,
)

# A token shorter than this is too easy to land one edit from a real word by
# coincidence ("pay" is one edit from "day", "way", "par"...).
_MIN_TYPO_TOKEN_LEN = 4
_MAX_TYPO_DISTANCE = 2

# Confidence per rule. Deterministic and ordered by how much structure the
# rule actually observed: an acronym is unmistakable, brandable is the
# residual bucket everything unmatched falls into.
_CONFIDENCE = {
    "acronym_short": 0.95,
    "acronym_four": 0.80,
    "geo": 0.90,
    "generic": 0.90,
    "emd": 0.80,
    "keyword": 0.75,
    "combination": 0.70,
    "brandable": 0.65,
    "emerging_tld": 0.85,
}


@dataclass
class ClassificationResult:
    domain: str
    sld: str
    tld: str
    category: str
    category_confidence: float
    subcategory: str
    words: list[str]
    known_words: list[str]
    unknown_tokens: list[str]
    typo_of: str | None
    typo_distance: int | None
    structural_notes: list[str] = field(default_factory=list)
    # Labels beyond the primary, each with its own confidence. A domain is
    # not obliged to be one thing: accounting.ai is a keyword domain AND a
    # domain on a trend-sensitive extension, and the two facts feed
    # different evidence. The primary still drives anything that needs a
    # single answer; the evidence plan is the union.
    secondary_categories: list[tuple[str, float]] = field(default_factory=list)

    @property
    def all_categories(self) -> list[tuple[str, float]]:
        return [(self.category, self.category_confidence), *self.secondary_categories]

    @property
    def is_typo(self) -> bool:
        return self.typo_of is not None

    @property
    def keywords(self) -> list[str]:
        """The tokens worth querying a keyword/sales tool with."""
        return [w for w in self.known_words if w in COMMERCIAL_HEADS or w in GENERIC_SINGLE_WORDS]


def split_domain(domain: str) -> tuple[str, str]:
    """`example.co.uk` -> ('example', 'co.uk').

    The trailing dot is stripped, because a fully-qualified domain name ends
    in one and nothing downstream expects it. `tulum.com.` is valid and
    arrives from anyone copying a name out of a DNS tool; left alone its TLD
    parses as `com.`, which is not `com` — so the premium-TLD check misses,
    the strategic gate reports ".com. is not a .com phenomenon", and the
    cohort query searches an extension that does not exist.
    """
    domain = (domain or "").strip().lower()
    domain = domain.removeprefix("http://").removeprefix("https://").removeprefix("www.")
    # Path first, THEN the trailing dot: stripping it earlier does nothing
    # to "tulum.com./x", which still yields a TLD of "com." once the path
    # comes off.
    domain = domain.split("/")[0].rstrip(".")
    sld, _, tld = domain.partition(".")
    return sld, tld or "com"


def _levenshtein(a: str, b: str, *, cap: int = 99) -> int:
    """Standard edit distance, with an early exit once every cell in a row
    exceeds `cap` — the caller only ever cares about distances ≤ 2."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
        if min(current) > cap:
            return cap + 1
        previous = current
    return previous[-1]


def check_typo(sld: str) -> tuple[str | None, int | None]:
    """Best-effort misspelling check over an SLD.

    Returns (corrected_word, edit_distance) when a segment of the SLD is a
    near-miss on a known word, else (None, None). A correctly spelled word —
    whether in our commercial list or merely in the dictionary — is never a
    typo of another word, so only *unrecognized* segments are tested
    — which is what keeps "carinsurance" (car is simply not in the lexicon)
    and "propertymanagment" (managment is one edit from management) apart.
    """
    if sld in SEGMENTABLE:
        return None, None

    best: tuple[str, int] | None = None
    for token in segment(sld):
        if is_known_word(token) or len(token) < _MIN_TYPO_TOKEN_LEN:
            continue
        # A real English word is not a misspelling of anything, however close
        # it sits to something in the commercial list. "apple" is two edits
        # from "app", and reading it as a typo turned apple.com into a
        # keyword domain built from the word "app".
        if is_dictionary_word(token):
            continue
        for word in SEGMENTABLE:
            # Only words that could possibly be within the distance cap.
            if abs(len(word) - len(token)) > _MAX_TYPO_DISTANCE:
                continue
            distance = _levenshtein(token, word, cap=_MAX_TYPO_DISTANCE)
            if 0 < distance <= _MAX_TYPO_DISTANCE and (best is None or distance < best[1]):
                best = (word, distance)
                if distance == 1:
                    break
    return best if best else (None, None)


def classify_domain(domain: str) -> ClassificationResult:
    """Classify `domain` into the APVA taxonomy.

    A misspelling is reported via `typo_of` but does NOT become the category:
    propertymanagment.com is still structurally a keyword domain, and it is
    the spelling deflator (scoring/spelling.py), not the category, that
    collapses its price. Keeping those separate is what lets the rationale
    say "keyword domain, deflated 95% for a misspelling" instead of losing
    the keyword evidence plan entirely.
    """
    sld, tld = split_domain(domain)
    notes: list[str] = []

    tokens = segment(sld)
    typo_of, typo_distance = check_typo(sld)
    misspelled = _typo_token(sld, typo_of) if typo_of else None
    if typo_of:
        notes.append(f"'{misspelled}' is a near-miss on '{typo_of}' (edit distance {typo_distance})")

    # Classify the domain the author *meant*. propertymanagment.com is
    # structurally a keyword domain that happens to be misspelled, not an
    # invented brand — reading 'managment' as an unknown token would file it
    # under Combination and run the wrong evidence plan. The misspelling is
    # carried separately, in typo_of, and priced by the spelling deflator.
    resolved = [typo_of if t == misspelled else t for t in tokens]
    known = [t for t in resolved if is_known_word(t)]
    unknown = [t for t in resolved if not is_known_word(t)]

    if tld in EMERGING_TLDS:
        notes.append(f".{tld} is an emerging/trend-sensitive TLD")
    if not sld.isalnum():
        notes.append("contains a hyphen or other non-alphanumeric character")
    if any(ch.isdigit() for ch in sld):
        notes.append("contains digits")

    category, confidence, subcategory, secondary = _apply_rules(
        sld, tld, resolved, known, unknown, notes)
    if secondary:
        notes.append(
            "also reads as " + ", ".join(f"{name} ({conf:.2f})" for name, conf in secondary))

    if known:
        notes.append(f"segments as: {' + '.join(tokens)}")

    return ClassificationResult(
        domain=f"{sld}.{tld}",
        sld=sld,
        tld=tld,
        category=category,
        category_confidence=confidence,
        subcategory=subcategory,
        words=tokens,
        known_words=known,
        unknown_tokens=unknown,
        typo_of=f"{sld.replace(misspelled, typo_of)}.{tld}" if typo_of else None,
        typo_distance=typo_distance,
        structural_notes=notes,
        secondary_categories=secondary,
    )


def _typo_token(sld: str, corrected: str) -> str:
    """The misspelled token `corrected` should replace, so typo_of reads as a
    whole domain ('propertymanagement.com') rather than just the word."""
    for token in segment(sld):
        if not is_known_word(token) and _levenshtein(token, corrected, cap=_MAX_TYPO_DISTANCE) <= _MAX_TYPO_DISTANCE:
            return token
    return sld


def _apply_rules(
    sld: str, tld: str, tokens: list[str], known: list[str], unknown: list[str], notes: list[str]
) -> tuple[str, float, str, list[tuple[str, float]]]:
    """The classification cascade, most specific rule first.

    Returns a primary label plus any SECONDARY labels that also apply. The
    cascade used to return exactly one, and the single answer was the bug:
    the category selects the evidence plan, so forcing a choice between
    "keyword" and "emerging TLD" for accounting.ai meant gathering one
    market's evidence and not the other's. A domain on .ai that contains a
    commercial keyword has both keyword economics and TLD trend exposure,
    and both are priced.
    """
    heads = [t for t in known if t in COMMERCIAL_HEADS or t in GENERIC_SINGLE_WORDS]
    geo = [t for t in known if t in GEO_TOKENS]
    secondary: list[tuple[str, float]] = []

    def with_tld(primary: str, confidence: float, subcategory: str):
        """Add the extension's own story where it is not already the primary."""
        extra = list(secondary)
        if tld in EMERGING_TLDS and primary != "emerging_tld":
            extra.append(("emerging_tld", _CONFIDENCE["emerging_tld"]))
        return primary, confidence, subcategory, extra

    # Geo leads: a location component changes the buyer persona (local
    # business / franchise) no matter what it is paired with. But it no
    # longer erases the keyword underneath it — carinsurance-chicago.com is
    # sold to a local buyer AND priced on insurance keyword economics.
    if geo:
        if heads:
            secondary.append(("keyword", _CONFIDENCE.get("keyword", 0.7)))
        return with_tld(
            "geo", _CONFIDENCE["geo"],
            f"location term '{geo[0]}'" + (f" + {heads[0]}" if heads else ""))

    # Acronym: a short all-alpha string that isn't a word. SEGMENTABLE is
    # the COMMERCIAL lexicon and knows what sells, not what English
    # contains — so "hop" passed it and was priced as a three-letter
    # acronym on rarity and liquidity, which is a different asset with a
    # different buyer.
    if (sld.isalpha() and len(sld) <= 4 and sld not in SEGMENTABLE
            and not is_dictionary_word(sld)):
        if tld in EMERGING_TLDS:
            notes.append(f"short acronym on .{tld} — rarity plus trend exposure")
        key = "acronym_short" if len(sld) <= 3 else "acronym_four"
        return with_tld("acronym", _CONFIDENCE[key], f"{len(sld)}-letter acronym")

    category, confidence, subcategory = _semantic_category(sld, tokens, known, unknown, heads)

    # On an emerging TLD, a single word or coined brand is PRIMARILY a bet on
    # the extension's trend cycle (APVA §3.3's "single-word .ai"), so the
    # labels swap places. Either way both survive.
    if tld in EMERGING_TLDS and category in ("generic", "brandable"):
        secondary.append((category, confidence))
        return "emerging_tld", _CONFIDENCE["emerging_tld"], f"{subcategory} on .{tld}", secondary

    return with_tld(category, confidence, subcategory)


def _semantic_category(
    sld: str, tokens: list[str], known: list[str], unknown: list[str], heads: list[str]
) -> tuple[str, float, str]:
    brandy = [t for t in known if t in BRAND_MODIFIERS]
    # A one-character leftover is a segmentation artifact, not a brand
    # element — "managementt" is a fat-fingered keyword, not a coined name.
    unknown = [t for t in unknown if len(t) >= 2]

    # Nothing commercial recognized. Two very different things land here, and
    # telling them apart is worth a dictionary lookup: a real English word
    # that simply is not in our commercial list, and an actually invented
    # string. voice.com sold for $30,000,000 — calling it "a coined or
    # invented name" because "voice" is absent from a 400-word curated list
    # routes a premium single-word .com to the wrong evidence plan and
    # disqualifies it from the tiers that price category-defining names.
    if not heads:
        if len(tokens) == 1 and is_dictionary_word(tokens[0]):
            return "generic", _CONFIDENCE["generic"], "single dictionary word"
        if unknown or brandy:
            return "brandable", _CONFIDENCE["brandable"], "coined or invented name"
        return "brandable", _CONFIDENCE["brandable"], "unclassified structural fallback"

    # A single common noun standing alone — the Generic row of Table 3.2.1.
    if len(tokens) == 1 and tokens[0] in GENERIC_SINGLE_WORDS:
        return "generic", _CONFIDENCE["generic"], "single common word"

    # An invented token or an evocative one fused to a real keyword — the
    # ZenBusiness.com case (APVA §3.2).
    if (unknown or brandy) and heads:
        lead = (brandy or unknown)[0]
        return "combination", _CONFIDENCE["combination"], f"brand element '{lead}' + keyword '{heads[-1]}'"

    # Every token is a recognized commercial word.
    if len(tokens) >= 2 and tokens[-1] in COMMERCIAL_HEADS:
        intent = [t for t in tokens[:-1] if t in COMMERCIAL_MODIFIERS]
        if intent:
            # Modifier + head reads as a search phrase someone types.
            return "emd", _CONFIDENCE["emd"], f"exact-match phrase '{' '.join(tokens)}'"
        return "emd", _CONFIDENCE["emd"], f"exact commercial term '{' '.join(tokens)}'"

    return "keyword", _CONFIDENCE["keyword"], f"contains keyword(s): {', '.join(heads)}"
