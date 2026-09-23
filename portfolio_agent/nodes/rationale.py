"""
The rationale generator — APVA §4.4, the capstone of the valuation and the
one place an LLM writes anything the user reads.

The design constraint that shapes this whole module: a valuation rationale
is the part a professional investor actually checks, so a hallucinated
figure in it is worse than no rationale at all. It does not merely mislead —
it discredits every correct number beside it.

So the model is given the finished valuation and asked only to explain it,
and then a post-validator extracts every numeral from what it wrote and
asserts that each one already exists in the structured output. A response
with an invented figure is discarded, not patched: the model is asked again
once, and if it fails twice the templated rationale is used instead. The
template is less readable and always correct, which is the right way round
for this trade.

Note what the validator does NOT do: it never edits the model's prose. A
validator that silently corrected numbers would produce text whose sentences
no longer follow from each other.
"""

import asyncio
import json
import math
import re

from portfolio_agent import config
from portfolio_agent.llm import get_chat_model, resolve
from portfolio_agent.logging_setup import error as log_error
from portfolio_agent.logging_setup import get_logger, info, log_step, run_context, warn
from portfolio_agent.prompts import RATIONALE_SECTION_PROMPT, RATIONALE_SYSTEM_PROMPT
from portfolio_agent.schema.rationale_view import RationaleView, Section, build_rationale_view
from portfolio_agent.schema.valuation_output import ValuationOutput
from portfolio_agent.state import PortfolioAgentState

log = get_logger(__name__)

# Numerals as they appear in prose: 1,250,000 / 1250000 / 0.85 / 12.5 / $1.5M.
#
# Two guards earn their complexity. The lookbehind and the `\.[A-Za-z]`
# lookahead stop evidence ids and domain names being read as figures —
# without them every rationale that correctly cites "ev3" or mentions
# "360.com" would be rejected. The magnitude suffix is the opposite problem:
# "$1.5M" would otherwise extract as the bare 1, which is inside the
# prose-integer allowance, so a model could round a figure into something
# unverifiable and pass. Scaling it to 1500000 makes it checkable.
#
# Spelled-out magnitudes are handled too, because a live run rejected a
# correct draft over "$30.75 million": the figure WAS in the output as
# 30,750,000, but the extractor saw a bare 30.75 and called it invented. A
# guard that rejects correct work is a guard people route around.
_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])"
    r"(\d{1,3}(?:,\d{3})+|\d+)"  # integer part, with or without separators
    r"(?:\.(\d+))?"  # optional fraction
    r"\s?(k|m|bn|b|thousand|million|billion)?"  # optional magnitude, letter or word
    r"(?![\w]|\.[A-Za-z])"  # not mid-word, and not a TLD
    , re.IGNORECASE,
)

_MAGNITUDES = {
    "k": 1_000, "thousand": 1_000,
    "m": 1_000_000, "million": 1_000_000,
    "b": 1_000_000_000, "bn": 1_000_000_000, "billion": 1_000_000_000,
}

# Words that claim the figures are not real. Narrow on purpose: these are
# the terms that actually appear when a model describes fabricated data, and
# a looser list would fire on ordinary valuation prose.
_FIXTURE_CLAIM_PATTERN = re.compile(
    r"\b(mock(ed)?|fixtures?|simulated|placeholder|fabricated|synthetic data|not real)\b",
    re.IGNORECASE,
)

# Integers this small are ordinary prose ("all four tiers", "two of the
# three comps") rather than claims about value. Allowing them avoids
# rejecting correct narratives over the word "three".
_PROSE_INTEGER_CEILING = 12

# The taxonomy in words. The enum values are terse and two of them are
# genuinely ambiguous out of context ("emd", "generic"), so the prompt always
# carries the expansion.
CATEGORY_LABELS = {
    "generic": "a single common dictionary word that names its own category",
    "brandable": "a coined or invented name with no keyword meaning",
    "keyword": "contains a commercial keyword, but is not an exact search phrase",
    "emd": "Exact Match Domain — it exactly matches a commercial search phrase",
    "acronym": "a short letter string; value comes from rarity and liquidity",
    "geo": "a location term, sold to local businesses and franchises",
    "emerging_tld": "on a newer, trend-sensitive extension such as .ai or .io",
    "combination": "a brand element fused to a keyword (e.g. ZenBusiness)",
}


_WORD_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_WORD_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_WORD_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
# An ordinal anywhere in a phrase makes the whole phrase a description
# rather than a quantity, so the phrase is discarded rather than
# half-parsed: "the twenty-fifth percentile" must not yield 20.
_WORD_ORDINALS = frozenset({
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth", "twentieth", "thirtieth", "fortieth", "fiftieth",
    "sixtieth", "seventieth", "eightieth", "ninetieth", "hundredth",
    "thousandth", "millionth",
})
_WORD_TOKEN = re.compile(r"[a-z]+")


def extract_number_words(text: str) -> list[float]:
    """Numbers written out as English words.

    The hole this closes: `26` was rejected as invented while `twenty-six`
    passed untouched, and "nine hundred thousand dollars" was invisible
    entirely. Since the rationale's one hard rule is that every figure in
    the prose exists in the structured output, a model could satisfy it by
    spelling the figure out.

    Ordinals are deliberately not parsed. "the second tier" and "the
    twenty-fifth percentile" are descriptions rather than quantity claims,
    and reading them as numbers would reject correct sentences.
    """
    values: list[float] = []
    current = 0.0          # the group being built, e.g. "twenty six"
    total = 0.0            # groups already closed by a scale word
    open_phrase = False

    def close():
        nonlocal current, total, open_phrase
        if open_phrase:
            values.append(total + current)
        current, total, open_phrase = 0.0, 0.0, False

    # Hyphens join a number phrase ("twenty-six"); everything else splits it.
    for token in _WORD_TOKEN.findall((text or "").lower().replace("-", " ")):
        if token in _WORD_UNITS:
            current += _WORD_UNITS[token]
            open_phrase = True
        elif token in _WORD_TENS:
            current += _WORD_TENS[token]
            open_phrase = True
        elif token in _WORD_SCALES:
            if not open_phrase:
                # "a thousand", "several million" — the scale alone is still
                # a quantity claim.
                current = 1.0
                open_phrase = True
            scale = _WORD_SCALES[token]
            if scale == 100:
                current *= scale
            else:
                total += current * scale
                current = 0.0
        elif token in _WORD_ORDINALS:
            # Discard, do not emit: this phrase names a position, not an
            # amount.
            current, total, open_phrase = 0.0, 0.0, False
        elif token in ("and", "a", "an"):
            continue  # "a hundred and twenty" is one number
        else:
            close()
    close()
    return values


def extract_numbers(text: str) -> list[float]:
    """Every numeral in a piece of prose, separators stripped and magnitude
    suffixes expanded — written-out numbers included, so that spelling a
    figure is not a way around the allowlist."""
    numbers = []
    for whole, fraction, suffix in _NUMBER_PATTERN.findall(text or ""):
        raw = whole.replace(",", "")
        value = float(f"{raw}.{fraction}") if fraction else float(raw)
        if suffix:
            value *= _MAGNITUDES[suffix.lower()]
        numbers.append(value)
    numbers.extend(extract_number_words(text))
    return numbers


def allowed_numbers(output: ValuationOutput) -> set[float]:
    """Every number the structured output actually asserts.

    Collected recursively from the whole serialized object — tiers, brand
    score, evidence datapoints, confidences — because a rationale citing a
    comp's price or a domain rating is citing the output, and a validator
    that only knew about tier prices would reject correct sentences.

    Probabilities are additionally admitted in percentage form, since "a 14%
    chance of sale" and "0.14" are the same claim and prose prefers the
    first.
    """
    allowed: set[float] = set(float(n) for n in range(_PROSE_INTEGER_CEILING + 1))
    _collect(output.model_dump(mode="json"), allowed)

    for tier in output.tiers.populated().values():
        if tier.sale_probability is not None:
            allowed.add(round(tier.sale_probability * 100, 2))
            allowed.add(float(round(tier.sale_probability * 100)))
    if output.category_confidence is not None:
        allowed.add(round(output.category_confidence * 100, 2))
        allowed.add(float(round(output.category_confidence * 100)))
    if output.brand_score is not None:
        allowed.add(float(round(output.brand_score.total)))
        allowed.update(float(round(v)) for v in output.brand_score.components.values())

    # Counts a narrative legitimately reports about the output itself.
    allowed.add(float(len(output.evidence)))
    allowed.add(float(len(output.tiers.populated())))
    allowed.add(float(len(output.warnings)))
    return allowed


def _collect(node, into: set[float]) -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        into.add(float(node))
        into.add(float(round(node)))
        # Every sensible rounding of a real figure is also that figure.
        # 0.0089 is how a person writes 0.008949999697506428, and rejecting
        # it as invented cost a live valuation its narrative twice over. This
        # is not a tolerance window — each admitted value is derived from a
        # number the output actually asserts, so an invented figure still has
        # nothing to round to.
        for places in range(1, 7):
            into.add(round(float(node), places))
        return
    if isinstance(node, dict):
        for value in node.values():
            _collect(value, into)
        return
    if isinstance(node, (list, tuple)):
        for value in node:
            _collect(value, into)
        return
    if isinstance(node, str):
        # Numbers already written into warnings and factor strings are part
        # of what the output asserts — "(×1.29)", "$57,037", "score 0.86".
        for number in extract_numbers(node):
            into.add(number)


def unsupported_numbers(text: str, output: ValuationOutput) -> list[float]:
    """Numerals in `text` that the structured output does not contain."""
    allowed = allowed_numbers(output)
    return [n for n in extract_numbers(text) if n not in allowed]


def provenance_violations(text: str, output: ValuationOutput) -> list[str]:
    """Whether the narrative describes where the numbers came from correctly.

    The number validator cannot catch this — provenance is a word, not a
    figure — and both directions do real damage. Calling a live valuation
    "mock" destroys trust in numbers that are sound; presenting fixture data
    as a real appraisal is how someone prices a domain off a hash of its own
    name.
    """
    claims_fixture = bool(_FIXTURE_CLAIM_PATTERN.search(text or ""))
    is_mock = any(item.mock for item in output.evidence)

    if is_mock and not claims_fixture:
        return ["fixture data is not disclosed — the narrative reads as a real appraisal"]
    if claims_fixture and not is_mock:
        return ["describes live market data as mock or fixture data"]
    return []


def _format(number: float) -> str:
    """Render a rejected figure the way the model wrote it, so the feedback
    and the warning both name something findable in the draft. %g turns
    42424242 into 4.24242e+07, which matches nothing anyone can search for."""
    return f"{int(number):,}" if float(number).is_integer() else f"{number:,}"


# NOTE: `build_prompt` and `generate_rationale` — a single call producing the
# whole narrative, with a whole-output numeric allowlist and an
# all-or-nothing template fallback — were removed when generation became
# sectioned. Two reasons, both learned from live runs.
#
# The allowlist was too wide: a tier's paragraph could quote another tier's
# price, which is how "the list_bin tier suggests a range of $425" passed
# while list_bin was not priced at all. Per-section slices make that figure
# unavailable rather than merely disallowed.
#
# And the fallback was too blunt: one bad figure anywhere replaced the entire
# narrative with a template. Failure is now scoped to the section that failed.
#
# `unsupported_numbers` and `provenance_violations` remain — they are still
# the right checks over a whole output, and the evals use them directly.


# ── Sectioned generation ───────────────────────────────────────────────────


def section_allowed_numbers(section: Section) -> set[float]:
    """Figures this section may state: its own slice, and nothing else.

    Per-section rather than per-output on purpose. A whole-output allowlist
    let a tier's paragraph quote another tier's price, which is exactly the
    cross-tier invention the slicing exists to prevent.
    """
    allowed: set[float] = {float(n) for n in range(_PROSE_INTEGER_CEILING + 1)}
    for number in section.numbers():
        allowed.add(number)
        allowed.add(float(round(number)))
        for places in range(1, 7):
            allowed.add(round(number, places))
        if 0 < number < 1:  # probabilities are read aloud as percentages
            allowed.add(round(number * 100, 2))
            allowed.add(float(round(number * 100)))
    return allowed


def foreign_tier_mentions(text: str, section: Section, view: RationaleView) -> list[str]:
    """Tier names a section referred to that are not its own.

    The numeric check cannot catch this: "the BIN tier suggests a range of
    $425" contains only real figures and is still false. Matching on the
    tier's own label is what makes rule 2 enforceable rather than requested.
    """
    if section.key == "tier_relations":
        # The one section whose job is to compare tiers. Its claims are
        # checked by relational_problems() instead, which is a stronger
        # check than "did it say the word".
        return []
    own = section.key.removeprefix("tier:") if section.key.startswith("tier:") else None
    lowered = (text or "").lower()
    mentions = []
    for name, tier_label in view.tier_labels.items():
        if name == own:
            continue
        # The distinctive head of the label — "Floor price", "Auction
        # price", "marketplace Buy-It-Now (BIN) price".
        head = tier_label.split("—")[0].strip().lower()
        if head and head in lowered:
            mentions.append(name)
    return mentions


def validate_section(text: str, section: Section, view: RationaleView) -> list[str]:
    """Everything wrong with a section's draft, as reasons."""
    problems = []
    allowed = section_allowed_numbers(section)
    invented = [n for n in extract_numbers(text) if n not in allowed]
    if invented:
        problems.append(
            "figures not present in this section's data: "
            + ", ".join(_format(n) for n in sorted(set(invented))[:6])
        )
    foreign = foreign_tier_mentions(text, section, view)
    if foreign:
        problems.append(
            f"refers to {', '.join(foreign)}, which this section was not given and cannot describe"
        )
    problems.extend(section_provenance_problems(text, section))
    problems.extend(relational_problems(text, section))
    return problems


# Words that assert one figure sits above another. Direction is a fact and
# is checkable; magnitude is a judgement and is supplied rather than policed.
_ABOVE = ("above", "higher than", "exceeds", "more than", "greater than", "over")
_BELOW = ("below", "lower than", "less than", "under", "beneath")
_SENTENCE = re.compile(r"[^.!?]+[.!?]?")

# The closed vocabulary relations.py assigns. Anything here that was not
# handed to the section is a magnitude the model chose for itself.
_MAGNITUDE_VOCABULARY = ("marginally", "moderately", "substantially", "dramatically")

# Ratios stated in ordinary English. These pass every numeric check —
# "doubling" contains no numeral — and are exactly as falsifiable as writing
# "2x", so they get checked against the computed multiples too. A live run
# produced "doubling", "tripling" and "a single multiple", all of which
# happened to be roughly right, which is not the same as being verified.
_RATIO_WORDS = {
    "halving": 0.5, "halves": 0.5, "half": 0.5,
    "doubling": 2.0, "doubles": 2.0, "double": 2.0, "twice": 2.0, "twofold": 2.0,
    "two-fold": 2.0,
    "tripling": 3.0, "triples": 3.0, "triple": 3.0, "threefold": 3.0, "three-fold": 3.0,
    "quadrupling": 4.0, "quadruples": 4.0, "quadruple": 4.0, "fourfold": 4.0,
    "four-fold": 4.0,
    "fivefold": 5.0, "five-fold": 5.0, "quintuple": 5.0,
    "tenfold": 10.0, "ten-fold": 10.0, "an order of magnitude": 10.0,
}

# A spelled ratio is accepted when it is the CLOSEST of these words to some
# computed step, measured in log space so the comparison is scale-fair. A
# fixed tolerance does not work here: three and four are only a third apart,
# so any band loose enough to accept "doubling" for 1.95x also accepts
# "quadrupling" for 3.3x, which is a different claim about the same number.
# Nearest-word needs no threshold and cannot be tuned wrong.


def _nearest_ratio_word(multiple: float) -> float:
    """The ratio word this multiple is closest to, in log space.

    Log space because ratios are multiplicative: 3x sits exactly between 2x
    and 4.5x on a ratio scale, while on a linear one it looks closer to 2x
    than to 4x and a 3.3x step would answer to "doubling".
    """
    candidates = sorted(set(_RATIO_WORDS.values()))
    return min(candidates, key=lambda c: abs(math.log(multiple / c)))


def relational_problems(text: str, section: Section) -> list[str]:
    """Comparisons between tiers that the tier points contradict.

    Only direction, and only in the section allowed to compare. A sentence
    saying the floor exceeds the auction price is wrong in a way no numeric
    check catches — both figures are real and the claim joining them is
    false, which is the same shape of error as quoting another tier's price
    inside a tier's own paragraph.
    """
    if section.key != "tier_relations":
        return []
    steps = (section.content or {}).get("steps") or []
    if not steps:
        return []

    problems = []
    for sentence in _SENTENCE.findall(text or ""):
        lowered = sentence.lower()
        for step in steps:
            lower_label = str(step.get("from", "")).lower()
            upper_label = str(step.get("to", "")).lower()
            lower_head = lower_label.split("—")[0].strip()
            upper_head = upper_label.split("—")[0].strip()
            if not lower_head or not upper_head:
                continue
            if lower_head not in lowered or upper_head not in lowered:
                continue
            # Which one does the sentence put on top?
            lower_pos = lowered.index(lower_head)
            upper_pos = lowered.index(upper_head)
            first, second = ((lower_head, upper_head) if lower_pos < upper_pos
                             else (upper_head, lower_head))
            between = lowered[min(lower_pos, upper_pos):max(lower_pos, upper_pos)]
            says_first_is_higher = any(word in between for word in _ABOVE)
            says_first_is_lower = any(word in between for word in _BELOW)
            if says_first_is_higher and first == lower_head:
                problems.append(
                    f"claims the {lower_head} sits above the {upper_head}, which the tier points "
                    f"contradict ({step.get('multiple')}x the other way)")
            elif says_first_is_lower and first == upper_head:
                problems.append(
                    f"claims the {upper_head} sits below the {lower_head}, which the tier points "
                    f"contradict ({step.get('multiple')}x the other way)")

    # Magnitude is supplied, so using a word that belongs to none of the
    # supplied steps is invention rather than judgement. Attribution between
    # steps is not attempted — that is genuinely ambiguous in prose — but a
    # gap called "marginal" when no step is marginal is not.
    supplied = {str(step.get("magnitude", "")).lower() for step in steps}
    lowered_text = (text or "").lower()
    for word in _MAGNITUDE_VOCABULARY:
        if word in lowered_text and word not in supplied:
            problems.append(
                f"describes a step as {word!r}, which matches none of the computed comparisons "
                f"({', '.join(sorted(supplied))})")

    # A ratio written as a word is a numeric claim that no numeric check can
    # see, because it contains no numeral.
    multiples = [float(step.get("multiple") or 0) for step in steps]
    multiples = [m for m in multiples if m > 0]
    if multiples:
        for phrase, claimed in sorted(_RATIO_WORDS.items(), key=lambda kv: -len(kv[0])):
            if not re.search(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])", lowered_text):
                continue
            if not any(_nearest_ratio_word(m) == claimed for m in multiples):
                better = {_nearest_ratio_word(m) for m in multiples}
                problems.append(
                    f"says {phrase!r}, a claim of about {claimed:g}x, when the computed steps are "
                    f"{', '.join(f'{m:g}x' for m in multiples)} — nearer "
                    f"{', '.join(f'{b:g}x' for b in sorted(better))}")
    return problems


def section_provenance_problems(text: str, section: Section) -> list[str]:
    """Whether a section describes where its numbers came from correctly.

    Every section carries a DATA SOURCE line, so this is checkable per
    section rather than only over the whole narrative. Both directions do
    real damage: calling a live valuation fake destroys trust in numbers
    that are sound, and presenting fixture data as a real appraisal is how
    someone prices a domain off a hash of its own name.

    Only the first section is required to DISCLOSE fixtures — repeating it in
    every paragraph would be noise — but no section may ever contradict it.
    """
    source = str((section.content or {}).get("data_source") or "")
    if not source:
        return []
    claims_fixture = bool(_FIXTURE_CLAIM_PATTERN.search(text or ""))
    is_mock = "MOCK" in source

    if claims_fixture and not is_mock:
        return ["describes live market data as mock or fixture data"]
    if is_mock and not claims_fixture and section.key == "classification":
        return ["fixture data is not disclosed — the opening section reads as a real appraisal"]
    return []


# Fields whose full contents belong in the structured output, not in prose.
# The template summarizes them; dumping them produced a paragraph nobody
# would read, which is its own kind of failure.
_TEMPLATE_SUMMARIZE_ONLY = {"tools_that_ran", "comparable_sales_used", "brand_score_components"}
_TEMPLATE_MAX_ITEMS = 3


def _section_template(section: Section) -> str:
    """The fallback for one section. Less readable, always correct.

    Summarizes rather than dumps. A template that renders every nested
    structure is technically accurate and practically unreadable, and an
    unreadable fallback is one people work around rather than fix.
    """
    parts = []
    for key, value in section.content.items():
        if key == "data_source" or value in (None, [], {}, ""):
            continue
        readable = key.replace("_", " ")

        if key in _TEMPLATE_SUMMARIZE_ONLY and isinstance(value, list):
            parts.append(f"{readable}: {len(value)} recorded in the structured output")
        elif isinstance(value, list):
            shown = [_short(v) for v in value[:_TEMPLATE_MAX_ITEMS]]
            more = f" (+{len(value) - _TEMPLATE_MAX_ITEMS} more)" if len(value) > _TEMPLATE_MAX_ITEMS else ""
            parts.append(f"{readable}: " + "; ".join(shown) + more)
        elif isinstance(value, dict):
            parts.append(f"{readable}: " + ", ".join(f"{k} {_short(v)}" for k, v in list(value.items())[:4]))
        else:
            parts.append(f"{readable}: {_short(value)}")
    return ". ".join(parts) + "." if parts else "No data for this section."


def _short(value, limit: int = 120) -> str:
    if isinstance(value, dict):
        primary = value.get("domain") or value.get("valuation") or value.get("source") or ""
        secondary = value.get("counted_as") or value.get("estimate") or value.get("reason") or ""
        rendered = f"{primary} {secondary}".strip() or ", ".join(f"{k}={v}" for k, v in list(value.items())[:2])
    else:
        rendered = str(value)
    return rendered if len(rendered) <= limit else rendered[:limit] + "…"


def generate_tier_relations_deterministic(section: Section) -> str:
    """F22.1: Deterministic 'How the tiers relate' section.
    
    The LLM repeatedly produces false ratio prose ("by a factor of 1" for 1.44×,
    "substantially... two times" for different-magnitude steps). This section is
    computed from actual tier points instead of generated, making the ratio bug
    structurally impossible.
    
    For each adjacent tier pair, compute the real ratio and pick the descriptor
    from a fixed table, then state the exact multiple.
    """
    steps = (section.content or {}).get("steps") or []
    if not steps:
        return "The tiers could not be compared because insufficient market data was available."
    
    # Fixed descriptor table (ratio ranges → descriptive words)
    # Thresholds chosen to match APVA §4.1.1 tier relationships
    def ratio_descriptor(ratio: float) -> str:
        if ratio < 1.3:
            return "modestly"
        elif ratio < 2.5:
            return "substantially"
        elif ratio < 5.0:
            return "sharply"
        else:
            return "dramatically"
    
    sentences = []
    for step in steps:
        from_tier = step.get("from", "")
        to_tier = step.get("to", "")
        from_point = step.get("from_point", 0)
        to_point = step.get("to_point", 0)
        multiple = step.get("multiple", 1.0)
        
        if not from_tier or not to_tier or from_point <= 0 or to_point <= 0:
            continue
        
        # Strip the "—" explanation from tier labels for readability
        from_label = from_tier.split("—")[0].strip()
        to_label = to_tier.split("—")[0].strip()
        
        descriptor = ratio_descriptor(multiple)
        
        sentences.append(
            f"{from_label} (${from_point:,}) to {to_label} (${to_point:,}) — "
            f"{descriptor}, about {multiple:.1f}×"
        )
    
    if not sentences:
        return "The tiers could not be compared due to insufficient data."
    
    prose = ". ".join(sentences) + "."
    
    # Add context about what the relationships mean
    context = (
        " These multiples reflect the different buyer pools and risk profiles at each level: "
        "investor demand in a competitive auction sits below brokered end-user negotiations, "
        "which in turn sit below premium Buy-It-Now asks that factor in seller patience and scarcity."
    )
    
    return prose + context


async def generate_section(llm, section: Section, view: RationaleView) -> tuple[str, list[str]]:
    """One section, validated, with one retry then a templated fallback.

    Failure is section-scoped. Losing one paragraph to a stubborn model is a
    far smaller loss than replacing an entire narrative with a template
    because one figure in it was wrong — which is what the whole-rationale
    fallback did.
    """
    # F22.1: Deterministic generation for tier_relations section.
    # The LLM repeatedly produces false ratio prose, so this section is computed
    # from tier points instead of generated.
    if section.key == "tier_relations":
        deterministic_prose = generate_tier_relations_deterministic(section)
        info(log, "rationale.section_deterministic", section=section.key, chars=len(deterministic_prose))
        return deterministic_prose, []
    
    prompt = (
        f"SECTION: {section.heading}\n"
        f"YOUR TASK: {section.instruction}\n\n"
        f"DATA FOR THIS SECTION:\n{json.dumps(section.content, indent=2, default=str)}\n\n"
        "Write the section."
    )
    problems: list[str] = []

    for attempt in range(1, config.RATIONALE_MAX_ATTEMPTS + 1):
        messages = [("system", RATIONALE_SECTION_PROMPT), ("human", prompt)]
        if problems:
            messages.append((
                "human",
                f"Your previous answer was rejected: {problems[-1]}. Rewrite it using only the data "
                "above, and drop any sentence you cannot support with it.",
            ))
        try:
            response = await llm.ainvoke(messages)
        except Exception as e:  # noqa: BLE001 — one section must not fail the valuation
            warn(log, "rationale.section_failed", section=section.key, error=f"{type(e).__name__}: {e}")
            return _section_template(section), [f"section '{section.key}' fell back to a template ({type(e).__name__})"]

        text = (response.content if isinstance(response.content, str) else str(response.content)).strip()
        found = validate_section(text, section, view)
        if not found:
            return text, ([] if attempt == 1 else
                          [f"section '{section.key}' was regenerated once after: {problems[-1]}"])
        warn(log, "rationale.section_rejected", section=section.key, attempt=attempt, problem=found[0])
        problems.append(found[0])

    return _section_template(section), [
        f"section '{section.key}' fell back to a template after {config.RATIONALE_MAX_ATTEMPTS} "
        f"attempts ({problems[-1]})"
    ]


async def generate_sectioned(output: ValuationOutput) -> tuple[str, list[str]]:
    """The whole narrative, one call per section, assembled deterministically."""
    view = build_rationale_view(output)
    llm = get_chat_model(node="rationale")
    if llm is None:
        info(log, "rationale.no_model", fallback="template", reason=resolve(node="rationale").reason)
        return _assemble([(s, _section_template(s)) for s in view.sections]), [
            f"rationale: no LLM provider configured — using templated sections. "
            f"({resolve(node='rationale').reason}) The numbers are unaffected."
        ]

    # Sections are independent by construction, so they run concurrently.
    with log_step(log, "rationale.sections", count=len(view.sections)) as step:
        results = await asyncio.gather(*(generate_section(llm, s, view) for s in view.sections))
        step.set(fallbacks=sum(1 for _, w in results if w))

    warnings = [w for _, section_warnings in results for w in section_warnings]
    return _assemble([(s, text) for s, (text, _) in zip(view.sections, results)]), warnings


def _assemble(sections: list[tuple[Section, str]]) -> str:
    """Fixed headers, fixed order. The model never controls structure."""
    return "\n\n".join(f"{section.heading}\n{text}" for section, text in sections if text.strip())


async def rationale(state: PortfolioAgentState) -> PortfolioAgentState:
    """Final node. Everything numeric is already decided before this runs."""
    with run_context(state.get("domain", ""), state.get("run_id")), log_step(log, "rationale") as span:
        output = ValuationOutput.model_validate(state["valuation"])
        narrative, warnings = await generate_sectioned(output)
        span.set(chars=len(narrative), sections_fallen_back=len(warnings))
        return _finish(output, narrative, warnings)


def _finish(output: ValuationOutput, narrative: str, warnings: list[str]) -> PortfolioAgentState:

    output.rationale = narrative
    if warnings:
        output.warnings = list(output.warnings) + warnings

    return {
        "valuation": output.model_dump(mode="json"),
        "rationale": narrative,
        "warnings": output.warnings,
    }
