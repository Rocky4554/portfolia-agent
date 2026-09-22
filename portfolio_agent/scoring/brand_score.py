"""
calculate_brand_score — APVA §4.3's composite: "a weighted average derived
from the outputs of several other specialized tools", normalized 0-100.

Pure function. No network, no LLM, no clock. Two runs on the same domain
produce byte-identical output, which is what lets evals/test_scoring.py
assert exact numbers and what stops the same domain being worth different
amounts on Tuesday.

The typo deflator is deliberately NOT folded into `total`. A misspelling
does not make a name badly structured or hard to say — propertymanagment is
as pronounceable as propertymanagement. What it destroys is the domain's
commercial value, so it travels alongside the score as a multiplier on the
valuation and the tier engine applies it. Mixing the two would produce a
brand score that lies about the name and a valuation that cannot explain
where its deflation came from.
"""

from portfolio_agent import config
from portfolio_agent.config import BRAND_SCORE_WEIGHTS
from portfolio_agent.schema.valuation_output import BrandScore
from portfolio_agent.scoring.memorability import calculate_memorability
from portfolio_agent.scoring.phonetic import calculate_phonetic_score
from portfolio_agent.scoring.sentiment import calculate_sentiment
from portfolio_agent.scoring.spelling import check_spelling
from portfolio_agent.scoring.structural import get_structural_metrics
from portfolio_agent.tools.linguistics import split_domain


def calculate_brand_score(
    domain: str, *, category: str | None = None, branded_search_volume: int | None = None
) -> BrandScore:
    """The 0-100 composite plus its parts.

    `components` and `weights` share keys and `total` is their weighted
    average, so a reader can recompute the headline number by hand — which
    is the whole point of publishing both.

    `category` selects the weights. Without it every domain was scored as
    though pronounceability mattered equally to a coined startup name and to
    an exact-match keyword domain — 25% in both cases — when a brandable
    lives on being sayable and an EMD is bought for the phrase in it. An
    unknown category falls back to the flat weights rather than guessing at
    one, because a wrong category applies the wrong emphasis silently.
    """
    sld, _ = split_domain(domain)

    structural = get_structural_metrics(domain)
    phonetic = calculate_phonetic_score(sld)
    memorability = calculate_memorability(structural, phonetic, branded_search_volume)
    sentiment = calculate_sentiment(sld)
    spelling = check_spelling(domain)

    components = {
        "structural": structural.score,
        "memorability": memorability.score,
    }
    # Same rule as sentiment: a component with nothing to measure is
    # EXCLUDED and the remaining weights renormalise. A numeric name scored
    # 0/100 for pronounceability, which is not a low score, it is no score
    # — and it dragged 163.ai's composite to 30 and then deflated the
    # end-user tier a further 20% for being "hard to say".
    if phonetic.available:
        components["phonetic"] = phonetic.score
    # Sentiment joins only when there is a reading. A component that is
    # unavailable is EXCLUDED and the remaining weights renormalize —
    # injecting a neutral would average a fabricated number into the score
    # and, because neutral sits high, drag every domain toward the same
    # value. Three unrelated words tied at exactly 70 before this.
    if sentiment.available and sentiment.score is not None:
        components["sentiment"] = sentiment.score
    key = (category or "").strip().lower()
    table = config.BRAND_SCORE_WEIGHTS_BY_CATEGORY.get(key)
    weighted_for = key if table else None
    table = table or BRAND_SCORE_WEIGHTS
    weights = {name: float(table[name]) for name in components if name in table}

    # Normalize by the weights actually present. This does double duty: a
    # mis-set config produces a wrong-but-bounded score rather than one
    # above 100, and an excluded component redistributes its weight across
    # the components that ARE measured instead of leaving a hole.
    total_weight = sum(weights.values()) or 1.0
    total = sum(components[name] * weights[name] for name in components) / total_weight

    return BrandScore(
        total=round(max(0.0, min(100.0, total)), 2),
        components={name: round(value, 2) for name, value in components.items()},
        weights=weights,
        weighted_for=weighted_for,
        typo_deflator=spelling.deflator,
    )


def brand_score_detail(
    domain: str, *, category: str | None = None, branded_search_volume: int | None = None
) -> dict:
    """The score plus every sub-score's reasoning, for evidence and the
    rationale. Separate from calculate_brand_score because the schema's
    BrandScore intentionally carries numbers only — the prose belongs in
    evidence, where it can be cited by id."""
    sld, _ = split_domain(domain)
    structural = get_structural_metrics(domain)
    phonetic = calculate_phonetic_score(sld)
    memorability = calculate_memorability(structural, phonetic, branded_search_volume)
    sentiment = calculate_sentiment(sld)
    spelling = check_spelling(domain)
    score = calculate_brand_score(
        domain, category=category, branded_search_volume=branded_search_volume)

    return {
        "domain": domain,
        "brand_score": score.total,
        "components": score.components,
        "weights": score.weights,
        "weighted_for": score.weighted_for,
        "typo_deflator": score.typo_deflator,
        "structural": {
            "char_length": structural.char_length,
            "word_count": structural.word_count,
            "hyphens": structural.hyphens,
            "digits": structural.digits,
            "tld": structural.tld,
            "notes": structural.notes,
        },
        "phonetic": {
            "syllables": phonetic.syllables,
            "soundex": phonetic.soundex,
            "metaphone": [phonetic.metaphone_primary, phonetic.metaphone_alternate],
            "longest_consonant_run": phonetic.longest_consonant_run,
            "notes": phonetic.notes,
        },
        "memorability": {
            "used_branded_search_volume": memorability.used_branded_search_volume,
            "notes": memorability.notes,
        },
        "sentiment": {
            "available": sentiment.available,
            "compound": sentiment.compound,
            "terms": sentiment.terms,
            "notes": sentiment.notes,
        },
        "spelling": {
            "is_typo": spelling.is_typo,
            "severity": spelling.severity,
            "corrected_form": spelling.corrected_form,
            "distance": spelling.distance,
            "deflator": spelling.deflator,
            "notes": spelling.notes,
        },
    }
