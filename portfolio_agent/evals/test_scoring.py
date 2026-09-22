"""
Offline regression eval for the deterministic scoring layer.
Run with: uv run python -m portfolio_agent.evals.test_scoring

No network, no keys, no model. Everything under portfolio_agent/scoring/ is
a pure function, and this file is what holds it to that: the expected values
below are exact, so any change to a weight, a penalty or a threshold shows up
here as a diff rather than as a quietly different valuation.

If a number below changes because you deliberately recalibrated something,
update it here in the same commit and say why in the message. If it changes
and you did not mean it to, that is the bug this file exists to catch.
"""

import sys

from portfolio_agent import config
from portfolio_agent.evals.golden_set import TYPO_REGRESSION_PAIRS
from portfolio_agent.scoring.brand_score import brand_score_detail, calculate_brand_score
from portfolio_agent.scoring.memorability import calculate_memorability
from portfolio_agent.scoring.phonetic import calculate_phonetic_score, estimate_syllables
from portfolio_agent.scoring.sentiment import calculate_sentiment
from portfolio_agent.scoring.spelling import check_spelling
from portfolio_agent.scoring.structural import get_structural_metrics

# ── Locked expectations ─────────────────────────────────────────────────────

EXPECTED_STRUCTURAL = {
    "cloud.ai": 100.0,  # 5 chars, one word — nothing to penalize
    "zylantix.com": 83.33,
    "zenbusiness.com": 46.33,  # two words, 11 chars
    "cheapinsurancequotes.com": 0.0,  # 20 chars, three words — a description
    "my-best-site4.net": 0.0,  # hyphens and a digit on top of four words
}

EXPECTED_PHONETIC = {
    "canva": 100.0,
    "zylantix": 100.0,
    "cloud": 90.0,  # one syllable — thin for a brand
    # Two readings (XRTS/XFRTS), but no cluster penalty: "schw" is two
    # sounds, not four, and a common surname is not hard to say. It scored
    # 39 while the run was measured in letters.
    "schwartz": 75.0,
    "zqxvlth": 0.0,  # unsayable
}

EXPECTED_SENTIMENT = {
    "cloud": 79.0,        # positive commercial connotation, curated
    "trustedcare": 92.29,
    "scamloans": 7.0,
    "plague": 21.0,       # negative, and previously indistinguishable from cloud
}

# Names we genuinely have no sentiment reading for. The component must be
# reported unavailable rather than scored, because a general sentiment
# lexicon is trained on social media and returns exactly 0.0 for coined
# words — a silence that used to be recorded as a neutral 70.
SENTIMENT_UNAVAILABLE = ("zylantix", "brellium", "nimbus")

# These moved when the fabricated neutral was removed. Two directions, both
# corrections: names with real positive connotation rose, and the weak names
# fell because a constant 70 had been propping them up. propertymanagment.com
# losing 7 points is the clearest case — a misspelling was being credited
# with average sentiment it had never earned.
EXPECTED_BRAND_TOTALS = {
    "nimbus.io": 100.0,
    "cloud.ai": 92.42,
    "canva.com": 95.31,
    "zylantix.com": 86.72,
    "zenbusiness.com": 68.11,
    "my-best-site4.net": 54.57,
    "propertymanagment.com": 33.28,
    "cheapinsurancequotes.com": 31.56,
}

EXPECTED_COMPONENTS = {
    "cloud.ai": {"structural": 100.0, "phonetic": 90.0, "memorability": 96.5, "sentiment": 79.0},
    # No sentiment key at all: the misspelling carries no readable
    # connotation, so the component is absent rather than invented.
    "propertymanagment.com": {"structural": 0.0, "phonetic": 70.0, "memorability": 36.5},
}

# APVA §4.3's regression requirement, both directions. The high group must
# stay high (a typo check that flags coined brandables is worse than no typo
# check at all) and the low group must collapse.
HIGH_BRAND_CLASS = ("cloud.ai", "canva.com", "nimbus.io", "zylantix.com")
DEFLATED_CLASS = ("propertymanagment.com", "managment.com", "insuranse.com", "hotells.com")

# Coined names that sit within two edits of an English word and must NOT be
# read as misspellings. canva.com is one edit from "canvas".
MUST_NOT_BE_TYPOS = ("canva.com", "lyft.com", "stripe.com", "zylantix.com", "brellium.com", "nimbus.io")


class Checks:
    def __init__(self):
        self.total = 0
        self.failures: list[str] = []

    def expect(self, condition, message: str):
        self.total += 1
        if not condition:
            self.failures.append(message)


def check_structural(c: Checks):
    for domain, expected in EXPECTED_STRUCTURAL.items():
        actual = get_structural_metrics(domain).score
        c.expect(actual == expected, f"structural({domain}) = {actual}, expected {expected}")

    # The property APVA states outright: score is inversely proportional to
    # length and word count.
    scores = [get_structural_metrics(d).score for d in ("ab.com", "abcdef.com", "abcdefghij.com", "abcdefghijklmnop.com")]
    c.expect(scores == sorted(scores, reverse=True), f"structural score must fall as length rises, got {scores}")

    hyphenated = get_structural_metrics("best-insurance.com").score
    plain = get_structural_metrics("bestinsurance.com").score
    c.expect(hyphenated < plain, f"a hyphen must cost points: {hyphenated} vs {plain}")


def check_phonetic(c: Checks):
    for sld, expected in EXPECTED_PHONETIC.items():
        actual = calculate_phonetic_score(sld).score
        c.expect(actual == expected, f"phonetic({sld}) = {actual}, expected {expected}")

    c.expect(estimate_syllables("zylantix") == 3, "zylantix should count 3 syllables")
    c.expect(estimate_syllables("cloud") == 1, "cloud should count 1 syllable")
    c.expect(estimate_syllables("") == 0, "an empty string has no syllables")

    ambiguous = calculate_phonetic_score("schwartz")
    c.expect(bool(ambiguous.metaphone_alternate),
             "Double Metaphone should return a second reading for 'schwartz'")
    c.expect(calculate_phonetic_score("zqxvlth").score < calculate_phonetic_score("zylantix").score,
             "a consonant pile-up must score below a pronounceable coined name")


def check_sentiment(c: Checks):
    for sld, expected in EXPECTED_SENTIMENT.items():
        actual = calculate_sentiment(sld).score
        c.expect(actual == expected, f"sentiment({sld}) = {actual}, expected {expected}")

    # The word that has to be caught isn't in the commercial lexicon, so this
    # also covers the token-rescue path in scoring/sentiment.py.
    scam = calculate_sentiment("scamloans")
    c.expect("scam" in scam.terms, f"'scam' must survive segmentation, got terms={scam.terms}")
    c.expect(calculate_sentiment("divorcelawyer").score < calculate_sentiment("cloud").score,
             "a negative industry term must score below a neutral one")

    # Absence of signal is reported as absence, not as a neutral score.
    for sld in SENTIMENT_UNAVAILABLE:
        reading = calculate_sentiment(sld)
        c.expect(not reading.available and reading.score is None,
                 f"sentiment({sld}) claims a reading ({reading.score}) it does not have")
        c.expect(reading.notes, f"an unavailable sentiment for {sld} must explain itself")

    # The regression this whole change exists for: three unrelated words must
    # not land on the same number by way of all being unreadable.
    scored = [calculate_sentiment(w).score for w in ("quantum", "forge", "plague")]
    c.expect(len(set(scored)) == 3,
             f"unrelated words must not tie on sentiment, got {scored}")


def check_pronounceable_clusters_are_not_penalised(c: Checks):
    """Consonant runs were counted in letters, so ordinary English was
    charged as unpronounceable.

    stripe, chrome, twitch, netflix, android and strava all took the
    "hard to say" deduction. Two reasons, both structural: a digraph like
    `ch` spells one sound and was counted as two, and a run spanning a
    syllable boundary is not a cluster at all — netflix is net + flix,
    never a "tfl" onset.
    """
    from portfolio_agent.scoring.phonetic import calculate_phonetic_score, is_pronounceable_run

    for sld in ("stripe", "chrome", "twitch", "netflix", "android", "strava",
                "splunk", "scribd", "brightside", "healthstream"):
        notes = calculate_phonetic_score(sld).notes
        c.expect(not any("hard to say" in n for n in notes),
                 f"{sld} is ordinary English and must not be charged as unpronounceable: {notes}")

    # And the check must still have teeth.
    for sld in ("zqxvlth", "xkcdz", "pfzthk"):
        notes = calculate_phonetic_score(sld).notes
        c.expect(any("hard to say" in n for n in notes),
                 f"{sld} IS hard to say and must still be penalised: {notes}")

    # A cluster is sayable if a syllable can start with it, or if it splits
    # into an ending plus a beginning.
    for run in ("str", "chr", "tch", "ght", "ndr", "lthstr", "rch"):
        c.expect(is_pronounceable_run(run), f"{run!r} is sayable English")
    for run in ("zqxvl", "xkcd", "vlth", "zxcv", "pfzt"):
        c.expect(not is_pronounceable_run(run), f"{run!r} is not sayable and must not pass")


def check_weights_follow_the_category(c: Checks):
    """One fixed weighting scored every domain as though the same things
    mattered to every buyer.

    Pronounceability was worth 25% to a coined startup name and 25% to an
    exact-match keyword domain. It is not: a brandable lives on surviving
    being said aloud, and nobody declined carinsurancequotes.com for being a
    mouthful. The buyer differs, so what is being paid for differs.
    """
    # The same domain, judged by two categories' criteria, must differ.
    brandable = calculate_brand_score("zylantix.com", category="brandable")
    generic = calculate_brand_score("zylantix.com", category="generic")
    c.expect(brandable.total != generic.total,
             f"category must change the composite: both scored {brandable.total}")
    c.expect(brandable.weights["phonetic"] > generic.weights["phonetic"],
             "a coined name must be judged on sound more than a dictionary word is")

    # A long keyword string is punished by brandable criteria and forgiven
    # by the criteria it is actually bought under.
    as_emd = calculate_brand_score("carinsurancequotes.com", category="emd")
    as_brandable = calculate_brand_score("carinsurancequotes.com", category="brandable")
    c.expect(as_emd.total != as_brandable.total,
             "an EMD and a brandable reading of the same string must not agree")

    # Every category in the taxonomy has weights, and they sum to 1.
    for name, table in config.BRAND_SCORE_WEIGHTS_BY_CATEGORY.items():
        c.expect(set(table) == set(config.BRAND_SCORE_WEIGHTS),
                 f"{name} weights cover different components than the default")
        c.expect(abs(sum(table.values()) - 1.0) < 1e-9,
                 f"{name} weights sum to {sum(table.values())}, not 1")

    # An unknown category falls back rather than guessing at an emphasis.
    unknown = calculate_brand_score("zylantix.com", category="not-a-category")
    flat = calculate_brand_score("zylantix.com")
    c.expect(unknown.total == flat.total,
             "an unrecognised category must fall back to the flat weights")
    c.expect(unknown.weighted_for is None and flat.weighted_for is None,
             "and must not claim to have been weighted for anything")

    # The score says which emphasis produced it, for the same reason it
    # publishes the weights at all.
    c.expect(brandable.weighted_for == "brandable",
             f"weighted_for = {brandable.weighted_for!r}, expected 'brandable'")

    # Still recomputable by hand from what it publishes.
    recomputed = (sum(brandable.components[k] * brandable.weights[k] for k in brandable.components)
                  / sum(brandable.weights.values()))
    c.expect(abs(recomputed - brandable.total) < 0.01,
             f"category-weighted total {brandable.total} is not its own weighted average "
             f"({recomputed})")


def check_an_axis_that_does_not_apply_is_excluded(c: Checks):
    """163.ai scored 0/100 for pronounceability.

    The scorer knew it had nothing to measure — its own note said "no
    pronounceable characters" — and published a zero anyway. That dragged
    the composite to 30 and then deflated the end-user tier a further 20%
    for being hard to say. A numeric name is not a badly-chosen word; it is
    a different kind of asset, and zero is not a low score but no score.
    Same rule as sentiment: exclude and renormalise.
    """
    from portfolio_agent.scoring.phonetic import calculate_phonetic_score

    numeric = calculate_phonetic_score("163")
    c.expect(not numeric.available, "a name with no letters cannot be scored on how it sounds")
    c.expect(numeric.notes and "excluded" in numeric.notes[0],
             f"and must say so rather than report a failure: {numeric.notes}")

    word = calculate_phonetic_score("hop")
    c.expect(word.available and word.score > 0, "a pronounceable name is still scored")

    score = calculate_brand_score("163.ai", category="emerging_tld")
    c.expect("phonetic" not in score.components,
             f"phonetics must be excluded, not scored zero: {sorted(score.components)}")
    c.expect(set(score.weights) == set(score.components),
             "and the weights must renormalise over what remains")
    recomputed = (sum(score.components[k] * score.weights[k] for k in score.components)
                  / sum(score.weights.values()))
    c.expect(abs(recomputed - score.total) < 0.01,
             f"total {score.total} is not the weighted average of its remaining parts")

    # A zero would have been far worse than an exclusion.
    c.expect(score.total > 45,
             f"excluding an inapplicable axis must not leave the composite at a "
             f"pronounceability-driven low: {score.total}")


def check_a_short_word_is_not_an_acronym(c: Checks):
    """hop.xyz was priced as a three-letter acronym.

    The rule's own comment said "a short all-alpha string that isn't a
    word", but it checked SEGMENTABLE — the COMMERCIAL lexicon, which knows
    what sells rather than what English contains. "hop" is a word that
    sells nothing, so it passed and was valued on acronym rarity and
    liquidity: a different asset with a different buyer.
    """
    from portfolio_agent.tools.linguistics import classify_domain

    for domain in ("hop.xyz", "cat.ai", "sun.com"):
        category = classify_domain(domain).category
        c.expect(category != "acronym",
                 f"{domain} is a dictionary word, not an acronym — got {category}")

    # And the rule must keep working for actual acronyms.
    for domain in ("abc.io", "xyz.com", "qqq.io"):
        category = classify_domain(domain).category
        c.expect(category == "acronym",
                 f"{domain} IS a short non-word and must stay an acronym — got {category}")


def check_memorability(c: Checks):
    real_word = calculate_memorability(
        get_structural_metrics("cloud.ai"), calculate_phonetic_score("cloud")
    )
    coined = calculate_memorability(
        get_structural_metrics("quorvexis.com"), calculate_phonetic_score("quorvexis")
    )
    c.expect(real_word.score > coined.score,
             f"a real word must out-recall a coined string: {real_word.score} vs {coined.score}")
    c.expect(not real_word.used_branded_search_volume,
             "memorability must report that it used the proxy when no volume was supplied")

    with_volume = calculate_memorability(
        get_structural_metrics("quorvexis.com"), calculate_phonetic_score("quorvexis"),
        branded_search_volume=200_000,
    )
    c.expect(with_volume.used_branded_search_volume and with_volume.score > coined.score,
             "real branded search volume must override the proxy upward")


def check_spelling_module(c: Checks):
    for domain in MUST_NOT_BE_TYPOS:
        report = check_spelling(domain)
        c.expect(not report.is_typo,
                 f"{domain} must not be read as a misspelling (got {report.corrected_form}, {report.severity})")

    for misspelled, correct in TYPO_REGRESSION_PAIRS:
        report = check_spelling(misspelled)
        c.expect(report.is_typo, f"{misspelled} should be flagged as a misspelling")
        c.expect(report.corrected_form == correct,
                 f"{misspelled} should correct to {correct}, got {report.corrected_form}")
        c.expect(report.deflator < 0.1,
                 f"{misspelled} should carry a heavy deflator, got {report.deflator}")
        c.expect(check_spelling(correct).deflator == 1.0,
                 f"the correctly spelled {correct} must carry no deflator")

    # A transposition is one keystroke, so it must grade as severely as a
    # substitution — this is the most common real typo class.
    c.expect(check_spelling("insruance.com").severity == "severe",
             "a transposition of a commercial term must grade 'severe'")

    # Severity is calibrated to APVA §4.4: ~98% off the comps-implied price.
    c.expect(config.TYPO_DEFLATORS["severe"] == 0.02,
             f"severe deflator should be 0.02 per APVA §4.4, got {config.TYPO_DEFLATORS['severe']}")


def check_brand_score(c: Checks):
    for domain, expected in EXPECTED_BRAND_TOTALS.items():
        actual = calculate_brand_score(domain).total
        c.expect(actual == expected, f"brand_score({domain}) = {actual}, expected {expected}")

    for domain, expected in EXPECTED_COMPONENTS.items():
        actual = calculate_brand_score(domain).components
        c.expect(actual == expected, f"components({domain}) = {actual}, expected {expected}")

    # The composite must be reproducible by hand from what it publishes.
    score = calculate_brand_score("zenbusiness.com")
    recomputed = sum(score.components[k] * score.weights[k] for k in score.components) / sum(score.weights.values())
    c.expect(abs(recomputed - score.total) < 0.01,
             f"total {score.total} is not the weighted average of its published parts ({recomputed})")
    # The published weights are the ones actually applied, so they cover
    # exactly the components present and carry their configured values. An
    # excluded component must take its weight with it rather than leave a
    # zero that still divides.
    c.expect(set(score.weights) == set(score.components),
             f"weights {sorted(score.weights)} do not match components {sorted(score.components)}")
    for name, weight in score.weights.items():
        c.expect(weight == float(config.BRAND_SCORE_WEIGHTS[name]),
                 f"weight for {name} is {weight}, not the configured {config.BRAND_SCORE_WEIGHTS[name]}")

    # And a score built from a reduced basis is still a correct weighted
    # average of that basis.
    partial = calculate_brand_score("zylantix.com")
    c.expect("sentiment" not in partial.components,
             "zylantix has no readable sentiment and must not be scored for it")
    recomputed = (sum(partial.components[k] * partial.weights[k] for k in partial.components)
                  / sum(partial.weights.values()))
    c.expect(abs(recomputed - partial.total) < 0.01,
             f"renormalized total {partial.total} is not the weighted average of its parts ({recomputed})")


def check_regression_pairs(c: Checks):
    """High-brand class stays high; misspelled class collapses."""
    for domain in HIGH_BRAND_CLASS:
        score = calculate_brand_score(domain)
        c.expect(score.total >= 80.0, f"{domain} should score >= 80 for brand potential, got {score.total}")
        c.expect(score.typo_deflator == 1.0, f"{domain} must carry no typo deflator, got {score.typo_deflator}")

    for domain in DEFLATED_CLASS:
        score = calculate_brand_score(domain)
        c.expect(score.typo_deflator <= 0.08,
                 f"{domain} should carry a heavy typo deflator, got {score.typo_deflator}")

    worst_high = min(calculate_brand_score(d).total for d in HIGH_BRAND_CLASS)
    best_deflated = max(
        calculate_brand_score(d).total * calculate_brand_score(d).typo_deflator for d in DEFLATED_CLASS
    )
    c.expect(best_deflated < worst_high,
             f"deflated class ({best_deflated}) must fall below the high class ({worst_high})")


def check_reproducibility(c: Checks):
    """Byte-identical across runs — the property that makes every expected
    value above meaningful."""
    for domain in ("cloud.ai", "propertymanagment.com", "zenbusiness.com"):
        first = calculate_brand_score(domain).model_dump()
        second = calculate_brand_score(domain).model_dump()
        c.expect(first == second, f"brand score for {domain} is not reproducible")
        c.expect(brand_score_detail(domain) == brand_score_detail(domain),
                 f"brand score detail for {domain} is not reproducible")


def main() -> int:
    c = Checks()
    for check in (
        check_structural,
        check_phonetic,
        check_sentiment,
        check_pronounceable_clusters_are_not_penalised,
        check_weights_follow_the_category,
        check_an_axis_that_does_not_apply_is_excluded,
        check_a_short_word_is_not_an_acronym,
        check_memorability,
        check_spelling_module,
        check_brand_score,
        check_regression_pairs,
        check_reproducibility,
    ):
        check(c)

    if c.failures:
        print(f"FAILED {len(c.failures)}/{c.total} checks:")
        for failure in c.failures:
            print(f"  - {failure}")
        return 1
    print(f"PASSED {c.total}/{c.total} checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
