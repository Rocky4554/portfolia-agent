"""
The regression set for classification and evidence routing.

Not a price-accuracy eval — that needs live services, real keys and a far
larger sample. This checks the part that is pure logic and must never
silently break, because everything downstream is routed by it: get the
category wrong and the domain runs the wrong evidence plan, which no amount
of good tier math afterwards can recover from.

At least three domains per category (APVA Table 3.2.1's taxonomy plus
`combination`), so a rule that overfits to one example fails here.

`expected_typo_of` is asserted independently of the category on purpose: a
misspelling must be *detected* without *becoming* the category — see
MIGRATION_NOTES.md.
"""

GOLDEN_SET = [
    # ── generic: single common words. Scarcity, trust, type-in traffic. ────
    {"domain": "insurance.com", "expected_category": "generic"},
    {"domain": "hotels.com", "expected_category": "generic"},
    {"domain": "coffee.com", "expected_category": "generic"},
    # Real dictionary words that are NOT in the curated commercial lexicon.
    # These used to classify as "coined or invented name" — voice.com is
    # recorded in knowledge/ma_cases.py as a $30M sale, so that was not a
    # cosmetic error. apple.com additionally had to survive being read as a
    # two-edit misspelling of "app".
    {"domain": "voice.com", "expected_category": "generic"},
    {"domain": "apple.com", "expected_category": "generic"},
    # ── brandable: coined, no keyword to search on. ────────────────────────
    {"domain": "zylantix.com", "expected_category": "brandable"},
    {"domain": "quorvexis.com", "expected_category": "brandable"},
    {"domain": "brellium.com", "expected_category": "brandable"},
    # ── keyword: a commercial term that isn't a top-tier generic. ──────────
    {"domain": "consulting.com", "expected_category": "keyword"},
    {"domain": "analytics.com", "expected_category": "keyword"},
    {"domain": "managment.com", "expected_category": "keyword", "expected_typo_of": "management.com"},
    # ── emd: exactly matches a commercial search phrase. ───────────────────
    {"domain": "carinsurance.com", "expected_category": "emd"},
    {"domain": "cheapinsurancequotes.com", "expected_category": "emd"},
    {"domain": "bestquotes.ai", "expected_category": "emd"},
    # The APVA §4.4 case: a keyword domain that happens to be misspelled.
    # It must keep its EMD evidence plan and be deflated on price instead.
    {"domain": "propertymanagment.com", "expected_category": "emd", "expected_typo_of": "propertymanagement.com"},
    # ── acronym: rarity and liquidity; length is the market. ───────────────
    {"domain": "xyz.com", "expected_category": "acronym"},
    {"domain": "qzr.com", "expected_category": "acronym"},
    {"domain": "abc.io", "expected_category": "acronym"},
    # ── geo: a location term changes the buyer persona outright. ───────────
    {"domain": "austinplumbing.com", "expected_category": "geo"},
    {"domain": "miamirealestate.com", "expected_category": "geo"},
    {"domain": "nyclawyers.com", "expected_category": "geo"},
    # ── emerging_tld: the TLD is the thesis, and the risk. ─────────────────
    {"domain": "cloud.ai", "expected_category": "emerging_tld"},
    {"domain": "onepay.ai", "expected_category": "emerging_tld"},
    {"domain": "data.io", "expected_category": "emerging_tld"},
    # Greedy segmentation used to split this into iso|top|e, because "top" is
    # in the commercial lexicon, making a single-word domain read as a
    # three-word compound.
    {"domain": "isotope.co", "expected_category": "emerging_tld"},
    # ── combination: brand fused to a keyword (APVA §3.2's ZenBusiness). ───
    {"domain": "zenbusiness.com", "expected_category": "combination"},
    {"domain": "novahealth.com", "expected_category": "combination"},
    {"domain": "apexlending.com", "expected_category": "combination"},
]

# Every category the golden set must exercise. Asserted against the playbook
# so adding a category without adding coverage fails the eval.
EXPECTED_CATEGORIES = {
    "generic", "brandable", "keyword", "emd", "acronym", "geo", "emerging_tld", "combination",
}

# Domains whose valuation must collapse relative to their correctly spelled
# sibling — the regression pair behind APVA §4.3's typo requirement.
# Real English words that must never be read as misspellings, split into
# parts, or called coined names — whether or not they are commercial terms.
REAL_WORD_DOMAINS = ("voice.com", "apple.com", "isotope.co", "orange.com", "summit.com")

# Strings with no dictionary meaning, which must stay brandable.
COINED_DOMAINS = ("zylantix.com", "quorvexis.com", "brellium.com")

# Single words that genuinely name a commercial category. The Strategic M&A
# tier is conditional on this, so a false negative costs a tier and a false
# positive invents one.
CATEGORY_DEFINING_DOMAINS = (
    "insurance.com", "hotels.com", "voice.com", "cars.com", "coffee.com", "slots.com",
)

# In the dictionary, and not categories. Proper nouns are the trap: they pass
# every "is this a real word" test, and crusoe.com was priced as a category
# killer on the grounds that "crusoe" names its own category.
NOT_CATEGORY_DEFINING_DOMAINS = (
    "crusoe.com", "everest.com", "tesla.com", "gatsby.com", "quixote.com",
)

TYPO_REGRESSION_PAIRS = [
    ("propertymanagment.com", "propertymanagement.com"),
    ("managment.com", "management.com"),
    ("insuranse.com", "insurance.com"),
]
