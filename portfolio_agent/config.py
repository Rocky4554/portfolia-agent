import os

from dotenv import load_dotenv

load_dotenv()

# ── LLM ────────────────────────────────────────────────────────────────────
# Deliberately NOT here. Everything about which model answers, with which
# credential and at what temperature lives in portfolio_agent/llm.py — the
# providers it knows, the per-job profiles, and which node uses which. This
# file holds the data-service configuration; putting the model settings here
# too meant nobody could find them.
#
#     uv run python -m portfolio_agent.llm_doctor
#
# prints the whole resolved picture and where every value came from.
#
# The two flags below stay here because they are about mock mode, not about
# models: see MOCK_MODE / MOCK_ALLOW_LLM further down.

NAMEBIO_API_KEY = os.environ.get("NAMEBIO_API_KEY", "")
NAMEBIO_BASE_URL = os.environ.get("NAMEBIO_BASE_URL", "https://namebio.com/api")

# ── Internal Namekart data services (valuation tools) ──────────────────────
# All three are ours, not vendors — no API key — but each is rate-limited or
# spends real upstream credits, so the defaults below matter. See
# tools/dvs.py, tools/sales_history.py and tools/comparables.py.

# Domain Valuation Service: cache + rate limiter in front of Estibot, GoDaddy
# GoValue and DotDB. DVS_SYSTEM_ACTOR is who DVS's rate limiter bills us as.
DVS_BASE_URL = os.environ.get("DVS_BASE_URL", "https://dvs.vps4.auctionhacker.com")
DVS_SYSTEM_ACTOR = os.environ.get("DVS_SYSTEM_ACTOR", "portfolio-agent")
# Max age of a cached appraisal before DVS re-fetches it upstream (spending
# credits). Estibot's own default is "any cached row is fresh", which served
# a year-old value in testing — hence an explicit bound here.
DVS_ESTIBOT_MAX_AGE_DAYS = int(os.environ.get("DVS_ESTIBOT_MAX_AGE_DAYS", "30"))
DVS_GDV_MAX_AGE_DAYS = int(os.environ.get("DVS_GDV_MAX_AGE_DAYS", "180"))

# Our Namebio service — an in-house mirror of NameBio's daily sales feed.
# Not the same thing as NAMEBIO_BASE_URL above, which is NameBio's paid API.
NAMEBIO_SERVICE_BASE_URL = os.environ.get("NAMEBIO_SERVICE_BASE_URL", "https://namebio.vps4.auctionhacker.com")

# Comparable AI Agent — semantic "similar sold domains" search.
COMPARABLE_AGENT_BASE_URL = os.environ.get("COMPARABLE_AGENT_BASE_URL", "https://comparable-agent.h.namekart.com")

# "read_only" is the only tier this repo implements (Phase 1 / APVA advisory tools).
# A future "portfolio_manager" tier adds mutating tools (reprice/pause/drop) behind
# their own confirmation flow — see guardrails.py. Reading this from env, not from
# a request parameter, means a deployment can't be talked into a higher tier by
# anything in a conversation.
PORTFOLIO_AGENT_TIER = os.environ.get("PORTFOLIO_AGENT_TIER", "read_only")


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ── Mock mode ──────────────────────────────────────────────────────────────
# Every tool that reaches outside this process answers from a deterministic
# fixture instead (tools/mocks.py) when this is on, so the whole graph runs
# end to end with no API keys, no network and no vendor credits spent — which
# is what makes the e2e evals runnable on any machine and in CI.
#
# Mock results are labeled mock=True all the way into the output's evidence
# items and a warning, so a mock number can never be presented as a real one.
# Off by default: a deployment must opt in to fake data, never drift into it.
MOCK_MODE = _flag("PORTFOLIO_AGENT_MOCK_MODE", False)

# Mock mode covers the LLM too. A model call is an external call — it costs
# money, it needs a key, and it makes a run non-deterministic — so "runs with
# no API keys and no network" has to include it, or an offline eval silently
# bills a live endpoint for every domain in its golden set. With this on and
# no override, get_chat_model() returns None and every LLM-dependent node
# degrades to its documented fallback.
#
# Set PORTFOLIO_AGENT_MOCK_ALLOW_LLM=true to combine fixture data with a real
# model, which is what you want when iterating on a prompt.
MOCK_ALLOW_LLM = _flag("PORTFOLIO_AGENT_MOCK_ALLOW_LLM", False)

# ── HTTP hardening ─────────────────────────────────────────────────────────
# Retries cover the recoverable failures only — timeouts, dropped
# connections, 429s and 5xx. A 4xx is a bad request and retrying it just
# spends the rate limit twice. Backoff is exponential from the base, and the
# base is configurable mainly so the offline evals can drive it to ~0.
HTTP_MAX_ATTEMPTS = int(os.environ.get("PORTFOLIO_AGENT_HTTP_MAX_ATTEMPTS", "3"))
HTTP_BACKOFF_BASE_S = float(os.environ.get("PORTFOLIO_AGENT_HTTP_BACKOFF_BASE_S", "0.25"))


# ── Brand scoring (APVA §4.3) ──────────────────────────────────────────────
# The four components calculate_brand_score averages, with the weights it
# averages them by. Here rather than hardcoded in scoring/brand_score.py
# because these are a calibration decision, not a code decision: they should
# be tuned against real sale outcomes, and tuning them must not require a
# code review. Keys must match the component names scoring/ produces.
#
# Weighted toward structure and sound because those are the two APVA names as
# the primary drivers for a brandable ("inversely proportional to character
# length and word count", "Soundex and Metaphone"), and because they are the
# two we measure directly rather than proxy.
BRAND_SCORE_WEIGHTS = {
    "structural": 0.30,  # simplicity & brevity
    "phonetic": 0.25,  # pronounceability
    "memorability": 0.25,  # recall proxy
    "sentiment": 0.20,  # keyword sentiment
}

# The same four components, weighted by what the category is actually bought
# for. One fixed set of weights says pronounceability matters equally to a
# coined startup name and to an exact-match keyword domain, and it does not:
# a brandable lives or dies on being sayable over a phone call, while nobody
# ever declined to buy carinsurancequotes.com because it is a mouthful. The
# buyer is different, so the thing being paid for is different.
#
# Weights within a category sum to 1. A component that turns out to be
# unavailable is excluded and the rest renormalise (scoring/brand_score.py),
# so these are the intended proportions rather than a guaranteed divisor.
BRAND_SCORE_WEIGHTS_BY_CATEGORY = {
    # Coined names. Sound is the asset — it has to survive being said aloud.
    "brandable": {"structural": 0.20, "phonetic": 0.40, "memorability": 0.25, "sentiment": 0.15},
    # A dictionary word is already memorable and already pronounceable;
    # what varies is how short and clean it is.
    "generic": {"structural": 0.35, "phonetic": 0.05, "memorability": 0.40, "sentiment": 0.20},
    # Sold to a local business. Recall and brevity carry it.
    "geo": {"structural": 0.30, "phonetic": 0.10, "memorability": 0.40, "sentiment": 0.20},
    # Bought for the commercial term in it, so connotation matters more and
    # phonetics barely register.
    "keyword": {"structural": 0.30, "phonetic": 0.05, "memorability": 0.35, "sentiment": 0.30},
    # An exact-match domain is bought for the search phrase itself.
    "emd": {"structural": 0.30, "phonetic": 0.05, "memorability": 0.30, "sentiment": 0.35},
    # Letters. Brevity is nearly the whole asset, and a string of letters
    # carries almost no keyword connotation to read.
    "acronym": {"structural": 0.50, "phonetic": 0.30, "memorability": 0.15, "sentiment": 0.05},
    # Startup-brandable in practice, so it inherits brandable's shape.
    "emerging_tld": {"structural": 0.20, "phonetic": 0.40, "memorability": 0.25, "sentiment": 0.15},
    # A brand element fused to a keyword sits between the two.
    "combination": {"structural": 0.25, "phonetic": 0.25, "memorability": 0.30, "sentiment": 0.20},
}

# Multiplicative deflators applied to the WHOLE valuation when the domain is
# a misspelling — not to the brand score alone, because a typo does not make
# a name badly structured, it makes it commercially close to worthless.
#
# "severe" is calibrated to APVA §4.4's worked example: propertymanagment.com
# is deflated ~98% from the base model's price, leaving only what the domain
# is genuinely worth (defensive registration, typo-traffic redirection).
TYPO_DEFLATORS = {
    "severe": 0.02,  # one edit from a high-value commercial term
    "moderate": 0.08,  # two edits — the intended word is still recognizable
    "mild": 0.25,  # a near-miss on a low-value or very short word
}

# A misspelling this far from any dictionary word is a coined name, not an
# error. "zylantix" is not a typo of anything.
TYPO_MAX_DISTANCE = 2


# ── Market trend search (APVA §3.3 / Table 7.1.1) ──────────────────────────
# Corroborating links only. The trend VERDICT is computed from the curated
# lists in knowledge/market_trends.py and never from a search result — see
# tools/trends.py. Either backend works; Jina is tried first when both are
# set. With neither, the verdict still stands and reports that it has no
# sources.
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
JINA_SEARCH_URL = os.environ.get("JINA_SEARCH_URL", "https://s.jina.ai")
GOOGLE_CSE_KEY = os.environ.get("GOOGLE_CSE_KEY", "")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID", "")

# ── SEO & traffic (Ahrefs-shaped; Table 7.1.1) ─────────────────────────────
AHREFS_API_KEY = os.environ.get("AHREFS_API_KEY", "")
AHREFS_BASE_URL = os.environ.get("AHREFS_BASE_URL", "https://api.ahrefs.com")


# ── Tier engine (APVA Table 4.1.1) ─────────────────────────────────────────
# Every constant the four market tiers are computed from. In config, not in
# pricing/tier_engine.py, because these are calibration against real sale
# outcomes — the kind of number that should change on evidence, without a
# code review, and be diffable when it does.

# Marketplaces that are genuinely competitive auctions. The Auction tier is
# anchored on these alone: a Sedo brokered sale and a GoDaddy expiry auction
# are different markets, and mixing them pulls the auction anchor toward
# negotiated end-user prices it should sit well below.
AUCTION_VENUES = {"godaddy", "namejet", "dropcatch", "dynadot", "park.io", "snapnames"}

# The auction tier is anchored on comps that should be auction clearings.
# When too few auction-venue sales exist to filter on, the anchor is built
# from brokered and retail sales instead — which clear ABOVE auction — so
# the anchor is biased high by a known amount and in a known direction.
#
# This used to be recorded as a deflationary factor with no multiplier
# attached, which is a silent no-op: the reader saw a caveat and the price
# ignored it. The haircut scales with the venue mix, from no reduction when
# every comp is an auction sale to AUCTION_RETAIL_HAIRCUT when none are.
AUCTION_RETAIL_HAIRCUT = 0.55

# Floor = auction × this. A quick wholesale exit gives up roughly half of
# what patience would earn.
FLOOR_DISCOUNT_FROM_AUCTION = 0.45
# Liquidity swings the floor either side of that discount: a 3-letter .com
# clears at close to auction, an awkward brandable needs a deeper cut.
FLOOR_LIQUIDITY_RANGE = (0.80, 1.20)
# Stands in for the real acquisition cost until a portfolio record supplies
# it. A floor price below what the domain cost is not a floor.
ACQUISITION_COST_PLACEHOLDER = 50

# Trend velocity (-1..1) scales each tier by at most this fraction. Auction
# moves most — investor demand is what a trend actually shifts. End-user
# demand is slower and moves less.
AUCTION_TREND_SENSITIVITY = 0.30
END_USER_TREND_SENSITIVITY = 0.15

# Brand score 0-100 maps linearly onto this multiplier for the End-User tier.
END_USER_BRAND_RANGE = (0.70, 1.35)
# .com is the premium an end user pays for (Table 4.1.1). Emerging TLDs sell
# to startups at par; everything else takes a discount.
END_USER_TLD_PREMIUM = {"com": 1.15, "emerging": 1.00, "other": 0.75}
# Existing SEO authority is a developed asset an end user can use on day one.
END_USER_SEO_MULTIPLIER = {"established": 1.25, "developing": 1.10, "undeveloped": 1.00, "unknown": 1.00}
# Keyword search volume, from no measurable demand to a major commercial term.
END_USER_KEYWORD_RANGE = (1.00, 1.30)
END_USER_KEYWORD_VOLUME_FOR_MAX = 250_000
# Worst-case multipliers for the two deflators Table 4.1.1 names for this
# tier. Spelling is handled separately and globally by TYPO_DEFLATORS.
END_USER_SENTIMENT_FLOOR = 0.65
END_USER_PHONETIC_FLOOR = 0.80

# List/BIN = end-user × a scarcity and industry-growth multiplier.
LIST_BIN_SCARCITY_RANGE = (1.10, 1.60)
# Sale probability against markup over the end-user price. Interpolated
# linearly between these points and clamped outside them. Calibrate against
# real marketplace outcomes — this is a shape, not a measurement.
LIST_BIN_PROBABILITY_CURVE = [
    (1.00, 0.35),
    (1.25, 0.22),
    (1.50, 0.14),
    (2.00, 0.07),
    (3.00, 0.03),
]

# How wide a tier's low/high band may be relative to its point price, when
# the comps distribution implies something wider or narrower than is useful.
TIER_BAND_LIMITS = (0.45, 2.20)

# The percentiles the low/high band is taken from. p25/p75 is the
# interquartile range and contains only half the distribution, which is the
# wrong promise for something labelled "low to high" — a band a buyer reads
# as the likely outcome should cover most of it, not 50%.
BAND_PERCENTILES = (0.10, 0.90)

# Below this many comparable sales, percentiles are noise and the market
# tiers are left unfilled rather than computed from an anecdote.
MIN_COMPS_FOR_TIERS = 3

# A category-defining domain is re-anchored on comparable ACQUISITIONS
# rather than on keyword comps, when at least this many exist for its shape.
#
# Why this is not a multiplier: a substring search for a category killer
# finds other names CONTAINING the word, which is a different and far
# cheaper market. insurance.com priced from autoinsurance.net comes out near
# $9,000 against a real $35.6M sale. The fix is not to inflate the wrong
# comps, it is to use the right ones — the handful of one-word .com sales in
# knowledge/ma_cases.py, read exactly as any other comps distribution.
#
# Deliberately narrow. This only reaches single-token .com domains that pass
# the common-noun frequency floor, which is the class that genuinely trades
# in the millions.
CATEGORY_KILLER_MIN_CASES = 3

# When an independent appraisal disagrees with the comps-derived price by
# this factor or more, say so loudly. It does NOT change the price — comps
# remain the anchor, per the spec — it makes a disagreement visible instead
# of silent.
#
# Ten is deliberately wide. Appraisals routinely differ by 2-3x and that is
# ordinary disagreement about an illiquid asset; an order of magnitude means
# the two are describing different markets. That is exactly what happens on
# a category-defining domain: comps find other names CONTAINING the word,
# while the appraiser is valuing the word itself. insurance.com priced from
# substring comps comes out near $9,000 against a real $35.6M sale.
APPRAISAL_DISAGREEMENT_FACTOR = 10.0

# No reported sale probability may reach 1.0. A liquidity adjustment on a
# high baseline probability can arithmetically exceed 1 and clamp there, and
# "100% chance of sale" is a claim about the future that no evidence
# supports — there is always a price nobody pays and a market that stops
# clearing. Capping below 1 keeps the number an estimate rather than a
# promise.
MAX_REPORTED_SALE_PROBABILITY = 0.95


# ── Targeted valuation (APVA Table 4.1.1, bottom two rows) ─────────────────
# A lead must clear this to count as "a specific, funded company". Below it,
# the company cannot write the cheque that justifies the tier, and the tier
# stays None — the spec's "no clear, high-value end-user identified".
MIN_LEAD_FUNDING = 1_000_000

# Funding bands → the multiplier on the End-User price. A better-funded buyer
# is not merely able to pay more; a premium domain is a smaller fraction of
# their raise, which is what actually moves a negotiation.
LEAD_FUNDING_BANDS = [
    (1_000_000, 1.15),
    (10_000_000, 1.40),
    (50_000_000, 1.75),
    (250_000_000, 2.20),
]

# Scales within the band by how much weaker the lead's current domain is —
# APVA's "domain solves a credibility problem". A funded company already on a
# strong .com has no gap to close and pays nothing extra.
CREDIBILITY_GAP_RANGE = (1.00, 1.30)

# Corpus frequency, per million words, above which a dictionary word counts
# as a common noun rather than a proper noun. Both are "in the dictionary",
# and only one of them can name a product category — without this,
# crusoe.com was priced as a category-defining domain on the grounds that
# "crusoe" names its own category.
#
# Calibrated on 37 category terms against 22 proper nouns: at 2.0 the
# category terms keep 36 and the proper nouns admit none. Raise it to be
# stricter about what reaches the Strategic M&A tier; lower it and names
# start qualifying.
CATEGORY_TERM_MIN_FREQUENCY_PPM = float(
    os.environ.get("PORTFOLIO_AGENT_CATEGORY_TERM_MIN_FREQUENCY_PPM", "2.0")
)

# Strategic M&A sits above the targeted lead price. This is the widest and
# least certain multiplier in the system: an M&A price is set by the
# acquirer's strategy, not by the domain market, so it is published as a
# broad range and only for genuinely category-defining names.
STRATEGIC_MA_MULTIPLIER = 2.5

# Bringing the category-defining comp set down to the level the market tiers
# describe.
#
# The four market tiers price a sale between a domain investor and a buyer.
# The category-defining comp set prices something else entirely: completed
# acquisitions in which the domain became a company's identity. Same names,
# different rung. Running the market-tier arithmetic over them unchanged
# takes p75 of a set of strategic outcomes and calls it an end-user price,
# which applies the end-user premium a second time — hello.com came out at
# $18.1M because the set's p75 is $18.0M.
#
# The divisor is not chosen, it is the engine's own chain read backwards:
# strategic = targeted x STRATEGIC_MA_MULTIPLIER, and targeted = end_user x
# a funding band x a credibility gap. Taking the median funding band and the
# midpoint of the credibility range gives the typical distance between an
# end-user price and a strategic one, and that is the distance to remove.
_MEDIAN_FUNDING_BAND = 1.575  # median of LEAD_FUNDING_BANDS' multipliers
_MID_CREDIBILITY_GAP = sum(CREDIBILITY_GAP_RANGE) / 2
STRATEGIC_TO_MARKET_DIVISOR = round(
    _MEDIAN_FUNDING_BAND * _MID_CREDIBILITY_GAP * STRATEGIC_MA_MULTIPLIER, 2)
STRATEGIC_MA_BAND = (0.6, 2.0)

# How much wider the strategic band gets when no acquirer has been
# identified. The point does not move: not knowing who would buy a
# category-defining domain makes the sale less likely, not the domain less
# valuable. What it costs is precision, so that is what is given up.
STRATEGIC_MA_NO_BUYER_WIDENING = 1.5

# Crunchbase (Table 7.1.1). No LinkedIn integration — the spec marks that
# row "High Risk" and it would mean scraping personal profiles.
CRUNCHBASE_API_KEY = os.environ.get("CRUNCHBASE_API_KEY", "")
CRUNCHBASE_BASE_URL = os.environ.get("CRUNCHBASE_BASE_URL", "https://api.crunchbase.com")

# ── Rationale generation ───────────────────────────────────────────────────
# The narrative is regenerated at most this many times when its numbers do
# not match the structured output, then falls back to the template. One
# retry: a model that invents a figure twice will invent it a third time, and
# the templated rationale is always correct.
RATIONALE_MAX_ATTEMPTS = 2

# End-to-end latency budget, seconds. Exceeding it is reported as a warning,
# not an error — a slow valuation is still a valuation.
LATENCY_BUDGET_S = float(os.environ.get("PORTFOLIO_AGENT_LATENCY_BUDGET_S", "60"))


# ── Funding collector (portfolio_agent/funding/) ───────────────────────────
# A scheduled news collector that supplies the one fact nothing else in this
# repo can: whether an identified company can write a cheque. Runs OUTSIDE
# the valuation graph — a valuation must not depend on a news site being up.
FUNDING_FEEDS = [
    f.strip()
    for f in os.environ.get(
        "PORTFOLIO_AGENT_FUNDING_FEEDS",
        "https://techcrunch.com/category/venture/feed/,"
        "https://techcrunch.com/category/startups/feed/,"
        "https://techcrunch.com/feed/",
    ).split(",")
    if f.strip()
]

# Postgres. Without it the collector writes newline-delimited JSON to
# FUNDING_STORE_PATH instead, so the pipeline is runnable before any database
# exists and the evals never need one.
FUNDING_DB_URL = os.environ.get("PORTFOLIO_AGENT_FUNDING_DB_URL", "")
FUNDING_STORE_PATH = os.environ.get(
    "PORTFOLIO_AGENT_FUNDING_STORE", "./data/funding/funding_events.jsonl")

# Hosts that are never the funded company: the publisher itself, social
# networks, and the infrastructure every page links to. Used when resolving a
# company's own URL from an article.
# A funding round older than this says little about whether the money is
# still there to spend. Roughly the window in which a raise is still
# "recent" to a founder deciding what to spend it on.
FUNDING_LEAD_MAX_AGE_DAYS = int(os.environ.get("PORTFOLIO_AGENT_FUNDING_LEAD_MAX_AGE_DAYS", "540"))

FUNDING_IGNORED_HOSTS = {
    "techcrunch.com", "tc.com", "strictlyvc.com", "yahoo.com", "google.com", "gstatic.com",
    "twitter.com", "x.com", "facebook.com", "linkedin.com", "instagram.com",
    "youtube.com", "reddit.com", "threads.net", "bsky.app", "mastodon.social",
    "wordpress.org", "wp.com", "gravatar.com", "doubleclick.net",
    "apple.com", "play.google.com", "amazon.com", "crunchbase.com",
    "wikipedia.org", "eventbrite.com", "sched.com",
}


# ── Marketplace listings (APVA Table 4.1.1, List Price BIN) ────────────────
# The one source in Table 4.1.1 that was never wired. A listing is what a
# seller ASKS; a comp is what a buyer PAID. They are different facts and are
# kept in different inputs — see pricing/tier_engine.py, which never lets an
# ask into the comps distribution.
#
# Each marketplace controls its own inventory and its own API access, so
# there is no single source for live asks. Providers are configured
# independently and any subset may be absent.

# Name.ai — our own marketplace. Listing prices need an authenticated token
# with the pricing:read scope.
# name.ai's own marketplace. The v1 API lives under /api/v1 on the app
# itself rather than a separate api. host, and authenticates with a key AND
# a secret as two headers.
NAMEAI_API_BASE_URL = os.environ.get("NAMEAI_API_BASE_URL", "https://name.ai")
NAMEAI_API_KEY = os.environ.get("NAMEAI_API_KEY", "").strip()
NAMEAI_API_SECRET = os.environ.get("NAMEAI_API_SECRET", "").strip()

# name.ai's marketplace commission, as a percentage of the SALE price.
# Deducted from it, never added to it: their orders.js is explicit that
# `seller_payout = sale_price - marketplace_commission`, so a $10,000 BIN
# nets $9,000 at 10%. The v1 API does not return the rate, so this is a
# default and anything computed from it is labelled as assumed.
NAMEAI_COMMISSION_PCT = float(os.environ.get("NAMEAI_COMMISSION_PCT", "10.0"))

# GoDaddy Auctions. Aftermarket listing availability and current bids.
GODADDY_API_BASE_URL = os.environ.get("GODADDY_API_BASE_URL", "https://api.godaddy.com")
GODADDY_API_KEY = os.environ.get("GODADDY_API_KEY", "")
GODADDY_API_SECRET = os.environ.get("GODADDY_API_SECRET", "")
GODADDY_CUSTOMER_ID = os.environ.get("GODADDY_CUSTOMER_ID", "MY")

# A listing older than this says little about current demand — an ask that
# has sat unsold for years is evidence about the seller, not the market.
LISTING_MAX_AGE_DAYS = int(os.environ.get("PORTFOLIO_AGENT_LISTING_MAX_AGE_DAYS", "540"))

# How far the BIN tier may be pulled toward a live asking price. Capped
# because an ask is not a transaction: consulting.ai is listed at ~$2.8M
# with 506 searches in a year, which is a seller's hope rather than a
# market. The tier moves, it does not surrender.
LISTING_BIN_PULL = float(os.environ.get("PORTFOLIO_AGENT_LISTING_BIN_PULL", "0.35"))

# How a seller's ask compares to what transactions support. Informational —
# no tier arithmetic reads these beyond the BIN ceiling. A label exists
# because "4.8x" means nothing to a reader deciding whether to list, and
# "ambitious" does.
ASK_GAP_BUCKETS = (
    (3.0, "aligned — the ask is within reach of what comparable sales support"),
    (20.0, "ambitious — the ask is well above comparable sales but not detached from them"),
    (float("inf"), "conviction pricing — the ask reflects a seller's belief rather than any "
                   "transaction, and domains priced this way routinely sit unsold for years"),
)

# How a single reference appraisal is chosen when several vendors answer.
# Median rather than min or max: one broken vendor should not unilaterally
# define the conflict. Estibot valued a clean single-word .io at $630 while
# GoValue said $12,565 — letting either drive it alone produces a number
# that is mostly about that vendor.
APPRAISAL_PRIMARY_STRATEGY = "median"

# Report a live ask that differs from our BIN price by this factor or more.
# Same principle as the appraisal cross-check: surface the disagreement
# rather than silently picking a winner.
LISTING_DISAGREEMENT_FACTOR = 5.0


# ── Comp ladder (WS1) ──────────────────────────────────────────────────────
# consulting.ai found one comparable sale and priced nothing. One sale is an
# anecdote, but "no answer" is not the only alternative to "a confident
# answer" — a human appraiser widens the search and says how far they had to
# go. The ladder does that explicitly, and carries the distance in
# comp_basis so the reader knows what kind of claim they are holding.

# Comps needed before a rung is considered sufficient. Below this the ladder
# descends to a weaker but broader basis.
COMP_LADDER_MIN_COMPS = 3

# A sale older than this stopped describing the current market.
COMP_MAX_AGE_MONTHS = 36

# TLDs searched when looking for this exact SLD sold elsewhere.
CROSS_TLD_SET = ("com", "co", "io", "net", "ai", "org")

# Confidence per rung. These scale tier spreads and sale probabilities, so
# they are the single knob controlling how loudly a thin valuation admits to
# being thin.
COMP_BASIS_CONFIDENCE = {
    "exact": 1.00,
    "sld_cross_tld": 0.60,
    "cohort": 0.45,
    "keyword_only": 0.25,
    # Five named acquisitions of structurally identical domains is real
    # evidence, but it is seventeen famous sales deep and survivorship-biased
    # by construction, so it never claims the confidence of direct sales.
    "category_defining": 0.50,
    "none": 0.00,
}

# An exact comp is worth two of anything else in the distribution; an
# own-book sale is worth one and a half (verified, but venue-specific).
COMP_WEIGHT_EXACT = 2.0
COMP_WEIGHT_INTERNAL = 1.5

# A sale reported to the trade press by the venue that made it. A real
# transaction and an interested witness: below a sale from our own book,
# which we verified, and above anything inferred from a cohort.
COMP_WEIGHT_REPORTED = 1.2

# Sent on every crawl. A crawler that does not say who it is cannot be
# asked to stop, which is the whole reason site owners block anonymous ones.
CRAWLER_USER_AGENT = os.environ.get(
    "CRAWLER_USER_AGENT", "portfolio-agent/1.0 (+domain valuation research)")

# Tier spread widens as confidence falls: spread = base / confidence, capped.
MAX_TIER_SPREAD = 4.0

# Cohort shape constraints.
COHORT_MAX_SLD_LENGTH = 12
COHORT_MIN_PHONETIC_SCORE = 60.0


# ── Dynadot ────────────────────────────────────────────────────────────────
# A second marketplace AND a second appraisal source, and the two are
# genuinely different inventories: consulting.ai is listed on GoDaddy at
# ~$2.8M and is absent from Dynadot's 5.9M listings entirely, while Dynadot
# will appraise it ($63,961) and GoDaddy's own appraiser says $2,220. No
# single provider covers the aftermarket, which is why marketplace.py is a
# registry rather than one integration.
#
# Endpoint paths are configurable because they have NOT been verified against
# the live API — see tools/marketplace.py. Correcting a path should not need
# a code change.
DYNADOT_API_KEY = os.environ.get("DYNADOT_API_KEY", "")
DYNADOT_BASE_URL = os.environ.get("DYNADOT_BASE_URL", "https://api.dynadot.com/restful/v1")
DYNADOT_LISTING_PATH = os.environ.get("DYNADOT_LISTING_PATH", "/market/listings/{domain}")
DYNADOT_APPRAISAL_PATH = os.environ.get("DYNADOT_APPRAISAL_PATH", "/domains/{domain}/appraisal")


# ── Internal comps (WS4) ───────────────────────────────────────────────────
# Our own brokerage book: closed sales, offers received, inquiry counts.
# The most trustworthy comps available — verified, first-party, and covering
# exactly the inventory we broker — and the only ones nobody else has.
INTERNAL_COMPS_URL = os.environ.get("INTERNAL_COMPS_URL", "")
INTERNAL_COMPS_TOKEN = os.environ.get("INTERNAL_COMPS_TOKEN", "")

# Own-book sales are better evidence than any aggregate, so a comp set
# containing enough of them is not honestly described by its weakest rung. A
# set resting partly on verified first-party sales gets at least
# cohort-grade confidence, whatever rung the ladder stopped on.
INTERNAL_COMPS_MIN_FOR_LIFT = 2

# An offer above the floor price is evidence someone will pay more than a
# quick exit would fetch, so it lifts the end-user tier. Capped, because one
# offer is one buyer and a buyer who walked away proves nothing.
INTERNAL_OFFER_MAX_LIFT = 1.25


# ── Length value window ────────────────────────────────────────────────────
# Where an extension's money actually sits, by SLD length. NameBio's .ai
# distribution is stark: $45.2M across 18,800 sales at 10 characters or
# fewer, against $1.8M across 1,345 sales at 11 or more. A name sitting on
# that boundary is not "slightly long" — it is at the edge of the window
# where the market transacts at all.
#
# This is a STRUCTURAL fact and applies to every category. Brand scoring is
# skipped for keyword and EMD domains, which is right — pronounceability and
# recall are brandable concerns — but length is not, and skipping structural
# scoring for them left accounting.ai with no length factor anywhere in its
# valuation.
LENGTH_VALUE_WINDOW = {
    "ai": 10,
    "io": 10,
    "co": 12,
    "com": 14,  # .com sustains longer names; the window is wider, not absent
}
LENGTH_VALUE_WINDOW_DEFAULT = 12

# Multiplier at and beyond the boundary. At the boundary exactly, the full
# discount does not apply — the name is inside the window, just at its edge.
LENGTH_BOUNDARY_FACTOR = 0.92
LENGTH_BEYOND_WINDOW_FACTOR = 0.60

# Cohort members must be structurally comparable, and length is most of what
# "structurally" means. Without this the .ai cohort was identical for
# accounting.ai and consulting.ai — 47 of 47 shared comps — because it was a
# property of the extension rather than of the domain, and three-character
# trophies like fin.ai priced a ten-character keyword.
COHORT_LENGTH_BAND = 4

# Competing qualified buyers are real leverage, but two interested parties
# is not an auction. Capped tightly.
MULTI_LEAD_MAX_LIFT = 1.12


# ── Cohort quality (F7) ────────────────────────────────────────────────────
# The mirror of the trophy guard. accounting.ai needed protecting from comps
# far ABOVE its class; forge.ai needed protecting from comps far below it —
# nine compound names (dataforge.ai $787, codeforge.ai $455) anchoring a
# five-character premium dictionary word. Both are the same error: a comp
# set whose quality does not match the subject's.
#
# Bands are ordered commodity < mid < premium.
SUBJECT_BAND_PREMIUM_MAX_LENGTH = 6
SUBJECT_BAND_PREMIUM_MIN_BRAND = 80.0
SUBJECT_BAND_COMMODITY_MIN_LENGTH = 13
SUBJECT_BAND_COMMODITY_MAX_BRAND = 45.0

# A comp more than one band from the subject is kept but nearly silenced.
# Kept, because dropping it would hide that the retrieval returned it;
# silenced, because it describes a different market.
CROSS_BAND_WEIGHT = 0.15

# A comp selling into a different vertical. Higher than the cross-band weight
# because a different industry is a weaker objection than a different quality
# of name: shape and TLD still carry information about what the market pays,
# where a commodity comp says nothing useful about a premium one.
CROSS_INDUSTRY_WEIGHT = 0.35

# How far two vendor appraisals of the same domain may diverge before their
# median stops being a reference worth arguing against. Estibot and GoValue
# three times apart are not two estimates of one number; they are two models
# that do not agree on what the domain is.
# Below this many vendor baselines, the reconciled appraisal is reported as
# under-covered. Taking a median of one is not reconciliation, and a vendor
# being unreachable must not look the same as a vendor agreeing.
# How readily each category finds a buyer at all — whether it has a standing
# investor bid or needs the one right buyer to walk past. A geo domain and a
# generic word can carry identical prices and completely different odds of
# clearing at them, which is why this belongs in the probability and not only
# in the price. Lived inline in tier_engine until now, against this repo's own
# rule that weights are configuration.
CATEGORY_LIQUIDITY = {
    "acronym": 1.00,
    "generic": 0.90,
    "emerging_tld": 0.70,
    "emd": 0.60,
    "keyword": 0.50,
    "combination": 0.40,
    "brandable": 0.30,
    "geo": 0.30,
}
CATEGORY_LIQUIDITY_DEFAULT = 0.50

# How much of a published sale probability comes from the vendor's own model.
# Using the vendor number alone lets one bad appraisal drive the headline
# figure a seller acts on; ignoring it throws away a model trained on more
# transactions than we will ever see. Weighted toward the vendor because it
# IS better informed, capped short of total because it has been wrong by
# orders of magnitude on exactly the domains that matter.
PROVIDER_PROBABILITY_WEIGHT = 0.75

# The Internet Archive's CDX endpoint measured ~35s per call, repeatedly,
# having first answered with a 429. The valuation budget is 60s and a live
# run takes about twelve, so this is off unless asked for. Its proper home is
# a cached backfill — a domain's 1998 snapshot will not change — rather than
# the hot path of a valuation.
ENABLE_WAYBACK = _flag("PORTFOLIO_AGENT_ENABLE_WAYBACK", False)

# SEC requires a User-Agent naming a real contact address and returns 403
# without one. There is deliberately no default: an address belongs to
# whoever is running this, and baking one in would send a person's email to
# a third party on behalf of every future operator. Unset means the tool
# reports itself unavailable, which is the honest state — the same as a
# missing API key everywhere else here.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "").strip()

# --- the public HTTP API (portfolio_agent/api.py) -----------------------
# A single shared secret, checked against the X-API-Key header. Unset means
# the endpoint refuses every request (503, not a silent open door) — this is
# a public network-facing surface once deployed, so there is deliberately no
# "unauthenticated is fine" default the way there is for the internal tools.
PUBLIC_API_KEY = os.environ.get("PUBLIC_API_KEY", "").strip()

# --- the search layer, which is metered and therefore rationed ---------
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "").strip()
EXA_API_KEY = os.environ.get("EXA_API_KEY", "").strip()

# A self-hosted SearXNG instance, tried before either paid provider — it is
# unmetered, so every search it answers is one Tavily/Exa never bills for.
# Unset means it is skipped, the same as a missing API key everywhere else
# here. Defaults to the sibling container in docker-compose.yml.
SEARXNG_BASE_URL = os.environ.get("SEARXNG_BASE_URL", "http://searxng:8080").strip()

# Where the cache and the spend counter live. The cache is the single
# biggest saving available: the domain a company uses and the shape of a
# keyword's results do not change, so paying twice for either is waste.
SEARCH_STATE_DIR = os.environ.get("SEARCH_STATE_DIR", "~/.portfolio-agent")

# A company does not move house often, and a keyword's result mix shifts
# over seasons rather than days.
SEARCH_CACHE_TTL_DAYS = int(os.environ.get("SEARCH_CACHE_TTL_DAYS", "90"))

# Hard monthly caps, well under the free allowances, so a runaway batch
# stops rather than bills. Over the cap the tool reports itself
# unavailable, which every consumer here already handles.
SEARCH_MONTHLY_CAP = {
    "tavily": int(os.environ.get("TAVILY_MONTHLY_CAP", "800")),   # of 1,000 free
    "exa": int(os.environ.get("EXA_MONTHLY_CAP", "1000")),        # ~$7 of $10
}

# How much of a result page must be commercial before the keyword counts as
# commercial intent. Deliberately not a bare majority: a 50/50 page is
# genuinely mixed, and the honest answer there is unclassified, which costs
# the domain nothing.
SERP_COMMERCIAL_SHARE = float(os.environ.get("SERP_COMMERCIAL_SHARE", "0.7"))

# How many results must be readable as commercial OR editorial before the
# share between them means anything. A coined word returns pages that are
# merely unrelated, and calling that a verdict is how absence of evidence
# becomes evidence.
SERP_MIN_CLASSIFIED = int(os.environ.get("SERP_MIN_CLASSIFIED", "3"))

# How many incumbent candidates get a (metered) domain lookup per valuation.
# The credibility gap is only worth measuring for companies large enough to
# buy, and spending a credit on the eighth-largest match is how a month's
# budget disappears into one batch run.
MAX_LEAD_DOMAIN_LOOKUPS = int(os.environ.get("MAX_LEAD_DOMAIN_LOOKUPS", "3"))

MIN_APPRAISAL_BASELINES = 2

VENDOR_SPREAD_UNRELIABLE = 3.0

# The floor under recency decay. An old sale is weaker evidence, never zero
# evidence: letting the curve run to nothing would quietly empty a thin comp
# set, which is a worse answer than an out-of-date one honestly labelled.
MIN_RECENCY_WEIGHT = 0.15

# Peers within one band of the subject, below which the set is under-powered.
MIN_PEER_COMPS = 3

# An under-powered set cannot support full confidence however many comps it
# has — nine commodity comps are not nine peers.
UNDERPOWERED_CONFIDENCE = 0.5

# When reaching for peers of a premium subject, only sales above this floor
# are plausibly in its class.
PREMIUM_PEER_PRICE_FLOOR = 75_000
