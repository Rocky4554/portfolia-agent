# Migration notes

Where the build spec (`dev_docs/valuation_build_spec.md`) conflicts with what
was already in this repo. Per the spec's execution order the spec wins, but
the conflict is recorded here rather than silently rewritten, so anyone
reading the old code understands why it changed.

---

## 1. `"typo"` is no longer a domain category

**Was:** `tools/linguistics.classify_domain` returned `category="typo"` for a
misspelled domain, short-circuiting before any structural rule ran.

**Now:** the spec's category enum (`schema/valuation_output.DomainCategory`)
has eight members and `typo` is not one of them. A misspelling is reported in
`typo_of` / `typo_distance` and priced by a multiplicative deflator
(`scoring/spelling.py`), while the domain keeps its structural category.

**Why it matters:** `propertymanagment.com` now classifies as `emd` and runs
the EMD evidence plan — keyword sales, keyword demand, SEO. Under the old
behaviour it got no evidence plan at all, which meant the rationale could not
say what APVA §4.4 requires it to say: that the correctly spelled sibling has
real comps and *this* one does not. You cannot make that comparison without
first gathering the sibling's evidence.

**If you depended on it:** check `result.is_typo` or `result.typo_of` instead
of `result.category == "typo"`.

## 2. `typo_of` names a domain, not a word

**Was:** `"management.com"` for `propertymanagment.com` — the corrected word
plus the original TLD, which is a domain that has nothing to do with the one
being valued.

**Now:** `"propertymanagement.com"` — the actual correctly spelled sibling,
which is the domain whose comps the deflator is measured against.

## 3. `ClassificationResult` gained fields

`domain`, `sld`, `tld`, `category_confidence`, `words`, `known_words`,
`unknown_tokens` and the `is_typo` / `keywords` properties. Nothing was
removed. `category_confidence` exists because the category now selects the
evidence plan, so callers need to know how much to trust the routing.

## 4. `ToolResult` gained `mock: bool`

Existing `ok_result(...)` / `error_result(...)` calls are unchanged and
default to `mock=False`; the new `mock_result(...)` sets it. Any code that
constructs a `ToolResult` dict literally (rather than through those helpers)
needs the new key.

## 5. `request_json` retries

Timeouts, dropped connections, 429s and 5xx are now retried with exponential
backoff (`config.HTTP_MAX_ATTEMPTS`, default 3). 4xx responses are not
retried. Errors from a retried call carry an attempt count
(`"HTTP 503 (3 attempts)"`), so warning text that was matched exactly may
need to match as a substring instead. Set
`PORTFOLIO_AGENT_HTTP_BACKOFF_BASE_S=0` in tests that drive failures.

## 6. `tools/namebio.py` is superseded, not deleted

The graph's comps now come from `tools/sales_history.py` (our own Namebio
mirror, already paid for) and `tools/comparables.py` (semantic comps for
coined names) via the category playbook. `tools/namebio.py` — which calls
NameBio's paid public API and carries its own hardcoded sample comps — is no
longer reachable from the valuation graph. It is left in place because it
still works and removing it is not this spec's scope, but new code should not
call it: its mock path substitutes the *same five sample sales* for every
domain, which is exactly the failure `config.MOCK_MODE` and
`tools/mocks.py` were built to avoid.

## 7. State keys

`PortfolioAgentState.estimated_value_low` / `estimated_value_high` still
exist and are populated, but they are now derived — the widest band across
populated tiers. The real answer is `state["valuation"]["tiers"]`, and a
domain with no comps has `None` for both where it previously got whatever the
LLM guessed.

`state["evidence"]` changed shape: it now holds `EvidenceItem` dicts
(`id`, `tool`, `source`, `timestamp`, `key_datapoints`, ...) rather than the
`{type, source, content, note}` envelopes `wrap_external_evidence` returns.
The untrusted-data `note` is still carried on every item.

## 8. `tools/base_appraisal.py` is a facade over `tools/dvs.py`

The spec (§3.1) asks for a `tools/base_appraisal.py` exposing
`get_base_appraisal`. That call already existed in `tools/dvs.py`, behind our
own cache and rate limiter.

The new module is a thin facade rather than a second implementation: calling
GoDaddy directly from it would spend credits DVS has already paid for and
bypass the freshness bounds `config.py` sets deliberately. It adds the
`underprices_premium` flag the tier engine needs and gives the tool the name
the spec refers to it by. `dvs.get_base_appraisal` is unchanged and still
works.

## 9. `tools/pricing.py` vs `portfolio_agent/pricing/`

Two different things with confusingly similar names, both required by the
spec:

- `tools/pricing.py` — `price_stats` and the v0 percentile tiers. Still used
  by `tools/sales_history.py` for its summary statistics and still exposed as
  the `compute_price_tiers` tool.
- `portfolio_agent/pricing/tier_engine.py` — the real APVA Table 4.1.1 tier
  engine, with adjustment factors, sale probabilities and evidence refs.

The valuation graph now uses the latter. `compute_price_tiers`'s flat
percentile mapping (floor=p25, auction=p50, end_user=p75, list=p90) was a
documented v0 placeholder and is no longer how tiers are computed.

## 10. `sale_probability` is on `floor` and `list_bin` only

APVA Table 4.1.1 annotates exactly those two tiers with "show selling
probability at predicted price". `auction` and `end_user` deliberately carry
`None`, and the tier eval asserts it. A probability on all four would imply
we can forecast a negotiated end-user sale, which we cannot.

## 11. Trademark screening is a stub, on purpose

Table 4.1.1 lists "potential trademark conflicts" as an End-User deflator.
`tier_engine.check_trademark_conflict` returns `status: "unknown"` and never
a verdict — a trademark conflict depends on class, jurisdiction and use in
commerce, and a plausible-looking guess would be acted on. The gap surfaces
as a warning on every valuation rather than being silently absent. Wiring a
real screening service is the only way to remove that warning.

## 12. Mock mode now covers the LLM

`config.MOCK_MODE` makes `llm.get_chat_model()` return `None` unless
`PORTFOLIO_AGENT_MOCK_ALLOW_LLM=true` is also set.

This was a real defect, not a tidy-up. A model call is an external call, so
"runs with no API keys and no network" has to include it. Before this change
the end-to-end eval appeared to run offline while making one live Bedrock
call per golden domain — roughly sixty per run, which is both slow and
billable, and which exhausted this account's daily token quota during
development.

Set `PORTFOLIO_AGENT_MOCK_ALLOW_LLM=true` to combine fixture data with a real
model, which is what you want when iterating on the rationale prompt.

## 13. The graph has a fourth node

`classify → gather_evidence → synthesize → rationale`. `synthesize` no longer
produces the final narrative; it produces the templated one, and `rationale`
replaces it with a generated narrative only if every number in that narrative
already exists in the structured output.

`nodes/graph.py` also now derives its `require_tier` calls from
`category_playbook.ALL_TOOL_IDS` rather than a hand-maintained list, so a tool
added to a category cannot skip the read-only tier check because nobody
remembered to list it.

## 14. `check_spelling` is cached and bounded

`pyspellchecker` generates edit-distance-2 candidates by brute force, which is
O((26·n)²) — over a second on a 20-character string, enough to dominate a
valuation's latency. Tokens longer than 8 characters now use single-edit
correction only, tokens over 20 characters are skipped (they are
concatenations; their parts are checked individually), and results are cached
per domain. No verdict in the eval set changed; the call went from ~1.5s to
under a millisecond.

## 15. LLM configuration moved out of `config.py` into `llm.py`

`config.LLM_PROVIDER` and `config.LLM_MODEL` are **gone**, along with
`ANTHROPIC_API_KEY` / `BEDROCK_API_KEY` / `BEDROCK_REGION` as config
attributes. Everything about models now lives in `portfolio_agent/llm.py`:
the provider registry, the per-job profiles, the node-to-profile mapping and
the resolution order.

**Why:** `config.py` also holds DVS URLs, brand-score weights, tier
calibration constants and freshness bounds. The model settings were four
screens down a file about something else, which is why "where do I change
the model?" kept needing an answer. `llm.py` already claimed in its own
docstring to be the only place that knows how to construct a client; now
that is true of configuration too.

**If you depended on them:** call `llm.resolve(node=...)`, which returns the
provider, model, settings and the provenance of each. `parse_instruction.py`
was the only caller and now reports `resolve(...).reason` instead of a bare
provider name — a more useful message, since it says *why* there is no model
rather than just that there is none.

**New:** per-node model selection. `PORTFOLIO_AGENT_LLM_RATIONALE_PROVIDER`,
`..._MODEL`, `..._TEMPERATURE` and the `MANAGER` equivalents override the
global setting for one profile, so the narrative and the structured parser
can run on different models.

**New:** `uv run python -m portfolio_agent.llm_doctor` prints the resolved
configuration with the provenance of every value, and `--ping` makes one
real call per node to prove the credentials work.

## 16. Structured logging, and `portfolio_agent/__init__.py` is no longer empty

The package now installs a `NullHandler` on its own logger at import, which
is the library-correct default: it emits nothing until an application calls
`logging_setup.setup_logging()`. `nodes/graph.py` calls `auto_setup()`
because `langgraph.json` points at it, so a deployed server logs without
anyone wiring it up; set `PORTFOLIO_AGENT_LOG_SETUP=off` if the host
application configures logging itself.

Only the `portfolio_agent` logger is configured and `propagate` is set to
`False`, so importing this package does not change how nameaiv1 logs and
does not duplicate records into its handlers.

`PortfolioAgentState` gained `run_id`, generated by the classify node and
carried through state — a contextvar set in one LangGraph node is gone by
the time the next one runs, so state is the only thing that crosses that
boundary.

## 17. Mistral default model is `ministral-8b-latest`, not `mistral-large-latest`

Verified against this account on 2026-09-21: `mistral-large-latest` is not
in the account's model list at all and returns `tier_not_allowed`, and
`mistral-medium-latest` / `mistral-small-latest` return a *persistent* 429
(three attempts, six seconds apart) rather than a transient one. Only
`ministral-8b-latest` serves this key's tier.

If the subscription changes, re-run `llm_doctor --ping` against a larger
model rather than assuming it works.

## 18. `.env` used a lowercase `mistral_api_key`

Environment variables are case-sensitive, so `os.environ["MISTRAL_API_KEY"]`
did not see it and the agent silently fell back to Bedrock. The name was
corrected in place (the value untouched, and a timestamped `.env.bak-*` was
left beside it).

## 19. An independent appraisal now contradicts the comps out loud

`get_domain_profile` has always fetched Estibot's automated appraisal, and
the pricing math has always ignored it — `TierInputs` had no field for it.
That was fine for ordinary domains and badly wrong for one class.

A live valuation of `insurance.com` priced it at **$8,970** end-user from 21
real comparable sales, while the Estibot value it had already retrieved in
the same run says **$36.49M**, against a documented real sale of $35.6M. The
comps are not wrong — they are `insuranceai.com`, `insurancer.com`,
`insuranceme.com`, domains *containing* the word. `insurance.com` is the
word. Those are different markets, and a category-killer .com has
essentially no comparable sales because they trade twice a decade.

`TierInputs.independent_appraisal` now carries it, and
`appraisal_disagreement()` emits a warning when it and the comps-derived
end-user price differ by `APPRAISAL_DISAGREEMENT_FACTOR` (default 10x) or
more, in either direction.

**It changes no price.** Comps remain the anchor, per the spec, and the tier
eval asserts that a conflict leaves every tier byte-identical. Reconciling
the two would mean deciding silently and per-domain whether to trust a
vendor's model or our own comps, and being wrong either way produces a
confident number with nothing attached to it. Surfacing the disagreement
hands that judgement to whoever has to act on it.

### Resolved in §23

## 20. A real dictionary word is no longer mistaken for a coined name

Three separate failures fired at once on `voice.com`, a domain
`knowledge/ma_cases.py` records selling for **$30,000,000**:

- `segment("isotope")` returned `iso|top|e`, because "top" is in the
  commercial lexicon. Greedy matching against a 400-word list mangles real
  words, and a single-word domain then reads as a three-word compound.
- `check_typo` read `apple` as a two-edit misspelling of `app`, turning
  apple.com into a keyword domain built from the word "app".
- `_semantic_category` had no way to tell "a real word absent from our
  commercial list" from "an invented string", so both became
  `brandable` / "coined or invented name".

The consequence was not cosmetic: the wrong category runs the wrong evidence
plan, and `is_category_defining()` — whose test the spec states as *"a single
dictionary word"* — was consulting our 400-word commercial list instead of a
dictionary, so voice.com was ruled not category-defining and disqualified
from the Strategic M&A tier.

`knowledge/dictionary.py` now holds the shared 160k-word lookup. It is a
separate module because both `tools/linguistics.py` and
`scoring/spelling.py` need it for different questions and must not import
each other. Three rules follow from it: a whole SLD that is a real word is
never split; a real word is never a typo; and a single real word is
`generic`, not coined.

No existing expectation changed — the new rules only fire where no
commercial term was recognized at all, which the golden set had no coverage
for. That gap is now closed with `REAL_WORD_DOMAINS` and `COINED_DOMAINS`.

## 21. Spelled-out magnitudes in the rationale validator

A live run rejected a *correct* draft over "$30.75 million". The figure was
in the output as `30,750,000`, but the extractor handled `$30.75M` and not
the word, so it saw a bare `30.75` and called it invented. `thousand`,
`million` and `billion` now scale like their letter forms. A guard that
rejects correct work is a guard people route around.

## 22. A dictionary holds names as well as words

Fixing §20 traded one error for another. Teaching the classifier that a real
word is not a coined name also taught it that **crusoe.com** is "a single
dictionary word that names its own category", and it priced a Strategic M&A
tier on that basis. `crusoe` passes every "is this a real word" test. It is
a proper noun.

`knowledge/dictionary.py` gains `frequency_per_million()` and
`is_common_noun()`. Corpus frequency is the only signal available without a
part-of-speech tagger, and the two populations barely overlap: measured
across 37 category terms against 22 proper nouns, a floor of **2.0 per
million** keeps 36 of the category terms and admits none of the proper
nouns. A name is rare in general prose however famous the thing it names —
`insurance` appears 29 times per million, `crusoe` 0.04.

`is_category_defining()` now takes curated terms on sight and requires
everything else to clear that floor. Its rejection quotes the frequency, so
the reasoning is visible rather than asserted.

The error this still makes is rejecting a genuine but uncommon category term
(`slots`, at 0.91 per million, from a $5.5M sale). That is the right
direction to fail in — a rejected term leaves an honestly unpriced tier with
a stated reason, while an accepted proper noun produces a confident price
built on a false premise. Demonstrated categories below the floor go in the
curated list instead, which is what the curated list is for.

Classification is deliberately unchanged: crusoe.com remains `generic`,
because a dictionary-word `.com` is a dictionary-word `.com` regardless of
what part of speech it is. Only the category-defining claim was wrong.

## 23. A category killer is priced against category killers

The §19 gap is closed. `insurance.com` was priced at **$9,120** end-user
against a documented **$35.6M** sale, and the fix is not a multiplier — it is
using the right comparable sales.

A substring search for `insurance` returns `autoinsurance.net` and
`insurancer.com`. Those are real sales in a real market, and it is not this
domain's market: `insurance.com` is not a domain containing the word, it is
the word. So when a domain is category-defining, the market tiers re-anchor
on comparable **acquisitions** of structurally identical domains from
`knowledge/ma_cases.py`, read through exactly the same percentile arithmetic
as any other comp set. No new multiplier exists.

Measured against seven public sale prices, end-user now lands between 0.38x
and 4.60x of the real figure, median 1.50x. Before it was roughly 4000x low.

Three things had to be right for this not to become blanket inflation:

**The gate is narrower than `is_category_defining()`.** That function also
admits two-word exact terms — "car insurance" genuinely is its category's
exact term — but every case in the comp set is a single word or a two-to-
three character string on `.com`. Re-anchoring `carinsurance.com` priced a
low-six-figure EMD at **$45.9M**. The comp set now applies only where the
domain matches its shape: one token, on `.com`.

**A domain is never its own comparable.** `insurance.com` is in the dataset,
and anchoring its valuation on its own sale price is circular.

**The selection must not be pre-sorted by price.** `comparable_cases()`
ranked by price descending within each length bucket, which is right when
citing a notable case and wrong when the selection becomes a distribution —
it biases the sample high before any arithmetic runs. `by_price=False` takes
a length-ordered slice instead.

The residual limitation is stated in the output rather than hidden: these
are seventeen of the most famous domain sales ever made, survivorship-biased
by construction because unremarkable eight-figure sales are never reported.
The comp set skews high, the warning says so, and the tiers are an order of
magnitude rather than an estimate.

## 24. Marketplace listings — the last unwired source in Table 4.1.1

APVA Table 4.1.1 names the primary adjustment source for the List Price
(BIN) tier as "Marketplace Listings". Every other tier's source was wired;
this one never was, and a screenshot of GoDaddy asking **~$2.8M for
consulting.ai** made the gap concrete — there was a live listing for the
exact domain being valued and the agent had no way to see it.

`tools/marketplace.py` adds a provider registry (Name.ai, GoDaddy Auctions,
extensible) behind one normalized result. A provider without credentials
reports itself unavailable rather than failing the lookup, because a
valuation with two of three marketplaces is still a valuation.

**The design rule, which is the whole point.** A listing is what a seller
ASKS; a comp is what a buyer PAID. `TierInputs.listings` is a separate field
from `TierInputs.comps` and nothing merges them. consulting.ai carries an ask
near $2.8M against one recorded sale in its class at $425 — putting that ask
into the comps distribution would not improve the valuation, it would poison
an anchor built from transactions with one seller's hope.

So only the **BIN tier** reads a listing, because the BIN tier is itself an
ask. Floor, auction and end-user are anchored on transactions and are
untouched; the eval asserts they do not move. The pull toward an ask is
capped at `LISTING_BIN_PULL` (0.35), and a divergence of 5x or more is
reported the same way an appraisal conflict is — surfaced, not silently
resolved.

### What this exposed about GoValue

GoDaddy's own `get_base_appraisal` returns **$2,220** for consulting.ai while
GoDaddy's own storefront asks **~$2.8M**. The same vendor disagrees with
itself by roughly 1,300x. That is not a defect here — it is direct
confirmation of the note already in `tools/dvs.py`, that GoValue models the
mass-market registrar tail and must never be the anchor.

### Unverified

Both adapters are written to their documented request shapes and have NOT
been run against the live services — neither credential is configured. The
response normalization is the part most likely to need correcting once a
real payload is seen, which is why every adapter returns through
`_normalize`/`_rows_from` rather than shaping its own output. The GoDaddy
MCP connector available in this environment returns `priceInfo: null` and
cannot substitute: it reports availability only.

## 25. Vendor float noise was failing correct rationales

A live run of consulting.ai lost its narrative twice and fell back to the
template, rejecting the figure `0.0089`. That was a false positive and the
bug was mine.

GoDaddy reports a sale probability as `0.008949999697506428`. That reached
the evidence at full binary precision, the rationale validator holds the
model to figures present in the output, and no one writes sixteen
significant digits in prose. The model rendered it as `0.0089` — correctly —
and was told it had invented a number.

Two changes, because the cause and the check were both wrong. Evidence
floats are now rounded to six decimal places in `build_evidence_item`, so
the noise never reaches the prompt or the output. And `allowed_numbers()`
admits every rounding of a collected figure to one through six decimal
places.

That second change is not a tolerance window. Each admitted value is derived
from a number the output actually asserts, so an invented figure still has
nothing to round to — `0.42` and `42424242` are rejected in the same test
that accepts `0.0089`.

### Still unsolved: the validator checks figures, not claims

The same run's narrative says "the **list_bin** tier, anchored to the single
comparable sale, suggests a range of $425". `list_bin` is not priced at all.
Every number in that sentence exists in the output, so the check passes; the
*characterisation* is wrong.

This is the third instance of the same class — earlier ones read `emd` as
"emerging market domain" and called a GoValue field a tier. A numbers-only
validator cannot police claims about what a number means. Catching these
needs a structural check of the assertions a narrative makes, not of the
figures it contains.
