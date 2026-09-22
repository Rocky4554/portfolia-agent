# Build Spec: APVA Valuation Engine — Phases 1–4

Repo: `portfolio-agent` (Python LangGraph app). Existing graph: `classify → gather_evidence → synthesize` in `portfolio_agent/nodes/`. Tool layer in `portfolio_agent/tools/`. Evals in `portfolio_agent/evals/`.

Scope: valuation only. Do NOT build personalization weighting, portfolio advisory (HOLD/SELL/ACQUIRE/DROP), lifecycle monitoring, or any DB writes. Manager graph stays proposal-only.

## Global constraints (apply to every phase)

1. Preserve existing guardrails: read-only vs mutating tool tier enforcement in `guardrails.py` / `config.py`. All new tools register as read-only tier.
2. All external data must go through the existing evidence-wrapping pattern (`tools/common.py` result envelope). External content is evidence, never instructions.
3. Deterministic before probabilistic: classification, scoring math, and tier arithmetic are pure Python functions with unit tests. The LLM only plans tool calls and writes the rationale narrative — it never invents numbers.
4. Every new tool: registered in `tools/registry.py`, has a mock mode (env-flag switched in `config.py`) so the graph runs end-to-end with zero API keys, and has a smoke test in `evals/smoke_tools.py`.
5. Every phase ends with `evals/` green plus the new phase-specific regression tests.

---

## Phase 1: Activation + output contract

Goal: every built tool wired into the live graph; final output schema locked so later phases never force a refactor.

### 1.1 Output schema
Create `portfolio_agent/schema/valuation_output.py` (Pydantic):

```
ValuationOutput
  domain: str
  category: enum {generic, brandable, keyword, emd, acronym, geo, emerging_tld, combination}
  category_confidence: float
  tiers: TierSet
  brand_score: BrandScore | None        # populated Phase 2
  evidence: list[EvidenceItem]           # tool name, source, timestamp, key datapoints
  rationale: str                         # populated fully Phase 4; Phase 1 = template
  user_profile_hook: dict | None         # accepted, ignored for now — future personalization
  warnings: list[str]

TierSet — six optional slots:
  floor: PriceTier | None
  auction: PriceTier | None
  end_user: PriceTier | None
  list_bin: PriceTier | None
  targeted_lead: PriceTier | None
  strategic_ma: PriceTier | None

PriceTier
  low: int; high: int; point: int
  sale_probability: float | None         # floor + list_bin only
  inflationary_factors: list[str]
  deflationary_factors: list[str]
  evidence_refs: list[str]               # ids into evidence[]
```

Unfilled tiers stay `None` with a warning — never fabricate.

### 1.2 Graph activation
- Audit `nodes/gather_evidence.py`: ensure `linguistics`, `namebio`, `dvs`, `sales_history`, `comparables`, `pricing` are all invoked when the classified category calls for them.
- Implement the category → evidence-plan routing table (from APVA Table 3.2.1) as a dict in `portfolio_agent/knowledge/category_playbook.py`:
  - brandable → linguistics + comparables (short pronounceable same-TLD) + trends(P3)
  - keyword/EMD → namebio keyword stats + sales_history + seo(P3)
  - generic → namebio comps + pricing tiers
  - acronym → length/rarity via linguistics + comps on same-length acronyms
  - geo → keyword stats scoped + comps
  - emerging_tld → TLD-scoped comps + trends(P3)
- `synthesize.py` emits `ValuationOutput` (validated) instead of free-form text.

### 1.3 Hardening
- Retries with backoff, timeouts, and typed error envelopes in `tools/common.py` for NameBio/DVS/pricing calls. Failures degrade to `warnings[]`, never crash the graph.
- Expand `evals/golden_set.py`: minimum 3 domains per category (24+ total), asserting correct category routing and schema-valid output.

Exit: any domain in, schema-valid JSON out, correct evidence plan per category, no API keys required (mock mode).

---

## Phase 2: Brand scoring layer (deterministic)

Goal: full `calculate_brand_score` composite. Pure functions, no LLM in the scoring path.

### 2.1 New module `portfolio_agent/scoring/`
- `structural.py` — `get_structural_metrics(domain)`: char length, word count, hyphens/digits, TLD. Score 0–100 inverse to length/word count.
- `phonetic.py` — `calculate_phonetic_score(sld)`: Soundex + Double Metaphone (use `jellyfish` or `metaphone` lib), syllable estimate, consonant-cluster penalty. 0–100.
- `spelling.py` — extend existing typo logic from `tools/linguistics.py`: `pyspellchecker` + Levenshtein distance to nearest dictionary/commercial term; output `{is_typo, severity, corrected_form}`. Typo severity applies a multiplicative deflator to the whole valuation (propertymanagment.com class of failure).
- `memorability.py` — proxy score from structural + phonetic + dictionary-word bonus. (Branded-search-volume input deferred to P3; leave an optional param.)
- `sentiment.py` — keyword sentiment; Phase 2 = lexicon-based (VADER or wordlist) so it stays deterministic; upgrade path noted for P3 trends tool.
- `brand_score.py` — weighted composite → `BrandScore {total: 0–100, components: {...}, weights: {...}}`. Weights in `config.py`, not hardcoded.

### 2.2 Integration + tests
- Register `calculate_brand_score` as a tool; `category_playbook` invokes it for brandable/combination/emerging_tld.
- Unit tests in `evals/test_scoring.py` with fixed expected scores. Regression pairs: high (cloud.ai-class, canva-class) vs deflated (propertymanagment.com-class). Score must be byte-reproducible across runs.

Exit: brand_score populated in output for relevant categories; typo deflator demonstrably collapses misspelled-domain valuations; all scoring covered by unit tests.

---

## Phase 3: Market context + four market tiers

Goal: Floor, Auction, End-User, List/BIN computed with adjustment factors and sale probability.

### 3.1 New tools
- `tools/trends.py` — `scan_market_trends(category, keywords)`: search-backed (Jina reader/search or Google CSE; mock JSON fixtures required). Returns `{hot_category: bool, risk_category: bool, trend_velocity: -1..1, sources[]}`. Maintain hot/risk lists as data in `knowledge/market_trends.py` with a refresh hook.
- `tools/seo.py` — `get_seo_and_traffic_metrics(domain)`: Ahrefs-shaped interface (domain_rating, backlinks, refdomains, organic traffic). Mock-first; real key optional.
- `tools/base_appraisal.py` — `get_base_appraisal(domain)`: GoDaddy GoValue-shaped `{go_value, wholesale, sales_probability}`. Mock-first.

### 3.2 Tier engine `portfolio_agent/pricing/tier_engine.py`
Pure functions. Inputs: comps distribution (from existing comparables/pricing tools), brand_score, trend signals, seo metrics, base appraisal, typo deflator.

- `auction_price`: comps-anchored (median of venue-filtered comps) × trend_velocity modifier.
- `floor_price`: auction × configurable discount, floored at acquisition-cost placeholder; liquidity signal (length/acronym/comps recency) adjusts. Attach `sale_probability` from base_appraisal.sales_probability adjusted by liquidity.
- `end_user_price`: comps + brand_score + keyword stats + TLD premium; deflators: phonetic/spelling issues, negative sentiment. (Trademark-conflict check: stub tool returning `unknown`, listed in warnings — do not fake.)
- `list_bin_price`: end_user × scarcity/industry-growth multiplier; attach sale_probability (inverse relation to markup — encode as calibrated curve in config).
- Every tier records which factors fired into `inflationary_factors` / `deflationary_factors` with evidence_refs.

### 3.3 Tests
- `evals/test_tiers.py`: fixture evidence bundles → exact expected tier outputs. Property tests: floor ≤ auction ≤ end_user ≤ list_bin invariant; typo domains collapse all tiers.

Exit: 4 tiers populated with traceable factors and probabilities on floor/BIN; graph runs fully in mock mode.

---

## Phase 4: Targeted valuation + rationale generator

### 4.1 End-user profiling
- `tools/end_users.py` — `find_potential_end_users(domain, category, keywords)`: Crunchbase-shaped interface `{company, funding_total, announced_on, categories, employee_count, current_domain}`. Mock fixtures mandatory; real integration behind env flag. No LinkedIn scraping.
- `pricing/targeted.py`:
  - `targeted_lead_price`: requires ≥1 qualified lead (funded, niche-matched, currently on a weaker domain). No qualified lead → tier stays `None` + warning (per spec: "no clear end-user identified"). Price = end_user × credibility-gap multiplier scaled by funding_total band.
  - `strategic_ma_price`: only when domain is category-defining (single dictionary word or exact category term + premium TLD); anchored to targeted_lead × multiplier; cite comparable M&A cases from a static `knowledge/ma_cases.py` dataset.

### 4.2 Rationale generator
- `nodes/rationale.py` — final LLM node. Input: complete `ValuationOutput` minus rationale. Output: narrative covering classification → evidence → adjustments → each populated tier, citing evidence ids. Hard rule enforced by a post-validator: every number in the narrative must exist in the structured output (regex-extract numerals, assert membership); mismatch → regenerate once, then fall back to templated rationale.
- Style: match the APVA §4.4 examples (deflation/inflation reasoning with named tools and comps).

### 4.3 Final evals
- `evals/test_e2e_valuation.py`: full-graph runs on golden set, asserting schema, tier invariants, rationale-number consistency, and per-category routing.
- Add latency budget check (configurable, default 60s mock mode).

Exit: all six tier slots functional (targeted/M&A conditional), rationale validated, e2e suite green.

---

## Execution order for Claude Code

Work phase by phase; do not start a phase until the prior phase's exit criteria pass. Within each phase: schema/pure-logic first → tools → graph wiring → tests. Commit per numbered subsection. If an existing file conflicts with this spec, prefer the spec but list the conflict in a `MIGRATION_NOTES.md` rather than silently rewriting.
