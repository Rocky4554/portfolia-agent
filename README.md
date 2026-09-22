# portfolio-agent

📖 **Full documentation, with architecture + sequence diagrams, lives in the `nameaiv1` repo:
`raunak_dev_docs/portfolio_agent/README.md`** (kept alongside the original APVA spec PDF).

The Portfolio Agent for name.ai, in two graphs:

- **Portfolio Valuation Agent** (APVA) — read-only. Takes a domain name, returns a
  validated `ValuationOutput`: an APVA category, up to six price tiers each with the
  factors that moved it and the evidence ids behind them, a brand score, and a written
  rationale whose every number is checked against the structured output before it is
  returned. No tool in this graph can change anything in name.ai's database.
- **Portfolio Manager Agent** (Tier 2) — also has **no database access of its own**.
  Given a natural-language instruction plus the caller's actual domains/folders (the
  only ground truth it's allowed to use), it returns a structured, validated list of
  *proposed* actions — drop a domain, restore one, create a folder, move a domain into
  one, or draft a brand-new registration. It never executes anything. nameaiv1 shows
  the proposal to the user, lets them approve or reject each action individually, and
  only then executes — re-checking ownership itself at that point, not trusting
  anything this agent said. See `portfolio_agent/guardrails.py` and
  `portfolio_agent/tools/action_schema.py` for how that boundary is enforced in code,
  and `nameaiv1/lib/server/portfolio-agent/execute-action.js` for the execution side.

Registering a new domain is real money at a real registrar, so
`draft_register_domain` is permanently draft-only — it only ever returns a link into
the real checkout flow (`/search?q=...`) for the user to complete themselves. No
action type in this repo, present or future, is allowed to skip that for a purchase.

## What's here

```
portfolio_agent/
├── state.py                   # Valuation graph's shared state shape
├── state_manager.py           # Manager graph's shared state shape
├── config.py                  # data-service env vars, weights, calibration constants
├── guardrails.py              # tool-tier enforcement + the "wrap external evidence" helper
├── llm.py                     # THE LLM CONFIG PAGE — providers, profiles, per-node models
├── cli.py                     # `python -m portfolio_agent.cli <domain>` — value one domain
├── llm_doctor.py              # shows what llm.py resolved to, and --ping proves it works
├── logging_setup.py           # structured logging: run correlation, timing, redaction
├── prompts.py                 # every system prompt, separate from node logic and model choice
├── schema/
│   └── valuation_output.py    # THE output contract — tiers, evidence, brand score, warnings
├── knowledge/                 # APVA's expert knowledge, as data rather than code
│   ├── lexicon.py             # the curated word lists classification segments on
│   ├── category_playbook.py   # Table 3.2.1: category -> evidence plan
│   ├── market_trends.py       # §3.3: hot/risk lists, with an as-of date and a refresh hook
│   └── ma_cases.py            # comparable acquisitions, for the Strategic M&A tier
├── scoring/                   # §4.3 brand score — pure functions, no network, no LLM
│   ├── structural.py          # simplicity & brevity
│   ├── phonetic.py            # Soundex + Double Metaphone pronounceability
│   ├── spelling.py            # misspelling detection, severity, and the valuation deflator
│   ├── memorability.py        # recall proxy (real branded search volume optional)
│   ├── sentiment.py           # VADER + a domain-industry wordlist
│   └── brand_score.py         # the weighted composite
├── pricing/                   # Table 4.1.1 tier math — pure functions
│   ├── tier_engine.py         # Floor / Auction / End-User / List-BIN
│   └── targeted.py            # Targeted Lead / Strategic M&A — both conditional
├── tools/
│   ├── linguistics.py         # classification into the APVA taxonomy, with confidence
│   ├── common.py              # shared HTTP + result envelope; retries, never raises
│   ├── mocks.py               # deterministic fixtures for every service (config.MOCK_MODE)
│   ├── dvs.py                 # DVS: Estibot profile, GoDaddy value, investor score, DotDB, leads
│   ├── sales_history.py       # our Namebio service: keyword sales stats + comps
│   ├── comparables.py         # Comparable AI Agent: semantically similar sold domains
│   ├── base_appraisal.py      # GoValue facade over dvs.py, with the underpricing flag
│   ├── trends.py              # scan_market_trends: rule-based verdict + searched citations
│   ├── seo.py                 # Ahrefs-shaped authority and traffic
│   ├── end_users.py           # Crunchbase-shaped funded-buyer profiling (no LinkedIn)
│   ├── pricing.py             # price stats — used for summary statistics, not for tiers
│   ├── namebio.py             # superseded; see MIGRATION_NOTES.md
│   ├── action_schema.py       # Tier 2's proposed-action contract + server-side id re-validation
│   └── registry.py            # all of the above as LangChain tools, output wrapped as evidence
├── nodes/
│   ├── classify.py            # pure logic: category, confidence, spelling verdict
│   ├── gather_evidence.py     # runs the category's evidence plan, concurrently
│   ├── synthesize.py          # assembles + validates the ValuationOutput. No LLM.
│   ├── rationale.py           # the ONLY LLM node; every number it writes is verified
│   ├── graph.py               # Valuation graph (`portfolio_agent`)
│   ├── parse_instruction.py   # Tier 2: instruction -> validated ActionPlan, no side effects
│   └── manager_graph.py       # Manager graph (`portfolio_manager_agent`)
└── evals/
    ├── golden_set.py          # 25 domains, 3+ per category, plus typo regression pairs
    ├── test_valuation.py      # classification, routing, guardrails, schema validity
    ├── test_tools.py          # the service tools against mocked HTTP
    ├── test_scoring.py        # brand scoring, exact expected values
    ├── test_tiers.py          # tier arithmetic, invariants, probability curve
    ├── test_e2e_valuation.py  # the real graph over the whole golden set, offline
    └── smoke_tools.py         # every tool against the LIVE services — run by hand
```

## How a valuation works

```
classify ──> gather_evidence ──> synthesize ──> rationale
   │               │                  │             │
   │               │                  │             └─ the only LLM call. Given the
   │               │                  │                finished numbers to explain;
   │               │                  │                every numeral it writes must
   │               │                  │                already exist in the output or
   │               │                  │                the draft is discarded.
   │               │                  └─ pure Python: tier engine -> validated
   │               │                     ValuationOutput. No model runs here.
   │               └─ runs the category's evidence plan concurrently. Any tool
   │                  failure becomes a warning, never an exception.
   └─ deterministic: APVA category + confidence + spelling verdict.
```

Three rules hold throughout, and most of the code exists to enforce them:

1. **Deterministic before probabilistic.** Classification, scoring and tier
   arithmetic are pure functions with exact-value tests. The model writes prose.
2. **Nothing is invented.** A tier that cannot be computed is `None` with a warning
   naming it — enforced by the schema, not by convention. Most domains legitimately
   have no Targeted Lead or Strategic M&A price.
3. **Everything is traceable.** Every adjustment records its multiplier, its reason
   and the evidence ids behind it.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) — `pyproject.toml`
declares them, `uv.lock` pins the exact resolved versions, and `uv sync` creates
`.venv` and installs them (including this package itself, which is what
`langgraph.json`'s `"dependencies": ["."]` refers to).

```bash
uv sync                # runtime deps + the dev group (LangGraph CLI); use --no-dev to skip it
cp .env.example .env   # set ONE model credential (e.g. MISTRAL_API_KEY); the rest is optional
```

## Configuring the LLM

All of it lives in one file, [`portfolio_agent/llm.py`](portfolio_agent/llm.py). Nothing
else in the repo imports a provider class, names a model, or sets a temperature — nodes
ask for `get_chat_model(node=...)` and get back a plain LangChain model. Three layers,
each overridable by env var so a deployment can change models without a deploy:

| Layer | What it decides | Where |
|---|---|---|
| `PROVIDERS` | which providers exist, their credential and default model | `llm.py` |
| `PROFILES` | temperature, token cap, timeout, retries — **per job** | `llm.py` |
| `NODE_PROFILES` | which profile each graph node uses | `llm.py` |

Set one credential and the provider is auto-detected, in the order
`mistral → anthropic → openai → google → groq → bedrock`. To see exactly what will be
called and **why**:

```bash
uv run python -m portfolio_agent.llm_doctor          # inspect — free, no calls
uv run python -m portfolio_agent.llm_doctor --ping   # one real call per node
```

The provenance column is the point. The question in production is never "what model is
configured" — it is "why is it calling *that* one", and every value reports whether it
came from a default in `llm.py` or from a named environment variable.

Two jobs, deliberately different settings. The **rationale** profile runs at temperature
0.2 because it is the one place the system writes prose a person reads and temperature 0
produces stilted repetition — safe only because every numeral it writes is verified
against the structured valuation before it is returned. The **manager** profile runs at
temperature 0 with no exceptions, because it emits domain and folder ids and creativity
there means a hallucinated id.

To split models across nodes — a strong model for the narrative, a cheap fast one for the
parser:

```bash
PORTFOLIO_AGENT_LLM_RATIONALE_PROVIDER=anthropic
PORTFOLIO_AGENT_LLM_MANAGER_MODEL=mistral-small-latest
```

There's no need to activate the venv — prefix commands with `uv run`. To add a
dependency, use `uv add <pkg>` (or `uv add --dev <pkg>`) rather than editing
`pyproject.toml` by hand, so the lockfile stays in step.

## Run the evals

No API keys, no network, no model calls — every suite runs fully offline:

```bash
uv run python -m portfolio_agent.evals.test_valuation      # classification + routing + schema
uv run python -m portfolio_agent.evals.test_tools          # service tools vs mocked HTTP
uv run python -m portfolio_agent.evals.test_scoring        # brand scoring, exact values
uv run python -m portfolio_agent.evals.test_tiers          # tier math + invariants
uv run python -m portfolio_agent.evals.test_llm_config     # provider resolution + overrides
uv run python -m portfolio_agent.evals.test_logging        # correlation, coverage, redaction
uv run python -m portfolio_agent.evals.test_e2e_valuation  # the real graph, whole golden set
```

`test_scoring` and `test_tiers` assert **exact** numbers. If one changes because you
recalibrated something deliberately, update it in the same commit and say why in the
message. If it changes and you did not mean it to, that is the bug they exist to catch.

## Valuation tools

Every tool returns the same `{source, ok, data, data_age_days, error, mock}` envelope
and never raises: a service being down, slow or rate-limited costs the valuation one
piece of evidence and one warning, and the remaining tiers are computed from what did
arrive. `tools/registry.py` exposes them as LangChain tools for an agent that picks its
own; the valuation graph calls the plain functions through the category playbook, so its
evidence gathering stays fixed and reproducible.

### Mock mode

```bash
PORTFOLIO_AGENT_MOCK_MODE=true uv run python -m portfolio_agent.evals.test_e2e_valuation
```

With `PORTFOLIO_AGENT_MOCK_MODE=true`, every external call — including the LLM — is
answered from a deterministic fixture derived from a hash of the domain. The whole graph
runs with no keys, no network and no vendor credits, and produces byte-identical output
on every machine. Fixtures are anchored on category, length and TLD, so `cloud.ai` comes
out expensive and `zylantix.com` cheap; a fixture layer that returned one flat number for
everything would make every downstream eval vacuous.

Mock results are flagged `mock=True` through the envelope, the evidence items and the
warnings, so a fixture number can never be presented as an appraisal. To iterate on the
rationale prompt against fixture data and a real model, add
`PORTFOLIO_AGENT_MOCK_ALLOW_LLM=true`.

To check the tools against the live services (real calls, a few upstream credits):

```bash
uv run python -m portfolio_agent.evals.smoke_tools insurance.com
```

Each module's docstring records what was broken upstream when it was built
(2026-09-14): Comparable Agent searches, live Estibot appraisals and leads,
and investorScore's LLM factors.

## Logs

Every step logs what it did, what it decided and how long it took. The aim is that a run
which went wrong in production can be reconstructed from its logs alone — the inputs (a
live market service's response at a moment in time) usually cannot be reproduced.

```bash
PORTFOLIO_AGENT_LOG_LEVEL=INFO uv run python -m portfolio_agent.evals.smoke_tools cloud.ai
```

```
13:51:16 INFO [99c309a8 propertymanagment.com] nodes.classify: classify.misspelling_detected  corrected_form=propertymanagement.com severity=severe deflator=0.02
13:51:16 INFO [99c309a8 propertymanagment.com] nodes.gather_evidence: tool.done  duration_ms=0.53 tool=get_domain_profile ok=True mock=True source=dvs_estibot
13:51:16 INFO [99c309a8 propertymanagment.com] nodes.synthesize: tier.computed  tier=end_user point=1467 up="554,254 monthly searches (×1.30) | .com premium (×1.15)" down="misspelling of a commercial term (×0.02)" evidence=ev3,ev5,ev1
13:51:25 INFO [99c309a8 propertymanagment.com] nodes.rationale: rationale.validated  attempt=1 numbers_written=31 unsupported=0
```

| Variable | Default | |
|---|---|---|
| `PORTFOLIO_AGENT_LOG_LEVEL` | `INFO` | `DEBUG` adds per-HTTP-attempt and per-anchor detail |
| `PORTFOLIO_AGENT_LOG_FORMAT` | `text` | `json` for a log aggregator |
| `PORTFOLIO_AGENT_LOG_FILE` | — | also append JSON records here |
| `PORTFOLIO_AGENT_LOG_SETUP` | — | `off` if the host application configures logging |

Five things worth knowing, because each avoids a specific failure:

- **Only the `portfolio_agent` logger is configured, never root.** This package is
  imported into nameaiv1's server; `logging.basicConfig()` from a library hijacks the host
  application's logging. Nothing is emitted until `setup_logging()` is called.
- **The run id travels in `contextvars`, and in graph state between nodes.** Evidence
  gathering runs its tools concurrently under asyncio, where a `threading.local` would
  leak one valuation's id into another's lines. `grep run=99c309a8` reconstructs a run.
- **Fields are structured, not interpolated.** The same record renders for a human or for
  an aggregator without either parsing the other's output.
- **Every `.done` line carries `duration_ms`.** "Which tool made this slow" cannot be
  answered retroactively from logs that did not record it.
- **Credentials and upstream payloads never reach a log.** Tools log summaries and counts,
  never raw bodies — those carry both PII and whatever text an attacker put on a web page.
  A redaction filter is the backstop, deliberately anchored so it does *not* fire on
  `max_tokens` or a token count; a filter that cries wolf gets ignored when it matters.

## Run a valuation

```bash
uv run python -m portfolio_agent.cli cloud.ai --mock --quiet
```

`--mock` answers every external service *and the model* from deterministic fixtures, so
this runs with no keys, no network and no spend. Drop it for a real valuation, add
`--json` for the raw `ValuationOutput`, `--log-level DEBUG` for every step.

**Full command reference, with a worked example, is in
[`dev_docs/RUNBOOK.md`](dev_docs/RUNBOOK.md).**

## Run as a LangGraph server (for nameaiv1 to call over HTTP)

```bash
uv run langgraph dev
```

If this fails with `Failed to spawn: langgraph`, the venv is stale from the project being
moved — `rm -rf .venv && uv sync --all-groups` fixes it. See
[`dev_docs/RUNBOOK.md`](dev_docs/RUNBOOK.md).

This serves the graph registered in `langgraph.json` under the assistant name
`"Portfolio Valuation Agent"`, using the same `createThread` / `runWait` HTTP shape the
existing negotiation agent already uses — see `nameaiv1/lib/server/portfolio-agent-client.js`.

## What's deliberately NOT here

- **Personalization.** APVA §2.2's user profile (risk appetite, investment strategy,
  budget, preferred TLDs) is accepted as `user_profile_hook`, echoed back untouched, and
  changes no price. The hook exists so adding it later is not a schema change that breaks
  every stored valuation.
- **Portfolio advisory** — HOLD / SELL / ACQUIRE / DROP recommendations (§5.1), profit
  forecasting (§5.2) and lifecycle monitoring (§6.1).
- **Trademark screening.** Table 4.1.1 lists trademark conflict as an End-User deflator.
  `check_trademark_conflict` returns `unknown` and warns on every valuation rather than
  guessing: the answer depends on class, jurisdiction and use in commerce, and a
  plausible-looking guess would be acted on.
- **LinkedIn end-user profiling.** Table 7.1.1 lists it and marks it "High Risk". It
  would mean scraping personal profiles for a valuation. Company-level fields only.
- **Anything that writes to name.ai's database.** The valuation graph only ever reads a
  domain name in and returns a structured valuation out.

Where this build's choices differ from what was in the repo before, see
[MIGRATION_NOTES.md](MIGRATION_NOTES.md).
