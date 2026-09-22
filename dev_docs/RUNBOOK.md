# Portfolio Agent — how to run it

Every command below has been run in this repo and produces the output shown.

**If a command fails with `Failed to spawn: <something>`, rebuild the venv.**

```bash
rm -rf .venv && uv sync --all-groups
```

This project was moved from `/Users/macbook/raunak/portfolio-agent` to
`/Users/macbook/raunak office/portfolio-agent` in September 2026. A virtual environment
stores absolute paths inside itself — in `bin/activate` and in every console script — so a
venv that survives a move still points at where the project used to be. The symptom is
confusing, because it reads like a missing install rather than a stale path:

```
error: Failed to spawn: `langgraph`
  Caused by: No such file or directory (os error 2)

warning: VIRTUAL_ENV=/Users/macbook/raunak/portfolio-agent/.venv does not match
         the project environment path `.venv` and will be ignored
```

The venv has been rebuilt and everything works. This note is here for the next time the
project moves. The rule is simply: **move the project, rebuild the venv.** Nothing about
the space in the folder name causes problems — uv handles it correctly.

If your shell still prints the `VIRTUAL_ENV` warning, it is holding the old value from an
earlier `source .venv/bin/activate`. Run `unset VIRTUAL_ENV` or open a new tab. In this
project you never need to activate anything — always use `uv run`.

---

## 1. Setup

You need [uv](https://docs.astral.sh/uv/). Nothing else — it manages Python and the venv.

```bash
cd "/Users/macbook/raunak office/portfolio-agent"

uv sync --all-groups        # runtime deps + the LangGraph CLI
cp .env.example .env        # then edit .env — see below
```

Do not activate the venv; prefix everything with `uv run`. To add a dependency use
`uv add <pkg>` rather than editing `pyproject.toml`, so `uv.lock` stays in step.

### What to put in `.env`

Only one thing is actually required: **one model credential**. Everything else has a
working default.

```bash
MISTRAL_API_KEY=your-key-here
```

Environment variable names are **case-sensitive**. `mistral_api_key` will not be found,
and the agent will silently fall back to another provider or to no model at all. If a key
seems to be ignored, check its case first.

The provider is auto-detected in the order
`mistral → anthropic → openai → google → groq → bedrock`. Everything about models —
which provider, which model, temperature, per-node overrides — lives in
`portfolio_agent/llm.py`.

---

## 2. Check the configuration before running anything

```bash
uv run python -m portfolio_agent.llm_doctor
```

Prints every provider, whether its credential is set, what each graph node will call, and
**where every value came from**. That last column is the point: the question is never
"what model is configured" but "why is it calling *that* one".

```
  provider     credential env               set   auto  default model
  mistral      MISTRAL_API_KEY              yes    yes  ministral-8b-latest
  anthropic    ANTHROPIC_API_KEY             no    yes  claude-sonnet-5
  bedrock      AWS_BEARER_TOKEN_BEDROCK     yes    yes  deepseek.v3.2

  active provider:   mistral

  rationale → profile rationale [available]
      mistral / ministral-8b-latest   temp=0.2  max_tokens=900
          provider     from auto-detected from MISTRAL_API_KEY
          model        from PROVIDERS['mistral'].default_model
```

Prove the credential actually works — one real call per node, three total:

```bash
uv run python -m portfolio_agent.llm_doctor --ping
```

```
PING   (real, billable calls — one per node)
    yes  default            replied 'Got it.'
    yes  parse_instruction  replied 'Got it.'
    yes  rationale          replied 'Got it.'
```

A `403 tier_not_allowed` here means the key is valid but the model is not in your
subscription tier — pick a different one, do not assume. A persistent `429` across several
attempts means the same thing in a different costume.

---

## 3. Run the tests

Seven suites, **1,527 checks**, all fully offline — no keys, no network, no model calls.
Run them before and after any change.

```bash
uv run python -m portfolio_agent.evals.test_valuation      # classification, routing, schema, guardrails
uv run python -m portfolio_agent.evals.test_tools          # service tools against mocked HTTP
uv run python -m portfolio_agent.evals.test_scoring        # brand scoring, exact expected values
uv run python -m portfolio_agent.evals.test_tiers          # tier arithmetic, invariants, probability curve
uv run python -m portfolio_agent.evals.test_llm_config     # provider resolution, per-node overrides
uv run python -m portfolio_agent.evals.test_logging        # run correlation, step coverage, redaction
uv run python -m portfolio_agent.evals.test_e2e_valuation  # the real graph over the whole golden set
```

All seven in one line:

```bash
for m in test_valuation test_tools test_scoring test_tiers test_llm_config test_logging test_e2e_valuation; do
  printf "%-22s " "$m"; uv run python -m portfolio_agent.evals.$m 2>&1 | tail -1
done
```

```
test_valuation         PASSED 66/66 checks.
test_tools             PASSED 41/41 checks.
test_scoring           PASSED 76/76 checks.
test_tiers             PASSED 383/383 checks.
test_llm_config        PASSED 64/64 checks.
test_logging           PASSED 215/215 checks.
test_e2e_valuation     PASSED 682/682 checks (slowest valuation 0.29s, budget 60s).
```

`test_scoring` and `test_tiers` assert **exact** numbers. If one changes because you
recalibrated something deliberately, update it in the same commit and say why in the
message. If it changes and you did not mean it to, that is the bug they exist to catch.

---

## 4. Value a domain

```bash
uv run python -m portfolio_agent.cli cloud.ai
```

| Flag | Effect |
|---|---|
| `--mock` | every external service **and the model** answered from deterministic fixtures — no keys, no network, no spend |
| `--mock-llm` | with `--mock`, still call the real model (for iterating on the rationale prompt) |
| `--json` | print the raw `ValuationOutput` instead of the report |
| `--quiet` / `-q` | suppress step logs, print only the result |
| `--log-level DEBUG` | add per-HTTP-attempt and per-anchor detail |
| `--log-format json` | one JSON object per log line |

Useful combinations:

```bash
# Offline demo — same numbers on any machine, costs nothing
uv run python -m portfolio_agent.cli cloud.ai --mock --quiet

# Fixture data, real model — for working on the rationale prompt
uv run python -m portfolio_agent.cli cloud.ai --mock --mock-llm

# Machine-readable, for piping into something else
uv run python -m portfolio_agent.cli cloud.ai --mock --json | jq '.tiers.end_user'

# Everything that happened, in detail
uv run python -m portfolio_agent.cli cloud.ai --mock --log-level DEBUG
```

---

## 5. A worked example, start to finish

This is the case the whole product is pitched against (APVA §4.4): a domain containing a
single-character misspelling of a valuable commercial term. First-generation appraisers
price it like its correctly spelled sibling. It is nearly worthless.

```bash
uv run python -m portfolio_agent.cli propertymanagment.com --mock --mock-llm
```

**What the agent decides, and why:**

```
  propertymanagment.com
  emd — confidence 0.8

  tier                  point          low         high   P(sale)  what it means
  -------------- ------------ ------------ ------------  --------  --------------------------------
  floor                  $224         $105         $405       71%  quick wholesale exit
  auction                $463         $208         $764            competitive investor auction
  end_user             $1,467         $685       $2,654            direct sale to a business
  list_bin             $2,066         $965       $3,737       17%  premium Buy-It-Now ask
  targeted_lead        $2,920       $1,364       $5,282            one identified, funded buyer
  strategic_ma     not priced                                      key asset in an acquisition
```

Read the adjustment trail — this is the part that distinguishes a valuation from a number:

```
  end_user  (evidence: ev3, ev5, ev1, ev4)
      + 554,254 monthly searches for the keyword (×1.30)
      + .com — the extension end users pay a premium for (×1.15)
      + existing SEO authority: established (DR 43.2) (×1.25)
      - misspelling of a commercial term (×0.02)
```

The keyword demand is real and it inflates the price three ways. Then the misspelling
deflates it by 98%, and the deflator is applied to **every** tier. Compare the correctly
spelled sibling:

```bash
uv run python -m portfolio_agent.cli propertymanagement.com --mock --quiet
```

`end_user` there is **$58,671** against this domain's **$1,467** — the same evidence, the
same tools, a 97.5% collapse caused by one letter. That is the failure mode APVA exists to
fix, working.

Two other things worth noticing in the output:

- **`strategic_ma` is `not priced`, and the reason is stated.** Most domains are not
  category-defining, and a system that always produces a Strategic M&A price is a system
  whose Strategic M&A price means nothing. An unfilled tier is always paired with a
  warning explaining it — that rule is enforced by the schema, not by convention.
- **Every number in the rationale exists in the structured output.** The model writes the
  narrative; a validator then extracts every numeral and rejects the draft if any of them
  is not in the valuation. On a live run against Mistral this fired for real — the model
  invented a figure twice on `carinsurance.com`, both drafts were discarded, and the
  templated rationale was used instead.

---

## 6. Run as a server

`langgraph.json` registers two graphs: `"Portfolio Valuation Agent"` and
`"Portfolio Manager Agent"`.

```bash
uv run langgraph dev
```

(`uv run python -m langgraph_cli dev` also works and is equivalent.) This serves
the same `createThread` / `runWait` HTTP shape that nameaiv1's existing negotiation agent
uses; the client side is `nameaiv1/lib/server/portfolio-agent-client.js`.

---

## 7. Check the live data services

The valuation tools talk to Namekart's own services (DVS, the Namebio sales mirror, the
Comparable AI Agent). To check whether they are up — **real calls, spends a few upstream
credits**:

```bash
uv run python -m portfolio_agent.evals.smoke_tools insurance.com
```

The same command with fixtures, which costs nothing and proves the wiring rather than the
services:

```bash
PORTFOLIO_AGENT_MOCK_MODE=true uv run python -m portfolio_agent.evals.smoke_tools carinsurance.com
```

```
  OK   get_domain_profile     estibot_value=22689 category='Finance -- Insurance' ...
  OK   get_base_appraisal     go_value=18615 list=37230 p(sale)=0.6103
  OK   scan_market_trends     velocity=0.0 hot=False risk=False sources=1
  OK   calculate_brand_score  65.05/100 {...} typo_deflator=1.0
  OK   check_trademark_conflict status=unknown (stub — never screens)

11/11 service tools OK
```

---

## 8. Logs

Every step logs what it did, what it decided and how long it took. The aim is that a run
which went wrong in production can be reconstructed from its logs alone — the inputs (a
live market service's response at a moment in time) usually cannot be reproduced.

```bash
PORTFOLIO_AGENT_LOG_LEVEL=DEBUG uv run python -m portfolio_agent.cli cloud.ai --mock
```

```
13:51:16 INFO [99c309a8 propertymanagment.com] nodes.classify: classify.misspelling_detected  corrected_form=propertymanagement.com severity=severe deflator=0.02
13:51:16 INFO [99c309a8 propertymanagment.com] nodes.gather_evidence: tool.done  duration_ms=0.53 tool=get_domain_profile ok=True mock=True source=dvs_estibot
13:51:16 INFO [99c309a8 propertymanagment.com] nodes.synthesize: tier.computed  tier=end_user point=1467 up="554,254 monthly searches (×1.30)" down="misspelling (×0.02)" evidence=ev3,ev5,ev1
13:51:25 INFO [99c309a8 propertymanagment.com] nodes.rationale: rationale.validated  attempt=1 numbers_written=31 unsupported=0
```

The bracketed `99c309a8` is the run id. Every line of one valuation carries it, so
`grep run=99c309a8` reconstructs a whole run — which matters because evidence gathering
runs its tools concurrently and the lines interleave.

| Variable | Default | |
|---|---|---|
| `PORTFOLIO_AGENT_LOG_LEVEL` | `INFO` | `DEBUG` adds per-HTTP-attempt detail |
| `PORTFOLIO_AGENT_LOG_FORMAT` | `text` | `json` for an aggregator |
| `PORTFOLIO_AGENT_LOG_FILE` | — | also append JSON records here |
| `PORTFOLIO_AGENT_LOG_SETUP` | — | `off` if the host application configures logging |

To capture a run to disk as JSON:

```bash
PORTFOLIO_AGENT_LOG_FILE=./run.log uv run python -m portfolio_agent.cli cloud.ai --mock
jq -r 'select(.event == "tier.computed") | "\(.tier) \(.point)"' run.log
```

Credentials never reach a log, and neither do raw upstream payloads — tools log summaries
and counts. Those payloads carry both PII (registrant names, emails) and whatever text
someone put on a web page, and a log viewer is one more place that text gets read.

---

## 9. When something goes wrong

**A model credential seems to be ignored.**
Check its case — `mistral_api_key` is not `MISTRAL_API_KEY`. Then run `llm_doctor`, which
prints whether each credential is `set` and which one won.

**`Failed to spawn: langgraph` (or any other console script).**
A stale venv, left over from the project being moved. `rm -rf .venv && uv sync --all-groups`.
See the top of this document.

**`403 tier_not_allowed` or a persistent `429` from Mistral.**
The key is valid; that model is not on your plan. `ministral-8b-latest` is the verified
default for the current key — `mistral-large-latest` is not in the account's model list at
all. If the subscription changes, re-run `llm_doctor --ping` against a bigger model rather
than assuming it works.

**The rationale is the short templated one, not prose.**
That is the fallback, and it is always disclosed in `warnings`. Three causes: no model
configured, mock mode on without `--mock-llm`, or the model wrote a number that is not in
the structured output twice in a row. The last one is the validator working — the tier
figures are unaffected.

**An eval is slow, or seems to hang.**
It should not be: the whole offline suite is seconds, and the slowest single valuation is
under a third of a second. If an "offline" run is slow, something is making a real network
call. `PORTFOLIO_AGENT_MOCK_MODE=true` covers the LLM as well as the data services, so a
slow mock run means something is bypassing it.

**A tier says `not priced`.**
Usually correct, not a bug. Read the matching warning: too few comparable sales, no
qualified end-user lead, or not a category-defining domain. The system declines to price
what it cannot support rather than guessing.

---

## Where things live

| I want to change... | File |
|---|---|
| which model, temperature, per-node model | `portfolio_agent/llm.py` |
| which evidence each category gathers | `portfolio_agent/knowledge/category_playbook.py` |
| brand score weights | `portfolio_agent/config.py` → `BRAND_SCORE_WEIGHTS` |
| tier discounts, multipliers, probability curve | `portfolio_agent/config.py` → tier engine section |
| hot / high-risk market categories | `portfolio_agent/knowledge/market_trends.py` |
| the words classification splits on | `portfolio_agent/knowledge/lexicon.py` |
| the rationale's wording | `portfolio_agent/prompts.py` |
| logging format, level, destination | `portfolio_agent/logging_setup.py` |

`MIGRATION_NOTES.md` in the repo root records every place this build's choices differ from
what was in the repo before, and why.

---

## The funding collector

Supplies the one fact nothing else in this repo can: whether an identified company can
write a cheque. Two of the six tiers — Targeted Lead and Strategic M&A — are premised on a
specific funded buyer, and without a funding figure they stay unpriced. This replaces
Crunchbase with public funding announcements.

Runs **outside** the valuation graph, on purpose: a valuation must not depend on a news
site being reachable. The collector writes to a store; the agent reads it.

### Run it

```bash
# Collect, print the report to stdout
uv run python -m portfolio_agent.funding.pipeline

# Collect, resolve each company's own domain, write the report to a file
uv run python -m portfolio_agent.funding.pipeline --resolve-domains --report reports/funding.md
```

**Use `--resolve-domains`.** Without it no lead can be scored: a funding article names the
company, the amount, the round and the date, but not their website — and the website is
what makes a credibility gap measurable. It costs one extra request per funding story
(three at a time, deliberately gentle).

### As a cron job

```cron
0 7 * * * cd "/Users/macbook/raunak office/portfolio-agent" && \
  uv run python -m portfolio_agent.funding.pipeline --resolve-domains \
  --report reports/funding-$(date +\%F).md >> logs/funding.log 2>&1
```

### Storage

Postgres when `PORTFOLIO_AGENT_FUNDING_DB_URL` is set, newline-delimited JSON at
`PORTFOLIO_AGENT_FUNDING_STORE` (default `./funding_events.jsonl`) otherwise. The table is
created on first run. Writes are idempotent on company + date + amount, so a cron job
re-reading yesterday's feed cannot double a company's funding total.

The JSON fallback is not a placeholder — it means the collector still gathers news when
the database is down, and the evals need no infrastructure.

### What a run produces

```
48 articles -> 3 funding events

  Crusoe          $3,900,000,000  ai          on crusoe.ai         ← lead: .ai, not .com
  Mazama Energy     $135,000,000  climate     on mazamaenergy.com
  Evvy               $40,000,000  healthtech  on evvy.com
```

The report ranks by **how much a better domain would help**, not by round size. A company
that raised $3.9B and already owns the matching `.com` is not a buyer; one on `.ai`, a
hyphenated name, or a `get`/`try` prefix of the name it wanted is a buyer with a visible
problem and the budget to fix it.

### How it reaches a valuation

`tools/end_users.py` tries Crunchbase first, then this store, then reports the tier as
unpriced. Nothing else changes — the funding band, the credibility-gap multiplier and the
Targeted Lead tier already worked and are covered by `evals/test_tiers.py`.

```bash
uv run python -m portfolio_agent.funding.pipeline --resolve-domains --quiet > /dev/null
uv run python -m portfolio_agent.cli crusoe.com          # now finds a $3.9B lead on crusoe.ai
```

### Known limits

- **Only companies whose domain we resolved become leads.** A funding row without one is a
  company, not a lead, and is dropped rather than passed through.
- **A raise with no stated figure is discarded.** Guessing a typical round size would
  invent the number the whole tier turns on.
- **VC fund raises are excluded.** "Sequoia raises $2B fund" has the same shape as "Crusoe
  raises $3.9B" and means something entirely different; a fund is not a domain buyer.
- Figures are as reported and not independently confirmed — a headline can bundle debt
  with equity, and rounds are often announced months after closing.

### Related: `find_end_user_leads` is returning unusable data

Separate from this pipeline, and our bug rather than Estibot's. The allowlist
`_LEAD_FIELDS` in `tools/dvs.py` requests `source_domain`, which is *the domain being
searched* — one distinct value across all 865 rows. The lead's own site is in a field
named `domain`, which is never requested, and the business name is in `site_title` rather
than the empty `whois_reg_org`. Every lead therefore scores a credibility gap of zero and
is discarded, while the tool reports `ok=True`. Fixing it would supply a second, free lead
source. Not yet done.
