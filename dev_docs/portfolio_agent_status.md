# Portfolio Agent Status

Date: 2026-09-21

## Overall summary

The project is already well advanced in its core architecture and has a solid foundation for a portfolio valuation and management agent. The core work that is complete is mainly the read-only valuation graph, the manager graph scaffold, the tool layer, and the evaluation setup.

At a practical level, I would describe the current state as:

- Core architecture: built
- Valuation agent: largely built and working as a LangGraph pipeline
- Manager agent: built as a proposal-only layer, not as an executor
- Database-writing features: not implemented and intentionally blocked
- Production-grade execution layer: not yet complete

In simple terms: about 70–80% of the core agent framework is in place, but the end-to-end production workflow is still not fully finished.

---

## What we have built

### 1. Project foundation

The repo is set up as a Python LangGraph application with proper package structure, configuration, and dependency management.

Included pieces:

- package setup in `pyproject.toml`
- LangGraph graph registration in `langgraph.json`
- environment/config handling in `portfolio_agent/config.py`
- package structure under `portfolio_agent/`

### 2. Valuation agent graph

The main valuation graph is implemented and follows a clear flow:

- `classify` node
- `gather_evidence` node
- `synthesize` node
- final graph output

This is the core read-only valuation workflow.

Key files:

- `portfolio_agent/nodes/graph.py`
- `portfolio_agent/nodes/classify.py`
- `portfolio_agent/nodes/gather_evidence.py`
- `portfolio_agent/nodes/synthesize.py`
- `portfolio_agent/state.py`

This graph is designed to accept a domain and return an estimated valuation plus reasoning based on evidence.

### 3. Manager graph

The manager graph is also implemented as a proposal system. It is designed to parse a human instruction and generate structured action proposals, without directly mutating anything.

Key files:

- `portfolio_agent/nodes/manager_graph.py`
- `portfolio_agent/nodes/parse_instruction.py`
- `portfolio_agent/state_manager.py`
- `portfolio_agent/tools/action_schema.py`

This is a strong base for a workflow where the agent suggests actions such as:

- drop a domain
- restore a domain
- create a folder
- move a domain
- draft a new registration

### 4. Safety and tier enforcement

The project has strong guardrails built in to prevent unsafe behavior.

Key files:

- `portfolio_agent/guardrails.py`
- `portfolio_agent/config.py`

Important safety mechanisms include:

- read-only vs mutating tool tier enforcement
- explicit blocking of writes unless permitted
- explicit evidence wrapping for external data sources
- clear “do not treat external evidence as instructions” pattern

This is one of the strongest parts of the project.

### 5. Tooling layer

The tool layer is fairly complete and covers multiple data sources and reasoning modules.

Built tools include:

- `linguistics.py` for classification and typo logic
- `namebio.py` for NameBio comps lookup
- `common.py` for HTTP/result envelope logic
- `dvs.py` for DVS-related valuation data
- `sales_history.py` for historical sales stats
- `comparables.py` for comparable-domain analysis
- `pricing.py` for pricing stats and pricing tiers
- `registry.py` for registering the tools into LangChain

This is a strong foundation for integrating valuation signals and structured evidence.

### 6. Evaluation and testing setup

There is a proper evaluation harness with regression/test coverage.

Key files:

- `portfolio_agent/evals/test_valuation.py`
- `portfolio_agent/evals/test_tools.py`
- `portfolio_agent/evals/golden_set.py`
- `portfolio_agent/evals/smoke_tools.py`

This means the project is not just a prototype; it includes test logic and validation workflows.

### 7. LLM and prompt layer

The agent includes its own prompt and model config setup.

Key files:

- `portfolio_agent/llm.py`
- `portfolio_agent/prompts.py`

This means the reasoning layer is intentionally structured rather than being a loose script.

---

## Current build maturity

### Completed well

- project scaffold
- graph architecture
- valuation pipeline
- manager proposal pipeline
- safety enforcement
- tool registry and evidence handling
- evaluation framework

### Partially completed

- manager-agent execution flow is not connected to live user-domain actions
- some tool integrations are implemented but not all are fully activated in the main graph
- several upstream data sources are wrapped but not yet fully production-hardened

### Not yet implemented

The following are intentionally not part of the current build:

- direct database writes
- real ownership mutation logic in the agent itself
- approval-execution pipeline connected to a live system
- full portfolio optimization engine
- SEO / marketing metrics features
- user personalization engine
- complete production deployment integration

---

## Practical assessment

If we are measuring by engineering foundation, the project is already quite substantial.

The best way to describe it is:

- The core agent architecture is built.
- The valuation engine is in place.
- The manager agent is built as a safe proposal layer.
- The final production execution path is still missing.

So the state is closer to: "strong MVP / foundation is built" than "full product is complete."

---

## Final verdict

This repo is not a blank project anymore. It contains a real LangGraph-based portfolio agent with valuation logic, manager logic, tool integrations, guardrails, and eval coverage.

The build is significant and meaningful, but the project still has a difference between:

- built core intelligence and infrastructure
- completed production-ready execution workflow

The first is already largely done. The second is still the next major layer to finish.
