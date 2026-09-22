from langgraph.graph import END, StateGraph

from portfolio_agent.guardrails import require_tier
from portfolio_agent.knowledge.category_playbook import ALL_TOOL_IDS
from portfolio_agent.nodes.classify import classify
from portfolio_agent.nodes.gather_evidence import gather_evidence
from portfolio_agent.nodes.rationale import rationale
from portfolio_agent.nodes.synthesize import synthesize
from portfolio_agent.logging_setup import auto_setup
from portfolio_agent.state import PortfolioAgentState

# langgraph.json points at this module, so this is the deployed entrypoint.
# Skipped if a host application already configured us, and by
# PORTFOLIO_AGENT_LOG_SETUP=off.
auto_setup()

# Every tool the playbook can route to is read-only (APVA advisory tools).
# Checked at build time from the playbook's own tool list rather than a
# hand-maintained copy, so a tool added to a category cannot skip the tier
# check by virtue of nobody remembering to list it here. A misconfigured
# deployment fails at startup, not mid-run.
for _tool_id in sorted(ALL_TOOL_IDS):
    require_tier(_tool_id, "read_only")
require_tier("classify_domain", "read_only")
require_tier("synthesize_valuation", "read_only")
require_tier("generate_rationale", "read_only")

_builder = StateGraph(PortfolioAgentState)
_builder.add_node("classify", classify)
_builder.add_node("gather_evidence", gather_evidence)
_builder.add_node("synthesize", synthesize)
# Last, and deliberately after every number is final: this is the only node
# an LLM writes user-facing text in, and it is given the completed valuation
# to explain rather than asked to produce one.
_builder.add_node("rationale", rationale)

_builder.set_entry_point("classify")
_builder.add_edge("classify", "gather_evidence")
_builder.add_edge("gather_evidence", "synthesize")
_builder.add_edge("synthesize", "rationale")
_builder.add_edge("rationale", END)

portfolio_agent = _builder.compile()
