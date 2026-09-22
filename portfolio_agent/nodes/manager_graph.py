from langgraph.graph import END, StateGraph

from portfolio_agent.nodes.parse_instruction import parse_instruction
from portfolio_agent.state_manager import PortfolioManagerState

# This graph only ever proposes. It has no tool that writes anywhere — see
# tools/action_schema.py. Mutating the user's actual domains/folders happens
# in nameaiv1, after a human confirms, using its own fresh ownership check —
# never trusted to this graph's output alone.

_builder = StateGraph(PortfolioManagerState)
_builder.add_node("parse_instruction", parse_instruction)
_builder.set_entry_point("parse_instruction")
_builder.add_edge("parse_instruction", END)

portfolio_manager_agent = _builder.compile()
