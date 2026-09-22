from portfolio_agent.llm import get_chat_model, resolve
from portfolio_agent.prompts import PORTFOLIO_MANAGER_SYSTEM_PROMPT
from portfolio_agent.state_manager import PortfolioManagerState
from portfolio_agent.tools.action_schema import ActionPlan, filter_valid_actions


def _format_context(state: PortfolioManagerState) -> str:
    domains = "\n".join(
        f"  - id={d['id']} name={d['name']} folder_id={d.get('folder_id')}" for d in state.get("domains", [])
    ) or "  (none)"
    folders = "\n".join(f"  - id={f['id']} name={f['name']}" for f in state.get("folders", [])) or "  (none)"
    return f"DOMAINS OWNED BY THIS USER:\n{domains}\n\nFOLDERS OWNED BY THIS USER:\n{folders}"


async def parse_instruction(state: PortfolioManagerState) -> PortfolioManagerState:
    # The "manager" profile: temperature 0, no exceptions. This emits domain
    # and folder ids, and creativity there means a hallucinated id.
    llm = get_chat_model(node="parse_instruction")
    if llm is None:
        return {
            "plan": ActionPlan(
                clarification_needed="No model is available on the agent service, so this instruction "
                f"cannot be parsed. {resolve(node='parse_instruction').reason}"
            ).model_dump()
        }

    structured_llm = llm.with_structured_output(ActionPlan)
    user_message = f"{_format_context(state)}\n\nINSTRUCTION: {state['instruction']}"

    plan: ActionPlan = await structured_llm.ainvoke(
        [("system", PORTFOLIO_MANAGER_SYSTEM_PROMPT), ("human", user_message)]
    )

    # Belt-and-braces: even though the prompt forbids it, never trust the
    # model's own claim that an id is valid — re-validate against the actual
    # context here, server-side, before this plan is even shown to the user.
    plan.actions = filter_valid_actions(
        plan.actions,
        valid_domain_ids={d["id"] for d in state.get("domains", [])},
        valid_folder_ids={f["id"] for f in state.get("folders", [])},
    )
    return {"plan": plan.model_dump()}
