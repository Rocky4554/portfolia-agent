"""
The structured output contract for the Portfolio Manager agent (Tier 2).

This agent NEVER touches a database and NEVER calls back into name.ai's API —
it only turns a natural-language instruction into a validated list of
proposed actions. nameaiv1 is the one that shows these to the user for
confirmation and, only after that, executes them with its own fresh
ownership check. See raunak_dev_docs / the portfolio-agent README for why
that split exists.

Every action referencing an existing domain or folder must use an id that
was present in the `portfolio_context` given to the agent (see
nodes/parse_instruction.py) — the agent is never allowed to invent or guess
an id. If the instruction is ambiguous about WHICH domain(s) it means (e.g.
"drop my old domains" with no names given), the agent must set
`clarification_needed` instead of guessing — guessing on a destructive
action is exactly the failure mode this schema exists to prevent.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

ActionType = Literal[
    "drop_domain",
    "restore_domain",
    "create_folder",
    "move_domain_to_folder",
    "draft_register_domain",
]

# Actions that change something the user already owns/has. All require an
# id from portfolio_context — never a name the model matched fuzzily.
_OWNERSHIP_SCOPED = {"drop_domain", "restore_domain", "move_domain_to_folder"}

# Actions with no real-world side effect until a human completes a separate,
# real checkout flow — the agent may only ever draft these.
_DRAFT_ONLY = {"draft_register_domain"}


class ProposedAction(BaseModel):
    type: ActionType
    domain_id: Optional[str] = Field(
        default=None, description="Must be an id from portfolio_context.domains — never invented."
    )
    folder_id: Optional[str] = Field(
        default=None, description="Must be an id from portfolio_context.folders, or null for a brand-new folder."
    )
    folder_name: Optional[str] = None
    domain_name: Optional[str] = Field(
        default=None, description="Only for draft_register_domain — a domain the user does not own yet."
    )
    reason: str = Field(description="One short sentence: why this action, in the user's own words where possible.")

    def is_draft_only(self) -> bool:
        return self.type in _DRAFT_ONLY

    def requires_ownership_check(self) -> bool:
        return self.type in _OWNERSHIP_SCOPED


class ActionPlan(BaseModel):
    actions: list[ProposedAction] = Field(default_factory=list)
    clarification_needed: Optional[str] = Field(
        default=None,
        description="Set this INSTEAD of guessing when the instruction doesn't clearly identify which "
        "domain(s)/folder(s) it means. Never populate both `actions` and this on the same response.",
    )


def filter_valid_actions(
    actions: list[ProposedAction], valid_domain_ids: set[str], valid_folder_ids: set[str]
) -> list[ProposedAction]:
    """Server-side re-validation of every id the model claims — the LLM's own
    adherence to the prompt is not trusted; this is the actual enforcement.
    An action referencing an id that isn't in the caller-supplied ground
    truth is dropped silently rather than surfaced, so a hallucinated id can
    never reach the confirmation UI, let alone execution.
    """
    clean = []
    for action in actions:
        if action.domain_id and action.domain_id not in valid_domain_ids:
            continue
        if action.folder_id and action.folder_id not in valid_folder_ids:
            continue
        clean.append(action)
    return clean
