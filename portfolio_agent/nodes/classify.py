import uuid

from portfolio_agent.logging_setup import get_logger, info, log_step, run_context
from portfolio_agent.scoring.spelling import check_spelling
from portfolio_agent.state import PortfolioAgentState
from portfolio_agent.tools.linguistics import classify_domain

log = get_logger(__name__)


def classify(state: PortfolioAgentState) -> PortfolioAgentState:
    """Pure-logic classification. Runs first because the category selects the
    evidence plan (knowledge/category_playbook.py) — everything downstream
    depends on getting this right, which is why it's deterministic Python
    with a golden set rather than an LLM call."""
    run_id = state.get("run_id") or uuid.uuid4().hex[:8]
    with run_context(state["domain"], run_id), log_step(log, "classify") as step:
        return _classify(state, run_id, step)


def _classify(state: PortfolioAgentState, run_id: str, step) -> PortfolioAgentState:
    result = classify_domain(state["domain"])

    # Two typo checks with different jobs, and this is where they meet.
    # classify_domain's is lexicon-based and exists to keep segmentation
    # honest; scoring/spelling.py runs the full dictionary and is the
    # authority on severity and the deflator. The deflator is computed for
    # EVERY domain, not just brand-scored ones — propertymanagment.com is an
    # EMD and still has to collapse.
    spelling = check_spelling(state["domain"])
    notes = list(result.structural_notes)
    if spelling.is_typo:
        notes.extend(spelling.notes)

    step.set(
        category=result.category,
        confidence=result.category_confidence,
        also=",".join(name for name, _ in result.secondary_categories) or None,
        subcategory=result.subcategory,
        segments="+".join(result.words),
        is_typo=bool(spelling.is_typo),
        typo_severity=spelling.severity,
        typo_deflator=spelling.deflator,
    )
    if spelling.is_typo:
        info(log, "classify.misspelling_detected",
             corrected_form=spelling.corrected_form, severity=spelling.severity,
             distance=spelling.distance, deflator=spelling.deflator)

    return {
        "run_id": run_id,
        "category": result.category,
        "category_confidence": result.category_confidence,
        "secondary_categories": result.secondary_categories,
        "subcategory": result.subcategory,
        "sld": result.sld,
        "tld": result.tld,
        "words": result.words,
        "keywords": result.keywords,
        "typo_of": spelling.corrected_form or result.typo_of,
        "typo_distance": spelling.distance if spelling.is_typo else result.typo_distance,
        "typo_severity": spelling.severity,
        "typo_deflator": spelling.deflator,
        "structural_notes": notes,
    }
