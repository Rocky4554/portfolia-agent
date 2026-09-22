"""
screen_trademarks — the deflator Table 4.1.1 names and nothing has ever applied.

**The gap.** Every valuation this engine has produced carries the same
warning: trademark conflict not screened, screen manually before relying on
the End-User tier. It has been honest and useless in equal measure.

**Why it stayed a stub, and what changed.** A conflict is a legal
determination turning on class, jurisdiction and use in commerce, so a
plausible-looking guess would be worse than silence — a reader acts on it.
That reasoning still holds for the general case and this does not attempt
it. What it does attempt is the case that needs no search: a domain
containing a household mark whose owner enforces against domains as routine
policy. nikeshoes.com does not require a database lookup to be a problem.

**USPTO is free but no longer anonymous.** developer.uspto.gov/ds-api now
redirects to data.uspto.gov and TSDR answers 401 with "you'll need to
register for an API key". So the real search sits behind USPTO_API_KEY,
unexercised here because no key is configured, and shaped to the documented
request rather than to a response anyone has seen. Everything below the key
is keyless and runs always.

**"No conflict found" is not "cleared", and the output must never let those
blur.** A domain that trips nothing has failed to trip an obvious wire and
has not been searched. The status vocabulary keeps them apart on purpose:
`likely_conflict`, `possible_conflict`, `no_obvious_conflict`, `unscreened`.
There is deliberately no `clear`.
"""

from portfolio_agent import config
from portfolio_agent.knowledge.famous_marks import ALSO_COMMON_WORDS, FAMOUS_MARKS
from portfolio_agent.knowledge.lexicon import segment
from portfolio_agent.logging_setup import get_logger, info, warn
from portfolio_agent.tools.common import ToolResult, error_result, mock_result, ok_result
from portfolio_agent.tools.linguistics import split_domain

log = get_logger(__name__)

SOURCE = "trademark_screen"

STATUSES = ("likely_conflict", "possible_conflict", "no_obvious_conflict", "unscreened")

# Applied to the end-user tier and above. A famous mark in the name does not
# reduce what the domain would fetch from a willing buyer — it reduces the
# set of buyers who can safely take it, and the odds the sale survives a
# letter from counsel.
CONFLICT_DEFLATORS = {
    "likely_conflict": 0.35,
    "possible_conflict": 0.75,
    "no_obvious_conflict": 1.0,
    "unscreened": 1.0,
}


async def screen_trademarks(domain: str) -> ToolResult:
    """Screen a domain against famous marks, and against USPTO where keyed.

    Returns `{status, marks, deflator, searched, detail}`.
    """
    domain = (domain or "").strip().lower()
    if not domain:
        return error_result(SOURCE, "domain is empty")

    if config.MOCK_MODE:
        return mock_result(SOURCE, _screen(domain, searched=False))

    result = _screen(domain, searched=False)
    if result["status"] != "no_obvious_conflict":
        info(log, "trademark.flagged", domain=domain, status=result["status"],
             marks=result["marks"])
    return ok_result(SOURCE, result)


def _screen(domain: str, *, searched: bool) -> dict:
    sld, _ = split_domain(domain)
    tokens = segment(sld)

    # Segmented words first, then a boundary pass, because the segmenter is
    # tuned for the commercial lexicon and does not know proper nouns: it
    # splits "targetpractice" into tar/get/practice and loses the mark
    # entirely.
    hits = [t for t in tokens if t in FAMOUS_MARKS]
    hits += [m for m in _boundary_marks(sld) if m not in hits]
    # The mark standing alone is a stronger signal than the mark inside a
    # compound: nike.com is the mark, nikeshoes.com trades on it, and both
    # are problems, but bestnikeshoesreview.com is unambiguous.
    exact = sld in FAMOUS_MARKS

    if not hits:
        return _result(domain, "no_obvious_conflict", [], searched)

    # A hit that is also an ordinary English word, used in a sense that has
    # nothing to do with the mark, is the false positive this screen would
    # otherwise generate constantly — "orange" is a colour and a fruit
    # before it is a telecom.
    # `not exact` is the whole compound test. Token count is not: the
    # segmenter leaves "orangejuice" whole, so counting tokens called a
    # fruit juice a telecom conflict.
    ambiguous = [h for h in hits if h in ALSO_COMMON_WORDS]
    if ambiguous and not exact:
        return _result(domain, "possible_conflict", hits, searched)
    return _result(domain, "likely_conflict", hits, searched)


def _boundary_marks(sld: str) -> list[str]:
    """Marks sitting at the start or end of the name, where what is left
    over is itself a word.

    Substring matching alone reads "meta" out of "metabolism" and "visa"
    out of "visualise"; requiring the remainder to be a real word admits
    nike+shoes and target+practice and rejects both of those. A screen that
    cries wolf gets switched off, which costs more than the cases it catches.
    """
    # The dictionary, not the commercial lexicon: the lexicon holds terms
    # that sell (insurance, loans, software) and does not know "practice",
    # "juice" or "orchard", which is exactly the vocabulary a famous mark
    # gets compounded with.
    from portfolio_agent.knowledge.dictionary import is_dictionary_word

    # A name that IS a word is not a compound of a mark and a word.
    # "shellfish" is one dictionary entry, not Shell plus fish, and reading
    # it as the latter is the kind of false positive that gets the whole
    # screen switched off.
    if is_dictionary_word(sld):
        return []

    found = []
    for mark in FAMOUS_MARKS:
        if len(mark) < 4 or mark == sld:
            continue
        for remainder in (sld[len(mark):] if sld.startswith(mark) else None,
                          sld[: -len(mark)] if sld.endswith(mark) else None):
            if remainder and len(remainder) >= 3 and is_dictionary_word(remainder):
                found.append(mark)
                break
    return found


def _result(domain: str, status: str, marks: list[str], searched: bool) -> dict:
    assert status in STATUSES, f"unknown status {status!r}"
    detail = {
        "likely_conflict": (
            f"contains {', '.join(marks)}, {'a mark' if len(marks) == 1 else 'marks'} whose owner "
            f"enforces against domains as routine policy. This is not a legal opinion, but it is "
            f"the class of name that draws one"),
        "possible_conflict": (
            f"contains {', '.join(marks)}, which is both a famous mark and an ordinary word. "
            f"Whether that is a conflict depends on the sense it is used in and the class it is "
            f"sold into — decidable by a lawyer, not by this"),
        "no_obvious_conflict": (
            "no famous mark appears in this name. That is NOT a clearance: no trademark search "
            "has been run, and an ordinary registered mark would not appear here at all"),
        "unscreened": "no screening was performed",
    }[status]
    return {
        "domain": domain,
        "status": status,
        "marks": marks,
        "deflator": CONFLICT_DEFLATORS[status],
        # False until a real search runs. The word "searched" rather than
        # "checked" because a wordlist check is not a search and the
        # difference is the whole point.
        "searched": searched,
        "detail": detail,
    }
