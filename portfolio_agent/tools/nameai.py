"""
check_availability — name.ai's public domain check.

**What it adds.** An EPP/registry availability answer and, for an
unregistered domain, the actual registration price. That second number
matters: the acquirability warning already says an unregistered domain "can
be had at registration cost" and had no figure to put on it. This supplies
one — $10.49 for a .com today.

**Public, no key.** `/api/domain/check` takes a POST and authenticates
nothing. That is the whole reason to prefer it over the marketplace API for
this particular question.

**What is deliberately ignored, and why — which is not that it is wrong.**
The response also carries `whois_summary` and `is_active_site`, and neither
is read here. They are name.ai's own database record of a domain it tracks,
populated from `dbData.domain.*`, with `expiry_date` hardcoded null because
the schema does not store it. For a domain name.ai does not manage they are
simply empty, which is correct for what they are and useless for what we
need. The service's actual WHOIS lives on a different route.

The same applies to `state_label`. "Registered Inactive" means no active
listing — a marketplace-inventory state, in their words "a domain we track
that just isn't for sale right now" — and not a claim about the website.
Read as a statement about liveness it looks wrong for google.com; read as
what it is, it is right, and their UI says exactly the same thing in
plainer words: "Taken. Already registered."

So only `available`, `price` and `listing` are read, because those are the
fields that mean the same thing for every domain rather than only for the
ones in name.ai's inventory. Our own lander check and RDAP answer the
liveness and registration questions, and they answer them for any domain.
"""

from portfolio_agent import config
from portfolio_agent.logging_setup import get_logger, info
from portfolio_agent.tools.common import ToolResult, error_result, mock_result, ok_result, request_json

log = get_logger(__name__)

SOURCE = "nameai_check"

_TIMEOUT_S = 25.0


async def check_availability(domain: str) -> ToolResult:
    """Registry availability for a domain, and its registration price if free.

    Returns `{available, registration_price, listed_on_nameai, state}`.
    """
    domain = (domain or "").strip().lower().rstrip(".")
    if not domain or "." not in domain:
        return error_result(SOURCE, "domain is empty or has no TLD")

    if config.MOCK_MODE:
        return mock_result(SOURCE, _mock_check(domain))

    body, error = await request_json(
        "POST", f"{config.NAMEAI_API_BASE_URL}/api/domain/check",
        json={"domain": domain},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=_TIMEOUT_S,
    )
    if error:
        return error_result(SOURCE, error)
    if not isinstance(body, dict):
        return error_result(SOURCE, f"check returned {type(body).__name__}, expected an object")

    available = body.get("available")
    if not isinstance(available, bool):
        # Null means the registry could not be reached, which is a failed
        # lookup and not an answer about the domain.
        return error_result(
            SOURCE, f"registry check returned no verdict (state {body.get('state_label')!r})")

    price = body.get("price")
    priced = float(price) if isinstance(price, (int, float)) else None
    result = {
        "domain": domain,
        "available": available,
        # Only a REGISTRATION price when the name can actually be
        # registered. isotope.co came back available=False with price=2888,
        # which is an aftermarket or premium figure and not the cost of
        # registering anything — calling it registration cost would put a
        # number on an action nobody can take.
        "registration_price": priced if available else None,
        "quoted_price": priced,
        "listed_on_nameai": bool(body.get("listing")),
        "state": body.get("state_label"),
        "note": "registry availability only — this endpoint's whois_summary, is_active_site "
                "and state_label describe name.ai's own inventory rather than the domain, so "
                "they are not read here; liveness comes from the lander check and registration "
                "from RDAP",
    }
    info(log, "nameai.checked", domain=domain, available=available,
         registration_price=result["registration_price"], quoted=priced)
    return ok_result(SOURCE, result)


def _mock_check(domain: str) -> dict:
    from portfolio_agent.tools.mocks import _seed

    seed = _seed(domain, "nameai_check")
    available = (seed % 7) == 0
    return {
        "domain": domain,
        "available": available,
        "registration_price": 10.49 if available else None,
        "listed_on_nameai": (seed % 5) == 0 and not available,
        "state": "Unregistered" if available else "Registered",
        "note": "fixture",
    }
