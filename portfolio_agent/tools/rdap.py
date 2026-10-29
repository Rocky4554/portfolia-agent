"""
get_registration — who holds this domain, since when, and until when.

**The gap this closes.** `tools/lander.py` can see what is SERVED at a
domain and nothing about the registration behind it. Its honest limit was
written into its own docstring: getaddrinfo cannot tell an unregistered
domain from a registered one publishing no A record, so the verdict was
`no_dns` and named both possibilities. cloud.ai is the live case — it does
not resolve and is certainly registered. RDAP answers that question
definitively, for free, with no key and no account.

**RDAP rather than WHOIS.** WHOIS is a text format that every registry
renders differently, so parsing it means maintaining a parser per registry
forever. RDAP is the IETF replacement, returns JSON, and rdap.org bootstraps
to whichever registry actually holds the record — which is why this follows
redirects where nothing else here does.

**What it does NOT give you.** The registrant. Since GDPR, .com and most
other registries redact it, and hello.com returns exactly one entity with
the role `registrar` — Squarespace Domains II LLC, which is where the domain
is parked administratively and says nothing about who owns it. Anyone
expecting an owner name from this will be disappointed by every major TLD.
The owner, where it is knowable at all, comes from the page the lander
detector already reads.

**What it does give you, and why it is worth having.** Age: hello.com was
registered in 1997, and a domain held for nearly thirty years is a different
asset from one registered last year — it has survived every renewal decision
since. Expiry, which is the acquisition calendar. And the EPP status codes,
where `redemptionPeriod` and `pendingDelete` mean a domain is falling out of
its owner's hands, which is the one moment a portfolio buyer most wants to
know about.
"""

from datetime import date, datetime, timezone

from portfolio_agent import config
from portfolio_agent.logging_setup import debug, get_logger, info
from portfolio_agent.tools.common import ToolResult, error_result, mock_result, ok_result, request_json

log = get_logger(__name__)

SOURCE = "rdap"

_TIMEOUT_S = 20.0
_BOOTSTRAP = "https://rdap.org/domain"

# EPP status codes that mean the registration is in trouble. A domain in
# redemption or pending delete is about to become available, which changes
# the acquisition question from "will they sell" to "can I catch the drop".
_DROPPING = {"redemptionperiod", "pendingdelete", "pendingrestore"}
# Codes that mean the owner has locked it down. Not a refusal to sell, but
# it does mean nothing is moving without them.
_LOCKED = {"clienttransferprohibited", "servertransferprohibited", "clienthold", "serverhold"}


async def get_registration(domain: str) -> ToolResult:
    """Registration facts for a domain, or an honest absence of them.

    Returns `{registered, registered_on, expires_on, age_years, registrar,
    statuses, dropping, locked, nameservers, registrant_available}`.
    """
    domain = (domain or "").strip().lower().rstrip(".")
    if not domain or "." not in domain:
        return error_result(SOURCE, "domain is empty or has no TLD")

    if config.MOCK_MODE:
        return mock_result(SOURCE, _mock_registration(domain))

    body, error = await request_json(
        "GET", f"{_BOOTSTRAP}/{domain}",
        headers={"Accept": "application/rdap+json"},
        timeout=_TIMEOUT_S,
        # rdap.org exists to redirect to the registry holding the record.
        follow_redirects=True,
    )
    if error and "HTTP 404" not in error:
        # rdap.org is one aggregator and a single point of failure. IANA
        # publishes the authoritative TLD -> RDAP-server map, so when the
        # aggregator is unreachable the registry can still be asked
        # directly. Borrowed from name.ai's own whois route, which does the
        # same thing for the same reason.
        direct, direct_error = await _via_iana_bootstrap(domain)
        if direct is not None:
            info(log, "rdap.via_bootstrap", domain=domain, aggregator_error=error)
            body, error = direct, None
        else:
            debug(log, "rdap.bootstrap_failed", domain=domain, error=direct_error)
    if error:
        if "HTTP 404" in error:
            # rdap.org answers 404 for two different things: a domain the
            # registry has no record of, and a TLD with no RDAP service at
            # all. isotope.co was the second — .co and .io have no entry in
            # IANA's bootstrap, so the 404 said nothing about the domain.
            # Only a TLD that IANA lists can have its 404 read as
            # unregistered.
            index, index_error = await _bootstrap_index()
            tld = domain.rsplit(".", 1)[-1]
            if index is not None and tld in index:
                info(log, "rdap.unregistered", domain=domain)
                return ok_result(SOURCE, _result(domain, registered=False))
            info(log, "rdap.unknown_no_service", domain=domain,
                 reason=index_error or f"IANA lists no RDAP server for .{tld}")
            return ok_result(SOURCE, _result(domain, registered=None, status="unknown_via_rdap"))
        return error_result(SOURCE, error)

    if not isinstance(body, dict):
        return error_result(SOURCE, f"RDAP returned {type(body).__name__}, expected an object")

    data = _parse(domain, body)
    info(log, "rdap.resolved", domain=domain, registered=data["registered"],
         age_years=data["age_years"], dropping=data["dropping"])
    return ok_result(SOURCE, data)


_bootstrap_cache: dict[str, str] | None = None
_IANA_BOOTSTRAP = "https://data.iana.org/rdap/dns.json"


async def _bootstrap_index() -> tuple[dict[str, str] | None, str | None]:
    """IANA's TLD -> RDAP server map, fetched once per process."""
    global _bootstrap_cache
    if _bootstrap_cache is None:
        body, error = await request_json(
            "GET", _IANA_BOOTSTRAP, timeout=_TIMEOUT_S, max_attempts=1)
        if error or not isinstance(body, dict):
            return None, error or "bootstrap returned an unexpected shape"
        index: dict[str, str] = {}
        for entry in body.get("services") or []:
            if isinstance(entry, list) and len(entry) == 2:
                tlds, urls = entry
                for name in tlds or []:
                    if urls:
                        index[str(name).lower()] = str(urls[0])
        # Cached for the process: IANA publishes this file roughly weekly.
        _bootstrap_cache = index
    return _bootstrap_cache, None


async def _via_iana_bootstrap(domain: str) -> tuple[dict | None, str | None]:
    """Ask the registry that actually holds the record.

    IANA publishes which RDAP server is authoritative for each TLD. Not
    every extension is listed — .io is absent today — so this is a fallback
    that can itself decline, which is why it returns an error rather than
    raising.
    """
    tld = domain.rsplit(".", 1)[-1]
    index, error = await _bootstrap_index()
    if index is None:
        return None, error

    base = index.get(tld.lower())
    if not base:
        return None, f"IANA lists no RDAP server for .{tld}"
    body, error = await request_json(
        "GET", f"{base.rstrip('/')}/domain/{domain}",
        headers={"Accept": "application/rdap+json"},
        timeout=_TIMEOUT_S, follow_redirects=True, max_attempts=1)
    if error:
        return None, error
    return (body, None) if isinstance(body, dict) else (None, "unexpected registry shape")


def _parse(domain: str, body: dict) -> dict:
    events = {}
    for event in body.get("events") or []:
        if isinstance(event, dict) and event.get("eventAction"):
            events[str(event["eventAction"]).lower()] = event.get("eventDate")

    statuses = [str(s).replace(" ", "").lower() for s in (body.get("status") or [])]
    registered_on = _as_date(events.get("registration"))

    registrar = None
    registrant_available = False
    for entity in body.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        roles = [str(r).lower() for r in (entity.get("roles") or [])]
        if "registrant" in roles:
            registrant_available = True
        if "registrar" in roles and registrar is None:
            registrar = _vcard_name(entity)

    return _result(
        domain,
        registered=True,
        registered_on=registered_on,
        expires_on=_as_date(events.get("expiration")),
        registrar=registrar,
        statuses=statuses,
        nameservers=[
            str(ns.get("ldhName")).lower()
            for ns in (body.get("nameservers") or []) if isinstance(ns, dict) and ns.get("ldhName")
        ],
        registrant_available=registrant_available,
    )


def _result(domain: str, *, registered: bool | None, registered_on: date | None = None,
            expires_on: date | None = None, registrar: str | None = None,
            statuses: list[str] | None = None, nameservers: list[str] | None = None,
            registrant_available: bool = False, status: str | None = None) -> dict:
    """Build RDAP result dict.
    
    Args:
        registered: True if registered, False if unregistered, None if unknown
        status: Optional status string like "unknown_via_rdap" for F19
    """
    statuses = statuses or []
    today = date.today()
    result = {
        "domain": domain,
        "registered": registered,
        "registered_on": registered_on.isoformat() if registered_on else None,
        "expires_on": expires_on.isoformat() if expires_on else None,
        # Held through every renewal decision since. An asset somebody has
        # paid to keep for twenty years is not the same as one bought in
        # January, whatever the two would fetch.
        "age_years": round((today - registered_on).days / 365.25, 1) if registered_on else None,
        "days_to_expiry": (expires_on - today).days if expires_on else None,
        "registrar": registrar,
        "statuses": statuses,
        "dropping": any(s in _DROPPING for s in statuses),
        "locked": any(s in _LOCKED for s in statuses),
        "nameservers": nameservers or [],
        # Stated explicitly because its absence is the norm and surprises
        # people: post-GDPR the registrant is redacted on .com and most
        # other registries, so this is almost always False.
        "registrant_available": registrant_available,
    }
    if status:
        result["status"] = status
    return result


def _vcard_name(entity: dict) -> str | None:
    """The display name out of an RDAP vCard array, which is jCard: a list
    of [name, params, type, value] rows rather than an object."""
    vcard = entity.get("vcardArray")
    if not isinstance(vcard, list) or len(vcard) < 2 or not isinstance(vcard[1], list):
        return entity.get("handle")
    for row in vcard[1]:
        if isinstance(row, list) and len(row) >= 4 and row[0] == "fn":
            return str(row[3])
    return entity.get("handle")


def _as_date(value) -> date | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return (parsed if parsed.tzinfo is None else parsed.astimezone(timezone.utc)).date()


def _mock_registration(domain: str) -> dict:
    from portfolio_agent.tools.mocks import _seed, _spread

    seed = _seed(domain, "rdap")
    if seed % 11 == 0:
        return _result(domain, registered=False)
    age_days = int(_spread(seed, 0, 200, 9000))
    registered_on = date.today() - __import__("datetime").timedelta(days=age_days)
    expires_on = date.today() + __import__("datetime").timedelta(days=int(_spread(seed, 1, 20, 900)))
    dropping = (seed % 17) == 0
    return _result(
        domain, registered=True, registered_on=registered_on, expires_on=expires_on,
        registrar=("Squarespace Domains II LLC", "GoDaddy.com, LLC", "Namecheap, Inc.")[seed % 3],
        statuses=["redemptionperiod"] if dropping else ["clienttransferprohibited"],
        nameservers=[f"ns1.{domain}", f"ns2.{domain}"],
        registrant_available=False,
    )
