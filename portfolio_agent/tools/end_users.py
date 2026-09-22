"""
find_potential_end_users — APVA Table 7.1.1's Crunchbase integration, and
the input to the Targeted Lead tier.

The thesis of that tier, in the spec's words: a specific, funded company
using a lesser domain, for which the domain "solves a credibility problem".
So a lead is only useful if we know three things — that they are funded,
that they are in the right niche, and what domain they are on *now*. The
third is what makes the gap measurable; without it there is a company, not a
lead.

**No LinkedIn.** Table 7.1.1 lists a LinkedIn alternative and marks it "High
Risk". It is not implemented and should not be: it would mean scraping
personal profiles for a valuation. Company-level fields only, which is also
what Crunchbase's organization search returns.

Mock-first. Real integration is behind CRUNCHBASE_API_KEY; without it the
tool reports that the signal is unavailable rather than substituting an
imagined buyer, because an invented lead produces an invented price at the
highest tier in the system.
"""

from portfolio_agent import config
from portfolio_agent.knowledge.lexicon import EMERGING_TLDS, PREMIUM_TLDS
from portfolio_agent.tools.common import ToolResult, error_result, mock_result, ok_result, request_json
from portfolio_agent.tools.linguistics import split_domain

SOURCE = "end_user_leads"

_TIMEOUT_S = 30.0
_MAX_LEADS = 10

# Company-level fields only. An allowlist rather than a denylist, so a field
# Crunchbase adds later cannot leak into a prompt or a log.
_LEAD_FIELDS = ("company", "funding_total", "announced_on", "categories", "employee_count", "current_domain")

# Prefixes and suffixes a company uses when it could not get the exact name —
# the visible symptom of the credibility gap this tier prices.
_COMPROMISE_AFFIXES = ("get", "try", "use", "join", "the", "my", "go", "hq", "app", "io", "hub", "co", "now")


async def find_potential_end_users(
    domain: str, category: str = "", keywords: list[str] | None = None
) -> ToolResult:
    """Funded companies in this domain's niche, with the domain each is on now.

    Returns `{leads: [...], qualified_count}` where each lead carries
    `company, funding_total, announced_on, categories, employee_count,
    current_domain` plus a computed `credibility_gap` — how much weaker
    their current domain is than this one.
    """
    domain = (domain or "").strip().lower()
    if not domain:
        return error_result(SOURCE, "domain is empty")
    keywords = [k for k in (keywords or []) if k]

    if config.MOCK_MODE:
        return mock_result(SOURCE, _build(domain, _mock_leads(domain, category, keywords)))

    if not config.CRUNCHBASE_API_KEY:
        # Fall back to our own funding collector before giving up. It carries
        # the same three facts the tier needs — who, how much, and what
        # domain they are on now — gathered from public funding
        # announcements rather than bought from a vendor. See
        # portfolio_agent/funding/.
        leads = await _leads_from_funding_store(domain, category, keywords)
        if leads:
            return ok_result(SOURCE, _build(domain, leads))
        return error_result(
            SOURCE,
            "no CRUNCHBASE_API_KEY configured and the funding collector has no matching "
            "companies — end-user profiling unavailable, so no targeted lead price",
        )

    body, error = await request_json(
        "POST",
        f"{config.CRUNCHBASE_BASE_URL}/v4/data/searches/organizations",
        headers={"X-cb-user-key": config.CRUNCHBASE_API_KEY},
        json={
            "field_ids": ["identifier", "funding_total", "last_funding_at", "categories", "num_employees_enum", "website"],
            "query": [{"type": "predicate", "field_id": "categories", "operator_id": "includes",
                       "values": keywords or [category]}],
            "limit": _MAX_LEADS,
        },
        timeout=_TIMEOUT_S,
    )
    if error:
        return error_result(SOURCE, error)
    entities = (body or {}).get("entities") if isinstance(body, dict) else None
    if entities is None:
        return error_result(SOURCE, "unexpected response shape")

    return ok_result(SOURCE, _build(domain, [_normalize(e) for e in entities[:_MAX_LEADS]]))


async def _leads_from_funding_store(domain: str, category: str, keywords: list[str]) -> list[dict]:
    """Funded companies from our own collector, in Crunchbase's shape.

    Only companies whose own domain we know are returned. A funding row
    without one is a company, not a lead — there is no gap to measure — and
    passing it through would let the tier price against a buyer whose
    situation we know nothing about.

    A funding announcement never carries the company's domain, so until a
    resolver existed this path returned nothing however much news the
    collector had gathered. It resolves the strongest few now, the same
    metered lookup the incumbent path uses and under the same cap.
    """
    from portfolio_agent.funding import store

    sld, _ = split_domain(domain)
    terms = [t for t in ([sld] + list(keywords or []) + ([category] if category else [])) if t]

    try:
        rows = store.open_store().search(
            keywords=terms, min_amount=config.MIN_LEAD_FUNDING,
            since=store.days_ago(config.FUNDING_LEAD_MAX_AGE_DAYS), limit=_MAX_LEADS,
        )
    except Exception:  # noqa: BLE001 — a missing store must not fail a valuation
        return []

    candidates = [
        {
            "company": row.get("company"),
            "funding_total": row.get("amount"),
            "announced_on": row.get("announced_on"),
            "categories": [c for c in (row.get("sector"), category) if c],
            "employee_count": None,  # not reported in funding announcements
            "current_domain": row.get("company_domain"),
            "evidence_url": row.get("article_url"),
            "lead_type": "funded_startup",
        }
        for row in rows if row.get("amount")
    ]
    if not candidates:
        return []

    # Resolve only what the collector could not supply, biggest rounds
    # first, and only up to the cap — the same budget discipline the
    # incumbent path follows, for the same reason.
    candidates.sort(key=lambda c: c.get("funding_total") or 0, reverse=True)
    unresolved = [c for c in candidates if not c.get("current_domain")]
    resolved = await _resolve_domains(
        unresolved[: config.MAX_LEAD_DOMAIN_LOOKUPS], domain)
    by_company = {c["company"]: c for c in resolved}
    merged = [by_company.get(c.get("company"), c) for c in candidates]
    return [c for c in merged if c.get("current_domain")]


async def find_public_incumbents(domain: str, keywords: list[str] | None = None) -> dict:
    """Established public companies in this domain's market.

    A second lead type, and one the funding collector structurally cannot
    see: a profitable incumbent does not appear in funding news because it
    has not raised, which is exactly why it has cash. Hartford Insurance
    files $28bn of revenue and would never surface in a search for recently
    funded companies.

    These come back as CANDIDATES. The targeted-lead tier prices the gap
    between the domain a company is on and the one being valued, and EDGAR
    does not report company websites — the field exists in the schema and is
    empty for most filers. Until that gap is measured there is a company,
    not a lead, and calling it one would price an imagined credibility
    problem.
    """
    from portfolio_agent.knowledge.industries import classify_industry
    from portfolio_agent.tools import edgar

    sld, _ = split_domain(domain)
    terms = [t for t in (keywords or []) if len(t) >= 3] or ([sld] if len(sld) >= 3 else [])
    if not terms:
        return {"candidates": [], "note": "no keyword long enough to match a company name"}

    industry = classify_industry(sld)
    found: list[dict] = []
    for term in terms[:2]:
        result = await edgar.find_public_incumbents(term, industry=industry, limit=5)
        if result["ok"]:
            found.extend((result["data"] or {}).get("candidates") or [])
        elif not found:
            return {"candidates": [], "note": result.get("error")}

    seen, unique = set(), []
    for candidate in found:
        if candidate.get("cik") not in seen:
            seen.add(candidate.get("cik"))
            unique.append(candidate)
    unique.sort(key=lambda c: c.get("revenue") or 0, reverse=True)
    shortlist = unique[:_MAX_LEADS]

    # Resolve the domain for the strongest few only. Each resolution is a
    # metered search, and the credibility gap is only worth measuring for
    # companies big enough to buy — spending a credit on the eighth-largest
    # match is how a month's budget disappears into a batch run.
    resolved = await _resolve_domains(shortlist[:config.MAX_LEAD_DOMAIN_LOOKUPS], domain)
    qualified = [c for c in resolved if c.get("current_domain")]
    return {
        "candidates": resolved + shortlist[config.MAX_LEAD_DOMAIN_LOOKUPS:],
        "industry": industry,
        "qualified_count": len(qualified),
        "note": (
            f"{len(qualified)} of {len(shortlist)} have a resolved domain and are leads; the rest "
            f"are candidates, because the credibility gap this tier prices is the distance "
            f"between the domain a company is ON and the one being valued"
            if qualified else
            "candidates, not qualified leads — no domain could be resolved, so no credibility "
            "gap can be measured and none has been priced"),
    }


async def _resolve_domains(candidates: list[dict], subject: str = "") -> list[dict]:
    """Attach each company's own domain, where a search can find it.

    EDGAR does not report websites, so this is the step that turns a filing
    into a lead. A failed or budget-exhausted lookup leaves `current_domain`
    None, which keeps the company a candidate — never a lead with an
    invented gap.
    """
    from portfolio_agent.tools.search import resolve_company_domain

    out = []
    for candidate in candidates:
        company = candidate.get("company")
        if not company:
            out.append(candidate)
            continue
        result = await resolve_company_domain(company)
        domain = (result["data"] or {}).get("domain") if result["ok"] else None
        # Computed here, where the domain first exists. Without it the lead
        # reaches the targeted tier with a gap of zero and is rejected by
        # the tier's own gate — collected, paid for, and then discarded one
        # step short of the price it exists to set.
        gap = credibility_gap(subject, domain) if (subject and domain) else 0.0
        out.append({**candidate, "current_domain": domain, "credibility_gap": gap,
                    "domain_source": "search" if domain else (result.get("error") or "unresolved")})
    return out


def _build(domain: str, leads: list[dict]) -> dict:
    scored = [dict(lead, credibility_gap=credibility_gap(domain, lead.get("current_domain"))) for lead in leads]
    scored.sort(key=lambda lead: (lead.get("funding_total") or 0), reverse=True)
    return {
        "domain": domain,
        "leads": scored,
        "qualified_count": sum(1 for lead in scored if (lead.get("funding_total") or 0) >= config.MIN_LEAD_FUNDING),
        "min_funding_threshold": config.MIN_LEAD_FUNDING,
    }


def _normalize(entity: dict) -> dict:
    properties = (entity or {}).get("properties") or {}
    website = properties.get("website") or {}
    return {
        "company": (properties.get("identifier") or {}).get("value"),
        "funding_total": (properties.get("funding_total") or {}).get("value_usd"),
        "announced_on": properties.get("last_funding_at"),
        "categories": [c.get("value") for c in properties.get("categories") or []],
        "employee_count": properties.get("num_employees_enum"),
        "current_domain": _hostname(website.get("value")),
    }


def _hostname(url: str | None) -> str | None:
    if not url:
        return None
    host = str(url).split("//")[-1].split("/")[0].lower()
    return host.removeprefix("www.") or None


def credibility_gap(target: str, current: str | None) -> float:
    """0-1: how much weaker the lead's current domain is than the target.

    0 means they are already on something just as good and the domain buys
    them nothing. 1 means a long, hyphenated, compromised name on a
    second-choice extension — the case where a premium domain visibly
    changes how the company reads.
    """
    if not current:
        return 0.0
    target_sld, target_tld = split_domain(target)
    current_sld, current_tld = split_domain(current)
    if current_sld == target_sld and current_tld == target_tld:
        return 0.0

    signals: list[float] = []

    # Extension. A funded company on a non-premium extension is the clearest
    # credibility gap there is.
    if current_tld in PREMIUM_TLDS:
        signals.append(0.0)
    elif current_tld in EMERGING_TLDS:
        signals.append(0.5)
    else:
        signals.append(1.0)

    # Length, relative to the target.
    if len(current_sld) > len(target_sld):
        signals.append(min(1.0, (len(current_sld) - len(target_sld)) / 8))
    else:
        signals.append(0.0)

    # Visible compromises: a hyphen, a digit, or a bolted-on affix around
    # the name they actually wanted.
    compromised = "-" in current_sld or any(ch.isdigit() for ch in current_sld)
    if not compromised:
        compromised = any(
            current_sld.startswith(affix) or current_sld.endswith(affix) for affix in _COMPROMISE_AFFIXES
        ) and target_sld in current_sld
    signals.append(1.0 if compromised else 0.0)

    return round(sum(signals) / len(signals), 4)


def _mock_leads(domain: str, category: str, keywords: list[str]) -> list[dict]:
    """Fixture leads, deterministic per domain.

    Not every domain gets a qualified buyer — roughly a third get none at
    all, which is the case the Targeted Lead tier has to handle correctly
    (stay None, warn, and do not guess). A fixture that always produced a
    funded buyer would leave that path untested.
    """
    from portfolio_agent.tools.mocks import _seed, _spread

    sld, tld = split_domain(domain)
    seed = _seed(domain, "end_users")

    if seed % 3 == 0:
        return []

    niche = (keywords or [category or "technology"])[0]
    count = 1 + (seed % 3)
    affixes = ("get", "try", "the", "my")
    return [
        {
            "company": f"{sld.capitalize()}{['Labs', 'Works', 'Technologies', 'Group'][i % 4]}",
            "funding_total": int(_spread(seed, i, 250_000, 180_000_000)),
            "announced_on": f"202{5 + (i % 2)}-0{1 + (i % 9)}-15",
            "categories": [niche, "software"],
            "employee_count": int(_spread(seed, i + 7, 5, 900)),
            # Deliberately weaker than the target: an affixed .com, a longer
            # name, or a second-choice extension.
            "current_domain": [
                f"{affixes[i % 4]}{sld}.com",
                f"{sld}app.io",
                f"{sld}-official.net",
            ][i % 3],
        }
        for i in range(count)
    ]
