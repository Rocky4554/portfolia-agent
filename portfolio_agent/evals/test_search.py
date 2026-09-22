"""
Offline regression eval for the metered search layer.
Run with: uv run python -m portfolio_agent.evals.test_search

This is the only layer here that costs money per call, so most of what
matters is about not making calls. Tavily gives 1,000 credits a month and
Exa about 1,400 searches for $10; a valuation that casually issues three
searches burns the month in a hundred domains, and finds out when a batch
comes back empty and a tier silently goes unpriced.

What is asserted, in order of importance:

  - Absence of evidence is not evidence. The first SERP reader defined
    commercial as "not editorial", so an invented word — whose results are
    merely unrelated, none of them Wikipedia — scored 100% commercial
    intent. Commercial now needs positive evidence.
  - A stale cache does not outlive a fixed classifier. That wrong verdict
    would have been served for ninety days after the fix without a cache
    version.
  - The cache actually prevents calls, since it is worth more than every
    other saving combined.
  - The budget refuses rather than overspends, and a throttle is reported
    as a failure, never as "this keyword has no commercial intent".
"""

import asyncio
import sys
import uuid

from portfolio_agent import config
from portfolio_agent.tools import search


class Checks:
    def __init__(self):
        self.total = 0
        self.failures = []

    def expect(self, condition, message):
        self.total += 1
        if not condition:
            self.failures.append(message)


def check_commercial_needs_positive_evidence(c: Checks):
    """The bug this exists for: zylantix at 100% commercial intent."""
    unrelated = [{"url": f"https://random{i}.example/page", "title": "An unrelated page"}
                 for i in range(8)]
    commercial, editorial, unknown = search._classify_results(unrelated)
    c.expect(commercial == 0,
             f"pages that are merely unrelated must not count as commercial, got {commercial}")
    c.expect(unknown == 8, f"they must be counted as unknown, got {unknown}")

    real_commercial = [
        {"url": "https://www.policygenius.com/auto-insurance/compare-car-insurance-quotes",
         "title": "Compare car insurance quotes"},
        {"url": "https://www.libertymutual.com/vehicle/auto-insurance",
         "title": "Car Insurance | Get an Auto Insurance Quote"},
    ]
    commercial, _, _ = search._classify_results(real_commercial)
    c.expect(commercial == 2, f"genuine commerce pages must be recognised, got {commercial}")

    editorial_pages = [{"url": "https://en.wikipedia.org/wiki/Photosynthesis", "title": "Photosynthesis"},
                       {"url": "https://www.britannica.com/science/photosynthesis", "title": "Photosynthesis"}]
    _, editorial, _ = search._classify_results(editorial_pages)
    c.expect(editorial == 2, f"reference pages must be recognised, got {editorial}")


def check_thin_evidence_is_unclassified(c: Checks):
    """Unclassified costs the domain nothing, which is where a coined word
    belongs — and the threshold that actually moves a price is the
    informational one, so being strict here is free."""
    from portfolio_agent.knowledge.intent import VOLUME_CREDIT

    c.expect(VOLUME_CREDIT["unclassified"] == VOLUME_CREDIT["commercial"] == 1.0,
             "an unclassified verdict must cost exactly what a commercial one does")
    c.expect(VOLUME_CREDIT["informational"] < 1.0,
             "and only the informational verdict may discount")
    c.expect(config.SERP_MIN_CLASSIFIED >= 2,
             "a verdict from one readable result is not a verdict")


def check_cache_prevents_calls(c: Checks):
    """Worth more than every other saving combined."""
    was_mock = config.MOCK_MODE
    config.MOCK_MODE = False
    original_search = search._search
    calls = []

    async def counting_search(provider, query, limit):
        calls.append(provider)
        return ([{"url": "https://acme-test-widget.example/", "title": "Acme"}], None, 0.001)

    search._search = counting_search
    try:
        # Unique per run. A shared name makes the FIRST call a cache hit on
        # every run after the first, so the test would pass while measuring
        # nothing — which is worse than not having it.
        company = f"Cache Probe Industries {uuid.uuid4().hex[:12]} LLC"
        asyncio.run(search.resolve_company_domain(company))
        first = len(calls)
        asyncio.run(search.resolve_company_domain(company))
        second = len(calls)
        c.expect(first == 1, f"the first lookup must call the provider, got {first}")
        c.expect(second == 1, f"the second must be served from cache, got {second} calls")
    finally:
        search._search = original_search
        config.MOCK_MODE = was_mock


def check_cache_is_versioned(c: Checks):
    """A fixed classifier must not keep serving its own old mistakes."""
    import inspect

    source = inspect.getsource(search)
    c.expect("_CACHE_VERSION" in source, "the cache must carry a version")
    c.expect("_CACHE_VERSION" in inspect.getsource(search._cached)
             and "_CACHE_VERSION" in inspect.getsource(search._cache),
             "and both read and write paths must apply it")


def check_budget_refuses_rather_than_overspends(c: Checks):
    was_mock = config.MOCK_MODE
    original_cap = dict(config.SEARCH_MONTHLY_CAP)
    config.MOCK_MODE = False
    config.SEARCH_MONTHLY_CAP["tavily"] = 0
    try:
        result = asyncio.run(search.serp_composition("a keyword never cached before xyzzy"))
        c.expect(not result["ok"], "an exhausted budget must refuse")
        error = result.get("error") or ""
        c.expect("budget exhausted" in error, f"and must say why, got {error!r}")
        c.expect("SEARCH_MONTHLY_CAP" in error, "and must name the lever to raise it")
        c.expect("no commercial" not in error.lower() and "informational" not in error.lower(),
                 "a budget failure must never read as a verdict about the keyword")
    finally:
        config.SEARCH_MONTHLY_CAP.clear()
        config.SEARCH_MONTHLY_CAP.update(original_cap)
        config.MOCK_MODE = was_mock


def check_directories_are_not_company_domains(c: Checks):
    """Returning linkedin.com as a company's domain would give every lead
    the same fabricated credibility gap."""
    for host in ("linkedin.com", "www.crunchbase.com", "bloomberg.com", "sec.gov"):
        c.expect(any(bad in host for bad in search._NOT_A_HOMEPAGE),
                 f"{host} must be excluded as a company homepage")
    c.expect(search._registrable("https://www.thehartford.com/about") == "thehartford.com",
             "a real homepage must reduce to its registrable domain")
    c.expect(search._registrable("not a url") is None, "junk must not become a domain")


def check_intent_falls_back_to_lexicon_first(c: Checks):
    """Search runs only where the free path is silent — the difference
    between a credit per valuation and a credit occasionally."""
    import inspect

    from portfolio_agent.knowledge import intent

    source = inspect.getsource(intent.resolve_intent)
    c.expect('!= "unclassified"' in source and "return" in source,
             "the lexicon result must short-circuit before any search call")
    c.expect(source.index("classify_intent") < source.index("serp_composition"),
             "the lexicon must be consulted first, not as a fallback after spending")

    was_mock = config.MOCK_MODE
    config.MOCK_MODE = True
    try:
        resolved = asyncio.run(intent.resolve_intent("carinsurancequotes"))
    finally:
        config.MOCK_MODE = was_mock
    c.expect(resolved["source"] == "lexicon",
             f"a name the lexicon knows must not reach a paid search, got {resolved['source']}")


def main() -> int:
    c = Checks()
    for fn in (check_commercial_needs_positive_evidence, check_thin_evidence_is_unclassified,
               check_cache_prevents_calls, check_cache_is_versioned,
               check_budget_refuses_rather_than_overspends,
               check_directories_are_not_company_domains,
               check_intent_falls_back_to_lexicon_first):
        fn(c)
    if c.failures:
        print(f"FAILED {len(c.failures)}/{c.total} checks:")
        for failure in c.failures:
            print(f"  - {failure}")
        return 1
    print(f"PASSED {c.total}/{c.total} checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
