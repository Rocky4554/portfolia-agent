"""
Namebio comps lookup — the one external data source confirmed available for
Phase 1. Real HTTP call when NAMEBIO_API_KEY is set; otherwise falls back to
a clearly-labeled mock so the graph is runnable end to end before the key is
wired in. The synthesis node is told explicitly when it's looking at mock
data, so it never reports a mock comp as a real one.
"""

import httpx

from portfolio_agent.config import NAMEBIO_API_KEY, NAMEBIO_BASE_URL

_MOCK_COMPS = {
    "acronym": [
        {"domain": "ZQK.com", "price": 42000, "date": "2025-11-02", "venue": "Sedo"},
        {"domain": "VXR.com", "price": 38500, "date": "2025-08-14", "venue": "Namebio"},
        {"domain": "TZB.com", "price": 51000, "date": "2026-02-20", "venue": "Sedo"},
    ],
    "brandable": [
        {"domain": "Zylo.com", "price": 28000, "date": "2025-12-01", "venue": "Sedo"},
        {"domain": "Kindra.com", "price": 34000, "date": "2026-01-18", "venue": "Namebio"},
    ],
    "keyword": [
        {"domain": "CheapInsurance.com", "price": 61000, "date": "2025-09-30", "venue": "Sedo"},
        {"domain": "BestQuotes.com", "price": 22000, "date": "2026-03-05", "venue": "Namebio"},
    ],
    "generic": [
        {"domain": "GenericExample.com", "price": 15000, "date": "2025-10-10", "venue": "Sedo"},
    ],
    "typo": [],
}


async def fetch_comps(category: str, sld: str) -> dict:
    """Returns {"mock": bool, "comps": [...]}. Never raises — a comps-fetch
    failure degrades to an empty/mock result rather than blocking the whole
    valuation, since a missing comp is recoverable (the LLM can say so in its
    rationale) and a crashed graph is not.
    """
    if not NAMEBIO_API_KEY:
        return {"mock": True, "comps": _MOCK_COMPS.get(category, [])}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{NAMEBIO_BASE_URL}/Comps",
                headers={"Authorization": f"Bearer {NAMEBIO_API_KEY}"},
                json={"query": sld, "category": category, "limit": 10},
            )
            resp.raise_for_status()
            data = resp.json()
            return {"mock": False, "comps": data.get("results", [])}
    except (httpx.HTTPError, ValueError):
        return {"mock": True, "comps": _MOCK_COMPS.get(category, [])}
