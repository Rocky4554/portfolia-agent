"""
The public HTTP surface: a generic web-search endpoint any app can call,
mounted onto the LangGraph API server via langgraph.json's "http.app" — no
separate service, same container, same deploy.

Kept separate from the two graphs on purpose. resolve_company_domain and
serp_composition (tools/search.py) are shaped for this agent's own
valuation logic — a domain, a commercial-intent verdict. A caller outside
this project — another app looking up a domain lead, a company, anything —
wants raw results, not that interpretation. web_search() in tools/search.py
is that generic path: same SearXNG-first, Tavily-fallback order, same disk
cache, no valuation-specific shape.

Authenticated with a single shared key (X-API-Key) against PUBLIC_API_KEY,
independent of whatever auth (if any) the LangGraph platform itself applies
to the graph routes — this endpoint is meant to be called by other apps
with no LangGraph client, so it needs to work on its own.
"""

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from portfolio_agent import config
from portfolio_agent.logging_setup import get_logger, warn
from portfolio_agent.tools.search import web_search

log = get_logger(__name__)


def _unauthorized(reason: str) -> JSONResponse:
    warn(log, "api.unauthorized", reason=reason)
    return JSONResponse({"error": reason}, status_code=401)


async def search_endpoint(request: Request) -> JSONResponse:
    if not config.PUBLIC_API_KEY:
        return JSONResponse({"error": "PUBLIC_API_KEY is not configured on this deployment"},
                            status_code=503)
    if request.headers.get("x-api-key") != config.PUBLIC_API_KEY:
        return _unauthorized("missing or invalid X-API-Key header")

    query = request.query_params.get("q", "")
    if not query.strip():
        return JSONResponse({"error": "query param 'q' is required"}, status_code=400)

    try:
        limit = int(request.query_params.get("limit", "8"))
    except ValueError:
        return JSONResponse({"error": "'limit' must be an integer"}, status_code=400)

    result = await web_search(query, limit)
    if not result["ok"]:
        return JSONResponse({"error": result["error"]}, status_code=502)
    return JSONResponse(result["data"])


async def health_endpoint(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


app = Starlette(routes=[
    Route("/search", search_endpoint, methods=["GET"]),
    Route("/search/health", health_endpoint, methods=["GET"]),
])
