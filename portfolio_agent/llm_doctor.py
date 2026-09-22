"""
Shows what portfolio_agent/llm.py actually resolved to, and optionally
proves it works.

    uv run python -m portfolio_agent.llm_doctor          # inspect only
    uv run python -m portfolio_agent.llm_doctor --ping   # make one real call per node

The point of the provenance column is the question you always end up asking
in production: not "what model is configured" but "why is it calling *that*
one". Every value says whether it came from a default in llm.py or from a
named environment variable.

--ping makes real, billable calls — one per distinct model, with a
three-token prompt. It is the only thing in this repo that calls a model
without being asked to by a graph, so it never runs by default.
"""

import asyncio
import sys

from portfolio_agent import config
from portfolio_agent.llm import (
    AUTO_DETECT_ORDER,
    NODE_PROFILES,
    PROFILES,
    describe,
    get_chat_model,
    resolve,
)

_GREEN, _RED, _DIM, _BOLD, _RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def _colour(text: str, code: str) -> str:
    return text if not sys.stdout.isatty() else f"{code}{text}{_RESET}"


def _mark(ok: bool) -> str:
    return _colour("  yes", _GREEN) if ok else _colour("   no", _RED)


def print_providers(info: dict) -> None:
    print(_colour("PROVIDERS", _BOLD) + _colour("   (edit PROVIDERS in portfolio_agent/llm.py)", _DIM))
    print(f"  {'provider':<12} {'credential env':<26} {'set':>5}  {'auto':>5}  default model")
    for provider in info["providers"]:
        auto = "  yes" if provider["auto_detectable"] else _colour("    -", _DIM)
        print(
            f"  {provider['key']:<12} {provider['credential_env']:<26} "
            f"{_mark(provider['configured'])}  {auto}  {provider['default_model']}"
        )
    print()
    print(f"  auto-detect order: {' → '.join(AUTO_DETECT_ORDER)}")
    active = info["active_provider"]
    print(f"  active provider:   {_colour(active, _GREEN) if active else _colour('none configured', _RED)}")
    print()


def print_profiles() -> None:
    print(_colour("PROFILES", _BOLD) + _colour("   (edit PROFILES / NODE_PROFILES in portfolio_agent/llm.py)", _DIM))
    for name, profile in PROFILES.items():
        used_by = [node for node, assigned in NODE_PROFILES.items() if assigned == name]
        print(f"  {_colour(name, _BOLD)}  temp={profile.temperature}  "
              f"max_tokens={profile.max_tokens}  timeout={profile.timeout_s}s  retries={profile.max_retries}")
        print(f"      used by: {', '.join(used_by) if used_by else 'any node without its own profile'}")
        if profile.purpose:
            print(_colour(f"      {profile.purpose}", _DIM))
    print()


def print_nodes(info: dict) -> None:
    print(_colour("WHAT EACH NODE WILL CALL", _BOLD))
    for node, resolved in info["nodes"].items():
        status = _colour("available", _GREEN) if resolved["available"] else _colour("unavailable", _RED)
        print(f"  {_colour(node, _BOLD)} → profile {resolved['profile']} [{status}]")
        print(f"      {resolved['provider'] or '-'} / {resolved['model'] or '-'} "
              f"  temp={resolved['temperature']}  max_tokens={resolved['max_tokens']}")
        print(f"      {resolved['reason']}")
        for field, source in sorted(resolved["sources"].items()):
            print(_colour(f"          {field:<12} from {source}", _DIM))
    print()


def print_mock(info: dict) -> None:
    if not info["mock_mode"]:
        return
    print(_colour("MOCK MODE IS ON", _BOLD))
    if info["llm_disabled_by_mock"]:
        print("  No model will be called at all: mock mode covers the LLM, because a model call is an")
        print("  external call. Every LLM-dependent node degrades to its documented fallback.")
        print("  Set PORTFOLIO_AGENT_MOCK_ALLOW_LLM=true to use fixture data with a real model.")
    else:
        print("  PORTFOLIO_AGENT_MOCK_ALLOW_LLM is set: fixture data, real (billable) model calls.")
    print()


async def ping(node: str) -> tuple[bool, str]:
    """One real call, as small as possible."""
    resolved = resolve(node=node)
    if not resolved.available:
        return False, f"skipped — {resolved.reason}"
    model = get_chat_model(node=node)
    if model is None:
        return False, "skipped — no model constructed"
    try:
        response = await model.ainvoke([("human", "Reply with the single word: ok")])
    except Exception as e:  # noqa: BLE001 — reporting the failure IS the job here
        return False, f"{type(e).__name__}: {e}"
    text = response.content if isinstance(response.content, str) else str(response.content)
    return True, f"replied {text.strip()[:60]!r}"


async def ping_all() -> int:
    print(_colour("PING", _BOLD) + _colour("   (real, billable calls — one per node)", _DIM))
    failures = 0
    for node in sorted(set(NODE_PROFILES) | {"default"}):
        ok, detail = await ping(node)
        failures += not ok
        print(f"  {_mark(ok)}  {node:<18} {detail}")
    print()
    return failures


def main(argv: list[str]) -> int:
    info = describe()

    print()
    print(_colour("LLM configuration — portfolio_agent/llm.py", _BOLD))
    print(_colour("Edit that file to change defaults; every value can also be set by env var.", _DIM))
    print()

    print_mock(info)
    print_providers(info)
    print_profiles()
    print_nodes(info)

    unavailable = [node for node, resolved in info["nodes"].items() if not resolved["available"]]
    if "--ping" in argv:
        if config.MOCK_MODE and not config.MOCK_ALLOW_LLM:
            print("  --ping does nothing in mock mode. Unset PORTFOLIO_AGENT_MOCK_MODE, or set")
            print("  PORTFOLIO_AGENT_MOCK_ALLOW_LLM=true, to make real calls.\n")
            return 1
        return 1 if asyncio.run(ping_all()) else 0

    if unavailable:
        print(f"  {len(unavailable)} node(s) have no model: {', '.join(unavailable)}")
        print("  Set a credential (e.g. MISTRAL_API_KEY) in .env, then re-run with --ping to verify.\n")
        return 1

    print("  All nodes have a model. Re-run with --ping to make one real call each and prove it.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
