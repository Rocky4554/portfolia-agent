"""
Offline regression eval for the LLM configuration page.
Run with: uv run python -m portfolio_agent.evals.test_llm_config

Constructs nothing and calls nothing — `llm.resolve` is pure, which is the
whole reason it is separate from `llm.get_chat_model`. What is checked here
is the part that silently goes wrong: which provider wins, which override
beats which, and that mock mode really does stop a model being called.

That last one is not a hypothetical. Before it was enforced, the end-to-end
eval appeared to run offline while making one live, billable call per golden
domain.
"""

import os
import sys
from contextlib import contextmanager

from portfolio_agent import config, llm


class Checks:
    def __init__(self):
        self.total = 0
        self.failures: list[str] = []

    def expect(self, condition, message: str):
        self.total += 1
        if not condition:
            self.failures.append(message)


# Every credential the page knows about, so a key in the developer's real
# environment cannot leak into a test and change which provider wins.
_ALL_CREDENTIAL_ENVS = [spec.credential_env for spec in llm.PROVIDERS.values() if spec.credential_env]
_ALL_OVERRIDE_ENVS = [name for name in os.environ if name.startswith("PORTFOLIO_AGENT_LLM_")]


@contextmanager
def env(**overrides):
    """Run with exactly the given LLM environment and nothing else."""
    saved = dict(os.environ)
    try:
        for name in set(_ALL_CREDENTIAL_ENVS + _ALL_OVERRIDE_ENVS):
            os.environ.pop(name, None)
        for name in list(os.environ):
            if name.startswith("PORTFOLIO_AGENT_LLM_"):
                os.environ.pop(name, None)
        for name, value in overrides.items():
            if value is not None:
                os.environ[name] = str(value)
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


@contextmanager
def mock_mode(on: bool, allow_llm: bool = False):
    was_mock, was_allow = config.MOCK_MODE, config.MOCK_ALLOW_LLM
    config.MOCK_MODE, config.MOCK_ALLOW_LLM = on, allow_llm
    try:
        yield
    finally:
        config.MOCK_MODE, config.MOCK_ALLOW_LLM = was_mock, was_allow


def check_registry(c: Checks):
    for key, spec in llm.PROVIDERS.items():
        c.expect(spec.key == key, f"PROVIDERS key {key!r} disagrees with its spec.key {spec.key!r}")
        c.expect(bool(spec.default_model), f"provider {key!r} has no default model")
        c.expect(bool(spec.package), f"provider {key!r} does not say which package it needs")

    unknown = set(llm.AUTO_DETECT_ORDER) - set(llm.PROVIDERS)
    c.expect(not unknown, f"AUTO_DETECT_ORDER names providers that do not exist: {sorted(unknown)}")

    # A credential-free provider must never auto-detect: it would count as
    # "configured" on any machine with the package installed and silently
    # win over a real one.
    for key in llm.AUTO_DETECT_ORDER:
        c.expect(bool(llm.PROVIDERS[key].credential_env),
                 f"{key!r} is auto-detectable but needs no credential — it would always win")

    # Every node must map to a profile that exists, or it silently gets
    # "default" settings that nobody chose for it.
    for node, profile in llm.NODE_PROFILES.items():
        c.expect(profile in llm.PROFILES, f"node {node!r} maps to unknown profile {profile!r}")
    c.expect("default" in llm.PROFILES, "there must be a 'default' profile")


def check_auto_detection(c: Checks):
    with mock_mode(False):
        with env(MISTRAL_API_KEY="k"):
            resolved = llm.resolve()
            c.expect(resolved.spec and resolved.spec.key == "mistral",
                     f"a lone MISTRAL_API_KEY should select mistral, got {resolved.spec and resolved.spec.key}")
            c.expect(resolved.model == llm.PROVIDERS["mistral"].default_model,
                     f"should fall back to the provider's default model, got {resolved.model}")
            c.expect("MISTRAL_API_KEY" in resolved.sources["provider"],
                     f"provenance should name the credential, got {resolved.sources['provider']}")

        # Order decides, not which key happens to be set.
        with env(MISTRAL_API_KEY="k", AWS_BEARER_TOKEN_BEDROCK="k", ANTHROPIC_API_KEY="k"):
            first = llm.AUTO_DETECT_ORDER[0]
            c.expect(llm.resolve().spec.key == first,
                     f"with several credentials set, {first!r} should win per AUTO_DETECT_ORDER")

        with env():
            resolved = llm.resolve()
            c.expect(not resolved.available and resolved.spec is None,
                     "with no credentials at all there must be no model")
            c.expect("MISTRAL_API_KEY" in resolved.reason or "provider" in resolved.reason.lower(),
                     f"the reason should say what to set, got {resolved.reason!r}")


def check_overrides(c: Checks):
    with mock_mode(False):
        # An explicit provider beats auto-detection.
        with env(AWS_BEARER_TOKEN_BEDROCK="k", ANTHROPIC_API_KEY="k", PORTFOLIO_AGENT_LLM_PROVIDER="anthropic"):
            c.expect(llm.resolve().spec.key == "anthropic", "PORTFOLIO_AGENT_LLM_PROVIDER must win")

        with env(MISTRAL_API_KEY="k", PORTFOLIO_AGENT_LLM_MODEL="mistral-small-latest"):
            c.expect(llm.resolve().model == "mistral-small-latest", "PORTFOLIO_AGENT_LLM_MODEL must win")

        # A per-profile override beats the global one, for that node only.
        with env(MISTRAL_API_KEY="k", ANTHROPIC_API_KEY="k",
                 PORTFOLIO_AGENT_LLM_PROVIDER="mistral",
                 PORTFOLIO_AGENT_LLM_RATIONALE_PROVIDER="anthropic"):
            c.expect(llm.resolve(node="rationale").spec.key == "anthropic",
                     "a per-profile provider override must beat the global one")
            c.expect(llm.resolve(node="parse_instruction").spec.key == "mistral",
                     "a per-profile override must not leak to other nodes")

        with env(MISTRAL_API_KEY="k", PORTFOLIO_AGENT_LLM_TEMPERATURE="0.9"):
            c.expect(llm.resolve(node="rationale").profile.temperature == 0.9,
                     "a global temperature override must apply")

        with env(MISTRAL_API_KEY="k",
                 PORTFOLIO_AGENT_LLM_TEMPERATURE="0.9",
                 PORTFOLIO_AGENT_LLM_RATIONALE_TEMPERATURE="0.25"):
            c.expect(llm.resolve(node="rationale").profile.temperature == 0.25,
                     "a per-profile temperature must beat the global one")
            c.expect(llm.resolve(node="parse_instruction").profile.temperature == 0.9,
                     "the global temperature still applies to other nodes")

        # A malformed override must not take the graph down.
        with env(MISTRAL_API_KEY="k", PORTFOLIO_AGENT_LLM_TEMPERATURE="hot"):
            resolved = llm.resolve(node="rationale")
            c.expect(resolved.profile.temperature == llm.PROFILES["rationale"].temperature,
                     "a malformed temperature must fall back to the profile default")
            c.expect("ignored" in resolved.sources["temperature"],
                     f"the provenance must say it was ignored, got {resolved.sources['temperature']}")

        with env(MISTRAL_API_KEY="k", PORTFOLIO_AGENT_LLM_PROVIDER="not-a-provider"):
            resolved = llm.resolve()
            c.expect(not resolved.available and "Unknown provider" in resolved.reason,
                     f"an unknown provider must be reported clearly, got {resolved.reason!r}")


def check_profiles_match_intent(c: Checks):
    """The two settings that are a correctness decision, not a preference."""
    c.expect(llm.PROFILES["manager"].temperature == 0.0,
             "the manager profile emits domain/folder ids — it must run at temperature 0")
    c.expect(llm.PROFILES["default"].temperature == 0.0, "the default profile must be deterministic")
    c.expect(0 < llm.PROFILES["rationale"].temperature <= 0.5,
             f"the rationale profile should be slightly warm, got {llm.PROFILES['rationale'].temperature}")


def check_mock_mode_blocks_calls(c: Checks):
    with env(MISTRAL_API_KEY="k"):
        with mock_mode(True):
            resolved = llm.resolve(node="rationale")
            c.expect(not resolved.available, "mock mode must make every node unavailable")
            c.expect("Mock mode" in resolved.reason, f"the reason must say so, got {resolved.reason!r}")
            c.expect(llm.get_chat_model(node="rationale") is None,
                     "mock mode must return no model — a model call is an external call")
            c.expect(llm.get_chat_model() is None, "mock mode must apply to the default node too")

        with mock_mode(True, allow_llm=True):
            c.expect(llm.resolve(node="rationale").available,
                     "PORTFOLIO_AGENT_MOCK_ALLOW_LLM must opt back in to real calls")


def check_describe(c: Checks):
    with mock_mode(False), env(MISTRAL_API_KEY="k"):
        info = llm.describe()
        c.expect(info["active_provider"] == "mistral", f"describe() got {info['active_provider']!r}")
        c.expect(set(info["nodes"]) >= set(llm.NODE_PROFILES) | {"default"},
                 "describe() must cover every mapped node plus the default")
        for node, resolved in info["nodes"].items():
            c.expect(bool(resolved["sources"]), f"describe() gives no provenance for {node!r}")
            c.expect(resolved["profile"] in llm.PROFILES, f"{node!r} reports an unknown profile")
        c.expect(any(p["key"] == "mistral" and p["configured"] for p in info["providers"]),
                 "describe() should report mistral as configured")


def main() -> int:
    c = Checks()
    for check in (
        check_registry,
        check_auto_detection,
        check_overrides,
        check_profiles_match_intent,
        check_mock_mode_blocks_calls,
        check_describe,
    ):
        check(c)

    if c.failures:
        print(f"FAILED {len(c.failures)}/{c.total} checks:")
        for failure in c.failures:
            print(f"  - {failure}")
        return 1
    print(f"PASSED {c.total}/{c.total} checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
