"""
THE LLM CONFIGURATION PAGE.

Everything about which model answers, how, and with what settings lives in
this one file. Nothing else in the repo imports a provider class, names a
model, or sets a temperature — nodes call `get_chat_model(node=...)` and get
back a plain LangChain chat model. That is what makes swapping models a
configuration change rather than a code change.

There are three things you can configure here, in increasing specificity,
and each can be set either by editing the defaults below or by setting an
environment variable (env always wins, so a deployment can change models
without a deploy):

  1. WHICH PROVIDER.  `PROVIDERS` lists every provider this repo knows how
     to construct, with the credential it needs and the package it wants.
     `AUTO_DETECT_ORDER` decides which one wins when several are
     configured. Override outright with PORTFOLIO_AGENT_LLM_PROVIDER.

  2. WHICH MODEL, AND HOW IT GENERATES.  `PROFILES` holds one named profile
     per job — temperature, token cap, timeout, retries. Different jobs
     genuinely want different settings: writing a rationale wants a little
     warmth, emitting a structured action plan wants none at all.

  3. WHICH PROFILE EACH NODE USES.  `NODE_PROFILES` maps a graph node to a
     profile, so you can point the rationale node at a strong model and the
     manager node at a cheap fast one without touching either node.

Run `uv run python -m portfolio_agent.llm_doctor` to see the whole resolved
picture — what is configured, what each node will actually call, and where
every value came from — and `--ping` to prove the credentials work.

Mock mode (config.MOCK_MODE) disables all of it and returns None, because a
model call is an external call. See `_mock_blocks_llm`.
"""

import os
from dataclasses import dataclass, replace

from langchain.chat_models import init_chat_model

from portfolio_agent import config
from portfolio_agent.logging_setup import debug, get_logger, info, warn

log = get_logger(__name__)

# ── 1. Providers ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProviderSpec:
    """One provider this repo knows how to construct."""

    key: str  # our name for it, and the value PORTFOLIO_AGENT_LLM_PROVIDER takes
    label: str  # for humans, in the doctor output
    langchain_provider: str  # what init_chat_model wants
    credential_env: str  # the env var holding the key ("" = no credential needed)
    default_model: str
    package: str  # the pip package that must be installed
    notes: str = ""


PROVIDERS: dict[str, ProviderSpec] = {
    "mistral": ProviderSpec(
        key="mistral",
        label="Mistral AI",
        langchain_provider="mistralai",
        credential_env="MISTRAL_API_KEY",
        default_model="ministral-8b-latest",
        package="langchain-mistralai",
        notes=(
            "Default verified against this account on 2026-09-21: mistral-large-latest is not in the "
            "account's model list at all and returns tier_not_allowed, and mistral-medium/small-latest "
            "return a persistent 429 rather than a transient one — this key's tier only serves "
            "ministral-8b. If the subscription changes, re-run llm_doctor --ping against a bigger model "
            "rather than assuming. Note that '-latest' aliases track Mistral's current release; pin an "
            "explicit version (e.g. ministral-8b-2512) if you need reproducible output."
        ),
    ),
    "anthropic": ProviderSpec(
        key="anthropic",
        label="Anthropic",
        langchain_provider="anthropic",
        credential_env="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-5",
        package="langchain-anthropic",
    ),
    "openai": ProviderSpec(
        key="openai",
        label="OpenAI",
        langchain_provider="openai",
        credential_env="OPENAI_API_KEY",
        default_model="gpt-4o",
        package="langchain-openai",
    ),
    "google": ProviderSpec(
        key="google",
        label="Google Gemini",
        langchain_provider="google_genai",
        credential_env="GOOGLE_API_KEY",
        default_model="gemini-2.0-flash",
        package="langchain-google-genai",
    ),
    "groq": ProviderSpec(
        key="groq",
        label="Groq",
        langchain_provider="groq",
        credential_env="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
        package="langchain-groq",
    ),
    "bedrock": ProviderSpec(
        key="bedrock",
        label="Amazon Bedrock",
        langchain_provider="bedrock_converse",
        credential_env="AWS_BEARER_TOKEN_BEDROCK",
        default_model="deepseek.v3.2",
        package="langchain-aws",
        notes=(
            "AWS_BEARER_TOKEN_BEDROCK is AWS's own env var for a Bedrock API key — botocore reads it "
            "directly, no AWS_ACCESS_KEY_ID needed. The default model id was verified against this "
            "account's list_foundation_models output; if it starts failing with ValidationException, "
            "re-run that rather than guessing a new id."
        ),
    ),
    "ollama": ProviderSpec(
        key="ollama",
        label="Ollama (local)",
        langchain_provider="ollama",
        credential_env="",  # local, no credential
        default_model="llama3.1",
        package="langchain-ollama",
        notes="Needs a local Ollama daemon. Never auto-detected — it has no credential to detect.",
    ),
}

# Which provider wins when more than one credential is present. Reorder this
# to change the preference. Ollama is deliberately absent: it has no
# credential, so including it would make it always "available" and it would
# silently win on a machine that merely has the package installed.
AUTO_DETECT_ORDER: tuple[str, ...] = ("mistral", "anthropic", "openai", "google", "groq", "bedrock")


# ── 2. Profiles ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelProfile:
    """How one job should generate.

    `provider` and `model` of None mean "whatever the resolved default is",
    which is what keeps a single PORTFOLIO_AGENT_LLM_PROVIDER switch working
    for every node at once. Set them to pin one job to a specific model.
    """

    name: str
    provider: str | None = None
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout_s: float | None = 60.0
    max_retries: int = 2
    purpose: str = ""


PROFILES: dict[str, ModelProfile] = {
    "default": ModelProfile(
        name="default",
        temperature=0.0,
        purpose="anything without a profile of its own",
    ),
    "rationale": ModelProfile(
        name="rationale",
        temperature=0.2,
        max_tokens=900,
        purpose=(
            "the valuation narrative. Slight warmth because this is the one place the system writes "
            "prose a person reads, and temperature 0 produces stilted repetition. It is safe to allow "
            "here precisely because the output is verified: every numeral is checked against the "
            "structured valuation before it is returned (nodes/rationale.py)."
        ),
    ),
    "manager": ModelProfile(
        name="manager",
        temperature=0.0,
        purpose=(
            "the portfolio manager's instruction parser. Temperature 0, no exceptions: this emits a "
            "structured action plan containing domain and folder ids, and creativity there means a "
            "hallucinated id."
        ),
    ),
}

# Which profile each node uses. A node not listed here gets "default".
NODE_PROFILES: dict[str, str] = {
    "rationale": "rationale",
    "parse_instruction": "manager",
}


# ── 3. Resolution ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResolvedModel:
    """A fully resolved decision, with provenance.

    `sources` records where each value came from — a default in this file or
    a named environment variable — because "why is it calling that model?"
    is the question you always end up asking, usually in production.
    """

    profile: ModelProfile
    spec: ProviderSpec | None
    model: str | None
    available: bool
    reason: str  # why it is or is not usable
    sources: dict[str, str]


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def provider_is_configured(spec: ProviderSpec) -> bool:
    """Ollama and anything else credential-free counts as configured only if
    explicitly selected — see AUTO_DETECT_ORDER."""
    return bool(_env(spec.credential_env)) if spec.credential_env else True


def configured_providers() -> list[ProviderSpec]:
    return [PROVIDERS[key] for key in AUTO_DETECT_ORDER if provider_is_configured(PROVIDERS[key])]


def _mock_blocks_llm() -> bool:
    """Mock mode covers the LLM too.

    A model call is an external call — it costs money, needs a key and makes
    a run non-deterministic — so "runs with no API keys and no network" has
    to include it, or an offline eval silently bills a live endpoint once per
    domain in its golden set. PORTFOLIO_AGENT_MOCK_ALLOW_LLM=true opts back
    in, which is what you want when iterating on a prompt.
    """
    return config.MOCK_MODE and not config.MOCK_ALLOW_LLM


def resolve(node: str | None = None, profile_name: str | None = None) -> ResolvedModel:
    """Work out exactly what `node` will call, and why.

    Resolution order for every value: an env var for this specific profile,
    then the global env var, then the profile's own field, then the
    provider's default. Pure — it constructs nothing and calls nothing.
    """
    name = profile_name or NODE_PROFILES.get(node or "", "default")
    profile = PROFILES.get(name, PROFILES["default"])
    prefix = f"PORTFOLIO_AGENT_LLM_{profile.name.upper()}"
    sources: dict[str, str] = {}

    # Generation settings: profile-specific env, then global env, then the
    # profile's own value.
    overrides = {}
    for field, env_suffix, cast in (
        ("temperature", "TEMPERATURE", float),
        ("max_tokens", "MAX_TOKENS", int),
        ("timeout_s", "TIMEOUT_S", float),
        ("max_retries", "MAX_RETRIES", int),
    ):
        for env_name in (f"{prefix}_{env_suffix}", f"PORTFOLIO_AGENT_LLM_{env_suffix}"):
            raw = _env(env_name)
            if raw:
                try:
                    overrides[field] = cast(raw)
                    sources[field] = env_name
                except ValueError:
                    # A malformed override must not take the graph down; the
                    # doctor reports it and the default stands.
                    sources[field] = f"{env_name} (ignored — {raw!r} is not a {cast.__name__})"
                break
        else:
            sources[field] = f"PROFILES[{profile.name!r}]"
    profile = replace(profile, **overrides)

    # Provider: profile-specific env, global env, the profile's pin, then
    # auto-detection.
    provider_key = _env(f"{prefix}_PROVIDER") or _env("PORTFOLIO_AGENT_LLM_PROVIDER")
    if provider_key:
        sources["provider"] = f"{prefix}_PROVIDER" if _env(f"{prefix}_PROVIDER") else "PORTFOLIO_AGENT_LLM_PROVIDER"
    elif profile.provider:
        provider_key, sources["provider"] = profile.provider, f"PROFILES[{profile.name!r}].provider"
    else:
        detected = configured_providers()
        if detected:
            provider_key = detected[0].key
            sources["provider"] = f"auto-detected from {detected[0].credential_env}"
        else:
            sources["provider"] = "nothing configured"

    if not provider_key:
        return ResolvedModel(
            profile, None, None, False,
            "No LLM provider configured. Set a credential (e.g. MISTRAL_API_KEY) or "
            "PORTFOLIO_AGENT_LLM_PROVIDER.",
            sources,
        )

    spec = PROVIDERS.get(provider_key)
    if spec is None:
        return ResolvedModel(
            profile, None, None, False,
            f"Unknown provider {provider_key!r}. Known: {', '.join(sorted(PROVIDERS))}.",
            sources,
        )

    # Model: profile-specific env, global env, the profile's pin, then the
    # provider's default.
    model = _env(f"{prefix}_MODEL") or _env("PORTFOLIO_AGENT_LLM_MODEL")
    if model:
        sources["model"] = f"{prefix}_MODEL" if _env(f"{prefix}_MODEL") else "PORTFOLIO_AGENT_LLM_MODEL"
    elif profile.model:
        model, sources["model"] = profile.model, f"PROFILES[{profile.name!r}].model"
    else:
        model, sources["model"] = spec.default_model, f"PROVIDERS[{spec.key!r}].default_model"

    if _mock_blocks_llm():
        return ResolvedModel(
            profile, spec, model, False,
            "Mock mode is on and PORTFOLIO_AGENT_MOCK_ALLOW_LLM is not set, so no model will be "
            "called. Every LLM-dependent node degrades to its documented fallback.",
            sources,
        )

    if not provider_is_configured(spec):
        return ResolvedModel(
            profile, spec, model, False,
            f"{spec.label} is selected but {spec.credential_env} is not set.",
            sources,
        )

    return ResolvedModel(profile, spec, model, True, f"{spec.label} / {model}", sources)


# ── 4. The factory every node calls ────────────────────────────────────────


def get_chat_model(temperature: float | None = None, *, node: str | None = None, profile: str | None = None):
    """A ready LangChain chat model, or None.

    Returns None rather than raising whenever no model should be called — no
    provider configured, a missing credential, or mock mode — so every caller
    can degrade to an honest fallback instead of failing a run that is
    otherwise complete. `resolve(...).reason` says which it was.

    `temperature` overrides the profile for this one call; prefer setting it
    on the profile so the choice is recorded in one place.
    """
    resolved = resolve(node=node, profile_name=profile)
    if not resolved.available or resolved.spec is None or resolved.model is None:
        # DEBUG, not WARNING: in mock mode this is the expected, correct
        # outcome, and a warning on every offline run is noise that teaches
        # people to ignore warnings.
        debug(log, "llm.unavailable", node=node or "default",
              profile=resolved.profile.name, reason=resolved.reason)
        return None

    info(log, "llm.resolved", node=node or "default", profile=resolved.profile.name,
         provider=resolved.spec.key, model=resolved.model,
         temperature=resolved.profile.temperature if temperature is None else temperature,
         provider_from=resolved.sources.get("provider"), model_from=resolved.sources.get("model"))

    settings = resolved.profile
    kwargs: dict = {
        "model_provider": resolved.spec.langchain_provider,
        "temperature": settings.temperature if temperature is None else temperature,
    }
    if settings.max_tokens is not None:
        kwargs["max_tokens"] = settings.max_tokens
    if settings.timeout_s is not None:
        kwargs["timeout"] = settings.timeout_s
    if settings.max_retries is not None:
        kwargs["max_retries"] = settings.max_retries
    if resolved.spec.key == "bedrock":
        kwargs["region_name"] = os.environ.get("BEDROCK_REGION", "us-east-1")

    try:
        return init_chat_model(resolved.model, **kwargs)
    except Exception as e:
        warn(log, "llm.kwargs_rejected", provider=resolved.spec.key, model=resolved.model,
             error=f"{type(e).__name__}: {e}",
             detail="retrying without max_tokens/timeout/max_retries")
        # Some providers reject kwargs others accept (max_retries and
        # timeout are the usual offenders). A settings mismatch should cost
        # the call its tuning, not the whole valuation.
        return init_chat_model(
            resolved.model,
            model_provider=resolved.spec.langchain_provider,
            temperature=kwargs["temperature"],
        )


def describe() -> dict:
    """The whole resolved picture, for llm_doctor and for diagnostics."""
    return {
        "mock_mode": config.MOCK_MODE,
        "mock_allow_llm": config.MOCK_ALLOW_LLM,
        "llm_disabled_by_mock": _mock_blocks_llm(),
        "providers": [
            {
                "key": spec.key,
                "label": spec.label,
                "credential_env": spec.credential_env or "(none needed)",
                "configured": provider_is_configured(spec),
                "auto_detectable": spec.key in AUTO_DETECT_ORDER,
                "default_model": spec.default_model,
                "package": spec.package,
                "notes": spec.notes,
            }
            for spec in PROVIDERS.values()
        ],
        "active_provider": (configured_providers()[0].key if configured_providers() else None),
        "nodes": {
            node: _describe_resolved(resolve(node=node))
            for node in sorted(set(NODE_PROFILES) | {"default"})
        },
    }


def _describe_resolved(resolved: ResolvedModel) -> dict:
    return {
        "profile": resolved.profile.name,
        "provider": resolved.spec.key if resolved.spec else None,
        "model": resolved.model,
        "temperature": resolved.profile.temperature,
        "max_tokens": resolved.profile.max_tokens,
        "timeout_s": resolved.profile.timeout_s,
        "max_retries": resolved.profile.max_retries,
        "available": resolved.available,
        "reason": resolved.reason,
        "sources": resolved.sources,
        "purpose": resolved.profile.purpose,
    }
