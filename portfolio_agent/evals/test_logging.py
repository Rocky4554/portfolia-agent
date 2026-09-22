"""
Offline regression eval for the logging layer.
Run with: uv run python -m portfolio_agent.evals.test_logging

Logging is the one part of a system that is never exercised by the tests of
anything else — it is invisible when it works and invisible when it does
not. These checks hold it to the three promises that make it worth having:

  - **Correlation.** Every line of one valuation carries the same run id, and
    two valuations never share one. Without that, concurrent runs interleave
    into an unreadable stream.
  - **Coverage.** A full graph run emits a line at every step. A gap here is
    a blind spot in production that nobody discovers until they need it.
  - **Safety.** No credential reaches a log, and the redaction does not fire
    on harmless fields — a filter that cries wolf on `max_tokens` trains
    everyone to ignore it, which is how the one that mattered gets ignored.
"""

import asyncio
import io
import json
import logging
import sys

from portfolio_agent import config
from portfolio_agent.logging_setup import (
    LOGGER_NAME,
    REDACTED,
    get_logger,
    info,
    log_step,
    run_context,
    setup_logging,
)

# Events a full valuation must emit. Each is a question someone asks of a
# production log: what was it classified as, which tools ran and how long did
# they take, how was each tier reached, and did the model behave.
REQUIRED_EVENTS = (
    "classify.done",
    "evidence.plan_selected",
    "tool.done",
    "evidence.done",
    "tiers.inputs",
    "tier.computed",
    "synthesize.done",
    "rationale.done",
)


class Checks:
    def __init__(self):
        self.total = 0
        self.failures: list[str] = []

    def expect(self, condition, message: str):
        self.total += 1
        if not condition:
            self.failures.append(message)


def capture(level="DEBUG", fmt="text"):
    buffer = io.StringIO()
    setup_logging(level, fmt, stream=buffer, force=True)
    return buffer


def run_graph(domain: str) -> None:
    from portfolio_agent.nodes.graph import portfolio_agent

    asyncio.run(portfolio_agent.ainvoke({"domain": domain}))


def check_setup_is_polite(c: Checks):
    """Importing this package must not change how the host application logs."""
    root_before = list(logging.getLogger().handlers)
    capture()
    logger = logging.getLogger(LOGGER_NAME)

    c.expect(logging.getLogger().handlers == root_before,
             "setup_logging must not touch the root logger — this package is embedded in another service")
    c.expect(logger.propagate is False,
             "the package logger must not propagate, or every line is duplicated into the host's handlers")
    c.expect(bool(logger.handlers), "setup_logging should install a handler on the package logger")

    # Idempotent: a second call without force must not stack handlers.
    count = len(logger.handlers)
    setup_logging()
    c.expect(len(logger.handlers) == count, "setup_logging must be idempotent")


def check_run_correlation(c: Checks):
    buffer = capture("INFO")
    run_graph("cloud.ai")
    first = _run_ids(buffer.getvalue())

    buffer = capture("INFO")
    run_graph("insurance.com")
    second = _run_ids(buffer.getvalue())

    c.expect(len(first) == 1, f"one valuation must use exactly one run id, saw {sorted(first)}")
    c.expect(len(second) == 1, f"one valuation must use exactly one run id, saw {sorted(second)}")
    c.expect(first and second and first != second, "two valuations must not share a run id")


def _run_ids(text: str) -> set[str]:
    ids = set()
    for line in text.splitlines():
        if "[" in line and "]" in line:
            inside = line.split("[", 1)[1].split("]", 1)[0].split()
            if inside and inside[0] != "-":
                ids.add(inside[0])
    return ids


def check_step_coverage(c: Checks):
    buffer = capture("INFO")
    run_graph("carinsurance.com")
    output = buffer.getvalue()

    for event in REQUIRED_EVENTS:
        c.expect(event in output, f"a full valuation must log {event!r}")

    c.expect(output.count("tool.done") >= 4,
             f"every planned tool must log its own line, saw {output.count('tool.done')}")
    c.expect("duration_ms=" in output, "timed steps must report duration_ms")
    c.expect("carinsurance.com" in output, "log lines must name the domain being valued")

    # Every .done line carries a duration, or "which step was slow" is
    # unanswerable after the fact.
    for line in output.splitlines():
        if ".done" in line:
            c.expect("duration_ms=" in line, f"a .done line without a duration: {line[:90]}")


def check_secrets_never_logged(c: Checks):
    buffer = capture("DEBUG")
    log = get_logger("portfolio_agent.evals.test_logging")
    secret = "sk-live-THIS-MUST-NEVER-APPEAR-0123456789"

    with run_context("example.com"):
        info(log, "test.credentials", api_key=secret, MISTRAL_API_KEY=secret, token=secret,
             authorization=f"Bearer {secret}")
        # The case the filter exists for: a third-party exception whose text
        # contains a header we never chose to log.
        info(log, f"upstream rejected request with Authorization: Bearer {secret}")
        # Fields that merely look credential-shaped must survive intact.
        info(log, "test.harmless", max_tokens=900, tokens=31, segments="property+management",
             credential_env="MISTRAL_API_KEY", model="ministral-8b-latest")

    output = buffer.getvalue()
    c.expect(secret not in output, "a credential reached the log output")
    c.expect(output.count(REDACTED) >= 4, f"expected several redactions, saw {output.count(REDACTED)}")
    c.expect("max_tokens=900" in output, "max_tokens must NOT be redacted — it is not a credential")
    c.expect("tokens=31" in output, "a token COUNT must not be redacted")
    c.expect("credential_env=MISTRAL_API_KEY" in output,
             "the NAME of a credential env var is safe and useful; only its value is secret")
    c.expect("model=ministral-8b-latest" in output, "the model name must not be redacted")


def check_untrusted_payloads_not_dumped(c: Checks):
    """Tool logs must carry summaries, not upstream bodies.

    Those bodies hold both PII (registrant names and emails) and whatever
    text an attacker put on a web page, and a log viewer is just another
    place that text gets read.
    """
    buffer = capture("INFO")
    run_graph("carinsurance.com")
    output = buffer.getvalue()

    for leak in ("whois_reg_org", "whois_reg_email", "site_emails", "'comps':", "descriptions"):
        c.expect(leak not in output, f"a raw upstream payload field reached the log: {leak!r}")


def check_json_format(c: Checks):
    buffer = capture("INFO", "json")
    run_graph("cloud.ai")

    lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
    c.expect(bool(lines), "json mode produced no output")

    parsed = []
    for line in lines:
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            c.expect(False, f"json mode emitted a line that is not JSON: {line[:90]}")

    for record in parsed:
        for field in ("ts", "level", "logger", "event", "run_id", "domain"):
            c.expect(field in record, f"a JSON record is missing {field!r}: {record}")

    c.expect(any(r["event"] == "tier.computed" and "point" in r for r in parsed),
             "structured fields must be top-level keys in JSON mode, not buried in the message")
    c.expect(len({r["run_id"] for r in parsed}) == 1, "JSON records from one run must share a run id")


def check_log_step_failure_path(c: Checks):
    buffer = capture("DEBUG")
    log = get_logger("portfolio_agent.evals.test_logging")

    raised = False
    try:
        with log_step(log, "test.step") as step:
            step.set(detail="before the failure")
            raise ValueError("deliberate")
    except ValueError:
        raised = True

    output = buffer.getvalue()
    c.expect(raised, "log_step must re-raise — recording a failure is not the same as swallowing it")
    c.expect("test.step.failed" in output, "a failing step must log .failed")
    c.expect("test.step.done" not in output, "a failing step must not also log .done")
    c.expect("duration_ms=" in output, "a failing step must still report how long it took")
    c.expect("ValueError" in output and "deliberate" in output, "the failure must name the exception")


def main() -> int:
    c = Checks()
    was_mock, was_allow = config.MOCK_MODE, config.MOCK_ALLOW_LLM
    config.MOCK_MODE, config.MOCK_ALLOW_LLM = True, False
    try:
        for check in (
            check_setup_is_polite,
            check_run_correlation,
            check_step_coverage,
            check_secrets_never_logged,
            check_untrusted_payloads_not_dumped,
            check_json_format,
            check_log_step_failure_path,
        ):
            check(c)
    finally:
        config.MOCK_MODE, config.MOCK_ALLOW_LLM = was_mock, was_allow
        # Leave logging quiet for whatever runs next.
        setup_logging("WARNING", "text", stream=io.StringIO(), force=True)

    if c.failures:
        print(f"FAILED {len(c.failures)}/{c.total} checks:", file=sys.stderr)
        for failure in c.failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"PASSED {c.total}/{c.total} checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
