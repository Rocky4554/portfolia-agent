"""
The Name.ai funding report.

Not a news digest. The question it answers is the one that matters to a
domain portfolio: **which companies just got money, and do they need a
better domain?**

A funded company already on a strong `.com` is not a lead however large its
round. A funded company on a hyphenated name, a second-choice extension, or
a `get`/`try` prefix of the name it actually wanted is a buyer with a visible
problem and the budget to fix it. The report sorts on that, not on round
size, because round size alone is a list of companies you cannot sell to.
"""

from datetime import date

from portfolio_agent.knowledge.lexicon import EMERGING_TLDS, PREMIUM_TLDS

# The prefixes and suffixes a company reaches for when it could not get the
# exact name. Their presence is the clearest signal in the whole report.
_COMPROMISE_AFFIXES = ("get", "try", "use", "join", "the", "my", "go", "hq", "app", "io", "co", "now")


def domain_weakness(domain: str | None) -> tuple[int, list[str]]:
    """0-100 on how much a better domain would help, and why.

    Returns (0, []) when there is no domain to judge — never a score, since
    "we do not know" and "they are fine" lead to opposite actions.
    """
    if not domain:
        return 0, []

    host = domain.lower().strip().removeprefix("www.")
    sld, _, tld = host.partition(".")
    tld = tld or ""
    reasons: list[str] = []
    score = 0

    if tld and tld not in PREMIUM_TLDS:
        if tld in EMERGING_TLDS:
            score += 30
            reasons.append(f".{tld} rather than .com")
        else:
            score += 45
            reasons.append(f".{tld} is outside the extensions buyers expect")

    if "-" in sld:
        score += 30
        reasons.append("hyphenated — cannot be said aloud unambiguously")
    if any(ch.isdigit() for ch in sld):
        score += 20
        reasons.append("contains digits")
    if len(sld) > 15:
        score += 15
        reasons.append(f"{len(sld)} characters long")

    for affix in _COMPROMISE_AFFIXES:
        if sld.startswith(affix) and len(sld) > len(affix) + 2:
            score += 35
            reasons.append(f"'{affix}' prefix — the bare name was taken")
            break

    return min(100, score), reasons


def build(rows: list[dict], *, since: date | None = None, today: date | None = None) -> str:
    """The report, as Markdown."""
    today = today or date.today()
    rows = list(rows)
    total = sum(r.get("amount") or 0 for r in rows)

    lines = [
        "# Name.ai funding report",
        "",
        f"Generated {today.isoformat()}"
        + (f" · covering funding announced since {since.isoformat()}" if since else ""),
        "",
        f"**{len(rows)} funding events** · **${total:,} raised in total**",
        "",
    ]

    if not rows:
        lines += [
            "No funding events in this window.",
            "",
            "That is a result, not a failure — most days a feed carries no company raises at "
            "all. If it persists for a week, check the collector is running and that the feeds "
            "in `config.FUNDING_FEEDS` still resolve.",
        ]
        return "\n".join(lines)

    # ── The part that is actually actionable ──────────────────────────────
    leads = []
    for row in rows:
        weakness, reasons = domain_weakness(row.get("company_domain"))
        if weakness > 0:
            leads.append((weakness, reasons, row))
    leads.sort(key=lambda entry: (entry[0], entry[2].get("amount") or 0), reverse=True)

    lines += ["## Domain leads", ""]
    if leads:
        lines += [
            "Funded companies on a domain that is visibly a compromise. Ranked by how much a "
            "better name would help them, then by budget.",
            "",
            "| Company | Raised | Round | On | Weakness | Why |",
            "|---|---|---|---|---|---|",
        ]
        for weakness, reasons, row in leads[:20]:
            lines.append(
                f"| {row.get('company')} | ${(row.get('amount') or 0):,} | {row.get('round_type') or '—'} "
                f"| `{row.get('company_domain')}` | {weakness}/100 | {'; '.join(reasons)} |"
            )
    else:
        resolved = sum(1 for r in rows if r.get("company_domain"))
        lines += [
            f"None. {resolved} of {len(rows)} events have a resolved company domain"
            + (
                "."
                if resolved
                else " — run the collector with `--resolve-domains` to fetch them, without which "
                "no lead can be scored."
            ),
        ]
    lines.append("")

    # ── Context ───────────────────────────────────────────────────────────
    by_sector: dict[str, list[dict]] = {}
    for row in rows:
        by_sector.setdefault(row.get("sector") or "unclassified", []).append(row)

    lines += ["## By sector", "", "| Sector | Events | Capital raised |", "|---|---|---|"]
    for sector, sector_rows in sorted(
        by_sector.items(), key=lambda kv: sum(r.get("amount") or 0 for r in kv[1]), reverse=True
    ):
        lines.append(
            f"| {sector} | {len(sector_rows)} | ${sum(r.get('amount') or 0 for r in sector_rows):,} |"
        )
    lines.append("")

    lines += ["## Largest rounds", "", "| Company | Raised | Round | Sector | Announced |", "|---|---|---|---|---|"]
    for row in sorted(rows, key=lambda r: r.get("amount") or 0, reverse=True)[:15]:
        lines.append(
            f"| {row.get('company')} | ${(row.get('amount') or 0):,} | {row.get('round_type') or '—'} "
            f"| {row.get('sector') or '—'} | {row.get('announced_on') or '—'} |"
        )

    lines += [
        "",
        "---",
        "",
        "Figures are as reported in the source articles and are not independently confirmed. "
        "A headline amount can bundle debt with equity, and a round is sometimes announced "
        "months after it closed.",
    ]
    return "\n".join(lines)
