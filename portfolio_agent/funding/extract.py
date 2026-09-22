"""
Turning a headline into a funding fact.

Pure functions and regular expressions, no model. This follows the same rule
as everything else that produces a number here: the LLM may explain a
figure, never invent one. A funding total drives a 1.15x-2.20x multiplier on
the Targeted Lead tier, so a hallucinated amount is a hallucinated price.

Three jobs, in order of difficulty:

  1. Is this a funding story at all? Most of a tech feed is not — TechCrunch
     runs conference promotion, analysis and product news through the same
     channel as company raises.
  2. Is the raiser a COMPANY, or an investor? "Sequoia raises $2B fund" has
     the same shape as "Crusoe raises $3.9B" and means something completely
     different. A VC firm is not a domain buyer, and letting one through
     would price a domain against a fund that will never buy it.
  3. What did it raise, and who is it? Amount, round, company name.
"""

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

# Verbs a raise is reported with. "Closes" is included because "closes its
# Series B" is standard, at the cost of the occasional "closes its doors".
_RAISE_VERBS = r"(?:raises|raised|raising|lands|landed|secures|secured|snags|snagged|nets|netted|closes|closed|bags|bagged|picks up|picked up|scores|scored|pulls in|pulled in)"

_HEADLINE = re.compile(
    r"^(?P<company>.{2,60}?)\s+" + _RAISE_VERBS + r"\s+(?P<amount>[^\s].{0,40}?)(?:\s|$|,|\.|—|–|-)",
    re.IGNORECASE,
)

# "$40M", "$3.9B", "$12 million", "€25M". Currency is captured so a non-USD
# figure is never silently treated as dollars.
_AMOUNT = re.compile(
    r"(?P<currency>[$€£¥])\s?(?P<value>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<scale>[KMBT]\b|thousand|million|billion|trillion)?",
    re.IGNORECASE,
)

_SCALES = {
    "k": 1_000, "thousand": 1_000,
    "m": 1_000_000, "million": 1_000_000,
    "b": 1_000_000_000, "billion": 1_000_000_000,
    "t": 1_000_000_000_000, "trillion": 1_000_000_000_000,
}

_ROUND = re.compile(
    r"\b(pre-?seed|seed|series\s+[A-J]\b|growth round|bridge round|"
    r"debt financing|strategic round|angel round)\b",
    re.IGNORECASE,
)

# A raise described with any of these is an investor raising a FUND, not a
# company raising capital. They are checked against the headline, because
# "raises $2B fund" and "raises $2B" differ by exactly this word.
_FUND_MARKERS = re.compile(
    r"\b(fund|funds|vehicle|LP|LPs|limited partners|SPV|"
    r"fund\s+[IVX]+|debut fund|second fund|third fund|opportunity fund)\b",
    re.IGNORECASE,
)

# Names ending in these belong to investors. Checked on the company name
# only, so "Capital One raises" is not caught by "Capital" appearing later.
_INVESTOR_SUFFIXES = (
    "ventures", "capital", "partners", "vc", "equity", "management",
    "advisors", "holdings", "fund", "funds", "asset management",
)

# Sector hints, used to match a lead against the domain being valued. Order
# matters: the first match wins, so put the specific before the general.
_SECTORS = (
    ("ai", ("ai", "artificial intelligence", "llm", "machine learning", "genai", "agentic")),
    ("fintech", ("fintech", "payments", "banking", "lending", "insurtech", "neobank")),
    ("healthtech", ("health", "biotech", "medical", "clinical", "therapeutics", "diagnostics")),
    ("security", ("security", "cyber", "infosec", "fraud", "compliance")),
    ("climate", ("climate", "energy", "solar", "battery", "fusion", "carbon")),
    ("devtools", ("developer", "devtools", "infrastructure", "database", "api", "observability")),
    ("commerce", ("ecommerce", "retail", "marketplace", "commerce", "logistics")),
    ("robotics", ("robot", "robotics", "drone", "autonomous", "defense")),
)

# Below this a round is too small to support the Targeted Lead tier at all —
# see config.MIN_LEAD_FUNDING. Kept separate so the extractor stays honest
# about what it found and the qualification stays in pricing/targeted.py.
# Words no company name begins with. Their presence means the headline
# described the company instead of naming it.
_CLAUSE_OPENERS = {
    "a", "an", "the", "that", "this", "these", "those", "which", "who",
    "whose", "it", "its", "their", "his", "her", "another", "one", "two",
    "and", "but", "as", "after", "before", "when", "while", "how", "why",
}

_IMPLAUSIBLE_BELOW = 10_000
_IMPLAUSIBLE_ABOVE = 100_000_000_000


@dataclass
class FundingEvent:
    """One company raising money, as reported."""

    company: str
    amount: int | None
    currency: str
    round_type: str | None
    announced_on: date | None
    sector: str | None
    article_title: str
    article_url: str
    source: str
    company_domain: str | None = None
    # Why this was or was not usable. Carried so a rejected story can be
    # explained without re-deriving it.
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Dedupe key. A raise is one event however many outlets report it."""
        stamp = self.announced_on.isoformat() if self.announced_on else "unknown"
        return f"{self.company.lower().strip()}|{stamp}|{self.amount or 0}"

    def as_row(self) -> dict:
        row = asdict(self)
        row["announced_on"] = self.announced_on.isoformat() if self.announced_on else None
        row["key"] = self.key
        return row


def parse_amount(text: str) -> tuple[int | None, str]:
    """The first currency amount in `text`, scaled. ('$3.9B' → 3_900_000_000).

    Returns (None, "") when there is no amount, and keeps the currency
    symbol so a euro round is never quietly read as dollars.
    """
    match = _AMOUNT.search(text or "")
    if not match:
        return None, ""
    try:
        value = float(match.group("value").replace(",", ""))
    except ValueError:
        return None, ""
    scale = (match.group("scale") or "").lower()
    value *= _SCALES.get(scale, 1)
    if not (_IMPLAUSIBLE_BELOW <= value <= _IMPLAUSIBLE_ABOVE):
        return None, match.group("currency")
    return int(value), match.group("currency")


def parse_round(text: str) -> str | None:
    match = _ROUND.search(text or "")
    return re.sub(r"\s+", " ", match.group(1)).title() if match else None


def detect_sector(text: str) -> str | None:
    lowered = (text or "").lower()
    for sector, keywords in _SECTORS:
        if any(re.search(rf"\b{re.escape(k)}\b", lowered) for k in keywords):
            return sector
    return None


def is_investor_name(company: str) -> bool:
    """Whether this name belongs to a fund rather than an operating company."""
    name = (company or "").lower().strip().rstrip(".,")
    return any(name.endswith(" " + suffix) or name == suffix for suffix in _INVESTOR_SUFFIXES)


def extract(item) -> FundingEvent | None:
    """A FundingEvent from a feed item, or None if it is not a company raise.

    `item` is anything with `.title`, `.summary`, `.link`, `.published` and
    `.source` — feeds.FeedItem in production, a fixture in the evals.
    """
    title = (item.title or "").strip()
    match = _HEADLINE.search(title)
    if not match:
        return None

    company = _clean_company(match.group("company"))
    if not company:
        return None

    amount, currency = parse_amount(title)
    if amount is None:
        # A raise with no stated figure cannot support the tier that needs
        # one, and guessing a typical round size would be inventing the
        # number the whole tier turns on.
        return None

    # A fund raise, not a company raise. Two independent signals, because
    # either alone misses cases: "Sequoia raises $2B" has no fund marker,
    # and "Acme raises $50M growth fund" has no investor suffix.
    if _FUND_MARKERS.search(title) or is_investor_name(company):
        return None

    text = f"{title}. {(item.summary or '').strip()}"
    published = item.published.date() if isinstance(item.published, datetime) else item.published

    return FundingEvent(
        company=company,
        amount=amount,
        currency=currency,
        round_type=parse_round(text),
        announced_on=published,
        sector=detect_sector(text),
        article_title=title,
        article_url=item.link,
        source=getattr(item, "source", "") or "",
    )


def _clean_company(raw: str) -> str:
    """The company name out of a headline's opening clause.

    Headlines are written as sentences, so the subject is often wrapped in
    narrative ("Exclusive: Acme raises...", "Fintech startup Acme raises...").
    Stripping the wrapper is what keeps the name matchable against a lead.
    """
    name = re.sub(r"\s+", " ", (raw or "").strip())
    name = re.sub(r"^(exclusive|scoop|report|breaking|update)\s*[:\-—]\s*", "", name, flags=re.IGNORECASE)
    # Investor endorsements lead the headline more often than the company
    # does: "Khosla-backed Mazama Energy raised $135M".
    name = re.sub(r"^[\w.\-]+[-\s]backed\s+", "", name, flags=re.IGNORECASE)
    # Leading descriptors: "AI startup Acme", "Indian fintech Acme".
    name = re.sub(
        r"^(?:[\w.\-]+\s+)*?(?:startup|company|firm|platform|maker|developer|unicorn|"
        r"scaleup|venture)\s+",
        "", name, flags=re.IGNORECASE,
    )
    # Adverbs that sit between the subject and the verb and get swept into
    # the name: "Mazama Energy just raised".
    name = re.sub(
        r"\s+(just|now|finally|reportedly|officially|quietly|apparently)$", "", name, flags=re.IGNORECASE
    )
    name = name.strip(" ,:;—–-")

    if not name or len(name) > 60 or name.count(" ") > 5:
        return ""
    # A relative clause is what is left when a headline never named the
    # company: "A startup that builds other startups raised $100M" leaves
    # "that builds other startups". A company name is not a sentence, so
    # anything opening with a function word is a parse failure, not a name.
    first = name.split()[0].lower()
    if first in _CLAUSE_OPENERS or re.search(r"\b(that|which|who|whose)\b", name, re.IGNORECASE):
        return ""
    return name
