"""
Fetching TechCrunch's RSS feeds.

RSS rather than scraping, deliberately: it is a published interface meant to
be consumed, it is stable, and it does not depend on page markup that
changes weekly. The cost is that TechCrunch's feed carries no article body
(no `content:encoded`), so a company's own URL is not in it — see
`resolve_company_url`.

Every fetch is bounded and never raises. A feed being down costs the run one
source, not the run.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import httpx

from portfolio_agent import config
from portfolio_agent.logging_setup import get_logger, info, log_step, warn

log = get_logger(__name__)

# The venture feed first: it is the same publication filtered to the beat we
# want, which means less noise to reject later.
DEFAULT_FEEDS = (
    "https://techcrunch.com/category/venture/feed/",
    "https://techcrunch.com/category/startups/feed/",
    "https://techcrunch.com/feed/",
)

_TIMEOUT_S = 30.0
# Identify ourselves. A feed reader that hides what it is gets blocked, and
# deserves to be.
_USER_AGENT = "portfolio-agent funding collector (+https://name.ai)"

_TAG_PATTERN = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class FeedItem:
    """One article, normalized across feeds."""

    title: str
    link: str
    summary: str
    published: datetime | None
    categories: tuple[str, ...]
    source: str

    @property
    def text(self) -> str:
        """Title and summary together — what the extractor reads."""
        return f"{self.title}. {self.summary}".strip()


async def fetch_feed(url: str) -> list[FeedItem]:
    """Every item in one feed. Returns [] on any failure, having logged it."""
    with log_step(log, "funding.fetch_feed", feed=url) as step:
        try:
            async with httpx.AsyncClient(
                timeout=_TIMEOUT_S, follow_redirects=True, headers={"User-Agent": _USER_AGENT}
            ) as client:
                response = await client.get(url)
        except httpx.HTTPError as e:
            warn(log, "funding.feed_unreachable", feed=url, error=f"{type(e).__name__}: {e}")
            step.set(items=0, ok=False)
            return []

        if response.status_code != 200:
            warn(log, "funding.feed_status", feed=url, status=response.status_code)
            step.set(items=0, ok=False)
            return []

        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as e:
            warn(log, "funding.feed_unparseable", feed=url, error=str(e))
            step.set(items=0, ok=False)
            return []

        items = [_to_item(node, url) for node in root.findall(".//item")]
        items = [item for item in items if item is not None]
        step.set(items=len(items), ok=True)
        return items


async def fetch_all(feeds: tuple[str, ...] | None = None) -> list[FeedItem]:
    """Every item across every configured feed, deduplicated by link.

    Feeds overlap — a venture story appears in the main feed too — so the
    same article arriving twice is normal rather than an error.
    """
    feeds = feeds or tuple(config.FUNDING_FEEDS)
    seen: dict[str, FeedItem] = {}
    for url in feeds:
        for item in await fetch_feed(url):
            seen.setdefault(item.link, item)
    info(log, "funding.fetched", feeds=len(feeds), unique_items=len(seen))
    return list(seen.values())


def _to_item(node: ET.Element, source: str) -> FeedItem | None:
    def text_of(tag: str) -> str:
        element = node.find(tag)
        return (element.text or "").strip() if element is not None else ""

    link = text_of("link")
    title = _strip_html(text_of("title"))
    if not link or not title:
        return None

    return FeedItem(
        title=title,
        link=link,
        summary=_strip_html(text_of("description")),
        published=_parse_date(text_of("pubDate")),
        categories=tuple(
            (c.text or "").strip() for c in node.findall("category") if (c.text or "").strip()
        ),
        source=source,
    )


def _strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", _TAG_PATTERN.sub(" ", value)).strip()


def _parse_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _candidate_hosts(html: str) -> list[str]:
    """Every outbound host in the page, in order, minus the ones that are
    never the subject."""
    hosts: list[str] = []
    for match in re.finditer(r'href="(https?://[^"]+)"', html):
        host = match.group(1).split("//", 1)[-1].split("/", 1)[0].lower().removeprefix("www.")
        if not host or host in hosts:
            continue
        if any(host == skip or host.endswith("." + skip) for skip in config.FUNDING_IGNORED_HOSTS):
            continue
        hosts.append(host)
    return hosts


def _name_tokens(company: str) -> list[str]:
    """The company name reduced to what a domain would plausibly be built
    from. "Mazama Energy" → ["mazamaenergy", "mazama"]."""
    letters = re.sub(r"[^a-z0-9 ]", "", (company or "").lower()).split()
    if not letters:
        return []
    joined = "".join(letters)
    tokens = [joined]
    if len(letters[0]) >= 4 and letters[0] != joined:
        tokens.append(letters[0])
    return tokens


def pick_company_host(hosts: list[str], company: str) -> str | None:
    """The host most likely to be the funded company's own site.

    Ranked by resemblance to the company name rather than by position,
    because a news page links its own properties long before it links the
    company it is writing about. The first honest attempt at this returned
    the publisher's newsletter for every single story.

    Falls back to the first outbound host only when the name matches
    nothing, and that fallback is weak — a caller should treat an
    unverified domain as unresolved rather than as fact.
    """
    tokens = _name_tokens(company)
    for host in hosts:
        sld = host.split(".")[0]
        if any(sld == token for token in tokens):
            return host
    for host in hosts:
        sld = host.split(".")[0]
        if any(token in sld or sld in token for token in tokens if len(token) >= 4):
            return host
    return None


async def resolve_company_url(article_url: str, company: str = "") -> str | None:
    """The funded company's own website, from the article page.

    This is the field that decides whether the whole pipeline is useful. A
    funding story gives the company, the amount, the round and the date — it
    does not give their domain, and the domain is what makes a credibility
    gap measurable. A $50M startup on getvoice.io is a buyer for voice.com;
    the same startup already on voice.com is not a buyer at all.

    Returns None rather than a guess when nothing on the page resembles the
    company. An unresolved lead is dropped; a wrongly resolved one produces
    a confident price against a company that was never involved.
    """
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT_S, follow_redirects=True, headers={"User-Agent": _USER_AGENT}
        ) as client:
            response = await client.get(article_url)
        if response.status_code != 200:
            return None
    except httpx.HTTPError:
        return None

    return pick_company_host(_candidate_hosts(response.text), company)
