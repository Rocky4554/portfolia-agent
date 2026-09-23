"""
Our own brokerage book, as comps and as demand signals.

The most trustworthy data in the system and the only data nobody else has.
A closed sale we brokered is verified in a way a scraped aftermarket record
is not: we know it completed, at what price, and on what date.

Two kinds of record, and the distinction is the same one that governs
marketplace listings — what somebody PAID versus what somebody WANTED:

  - **Closed sales** are comps. They enter the ladder at every rung, marked
    `venue="internal"`, weighted 1.5x: verified, but from one venue, so they
    should outweigh a scraped record without drowning a broader market.
  - **Offers and inquiries are NOT comps.** An offer is a price nobody paid
    and an inquiry is not even that. They feed a demand signal and, where an
    offer beats the floor, a capped lift on the end-user tier. Letting an
    offer into the comps distribution would price a domain off a
    conversation.

**Privacy.** Counterparty identity is stripped at the tool boundary, not
downstream. A comp carries price, date and venue; who bought it does not
leave this function. A broker's book contains names, emails and negotiation
history, and none of it needs to reach a prompt or a log to value a domain.
"""

import csv
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from portfolio_agent import config
from portfolio_agent.logging_setup import debug, get_logger, info
from portfolio_agent.tools.common import ToolResult, error_result, mock_result, ok_result, request_json

log = get_logger(__name__)

SOURCE = "internal_comps"
SIGNALS_SOURCE = "internal_signals"

_TIMEOUT_S = 25.0

# Fields that may leave this module. An allowlist, not a denylist: a broker
# record gains columns over time and a new one must not escape by default.
_SALE_FIELDS = ("domain", "price", "date", "venue")
_SIGNAL_FIELDS = ("inquiries_12m", "best_offer", "best_offer_date", "closed_sales")


@dataclass
class InternalSignals:
    """Demand, as distinct from price."""

    inquiries_12m: int = 0
    best_offer: int | None = None
    best_offer_date: str | None = None
    closed_sales: list[dict] = field(default_factory=list)


async def fetch_internal_comps(
    keywords: list[str] | None = None,
    tld: str | None = None,
    category: str = "",
    *,
    before: str | date | None = None,
    exclude_domain: str | None = None,
) -> ToolResult:
    """Closed sales from our own book, in the shape the ladder reads comps.

    Mirrors the signature of the external sales tools deliberately: the
    ladder should not have to know which of its comps came from our book
    and which from a public feed, beyond the venue marking that sets their
    weight.
    """
    keywords = [k for k in (keywords or []) if k]
    tld = (tld or "").lstrip(".").lower()

    if config.MOCK_MODE:
        return mock_result(SOURCE, _mock_comps(keywords, tld), data_age_days=9)
    if config.INTERNAL_COMPS_CSV:
        try:
            comps = load_book(
                config.INTERNAL_COMPS_CSV, before=before, exclude_domain=exclude_domain,
            )
        except OSError as exc:
            return error_result(SOURCE, f"could not read INTERNAL_COMPS_CSV: {exc}")
        info(log, "internal_comps.fetched", keywords=len(keywords), tld=tld,
             comps=len(comps), source="csv", before=str(before or ""),
             excluded=exclude_domain or "")
        return ok_result(SOURCE, {"keywords": keywords, "tld": tld, "comp_count": len(comps), "comps": comps})
    if not (config.INTERNAL_COMPS_URL and config.INTERNAL_COMPS_TOKEN):
        return error_result(SOURCE, "no INTERNAL_COMPS_URL / INTERNAL_COMPS_TOKEN configured")

    body, error = await request_json(
        "GET",
        f"{config.INTERNAL_COMPS_URL}/sales",
        params={"keywords": ",".join(keywords), "tld": tld, "category": category},
        headers={"Authorization": f"Bearer {config.INTERNAL_COMPS_TOKEN}", "Accept": "application/json"},
        timeout=_TIMEOUT_S,
    )
    if error:
        return error_result(SOURCE, error)

    rows = _rows(body, ("sales", "results", "data"))
    comps = [_strip_sale(row) for row in rows]
    comps = [c for c in comps if c["price"] > 0]
    comps = _apply_holdout(comps, before=before, exclude_domain=exclude_domain)
    info(log, "internal_comps.fetched", keywords=len(keywords), tld=tld, comps=len(comps))
    return ok_result(SOURCE, {"keywords": keywords, "tld": tld, "comp_count": len(comps), "comps": comps})


async def fetch_internal_signals(domain: str) -> ToolResult:
    """Demand for one specific domain: inquiries, the best offer, past sales.

    None of this is a comp. An offer is a price nobody paid; an inquiry is
    somebody asking. They answer "is anyone interested", which the comps
    cannot, and they must never be mistaken for evidence of what a domain
    fetches.
    """
    domain = (domain or "").strip().lower()
    if not domain:
        return error_result(SIGNALS_SOURCE, "domain is empty")

    if config.MOCK_MODE:
        return mock_result(SIGNALS_SOURCE, _mock_signals(domain))
    if not (config.INTERNAL_COMPS_URL and config.INTERNAL_COMPS_TOKEN):
        return error_result(SIGNALS_SOURCE, "no INTERNAL_COMPS_URL / INTERNAL_COMPS_TOKEN configured")

    body, error = await request_json(
        "GET",
        f"{config.INTERNAL_COMPS_URL}/signals",
        params={"domain": domain},
        headers={"Authorization": f"Bearer {config.INTERNAL_COMPS_TOKEN}", "Accept": "application/json"},
        timeout=_TIMEOUT_S,
    )
    if error:
        return error_result(SIGNALS_SOURCE, error)

    raw = body if isinstance(body, dict) else {}
    return ok_result(SIGNALS_SOURCE, _strip_signals(domain, raw))


# ── Boundary ───────────────────────────────────────────────────────────────


def load_book(
    path: str | Path,
    *,
    before: str | date | None = None,
    exclude_domain: str | None = None,
) -> list[dict]:
    """Closed sales from a CSV, reduced to comp fields and holdout-filtered.

    The same file can be the ground-truth list and the book. On the turn
    that grades one row, that domain is excluded and anything dated on or
    after its sale is excluded, so the other rows are comps and the row
    itself is not.
    """
    cutoff = _parse_day(before)
    subject = (exclude_domain or "").strip().lower()
    comps = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip() and not line.lstrip().startswith("#")]
    for row in csv.DictReader(lines):
        sale = _strip_sale(row)
        if sale["price"] <= 0 or not sale["domain"]:
            continue
        if not _kept(sale, cutoff=cutoff, subject=subject):
            continue
        comps.append(sale)
    return comps


def _apply_holdout(comps: list[dict], *, before, exclude_domain: str | None) -> list[dict]:
    cutoff = _parse_day(before)
    if cutoff is None and not exclude_domain:
        return comps
    subject = (exclude_domain or "").strip().lower()
    return [c for c in comps if _kept(c, cutoff=cutoff, subject=subject)]


def _kept(sale: dict, *, cutoff: date | None, subject: str) -> bool:
    """A book row may be a comp only if it was already closed, and it is not the answer."""
    if subject and sale["domain"] == subject:
        return False
    if cutoff is None:
        return True
    sold = _parse_day(sale.get("date"))
    return sold is not None and sold < cutoff


def _parse_day(value) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _money(value) -> float:
    if value is None or value == "":
        return 0.0
    text = str(value).replace("$", "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def _strip_sale(row: dict) -> dict:
    """A closed sale, reduced to the four fields a comp needs.

    Everything else — buyer, seller, broker, negotiation notes — stops
    here. A valuation needs to know a domain sold for $40,000 in March; it
    does not need to know who bought it, and a log that records that is a
    log nobody can share.
    """
    return {
        "domain": str(row.get("domain") or "").lower(),
        "price": _money(row.get("price") or row.get("sale_price") or row.get("actual_price")),
        "date": str(row.get("date") or row.get("sale_date") or row.get("closed_at") or "")[:10] or None,
        "venue": "internal",
    }


def _strip_signals(domain: str, raw: dict) -> dict:
    signals = InternalSignals(
        inquiries_12m=int(raw.get("inquiries_12m") or raw.get("inquiries") or 0),
        best_offer=_positive(raw.get("best_offer") or raw.get("highest_offer")),
        best_offer_date=str(raw.get("best_offer_date") or "")[:10] or None,
        closed_sales=[_strip_sale(r) for r in _rows(raw, ("closed_sales", "sales"))],
    )
    return {
        "domain": domain,
        "inquiries_12m": signals.inquiries_12m,
        "best_offer": signals.best_offer,
        "best_offer_date": signals.best_offer_date,
        "closed_sales": [s for s in signals.closed_sales if s["price"] > 0],
        "note": (
            "Offers and inquiries are demand signals, not comparable sales. An offer is a price "
            "nobody paid."
        ),
    }


def _rows(body, keys: tuple[str, ...]) -> list[dict]:
    if isinstance(body, list):
        return [r for r in body if isinstance(r, dict)]
    if not isinstance(body, dict):
        return []
    for key in keys:
        value = body.get(key)
        if isinstance(value, list):
            return [r for r in value if isinstance(r, dict)]
    return []


def _positive(value):
    return int(value) if isinstance(value, (int, float)) and value > 0 else None


# ── Fixtures ───────────────────────────────────────────────────────────────


def _mock_comps(keywords: list[str], tld: str) -> dict:
    """Own-book sales, shaped like real brokerage records.

    Sparse on purpose. A brokerage closes a handful of sales in any one
    niche, not fourteen, and a fixture that supplied plenty would hide the
    case this feed exists for: a couple of verified own-book sales lifting a
    thin market off its weakest rung.
    """
    from portfolio_agent.tools.mocks import _seed, _spread, anchor_value

    keyword = (keywords or ["domain"])[0]
    tld = tld or "com"
    seed = _seed("internal", keyword, tld)
    anchor = anchor_value(f"{keyword}.{tld}")
    comps = [
        {
            "domain": f"{keyword}{suffix}.{tld}",
            "price": round(anchor * _spread(seed, i, 0.4, 1.8), 2),
            "date": f"2026-0{3 + i}-1{i}",
            "venue": "internal",
        }
        for i, suffix in enumerate(("hq", "pro"))
    ]
    return {"keywords": keywords, "tld": tld, "comp_count": len(comps), "comps": comps}


def _mock_signals(domain: str) -> dict:
    from portfolio_agent.tools.mocks import _seed, _spread, anchor_value

    seed = _seed("internal_signals", domain)
    anchor = anchor_value(domain)
    has_offer = seed % 3 != 0
    return {
        "domain": domain,
        "inquiries_12m": int(_spread(seed, 0, 0, 40)),
        # An offer sits below what the market would pay — that is what makes
        # it an offer and not a sale.
        "best_offer": int(anchor * _spread(seed, 1, 0.15, 0.75)) if has_offer else None,
        "best_offer_date": "2026-07-14" if has_offer else None,
        "closed_sales": [],
        "note": (
            "Offers and inquiries are demand signals, not comparable sales. An offer is a price "
            "nobody paid."
        ),
    }
