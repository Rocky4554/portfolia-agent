"""
Where collected funding events live.

Two backends behind one interface. Postgres when `FUNDING_DB_URL` is set,
newline-delimited JSON on disk otherwise.

The fallback is not a toy. It keeps the whole pipeline runnable before any
database exists, keeps the evals free of infrastructure, and means a cron
job on a machine that has lost its database degrades to collecting rather
than to failing. The agent reads through the same interface either way, so
nothing downstream knows or cares which is in use.

Writes are idempotent on `FundingEvent.key` (company + date + amount). The
same raise reported twice, or a feed re-serving yesterday's items, must not
become two leads and must not double a company's funding total.
"""

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from portfolio_agent import config
from portfolio_agent.funding.extract import FundingEvent
from portfolio_agent.logging_setup import get_logger, info, warn

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS funding_events (
    key            TEXT PRIMARY KEY,
    company        TEXT NOT NULL,
    company_domain TEXT,
    amount         BIGINT,
    currency       TEXT,
    round_type     TEXT,
    announced_on   DATE,
    sector         TEXT,
    article_title  TEXT,
    article_url    TEXT,
    source         TEXT,
    collected_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS funding_events_announced_idx ON funding_events (announced_on DESC);
CREATE INDEX IF NOT EXISTS funding_events_sector_idx    ON funding_events (sector);
CREATE INDEX IF NOT EXISTS funding_events_amount_idx    ON funding_events (amount DESC);
"""

_COLUMNS = (
    "key", "company", "company_domain", "amount", "currency", "round_type",
    "announced_on", "sector", "article_title", "article_url", "source",
)


@dataclass
class StoreResult:
    written: int
    skipped: int
    backend: str


class JsonlStore:
    """Append-only JSON lines, deduplicated by key on read and write.

    Rewrites the whole file on save. That is fine at this scale — a funding
    feed produces single-digit events per run — and it keeps the file a
    valid, sorted, human-readable record rather than an append log with
    stale duplicates in it.
    """

    backend = "jsonl"

    def __init__(self, path: str | None = None):
        self.path = Path(path or config.FUNDING_STORE_PATH)

    def ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        rows: dict[str, dict] = {}
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a truncated write must not lose the whole file
            if row.get("key"):
                rows[row["key"]] = row
        return rows

    def upsert_many(self, events: list[FundingEvent]) -> StoreResult:
        self.ensure_schema()
        existing = self._load()
        written = skipped = 0
        for event in events:
            row = event.as_row()
            if row["key"] in existing:
                skipped += 1
            else:
                written += 1
            existing[row["key"]] = {**row, "collected_at": _now()}
        ordered = sorted(existing.values(), key=lambda r: (r.get("announced_on") or "", r.get("company") or ""))
        self.path.write_text("\n".join(json.dumps(r, default=str) for r in ordered) + "\n")
        return StoreResult(written, skipped, self.backend)

    def search(self, *, keywords=(), min_amount=0, since=None, limit=50) -> list[dict]:
        rows = [r for r in self._load().values() if (r.get("amount") or 0) >= min_amount]
        if since:
            rows = [r for r in rows if (r.get("announced_on") or "") >= since.isoformat()]
        if keywords:
            lowered = [k.lower() for k in keywords if k]
            rows = [r for r in rows if _matches(r, lowered)]
        rows.sort(key=lambda r: (r.get("amount") or 0), reverse=True)
        return rows[:limit]

    def recent(self, *, since=None, limit=200) -> list[dict]:
        rows = list(self._load().values())
        if since:
            rows = [r for r in rows if (r.get("announced_on") or "") >= since.isoformat()]
        rows.sort(key=lambda r: (r.get("announced_on") or "", r.get("amount") or 0), reverse=True)
        return rows[:limit]


class PostgresStore:
    """Postgres via psycopg 3. Imported lazily so the driver is only needed
    by deployments that actually use a database."""

    backend = "postgres"

    def __init__(self, dsn: str):
        self.dsn = dsn

    def _connect(self):
        import psycopg  # noqa: PLC0415 — lazy on purpose

        return psycopg.connect(self.dsn)

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(SCHEMA)
            conn.commit()

    def upsert_many(self, events: list[FundingEvent]) -> StoreResult:
        if not events:
            return StoreResult(0, 0, self.backend)
        self.ensure_schema()
        placeholders = ", ".join(["%s"] * len(_COLUMNS))
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _COLUMNS if c != "key")
        # ON CONFLICT rather than a pre-read: two collectors running at once
        # (a cron overlapping a manual run) must not race into duplicates.
        sql = (
            f"INSERT INTO funding_events ({', '.join(_COLUMNS)}) VALUES ({placeholders}) "
            f"ON CONFLICT (key) DO UPDATE SET {updates} "
            f"RETURNING (xmax = 0) AS inserted"
        )
        written = skipped = 0
        with self._connect() as conn:
            for event in events:
                row = event.as_row()
                result = conn.execute(sql, [row.get(c) for c in _COLUMNS]).fetchone()
                if result and result[0]:
                    written += 1
                else:
                    skipped += 1
            conn.commit()
        return StoreResult(written, skipped, self.backend)

    def search(self, *, keywords=(), min_amount=0, since=None, limit=50) -> list[dict]:
        clauses, params = ["COALESCE(amount, 0) >= %s"], [min_amount]
        if since:
            clauses.append("announced_on >= %s")
            params.append(since)
        if keywords:
            terms = [k for k in keywords if k]
            if terms:
                ors = " OR ".join(
                    ["(company ILIKE %s OR sector ILIKE %s OR article_title ILIKE %s)"] * len(terms)
                )
                clauses.append(f"({ors})")
                for term in terms:
                    params.extend([f"%{term}%"] * 3)
        params.append(limit)
        sql = (
            f"SELECT {', '.join(_COLUMNS)} FROM funding_events "
            f"WHERE {' AND '.join(clauses)} ORDER BY amount DESC NULLS LAST LIMIT %s"
        )
        return self._rows(sql, params)

    def recent(self, *, since=None, limit=200) -> list[dict]:
        clauses, params = ["TRUE"], []
        if since:
            clauses.append("announced_on >= %s")
            params.append(since)
        params.append(limit)
        sql = (
            f"SELECT {', '.join(_COLUMNS)} FROM funding_events "
            f"WHERE {' AND '.join(clauses)} ORDER BY announced_on DESC NULLS LAST, amount DESC LIMIT %s"
        )
        return self._rows(sql, params)

    def _rows(self, sql: str, params: list) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {c: (v.isoformat() if isinstance(v, (date, datetime)) else v) for c, v in zip(_COLUMNS, row)}
            for row in rows
        ]


def open_store(dsn: str | None = None, path: str | None = None):
    """Postgres when a DSN is configured, JSON lines otherwise.

    A Postgres failure falls back rather than raising: a collector that
    cannot reach its database should still gather the news, so the run is
    recoverable once the database returns.
    """
    dsn = dsn if dsn is not None else config.FUNDING_DB_URL
    if dsn:
        store = PostgresStore(dsn)
        try:
            store.ensure_schema()
            return store
        except Exception as e:  # noqa: BLE001 — degrade, never fail the collector
            warn(log, "funding.postgres_unavailable", error=f"{type(e).__name__}: {e}",
                 fallback=config.FUNDING_STORE_PATH)
    return JsonlStore(path)


def _matches(row: dict, keywords: list[str]) -> bool:
    haystack = " ".join(
        str(row.get(f) or "") for f in ("company", "sector", "article_title", "company_domain")
    ).lower()
    return any(k in haystack for k in keywords)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def days_ago(days: int) -> date:
    return (datetime.now(timezone.utc) - timedelta(days=days)).date()
