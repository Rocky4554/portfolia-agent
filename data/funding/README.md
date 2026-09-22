# Funding events

The TechCrunch funding collector writes here, one JSON object per line.

## What it is

Recently-funded companies, gathered from public funding announcements. It
feeds the **targeted lead** tier: a company that has just raised is a buyer
with cash and, often, a compromise domain it settled for when it was three
people.

This is the *startup* half of lead discovery. The *incumbent* half comes
from SEC filings (`tools/edgar.py`) and finds the opposite kind of buyer —
profitable, established, never raised, and invisible to funding news
precisely because not being news is the same fact as not having raised.

## Why a file and not a database

`store.py` speaks to Postgres when `PORTFOLIO_AGENT_FUNDING_DB_URL` is set
and falls back to JSON lines when it is not. Lines are the right default: the
collector is useful the moment it runs, the file is greppable, and a
valuation should not need a database to be reproducible. Postgres becomes
worth it when the store outgrows a file or more than one machine writes to it.

## Running it

    uv run python -m portfolio_agent.funding.pipeline

Append-only and deduplicated on the announcement URL, so re-running it is
safe and picks up only what is new. Schedule it daily; funding news is
perishable and a lead that raised two years ago is not a lead.

## Reading it

    uv run python -m portfolio_agent.funding.report

## What is NOT in here

Anything about a company's current domain. The credibility gap this feeds is
the distance between the domain a company is ON and the one being valued, and
that is resolved separately (`tools/search.py`, one metered lookup per
candidate). A funding row without a resolved domain is a company, not a lead.
