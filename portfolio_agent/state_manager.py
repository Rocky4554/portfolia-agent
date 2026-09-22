from typing import TypedDict


class PortfolioContextDomain(TypedDict):
    id: str
    name: str
    folder_id: str | None


class PortfolioContextFolder(TypedDict):
    id: str
    name: str


class PortfolioManagerState(TypedDict, total=False):
    instruction: str
    domains: list[PortfolioContextDomain]
    folders: list[PortfolioContextFolder]
    plan: dict  # serialized ActionPlan
