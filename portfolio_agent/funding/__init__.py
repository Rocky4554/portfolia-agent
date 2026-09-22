"""
Funding intelligence from news feeds.

Exists to supply the one fact the valuation agent cannot get anywhere else:
whether an identified company can actually write a cheque. Two of the six
price tiers (Targeted Lead, Strategic M&A) are premised on a specific funded
buyer, and without a funding figure they stay unpriced.

Deliberately separate from the valuation graph. This is a scheduled
collector that writes to a store; the agent reads that store. Nothing here
runs inside a valuation, because a valuation must not depend on a news site
being up.
"""
