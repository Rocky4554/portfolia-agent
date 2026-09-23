"""
Per-TLD authoritative RDAP base URLs for ccTLDs the aggregator handles poorly.

F19: RDAP 404 from rdap.org (the proxy) does not mean unregistered — it
means the proxy has coverage gaps. isotope.co returned 404 from rdap.org but
is obviously registered (listed for sale on name.ai).

This file provides direct registry endpoints for TLDs where the proxy fails.
IANA's bootstrap already handles most TLDs; these are the known exceptions.

Format: TLD → authoritative RDAP base URL (without /domain/ path)
"""

# ccTLDs and other extensions with known registry RDAP endpoints
# that rdap.org doesn't reliably proxy
AUTHORITATIVE_RDAP_ENDPOINTS = {
    # .co (Colombia) — rdap.org frequently 404s
    "co": "https://rdap.nic.co",
    
    # .ai (Anguilla) — covered by IANA bootstrap, but kept for reference
    # "ai": "https://rdap.nic.ai",  # Usually works via rdap.org
    
    # .io (British Indian Ocean Territory) — not in IANA bootstrap
    "io": "https://rdap.nic.io",
    
    # Add more as gaps are discovered
}


def get_authoritative_endpoint(tld: str) -> str | None:
    """Return the authoritative RDAP endpoint for this TLD, if known.
    
    Returns None if the TLD should be queried via rdap.org or IANA bootstrap.
    """
    tld = (tld or "").lower().strip()
    return AUTHORITATIVE_RDAP_ENDPOINTS.get(tld)
