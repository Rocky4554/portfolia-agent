"""
Deterministic scoring. Every function here is pure: same domain in, same
numbers out, no network, no LLM, no clock. That is what makes the scores
testable against fixed expected values and reproducible across runs — the
LLM's only job in this system is to explain these numbers, never to produce
them.
"""
