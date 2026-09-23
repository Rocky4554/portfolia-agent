"""
Finding enough comparable sales to price from, and saying how hard it was.

consulting.ai found one sale at $425 and priced nothing. One sale is an
anecdote and refusing to extrapolate from it was right — but "no answer" is
not the only alternative to "a confident answer". A human appraiser handed a
thin market widens the search and then tells you how far they had to go:
*"no .ai consulting sales to speak of, but consulting.com went for $X and
single-word .ai names are trading around $Y."* That is still a valuation.
It is a different KIND of valuation, and the difference is the thing that
must survive into the output.

So the ladder descends through progressively broader bases, stops at the
first rung with enough comps, and records which rung it stopped on:

    exact          sales of this keyword on this TLD
    sld_cross_tld  this exact name, sold on other extensions
    cohort         structurally similar names on this TLD
    keyword_only   aggregate keyword statistics, no individual sales
    none           nothing usable

Two rules keep this from becoming a machine for manufacturing confidence.

**Rungs are cumulative.** A comp found at a stronger rung is never discarded
when the ladder descends — it just was not enough on its own. An exact comp
carries double weight in the distribution, so one real sale still dominates
five cohort members.

**Confidence falls with the rung, and the tier engine widens its ranges
accordingly.** A cohort-based price is not a slightly worse exact price; it
is a wider claim, and it must look wider.
"""

import statistics

from dataclasses import dataclass, field
from datetime import date, datetime

from portfolio_agent import config
from portfolio_agent.knowledge.industries import classify_industry
from portfolio_agent.knowledge.tld_lifecycle import phase_for
from portfolio_agent.knowledge.tld_lifecycle import staleness_warning as lifecycle_staleness
from portfolio_agent.knowledge.tld_transfer import transfer_ratio
from portfolio_agent.tools.linguistics import split_domain
from portfolio_agent.logging_setup import debug, get_logger, info
from portfolio_agent.schema.valuation_output import CompBasis, CompRef

log = get_logger(__name__)


@dataclass
class CompSet:
    """The comps the tiers will be built from, and their provenance."""

    comps: list[CompRef] = field(default_factory=list)
    basis: CompBasis = CompBasis.NONE
    confidence: float = 0.0
    steps_taken: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def prices(self) -> list[int]:
        return sorted(c.adjusted_price for c in self.comps if c.adjusted_price > 0)

    def weighted_prices(self) -> list[int]:
        """Prices repeated by weight, so a weighted median is just a median.

        Cruder than carrying weights into a percentile function, and
        deliberately so: the arithmetic downstream stays ordinary, which
        means the tier engine has one code path rather than two and the
        exact-comp weighting cannot quietly stop applying.
        """
        expanded: list[int] = []
        for comp in self.comps:
            if comp.adjusted_price > 0:
                expanded.extend([comp.adjusted_price] * max(1, round(comp.weight)))
        return sorted(expanded)

    def as_dicts(self, today: date | None = None) -> list[dict]:
        """The shape tier_engine reads comps in.

        Carries each comp's age and its OWN extension's evidence half-life.
        The comp's extension rather than the subject's, because the question
        the half-life answers is whether that sale still describes the market
        it happened in — the cross-extension adjustment is transfer_ratio's
        job and is already applied to the price.
        """
        rows = []
        for c in self.comps:
            if c.adjusted_price <= 0:
                continue
            comp_tld = str(c.domain or "").rpartition(".")[2].lower()
            rows.append({
                "domain": c.domain,
                "price": float(c.adjusted_price),
                "date": c.date,
                "marketplace": c.venue,
                "weight": c.weight,
                "age_days": _comp_age_days(c.date, today),
                "comp_half_life_days": phase_for(comp_tld).comp_half_life_days,
                "tld_phase": phase_for(comp_tld).phase,
            })
        return rows


def _comp_age_days(sale_date, today: date | None) -> int | None:
    if not sale_date:
        return None
    try:
        parsed = (sale_date if isinstance(sale_date, date)
                  else datetime.fromisoformat(str(sale_date)[:10]).date())
    except (ValueError, TypeError):
        return None
    return max(0, ((today or date.today()) - parsed).days)


@dataclass
class LadderInputs:
    """Everything the ladder reads, already fetched.

    Taking data rather than tools keeps the ladder a pure function: the
    evals build one of these from fixtures and assert exact rungs without
    any network or monkeypatching.
    """

    domain: str
    sld: str
    tld: str
    category: str
    keywords: list[str] = field(default_factory=list)
    # Raw sale rows: {domain, price, date, marketplace}
    keyword_sales: list[dict] = field(default_factory=list)
    similar_sales: list[dict] = field(default_factory=list)
    cross_tld_sales: list[dict] = field(default_factory=list)
    cohort_sales: list[dict] = field(default_factory=list)
    internal_sales: list[dict] = field(default_factory=list)
    # {keyword: {avg_price, total_sales}}
    keyword_stats: dict = field(default_factory=dict)
    # Quality signals about the SUBJECT, used to judge whether the comps
    # returned are in its class at all.
    brand_score: float | None = None
    structural: dict | None = None
    today: date | None = None
    # Exclusive upper bound for a time-holdout. Sales of `domain` itself,
    # sales dated on or after this day, and undated rows are dropped, so a
    # backtest cannot be graded on the sale it is predicting.
    as_of: date | None = None


PREMIUM, MID, COMMODITY = "premium", "mid", "commodity"
_BAND_RANK = {COMMODITY: 0, MID: 1, PREMIUM: 2}


def subject_band(inputs: LadderInputs) -> str:
    """Which market the domain being valued actually belongs to.

    Deterministic, from signals already gathered. A five-character
    dictionary word with a high brand score trades in a different market
    from a thirteen-character compound, and a comp set that ignores the
    difference will anchor one on the other.
    """
    from portfolio_agent.knowledge.dictionary import is_dictionary_word

    structural = inputs.structural or {}
    length = structural.get("char_length") or len(inputs.sld)
    words = structural.get("word_count") or 1
    brand = inputs.brand_score

    if length >= config.SUBJECT_BAND_COMMODITY_MIN_LENGTH or words > 1:
        return COMMODITY
    if brand is not None and brand < config.SUBJECT_BAND_COMMODITY_MAX_BRAND:
        return COMMODITY
    if (
        length <= config.SUBJECT_BAND_PREMIUM_MAX_LENGTH
        and is_dictionary_word(inputs.sld)
        and (brand is None or brand >= config.SUBJECT_BAND_PREMIUM_MIN_BRAND)
    ):
        return PREMIUM
    return MID


def comp_band(domain: str) -> str:
    """The same judgement for a comp, from structure alone.

    No brand score is available for a comp, so this is a structural proxy —
    which is most of what the band means anyway.
    """
    from portfolio_agent.knowledge.dictionary import is_dictionary_word
    from portfolio_agent.knowledge.lexicon import segment

    sld = str(domain or "").lower().partition(".")[0]
    if not sld:
        return COMMODITY
    if len(sld) >= config.SUBJECT_BAND_COMMODITY_MIN_LENGTH or len(segment(sld)) > 1:
        return COMMODITY
    if len(sld) <= config.SUBJECT_BAND_PREMIUM_MAX_LENGTH and is_dictionary_word(sld):
        return PREMIUM
    return MID


def band_distance(subject: str, comp: str) -> int:
    return abs(_BAND_RANK[subject] - _BAND_RANK[comp])


def _apply_band_weights(result: CompSet, inputs: LadderInputs) -> str:
    """Silence comps that describe a different market, and say which.

    Down-weighted rather than dropped: the retrieval did return them, and
    hiding that would make a thin set look healthy. At 0.15 they are present
    and visible without steering the anchor.
    """
    band = subject_band(inputs)
    for comp in result.comps:
        if band_distance(band, comp_band(comp.domain)) >= 2:
            comp.weight = config.CROSS_BAND_WEIGHT
            comp.adjustment_reason = (
                (comp.adjustment_reason + "; ") if comp.adjustment_reason else ""
            ) + f"different quality band from the subject ({comp_band(comp.domain)} vs {band})"
    return band


def _apply_industry_weights(result: CompSet, inputs: LadderInputs) -> None:
    """Quieten comps that sell into a different market, and say when they are
    ALL that is left.

    Retrieval matches on shape, and shape is not a market: a fifteen-letter
    two-word .com is priced by whether the words are "cloud storage" or
    "divorce lawyer". Down-weighted rather than dropped, for the same reason
    as the quality bands — the sale happened, and hiding it would make a
    thin set look healthy.

    The note matters more than the weight. Percentile weights are relative,
    so if EVERY comp is cross-industry, down-weighting them uniformly moves
    nothing at all — the set is simply the wrong market and no arithmetic
    fixes that. Only a warning can.
    """
    subject = classify_industry(inputs.sld)
    if subject is None:
        return

    classified = 0
    mismatched = []
    for comp in result.comps:
        comp_sld, _ = split_domain(comp.domain)
        comp_industry = classify_industry(comp_sld)
        if comp_industry is None:
            continue  # absence of evidence, never evidence of mismatch
        classified += 1
        if comp_industry != subject:
            mismatched.append(comp)
            comp.weight = min(comp.weight, config.CROSS_INDUSTRY_WEIGHT)
            comp.adjustment_reason = (
                (comp.adjustment_reason + "; ") if comp.adjustment_reason else ""
            ) + f"different industry from the subject ({comp_industry} vs {subject})"

    if not mismatched:
        return
    if len(mismatched) == classified:
        result.notes.append(
            f"every comp whose industry could be identified sells into a different market than "
            f"{subject} ({classified} of {len(result.comps)} classified). Down-weighting them "
            f"changes nothing when there is nothing to weight them against, so this price rests "
            f"on the wrong industry and should be read as a shape comparison, not a market one"
        )
    else:
        result.notes.append(
            f"{len(mismatched)} of {classified} industry-identified comps sell into a different "
            f"market than {subject} and were down-weighted to {config.CROSS_INDUSTRY_WEIGHT}"
        )
    info(log, "comp_ladder.industry_weighted", subject=subject, classified=classified,
         mismatched=len(mismatched), total=len(result.comps))


def _note_comp_dating(result: CompSet, inputs: LadderInputs, today: date | None = None) -> None:
    """Say when the comps predate the market they are being used to describe.

    The recency weighting already quietens them, but weighting alone is not
    a disclosure: a reader seeing a .ai price built entirely from 2022 sales
    should be told, because the extension has had a whole boom since and no
    amount of down-weighting turns old evidence into new evidence.
    """
    phase = phase_for(inputs.tld)
    ages = [a for a in (_comp_age_days(c.date, today) for c in result.comps) if a is not None]
    if not ages:
        return
    median_age = int(statistics.median(ages))
    if median_age <= phase.comp_half_life_days:
        return
    result.notes.append(
        f"the median comparable sale is {median_age} days old, past the {phase.comp_half_life_days}-day "
        f"evidence half-life for .{inputs.tld} ({phase.phase}: {phase.note}). They are weighted down "
        f"for age, but a price built from them describes the market those sales happened in"
    )
    stale = lifecycle_staleness(today)
    if stale:
        result.notes.append(stale)
    info(log, "comp_ladder.comps_predate_phase", tld=inputs.tld, phase=phase.phase,
         median_age_days=median_age, half_life=phase.comp_half_life_days)


def peer_count(result: CompSet, band: str) -> int:
    """Comps within one band of the subject — the only ones that anchor it."""
    return sum(1 for c in result.comps if band_distance(band, comp_band(c.domain)) <= 1)


def build_comp_set(
    inputs: LadderInputs, min_comps: int | None = None, min_peers: int | None = None
) -> CompSet:
    """Descend the ladder until there are enough USABLE comps, and report
    the rung reached."""
    minimum = min_comps or config.COMP_LADDER_MIN_COMPS
    min_peers = config.MIN_PEER_COMPS if min_peers is None else min_peers
    band = subject_band(inputs)
    result = CompSet()

    for step, basis in (
        (_step_exact, CompBasis.EXACT),
        (_step_cross_tld, CompBasis.SLD_CROSS_TLD),
        (_step_cohort, CompBasis.COHORT),
        (_step_keyword_only, CompBasis.KEYWORD_ONLY),
    ):
        added = step(inputs)
        result.comps.extend(_dedupe(added, result.comps))
        result.steps_taken.append(f"{basis.value}:{len(added)}")

        # A rung is sufficient only when it has enough comps AND enough of
        # them are in the subject's class. Counting comps alone is what let
        # forge.ai stop at `exact` on nine compound names — dataforge.ai at
        # $787 anchoring a five-character dictionary word — while forty-four
        # peers sat one rung below, unreached, because the count had already
        # been satisfied.
        peers = peer_count(result, band)
        if len(result.comps) >= minimum and peers >= min_peers:
            result.basis = basis
            result.confidence = config.COMP_BASIS_CONFIDENCE[basis.value]
            _apply_band_weights(result, inputs)
            _apply_industry_weights(result, inputs)
            _note_comp_dating(result, inputs, inputs.as_of or inputs.today)
            _note_holdout(result, inputs)
            _apply_peer_floor(result, inputs, band)
            _apply_internal_lift(result)
            _annotate(result, inputs, basis)
            info(log, "comp_ladder.resolved", basis=basis.value, comps=len(result.comps),
                 peers=peers, band=band, confidence=result.confidence,
                 steps=",".join(result.steps_taken))
            return result
        debug(log, "comp_ladder.step_insufficient", basis=basis.value, found=len(added),
              running_total=len(result.comps), peers=peers, needed=minimum, peers_needed=min_peers)

    # Every rung tried and still short. The two ways of being short are not
    # equally recoverable.
    #
    # Too few PEERS is a composition problem: there are enough sales to form
    # a distribution, they are just not all in the subject's class. That is
    # worth pricing at reduced confidence, because the alternative is
    # discarding real sales and saying nothing.
    #
    # Too few COMPS is not recoverable at all. One sale is an anecdote
    # whatever its quality, and a ladder built on it would be arithmetic
    # performed on a single data point.
    if len(result.comps) >= minimum:
        result.basis = CompBasis.KEYWORD_ONLY if any(
            c.source == "keyword_stats" for c in result.comps
        ) else CompBasis.COHORT
        result.confidence = min(
            config.COMP_BASIS_CONFIDENCE[result.basis.value], config.UNDERPOWERED_CONFIDENCE
        )
        _apply_band_weights(result, inputs)
        _apply_industry_weights(result, inputs)
        _note_comp_dating(result, inputs, inputs.as_of or inputs.today)
        _note_holdout(result, inputs)
        _apply_peer_floor(result, inputs, band)
        result.notes.append(
            f"every basis was tried and none produced {minimum} comps with {min_peers} in the "
            f"subject's class; the {len(result.comps)} found are used at reduced confidence"
        )
        info(log, "comp_ladder.exhausted_but_usable", comps=len(result.comps),
             basis=result.basis.value, confidence=result.confidence)
        return result

    result.basis = CompBasis.NONE
    result.confidence = 0.0
    _note_holdout(result, inputs)
    result.notes.append(
        f"no comparable sales at any basis — {len(result.comps)} found, {minimum} needed"
    )
    info(log, "comp_ladder.exhausted", comps=len(result.comps))
    return result


# ── Rungs ──────────────────────────────────────────────────────────────────


def _is_exact_match(subject_sld: str, comp_sld: str, terms: list[str]) -> bool:
    """Check if a comp's SLD is an exact match for the subject.
    
    For single-token subjects (pay, isotope), an "exact" comp must have the
    SAME SLD, not merely contain the subject term. Allowed equivalences:
    - Exact equality: pay == pay
    - Plural variants: isotope == isotopes
    - Hyphen variants: pay == pay-pal is NOT equal (different word)
    
    Multi-token subjects use the old substring logic since they're already
    compound names where the keyword appearing is meaningful.
    """
    subject_sld = subject_sld.lower().strip()
    comp_sld = comp_sld.lower().strip()
    
    # Exact equality always matches
    if subject_sld == comp_sld:
        return True
    
    # Plural equivalence: isotope <-> isotopes
    if subject_sld + "s" == comp_sld or comp_sld + "s" == subject_sld:
        return True
    
    # If subject is a single token (no hyphens, no numbers, clean word),
    # require strict equality. Multi-token or compound subjects can use
    # substring matching since they're already compound names.
    from portfolio_agent.knowledge.lexicon import segment
    subject_segments = segment(subject_sld)
    
    # Single-token subject: strict match only (already checked above)
    if len(subject_segments) <= 1 and "-" not in subject_sld:
        return False
    
    # Multi-token subject: fall back to substring match
    # (the old behavior for compound names)
    return any(term in comp_sld for term in terms)


def _step_exact(inputs: LadderInputs) -> list[CompRef]:
    """Sales of this keyword on this TLD. The thing itself.

    "The thing itself" has to mean it. This rung used to accept the semantic
    search's output too, and semantic neighbours are not exact comps:
    forge.ai drew daft.ai, ide.ai and icode.ai, was told its basis was
    `exact` at confidence 1.0, and never reached for the cohort that
    actually defines its class. Ten neighbours read as certainty.

    A comp qualifies here only if it is the SAME asset class — for a
    single-token subject, the SLD must equal the subject SLD (different TLD
    allowed) OR match under a defined equivalence (plural/hyphen).
    paydayloans.ai is NOT an exact comp for pay.ai — it's a different SLD.
    Everything else is a cohort member and is priced as one, at cohort
    confidence.
    """
    terms = [t.lower() for t in (inputs.keywords or [inputs.sld]) if t]
    comps = []
    rejected_substring = 0
    for row in list(inputs.keyword_sales) + list(inputs.internal_sales):
        tld = _tld_of(row)
        if tld and tld != inputs.tld:
            continue
        if not _admissible(row, inputs):
            continue
        sld = str(row.get("domain") or "").lower().partition(".")[0]
        internal = row in inputs.internal_sales
        # Own-book sales are matched by the caller's query, so they are
        # trusted without a substring check; public feed rows are not.
        if not internal and terms:
            # F17.1: Exact match means SAME ASSET CLASS, not "substring appears".
            # For a single-token subject, the comp's SLD must equal the subject
            # SLD or match under defined equivalence (plural, hyphen variants).
            if not _is_exact_match(inputs.sld, sld, terms):
                rejected_substring += 1
                continue
        comps.append(
            _comp(
                row,
                source="internal_comps" if internal else "sales_history",
                weight=config.COMP_WEIGHT_INTERNAL if internal else config.COMP_WEIGHT_EXACT,
            )
        )
    if rejected_substring > 0:
        info(log, "comp_ladder.exact_match_rejected", count=rejected_substring,
             reason="substring_not_equal", subject=inputs.sld)
    return comps


def _internal_comps(inputs: LadderInputs, tld_must_match: bool) -> list[CompRef]:
    """Own-book sales, available at every rung.

    Verified and first-party, so they outweigh a scraped record — but from
    one venue, so they must not drown a broader market either. Hence 1.5x,
    between a cohort member's 1.0 and an exact comp's 2.0.
    """
    comps = []
    for row in inputs.internal_sales:
        tld = _tld_of(row)
        if tld_must_match and tld and tld != inputs.tld:
            continue
        if not _admissible(row, inputs):
            continue
        comps.append(
            _comp(row, source="internal_comps", weight=config.COMP_WEIGHT_INTERNAL,
                  adjustment_reason="verified sale from our own book")
        )
    return comps


def _step_cross_tld(inputs: LadderInputs) -> list[CompRef]:
    """This exact name, sold on another extension.

    The strongest evidence available when the domain's own extension has no
    history: it is the same name, and the only question is what the
    extension is worth. That question has an answer in
    knowledge/tld_transfer.py, and the answer is applied per comp and stated.
    """
    comps = _internal_comps(inputs, tld_must_match=False)
    for row in inputs.cross_tld_sales:
        sld, _, tld = str(row.get("domain") or "").lower().partition(".")
        if sld != inputs.sld or not tld or tld == inputs.tld:
            continue
        if tld not in config.CROSS_TLD_SET and inputs.tld not in config.CROSS_TLD_SET:
            continue
        if not _admissible(row, inputs):
            continue
        ratio = transfer_ratio(tld, inputs.tld)
        comps.append(
            _comp(row, source="sales_history", adjustment=ratio.multiplier,
                  adjustment_reason=ratio.describe(tld, inputs.tld))
        )
    return comps


def _step_cohort(inputs: LadderInputs) -> list[CompRef]:
    """Structurally similar names on this TLD.

    Not "similar meaning" — similar SHAPE. A single dictionary word on .ai
    is priced by the market as a member of that class, largely independent
    of which word it is, which is why a cohort works at all where semantic
    comps do not.

    The cohort members carry no per-comp adjustment. Their SPREAD is the
    information: the tier engine takes p25-p75 of the cohort rather than its
    extremes, so one outlier sale cannot define a class.
    """
    # Semantic neighbours belong HERE, not on the exact rung — they are
    # structurally similar names, which is precisely what a cohort is.
    comps = _internal_comps(inputs, tld_must_match=True)
    for row in list(inputs.similar_sales) + list(inputs.cohort_sales):
        tld = _tld_of(row)
        if tld and tld != inputs.tld:
            continue
        if not _admissible(row, inputs):
            continue
        comps.append(_comp(row, source="comparables", adjustment_reason="same structural cohort"))
    return _dedupe_by_domain(comps)


def _dedupe_by_domain(comps: list[CompRef]) -> list[CompRef]:
    seen, kept = set(), []
    for comp in comps:
        key = comp.domain.lower()
        if key not in seen:
            seen.add(key)
            kept.append(comp)
    return kept


def _step_keyword_only(inputs: LadderInputs) -> list[CompRef]:
    """Aggregate keyword statistics, with no individual sale behind them.

    Skipped on a holdout. The aggregate is undated, so it may already
    contain the sale being predicted, and a backtest that used it would be
    grading the engine on an average that includes the answer.

    The weakest rung by a distance, and it is included because the
    alternative is silence. A synthesized comp says "sales containing this
    keyword average $X across Y transactions" — which is a real fact about a
    real market, just not about any one domain. It is labeled as synthesized
    so nothing downstream can mistake it for a sale that happened.
    """
    if inputs.as_of is not None:
        return []
    comps = []
    for keyword, stats in (inputs.keyword_stats or {}).items():
        count = stats.get("total_sales") or stats.get("count") or 0
        if count <= 0:
            continue

        # The PERCENTILES, not just the average. Three reasons, and the
        # first is the one that matters: a single average is one comp, and
        # one comp never reaches the three-comp minimum, so a
        # single-keyword domain could never reach this rung at all. The
        # second is that these are real percentiles of real sales — p25 and
        # p75 describe that market more honestly than its mean. The third is
        # that the tier engine derives its band width from the spread, and
        # an average has no spread.
        points = [
            ("25th percentile", stats.get("p25")),
            ("median", stats.get("median") or stats.get("avg_price") or stats.get("average_price")),
            ("75th percentile", stats.get("p75")),
        ]
        for label, value in points:
            if not value or value <= 0:
                continue
            comps.append(
                CompRef(
                    domain=f"[{label} of {count} sales containing '{keyword}']",
                    price=int(value),
                    date=None,
                    venue="aggregate",
                    source="keyword_stats",
                    adjustment=1.0,
                    adjustment_reason=(
                        f"synthesized from the {label} of {count} sales — a statistic, not a transaction"
                    ),
                    weight=1.0,
                )
            )
    return comps


# ── Helpers ────────────────────────────────────────────────────────────────


def _apply_peer_floor(result: CompSet, inputs: LadderInputs, band: str) -> None:
    """A set with no peers cannot support full confidence, however large.

    forge.ai drew nine comps — dataforge.ai at $787, codeforge.ai at $455 —
    every one of them a compound name, and was told its basis was `exact` at
    confidence 1.0. Nine commodity comps are not nine peers, and the count
    was doing the reassuring while the composition did the damage.
    """
    peers = peer_count(result, band)
    if peers >= config.MIN_PEER_COMPS:
        return

    result.notes.append(
        f"cohort lacks same-tier comps: only {peers} of {len(result.comps)} are within one quality "
        f"band of a {band} subject, so the anchor may understate what this domain's own class "
        f"trades at"
    )
    if result.confidence > config.UNDERPOWERED_CONFIDENCE:
        result.confidence = config.UNDERPOWERED_CONFIDENCE
        result.notes.append(
            f"confidence capped at {config.UNDERPOWERED_CONFIDENCE:.2f} because the comp set has "
            f"fewer than {config.MIN_PEER_COMPS} peers, whatever rung it reached"
        )


def _apply_internal_lift(result: CompSet) -> None:
    """Verified own-book sales are better evidence than the rung implies.

    The rung records how far the SEARCH had to widen, which is not the same
    question as how good the evidence is. A set that stopped at
    `keyword_only` but contains two sales we brokered ourselves is not
    resting on aggregates — most of its weight is on transactions we can
    verify. Reporting 0.25 confidence there would understate it as badly as
    reporting 1.0 would overstate it.

    The basis label is left alone. It is a factual record of where the
    comps came from, and rewriting it would hide the widening that actually
    happened.
    """
    own_book = [c for c in result.comps if c.source == "internal_comps"]
    floor = config.COMP_BASIS_CONFIDENCE["cohort"]
    if len(own_book) < config.INTERNAL_COMPS_MIN_FOR_LIFT or result.confidence >= floor:
        return

    result.confidence = floor
    result.notes.append(
        f"confidence raised to {floor:.2f} because {len(own_book)} of these comps are verified sales "
        f"from our own brokerage book — the search had to widen, but what it found is first-party "
        f"and confirmed, not an aggregate"
    )


def _annotate(result: CompSet, inputs: LadderInputs, basis: CompBasis) -> None:
    """The sentence a reader needs about how this price was reached."""
    if basis is CompBasis.EXACT:
        return
    if basis is CompBasis.SLD_CROSS_TLD:
        result.notes.append(
            f"no .{inputs.tld} sales of this term, so '{inputs.sld}' was priced from its sales on other "
            f"extensions with a TLD transfer ratio applied to each"
        )
    elif basis is CompBasis.COHORT:
        result.notes.append(
            f"no sales of this name on any extension, so it was priced from the cohort of structurally "
            f"similar .{inputs.tld} domains — the class, not the name"
        )
    elif basis is CompBasis.KEYWORD_ONLY:
        result.notes.append(
            "pricing based on keyword aggregates only — no individual comparable sale was found, and "
            "the figures below are derived from averages across a keyword's sales rather than from any "
            "transaction involving a domain like this one"
        )


def _comp(row: dict, *, source: str, adjustment: float = 1.0,
          adjustment_reason: str | None = None, weight: float = 1.0) -> CompRef:
    return CompRef(
        domain=str(row.get("domain") or "unknown"),
        price=int(float(row.get("price") or 0)),
        date=_date_of(row),
        venue=row.get("marketplace") or row.get("platform") or row.get("venue"),
        source=source,
        adjustment=adjustment,
        adjustment_reason=adjustment_reason,
        weight=weight,
    )


def _dedupe(candidates: list[CompRef], existing: list[CompRef]) -> list[CompRef]:
    """A sale found at two rungs is one sale, and it keeps its stronger form."""
    seen = {c.domain.lower() for c in existing}
    kept = []
    for comp in candidates:
        key = comp.domain.lower()
        if key in seen or comp.price <= 0:
            continue
        seen.add(key)
        kept.append(comp)
    return kept


def _tld_of(row: dict) -> str:
    return str(row.get("domain") or "").lower().partition(".")[2]


def _date_of(row: dict) -> str | None:
    value = row.get("date") or row.get("saleDate")
    return str(value)[:10] if value else None


def _note_holdout(result: CompSet, inputs: LadderInputs) -> None:
    if inputs.as_of is None:
        return
    result.notes.append(
        f"holdout as of {inputs.as_of.isoformat()}: {inputs.domain} itself, any sale on or after "
        f"that date, and undated keyword aggregates are excluded, so this price is not graded "
        f"on the sale it is predicting"
    )
    info(log, "comp_ladder.holdout", domain=inputs.domain, as_of=inputs.as_of.isoformat())


def _admissible(row: dict, inputs: LadderInputs) -> bool:
    """Whether this sale may enter the ladder.

    Live runs keep the old rule: undated sales stay, and only sales older
    than the staleness window are dropped. A holdout is stricter, because
    the question has changed from "is this sale stale?" to "could this sale
    have been known on the morning of the sale we are grading?"
    """
    if inputs.as_of is None:
        return _recent_enough(row, inputs.today)
    domain = str(row.get("domain") or "").strip().lower()
    if domain and domain == inputs.domain.strip().lower():
        return False
    raw = _date_of(row)
    if not raw:
        return False
    try:
        sold = date.fromisoformat(raw)
    except ValueError:
        return False
    if sold >= inputs.as_of:
        return False
    return _recent_enough(row, inputs.as_of)


def _recent_enough(row: dict, today: date | None) -> bool:
    """A sale from five years ago describes a market that no longer exists."""
    raw = _date_of(row)
    if not raw:
        return True  # undated sales are kept; an unknown date is not a stale one
    try:
        sold = date.fromisoformat(raw)
    except ValueError:
        return True
    age_months = ((today or date.today()) - sold).days / 30.44
    return age_months <= config.COMP_MAX_AGE_MONTHS
