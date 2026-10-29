"""A transparent break-even scenario, deliberately separate from measured usage."""
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction


def _decimal(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{name} must be a finite nonnegative number") from None
    if not result.is_finite() or result < 0 or result > Decimal("1e12"):
        raise ValueError(f"{name} must be finite and between 0 and 1e12")
    if len(result.as_tuple().digits) > 100 or abs(result.as_tuple().exponent) > 100:
        raise ValueError(f"{name} supports at most 100 digits and decimal exponent magnitude 100")
    return result


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10**9:
        raise ValueError(f"{name} must be an integer between 0 and 1,000,000,000")
    return value


def estimate_compaction(*, current_tokens, retained_tokens, future_turns,
                        cache_read_usd_per_million, rewrite_cost_usd, summary_cost_usd):
    """Compare costs attributable to a replaced span under stated fixed assumptions."""
    current = _count(current_tokens, "current_tokens")
    retained = _count(retained_tokens, "retained_tokens")
    turns = _count(future_turns, "future_turns")
    if retained > current:
        raise ValueError("retained_tokens cannot exceed current_tokens")
    rate = _decimal(cache_read_usd_per_million, "cache_read_usd_per_million")
    rewrite = _decimal(rewrite_cost_usd, "rewrite_cost_usd")
    summary = _decimal(summary_cost_usd, "summary_cost_usd")
    # Products and sums terminate in base ten. Reserve enough significant digits
    # to retain tiny differences; use an exact rational for strict break-even.
    values = (rate, rewrite, summary)
    precision = max(28, max([0] + [v.adjusted() for v in values])
                    - min([0] + [v.as_tuple().exponent for v in values]) + 32)
    with localcontext() as context:
        context.prec = precision
        per_turn = Decimal(current - retained) * rate / Decimal(1_000_000)
        upfront = rewrite + summary
        keep = Decimal(turns * current) * rate / Decimal(1_000_000)
        compact = upfront + Decimal(turns * retained) * rate / Decimal(1_000_000)
        savings = keep - compact
        threshold = (int(Fraction(upfront) // Fraction(per_turn)) + 1 if per_turn else None)
    return {
        "schema_version": 1,
        "evidence_type": "scenario_estimate",
        "certifies_customer_savings": False,
        "inputs": {"current_tokens": current, "retained_tokens": retained, "future_turns": turns,
                   "cache_read_usd_per_million": str(rate), "rewrite_cost_usd": str(rewrite),
                   "summary_cost_usd": str(summary)},
        "keep_cost_usd": format(keep, "f"),
        "compact_cost_usd": format(compact, "f"),
        "estimated_net_savings_usd": format(savings, "f"),
        "cached_read_savings_per_turn_usd": format(per_turn, "f"),
        "first_strictly_cheaper_turn": threshold,
        "cheaper_at_supplied_horizon": compact < keep,
        "equation": "net savings = future turns × removed tokens × cache-read rate / 1,000,000 − rewrite cost − summary cost",
        "assumptions": [
            "All modeled future reads hit the cache at the supplied constant rate; expiry and routing misses are excluded.",
            "The rewrite estimate must include the entire affected suffix, not just the shortened tool output.",
            "Future turn count is supplied by the user, not predicted or known by KVPilot.",
            "Task behavior, output costs, retries and quality are assumed unchanged; this estimate does not validate them.",
            "Costs cover only the modeled changed context; other task costs must be measured in a live comparison.",
        ],
    }
