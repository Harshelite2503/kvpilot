"""Reproducible token accounting, without invoice or savings certification."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, ROUND_HALF_EVEN, localcontext

from .schema import ContextDataError, Pricing, RATE_KEYS, Trace, token_categories


SCOPE_EXCLUSIONS = [
    "No invoice reconciliation: totals use supplied provider token counts and a user-supplied flat rate card.",
    "No provider API calls, current-price verification, cache-hit simulation, or causal savings claim.",
    "Nonzero server-tool usage, server-side iteration billing, and nontext usage counters are unsupported and rejected.",
    "Taxes, credits, negotiated discounts, batch/service-tier/geography adjustments, and external tool/runtime charges are not modeled.",
    "A single flat rate card does not infer or apply per-request long-context thresholds, premium/fast modes, or mixed pricing tiers.",
    "Provider/model checks compare declared trace and pricing labels, not original response.model fields; the exporter must use the actual returned model and exclude mixed-model runs.",
    "Text-only Messages usage is assumed; traces must not hide unreported modality or ancillary charges.",
]


def _decimal_text(value):
    if value == 0:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def audit_costs(trace: Trace, pricing: Pricing) -> dict:
    """Bill every request once, including failed tasks and compaction work."""
    for key in ("provider", "model"):
        if trace.metadata[key] != pricing.metadata[key]:
            raise ContextDataError(f"trace/pricing {key} mismatch")
    categories = [token_categories(request["usage"]) for request in trace.requests]
    token_totals = {key: sum(counts[key] for counts in categories) for key in RATE_KEYS}
    # Enough precision to sum the exact decimal products even when price
    # categories use very different decimal scales. Division by 1e6 is exact.
    rates = list(pricing.rates_usd_per_million.values())
    max_adjusted = max([0] + [rate.adjusted() for rate in rates])
    min_exponent = min([0] + [rate.as_tuple().exponent for rate in rates])
    precision = max(28, max_adjusted - min_exponent + max(len(str(n)) for n in token_totals.values()) + 12)
    with localcontext() as ctx:
        ctx.prec = precision
        task_costs = {task["task_id"]: Decimal(0) for task in trace.tasks}
        requests = []
        total, compaction = Decimal(0), Decimal(0)
        compaction_count = 0
        for request, counts in zip(trace.requests, categories):
            breakdown = {key: Decimal(counts[key]) * pricing.rates_usd_per_million[key] / Decimal(1_000_000)
                         for key in RATE_KEYS}
            cost = sum(breakdown.values(), Decimal(0))
            task_costs[request["task_id"]] += cost
            total += cost
            if request["kind"] == "compaction":
                compaction += cost
                compaction_count += 1
            row = deepcopy(request)
            row.update(token_categories=counts,
                       cost_breakdown_usd={key: _decimal_text(value) for key, value in breakdown.items()},
                       total_cost_usd=_decimal_text(cost))
            requests.append(row)
        tasks = [{"task_id": task["task_id"], "success": task["success"],
                  "duration_ms": task.get("duration_ms"), "total_cost_usd": _decimal_text(task_costs[task["task_id"]])}
                 for task in trace.tasks]
        successes = sum(task["success"] for task in trace.tasks)
        with localcontext() as ratio_context:
            ratio_context.prec = 28
            ratio_context.rounding = ROUND_HALF_EVEN
            per_success = _decimal_text(total / successes) if successes else None
    price_metadata = deepcopy(pricing.metadata)
    price_metadata.update(source_sha256=pricing.source_sha256,
                          rates_usd_per_million={key: _decimal_text(pricing.rates_usd_per_million[key]) for key in RATE_KEYS})
    return {
        "schema_version": 1, "complete": True, "metadata": deepcopy(trace.metadata),
        "evidence_type": "synthetic_example" if trace.metadata["evidence_type"] == "synthetic" else "observed_usage_accounting",
        "trace_sha256": trace.source_sha256, "pricing": price_metadata,
        "requests": requests, "tasks": tasks,
        "totals": {"total_cost_usd": _decimal_text(total), "request_count": len(requests),
                   "successful_tasks": successes, "failed_tasks": len(tasks) - successes,
                   "cost_per_successful_task_usd": per_success, "tokens": token_totals,
                   "compaction_request_count": compaction_count,
                   "compaction_request_cost_usd": _decimal_text(compaction)},
        "cost_per_successful_task_definition": "All request costs, including compaction and failed-task work, divided by successful tasks; null if no tasks succeed.",
        "cost_per_successful_task_rounding": "28 significant decimal digits, ROUND_HALF_EVEN; request/category/task totals are not rounded to currency cents.",
        "scope_exclusions": SCOPE_EXCLUSIONS.copy(),
    }
