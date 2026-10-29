"""Compare paired context-cost audits without claiming causal or quality proof."""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from typing import Any


_CONTROLS = (
    "dataset_id", "workload_version", "evaluator_id", "provider", "model",
    "cache_protocol", "evidence_type",
)
_EVIDENCE = {"synthetic": "synthetic_example", "observed": "observed_usage_accounting"}


def _decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{field} must be a finite nonnegative number")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{field} must be a finite nonnegative number") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{field} must be a finite nonnegative number")
    return result


def _count(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _money(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _exact_precision(values: list[Decimal], terms: int = 1) -> int:
    """Keep sums/products of the supplied finite decimals exact."""
    return max(50, max(value.adjusted() for value in values)
               - min(value.as_tuple().exponent for value in values)
               + len(str(terms)) + 12)


def _snapshot(value: dict, field: str) -> dict:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain JSON-safe values") from exc
    return deepcopy(value)


def _validate(report: dict, label: str) -> dict:
    if not isinstance(report, dict):
        raise ValueError(f"{label} must be an audit object")
    if report.get("schema_version") != 1 or type(report.get("schema_version")) is not int:
        raise ValueError(f"{label}.schema_version must be 1")
    if report.get("complete") is not True:
        raise ValueError(f"{label} must be a complete audit")
    trace_sha256 = report.get("trace_sha256")
    if "trace_sha256" in report and (not isinstance(trace_sha256, str)
                                     or re.fullmatch(r"[0-9a-fA-F]{64}", trace_sha256) is None):
        raise ValueError(f"{label}.trace_sha256 must contain exactly 64 hexadecimal characters")
    metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{label}.metadata must be an object")
    for key in ("run_id", "agent_revision", *_CONTROLS):
        _text(metadata.get(key), f"{label}.metadata.{key}")
    metadata = _snapshot(metadata, f"{label}.metadata")
    evidence = _EVIDENCE.get(metadata["evidence_type"])
    if evidence is None or report.get("evidence_type") != evidence:
        raise ValueError(f"{label} has inconsistent or unsupported evidence_type")
    pricing = report.get("pricing")
    if not isinstance(pricing, dict) or not pricing:
        raise ValueError(f"{label}.pricing must contain the price card and provenance")
    for key in ("pricing_id", "currency", "source", "effective_date", "provider", "model", "source_sha256"):
        _text(pricing.get(key), f"{label}.pricing.{key}")
    if pricing["provider"] != metadata["provider"] or pricing["model"] != metadata["model"]:
        raise ValueError(f"{label} price card does not match its provider/model")
    if pricing["currency"] != "USD":
        raise ValueError(f"{label} price card must use USD")
    rates = pricing.get("rates_usd_per_million")
    if not isinstance(rates, dict) or not rates:
        raise ValueError(f"{label} price card has no rates")
    for key, rate in rates.items():
        _decimal(rate, f"{label}.pricing.rates_usd_per_million.{key}")
    pricing = _snapshot(pricing, f"{label}.pricing")
    rows = report.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label}.tasks must be a nonempty list")
    tasks = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} task must be an object")
        task_id = _text(row.get("task_id"), f"{label}.task_id")
        if task_id in tasks:
            raise ValueError(f"{label} has duplicate task_id: {task_id}")
        if type(row.get("success")) is not bool:
            raise ValueError(f"{label} task {task_id} success must be a boolean")
        cost = _decimal(row.get("total_cost_usd"), f"{label}.{task_id}.total_cost_usd")
        duration = row.get("duration_ms")
        if duration is not None:
            if type(duration) not in (int, float):
                raise ValueError(f"{label}.{task_id}.duration_ms must be a number or null")
            try:
                duration = float(duration)
            except OverflowError as exc:
                raise ValueError(f"{label}.{task_id}.duration_ms must be finite") from exc
            if not math.isfinite(duration) or duration < 0:
                raise ValueError(f"{label}.{task_id}.duration_ms must be finite and nonnegative")
        tasks[task_id] = {"success": row["success"], "cost": cost, "duration_ms": duration}
    totals = report.get("totals")
    if not isinstance(totals, dict):
        raise ValueError(f"{label}.totals must be an object")
    total = _decimal(totals.get("total_cost_usd"), f"{label}.totals.total_cost_usd")
    successes = _count(totals.get("successful_tasks"), f"{label}.totals.successful_tasks")
    failures = _count(totals.get("failed_tasks"), f"{label}.totals.failed_tasks")
    if successes != sum(row["success"] for row in tasks.values()) or failures != len(tasks) - successes:
        raise ValueError(f"{label} task outcome counts disagree with totals")
    with localcontext() as sum_context:
        sum_context.prec = _exact_precision([total, *(row["cost"] for row in tasks.values())], len(tasks))
        if total != sum((row["cost"] for row in tasks.values()), Decimal(0)):
            raise ValueError(f"{label} task costs disagree with total_cost_usd")
    supplied_cps = totals.get("cost_per_successful_task_usd")
    with localcontext() as ratio_context:
        ratio_context.prec = 28
        ratio_context.rounding = ROUND_HALF_EVEN
        cps = total / successes if successes else None
    if cps is None:
        if supplied_cps is not None:
            raise ValueError(f"{label} cost per success must be null with no successes")
    else:
        supplied_cps = _decimal(supplied_cps, f"{label}.totals.cost_per_successful_task_usd")
        # Match billing's declared precision exactly; gates below use the exact
        # total/count fractions, never this rounded display value.
        if supplied_cps != cps:
            raise ValueError(f"{label} cost per success disagrees with totals")
    durations = [row["duration_ms"] for row in tasks.values()]
    p95 = None if any(x is None for x in durations) else sorted(durations)[math.ceil(0.95 * len(durations)) - 1]
    return {"metadata": metadata, "trace_sha256": trace_sha256, "tasks": tasks, "total": total, "successes": successes,
            "failures": failures, "cps": cps, "p95": p95, "pricing": pricing}


def _gate(status: str, *reasons: str) -> dict:
    return {"status": status, "reasons": list(reasons)}


def compare_reports(
    baseline: dict,
    candidate: dict,
    *,
    min_savings_pct: float = 0,
    max_p95_regression_pct: float = 5,
) -> dict:
    """Check matched tasks, cost per success, outcomes, and task-wall-clock p95.

    Malformed audits raise ValueError. Valid but incompatible audits return an
    inconclusive result without calculating cross-run deltas. A pass is a
    descriptive check against supplied evaluator labels, not a quality guarantee
    or a statistical/causal finding. Synthetic passes are only mechanical tests.
    """
    with localcontext() as context:
        context.prec = 50
        minimum = _decimal(min_savings_pct, "min_savings_pct")
        maximum = _decimal(max_p95_regression_pct, "max_p95_regression_pct")
        if minimum > 100:
            raise ValueError("min_savings_pct must be between 0 and 100")
        left, right = _validate(baseline, "baseline"), _validate(candidate, "candidate")
        mismatch = [f"metadata mismatch: {key}" for key in _CONTROLS
                    if left["metadata"][key] != right["metadata"][key]]
        if left["metadata"]["run_id"] == right["metadata"]["run_id"]:
            mismatch.append("baseline and candidate run_id must differ")
        if set(left["tasks"]) != set(right["tasks"]):
            mismatch.append("baseline and candidate must contain exactly the same task IDs")
        if left["pricing"] != right["pricing"]:
            mismatch.append("price card or its provenance differs")
        limitations = [
            "Task success is supplied by the evaluator; this comparison does not rerun tasks or infer answer quality.",
            "Deltas are descriptive, with no statistical significance or causal savings claim.",
            "Task duration is end-to-end wall-clock supplied per task; request latencies are not summed.",
        ]
        # Comparison reports must retain the accounting scope, not just the
        # numerical totals, so a token charge is never labeled a complete bill.
        for audit in (baseline, candidate):
            for exclusion in audit.get("scope_exclusions", []):
                if exclusion not in limitations:
                    limitations.append(exclusion)
        if baseline["evidence_type"] == "synthetic_example" or candidate["evidence_type"] == "synthetic_example":
            limitations.append("Synthetic examples cannot certify customer savings or production performance.")
        result = {
            "schema_version": 1, "comparison_type": "paired_context_cost_audit",
            "baseline_run_id": left["metadata"]["run_id"],
            "candidate_run_id": right["metadata"]["run_id"],
            "baseline_trace_sha256": left["trace_sha256"],
            "candidate_trace_sha256": right["trace_sha256"],
            "baseline_metadata": left["metadata"], "candidate_metadata": right["metadata"],
            "pricing": left["pricing"] if left["pricing"] == right["pricing"] else None,
            "evidence_type": baseline["evidence_type"] if not mismatch else "incompatible_audits",
            "comparable": not mismatch, "certifies_customer_savings": False,
            "thresholds": {"min_savings_pct": str(minimum), "max_p95_regression_pct": str(maximum)},
            "p95_method": "nearest_rank_over_task_wall_clock_including_failed_tasks",
            "summary": None, "paired_outcomes": None, "limitations": limitations,
            "monetary_rounding": "Cost per success: 28 significant decimal digits, ROUND_HALF_EVEN, matching billing. Savings gates compare exact total/count fractions; percentages are descriptive Decimal quotients.",
        }
        if left["trace_sha256"] is None or right["trace_sha256"] is None:
            limitations.append("A supplied audit lacks a trace hash; source trace identity cannot be checked from this comparison.")
        if mismatch:
            result.update(status="inconclusive", passed=False, reasons=mismatch, gates={
                "comparability": _gate("inconclusive", *mismatch),
                **{name: _gate("inconclusive", "Incompatible audits; no comparison performed.")
                   for name in ("quality", "cost", "latency")},
            })
            return result
        task_ids = sorted(left["tasks"])
        regressions = [key for key in task_ids if left["tasks"][key]["success"] and not right["tasks"][key]["success"]]
        improvements = [key for key in task_ids if not left["tasks"][key]["success"] and right["tasks"][key]["success"]]
        result["paired_outcomes"] = {
            "task_count": len(task_ids), "regressions": regressions, "improvements": improvements,
            "passed_both": sum(left["tasks"][key]["success"] and right["tasks"][key]["success"] for key in task_ids),
            "failed_both": sum(not left["tasks"][key]["success"] and not right["tasks"][key]["success"] for key in task_ids),
        }
        cost_savings = None
        if left["cps"] is None or right["cps"] is None:
            cost_gate = _gate("inconclusive", "Both runs need at least one successful task for cost per success.")
        elif left["cps"] == 0:
            cost_gate = _gate("inconclusive", "Baseline cost per success is zero; percentage savings is undefined.")
        else:
            with localcontext() as cost_context:
                cost_context.prec = _exact_precision([left["total"], right["total"], minimum], len(task_ids)) * 2
                denominator = left["total"] * right["successes"]
                numerator = (denominator - right["total"] * left["successes"]) * 100
                passes_cost = numerator >= minimum * denominator
                cost_savings = numerator / denominator
            cost_gate = _gate("pass") if passes_cost else _gate("fail", "Cost-per-success savings is below the required threshold.")
        latency_change = None
        if left["p95"] is None or right["p95"] is None:
            latency_gate = _gate("inconclusive", "Every matched task in both runs needs duration_ms for latency evidence.")
        elif left["p95"] == 0:
            latency_gate = _gate("inconclusive", "Baseline p95 task duration is zero; percentage change is undefined.")
        else:
            latency_change = (Decimal(str(right["p95"])) / Decimal(str(left["p95"])) - 1) * 100
            latency_gate = _gate("pass") if latency_change <= maximum else _gate("fail", "Task-wall-clock p95 regression exceeds the allowed threshold.")
        gates = {
            "comparability": _gate("pass"),
            "quality": _gate("fail", "Previously successful tasks now fail: " + ", ".join(regressions)) if regressions else _gate("pass"),
            "cost": cost_gate, "latency": latency_gate,
        }
        status = "fail" if any(gate["status"] == "fail" for gate in gates.values()) else (
            "inconclusive" if any(gate["status"] == "inconclusive" for gate in gates.values()) else "pass")
        result.update(status=status, passed=status == "pass", gates=gates,
                      reasons=[reason for gate in gates.values() for reason in gate["reasons"]])
        with localcontext() as delta_context:
            delta_context.prec = _exact_precision([left["total"], right["total"]])
            total_delta = right["total"] - left["total"]
        result["summary"] = {
            "baseline_total_cost_usd": str(left["total"]), "candidate_total_cost_usd": str(right["total"]),
            "total_cost_delta_usd": str(total_delta),
            "baseline_successful_tasks": left["successes"], "candidate_successful_tasks": right["successes"],
            "baseline_failed_tasks": left["failures"], "candidate_failed_tasks": right["failures"],
            "baseline_cost_per_successful_task_usd": _money(left["cps"]),
            "candidate_cost_per_successful_task_usd": _money(right["cps"]),
            "cost_per_success_savings_pct": _money(cost_savings),
            "baseline_p95_task_duration_ms": left["p95"], "candidate_p95_task_duration_ms": right["p95"],
            "p95_task_duration_regression_pct": _money(latency_change),
        }
        return result
