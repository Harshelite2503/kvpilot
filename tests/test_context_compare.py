from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal

import pytest

from kvpilot.context.compare import compare_reports


def audit(run_id="baseline", costs=("2", "3", "5"), successes=(True, True, False), durations=(100, 200, 300)):
    tasks = [{"task_id": f"case-{i}", "success": success, "duration_ms": duration, "total_cost_usd": cost}
             for i, (cost, success, duration) in enumerate(zip(costs, successes, durations, strict=True))]
    total = sum(map(Decimal, costs))
    count = sum(successes)
    return {
        "schema_version": 1, "complete": True, "evidence_type": "observed_usage_accounting",
        "trace_sha256": hashlib.sha256(run_id.encode()).hexdigest(),
        "metadata": {
            "run_id": run_id, "dataset_id": "customer-holdout", "workload_version": "v1",
            "evaluator_id": "unit-tests-v1", "agent_revision": "agent-a", "provider": "test",
            "model": "fixed-model", "cache_protocol": "test-protocol-v1", "evidence_type": "observed",
        },
        "pricing": {
            "schema_version": 1, "pricing_id": "frozen-prices", "currency": "USD",
            "source": "https://example.invalid/customer-contract", "effective_date": "2026-10-29",
            "provider": "test", "model": "fixed-model", "rates_usd_per_million": {"input": "2"},
            "source_sha256": "a" * 64,
        },
        "tasks": tasks,
        "totals": {
            "total_cost_usd": str(total), "successful_tasks": count, "failed_tasks": len(tasks) - count,
            "cost_per_successful_task_usd": str(total / count) if count else None,
        },
    }


def test_pass_counts_failed_task_work_and_uses_task_wall_clock():
    baseline = audit()
    candidate = audit("candidate", costs=("1", "1", "3"), durations=(120, 210, 310))
    candidate["metadata"]["agent_revision"] = "agent-b"
    # A request trace is not a task stopwatch and must not drive p95.
    baseline["requests"] = [{"duration_ms": 10000000}]
    candidate["requests"] = [{"duration_ms": 1}]
    result = compare_reports(baseline, candidate, min_savings_pct=40)
    assert result["passed"] is True
    assert result["status"] == "pass"
    assert result["summary"]["candidate_cost_per_successful_task_usd"] == "2.5"
    assert Decimal(result["summary"]["cost_per_success_savings_pct"]) == 50
    assert result["summary"]["candidate_p95_task_duration_ms"] == 310
    assert result["paired_outcomes"]["failed_both"] == 1
    assert result["certifies_customer_savings"] is False
    json.dumps(result, allow_nan=False)


def test_cost_regression_fails():
    result = compare_reports(audit(), audit("candidate", costs=("2", "3", "6")))
    assert result["status"] == "fail"
    assert result["gates"]["cost"]["status"] == "fail"
    assert Decimal(result["summary"]["cost_per_success_savings_pct"]) == -10


def test_swapped_successes_are_quality_regression_even_with_same_aggregate():
    candidate = audit("candidate", costs=("1", "1", "1"), successes=(True, False, True))
    result = compare_reports(audit(), candidate)
    assert result["status"] == "fail"
    assert result["paired_outcomes"]["regressions"] == ["case-1"]
    assert result["paired_outcomes"]["improvements"] == ["case-2"]
    assert result["gates"]["quality"]["status"] == "fail"


def test_no_reordering_dependency_and_all_task_durations_feed_nearest_rank_p95():
    baseline = audit(costs=("1",) * 20, successes=(True,) * 20, durations=tuple(range(1, 21)))
    candidate = copy.deepcopy(baseline)
    candidate["metadata"]["run_id"] = "candidate"
    candidate["tasks"].reverse()
    result = compare_reports(baseline, candidate)
    assert result["passed"]
    assert result["summary"]["baseline_p95_task_duration_ms"] == 19


@pytest.mark.parametrize("where", ["baseline", "candidate"])
def test_missing_any_task_duration_prevents_pass(where):
    baseline, candidate = audit(), audit("candidate")
    target = baseline if where == "baseline" else candidate
    target["tasks"][2].pop("duration_ms")  # A failed task is still part of elapsed work.
    result = compare_reports(baseline, candidate)
    assert result["status"] == "inconclusive"
    assert result["passed"] is False
    assert result["gates"]["latency"]["status"] == "inconclusive"


def test_latency_regression_fails_and_threshold_equality_passes():
    assert compare_reports(audit(), audit("candidate", durations=(100, 200, 315)))["passed"]
    result = compare_reports(audit(), audit("candidate", durations=(100, 200, 316)))
    assert result["gates"]["latency"]["status"] == "fail"


@pytest.mark.parametrize("field", ["dataset_id", "workload_version", "evaluator_id", "cache_protocol", "run_id"])
def test_incompatible_metadata_has_no_computed_comparison(field):
    baseline, candidate = audit(), audit("candidate")
    candidate["metadata"][field] = baseline["metadata"][field] if field == "run_id" else "other"
    result = compare_reports(baseline, candidate)
    assert result["status"] == "inconclusive"
    assert result["comparable"] is False
    assert result["summary"] is None
    assert result["paired_outcomes"] is None


@pytest.mark.parametrize("field", ["provider", "model"])
def test_different_provider_or_model_rejected_even_with_matching_pricecards(field):
    baseline, candidate = audit(), audit("candidate")
    candidate["metadata"][field] = "other"
    candidate["pricing"][field] = "other"
    assert compare_reports(baseline, candidate)["comparable"] is False


def test_mixed_evidence_modes_are_not_compared():
    candidate = audit("candidate")
    candidate["metadata"]["evidence_type"] = "synthetic"
    candidate["evidence_type"] = "synthetic_example"
    result = compare_reports(audit(), candidate)
    assert result["status"] == "inconclusive"
    assert result["summary"] is None


def test_synthetic_pass_is_explicitly_not_customer_evidence():
    baseline, candidate = audit(), audit("candidate")
    for report in (baseline, candidate):
        report["metadata"]["evidence_type"] = "synthetic"
        report["evidence_type"] = "synthetic_example"
    result = compare_reports(baseline, candidate)
    assert result["passed"]
    assert result["evidence_type"] == "synthetic_example"
    assert not result["certifies_customer_savings"]
    assert any("Synthetic" in item for item in result["limitations"])


@pytest.mark.parametrize("field,value", [("source_sha256", "b" * 64), ("pricing_id", "different"),
                                        ("rates_usd_per_million", {"input": "3"})])
def test_price_card_or_provenance_changes_are_incompatible(field, value):
    candidate = audit("candidate")
    candidate["pricing"][field] = value
    assert compare_reports(audit(), candidate)["comparable"] is False


def test_unmatched_task_ids_are_not_compared():
    candidate = audit("candidate")
    candidate["tasks"][0]["task_id"] = "unseen-case"
    result = compare_reports(audit(), candidate)
    assert not result["comparable"]
    assert result["summary"] is None


def test_missing_task_records_with_reconciled_totals_are_incompatible():
    candidate = audit("candidate", costs=("2", "3"), successes=(True, True), durations=(100, 200))
    assert compare_reports(audit(), candidate)["comparable"] is False


def test_duplicate_task_ids_are_malformed():
    candidate = audit("candidate")
    candidate["tasks"][1]["task_id"] = "case-0"
    with pytest.raises(ValueError, match="duplicate"):
        compare_reports(audit(), candidate)


@pytest.mark.parametrize("costs,successes,durations,gate", [
    (("0", "0", "0"), (True, True, False), (100, 200, 300), "cost"),
    (("2", "3", "5"), (False, False, False), (100, 200, 300), "cost"),
    (("2", "3", "5"), (True, True, False), (0, 0, 0), "latency"),
])
def test_zero_denominators_are_inconclusive(costs, successes, durations, gate):
    baseline = audit(costs=costs, successes=successes, durations=durations)
    candidate = audit("candidate", costs=costs, successes=successes, durations=durations)
    result = compare_reports(baseline, candidate)
    assert result["status"] == "inconclusive"
    assert result["gates"][gate]["status"] == "inconclusive"
    json.dumps(result, allow_nan=False)


def test_candidate_zero_cost_and_duration_are_valid_improvements():
    candidate = audit("candidate", costs=("0", "0", "0"), durations=(0, 0, 0))
    result = compare_reports(audit(), candidate, min_savings_pct=100)
    assert result["passed"]
    assert Decimal(result["summary"]["cost_per_success_savings_pct"]) == 100


@pytest.mark.parametrize("field,value", [
    ("min_savings_pct", -1), ("min_savings_pct", 101), ("min_savings_pct", float("nan")),
    ("min_savings_pct", True), ("max_p95_regression_pct", -1),
    ("max_p95_regression_pct", float("inf")), ("max_p95_regression_pct", "NaN"),
])
def test_bad_gate_thresholds_rejected(field, value):
    with pytest.raises(ValueError):
        compare_reports(audit(), audit("candidate"), **{field: value})


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), "200", 10**400])
def test_invalid_durations_rejected(value):
    candidate = audit("candidate")
    candidate["tasks"][0]["duration_ms"] = value
    with pytest.raises(ValueError, match="duration_ms"):
        compare_reports(audit(), candidate)


@pytest.mark.parametrize("value", [True, -1, "NaN", "Infinity", "-0.01", "bad"])
def test_invalid_money_rejected(value):
    candidate = audit("candidate")
    candidate["tasks"][0]["total_cost_usd"] = value
    with pytest.raises(ValueError):
        compare_reports(audit(), candidate)


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_success_labels_are_strict_booleans(value):
    candidate = audit("candidate")
    candidate["tasks"][0]["success"] = value
    with pytest.raises(ValueError, match="success must be a boolean"):
        compare_reports(audit(), candidate)


@pytest.mark.parametrize("field,value", [("total_cost_usd", "1"), ("successful_tasks", 3),
                                       ("failed_tasks", True), ("cost_per_successful_task_usd", "1")])
def test_inconsistent_totals_rejected(field, value):
    candidate = audit("candidate")
    candidate["totals"][field] = value
    with pytest.raises(ValueError):
        compare_reports(audit(), candidate)


def test_rounded_cost_per_success_is_accepted_but_not_used_for_arithmetic():
    baseline = audit(successes=(True, True, True))
    candidate = audit("candidate", costs=("1", "2", "2"), successes=(True, True, True))
    result = compare_reports(baseline, candidate)
    assert result["passed"]
    assert abs(Decimal(result["summary"]["cost_per_success_savings_pct"]) - 50) < Decimal("1e-45")


def test_incomplete_audit_and_inconsistent_evidence_type_rejected():
    candidate = audit("candidate")
    candidate["complete"] = False
    with pytest.raises(ValueError, match="complete"):
        compare_reports(audit(), candidate)
    candidate["complete"] = True
    candidate["evidence_type"] = "synthetic_example"
    with pytest.raises(ValueError, match="evidence_type"):
        compare_reports(audit(), candidate)


def test_comparison_keeps_detached_source_and_price_snapshots():
    baseline, candidate = audit(), audit("candidate")
    result = compare_reports(baseline, candidate)
    assert result["baseline_trace_sha256"] == baseline["trace_sha256"]
    assert result["candidate_trace_sha256"] == candidate["trace_sha256"]
    assert result["baseline_metadata"] == baseline["metadata"]
    assert result["candidate_metadata"] == candidate["metadata"]
    assert result["pricing"] == baseline["pricing"]
    baseline["metadata"]["agent_revision"] = "later-change"
    candidate["metadata"]["agent_revision"] = "another-change"
    baseline["pricing"]["rates_usd_per_million"]["input"] = "100"
    assert result["baseline_metadata"]["agent_revision"] == "agent-a"
    assert result["candidate_metadata"]["agent_revision"] == "agent-a"
    assert result["pricing"]["rates_usd_per_million"]["input"] == "2"


@pytest.mark.parametrize("value", [None, "", "x" * 64, "a" * 63, "a" * 65, 12])
def test_supplied_source_hash_must_be_64_hex_characters(value):
    candidate = audit("candidate")
    candidate["trace_sha256"] = value
    with pytest.raises(ValueError, match="trace_sha256"):
        compare_reports(audit(), candidate)


def test_missing_optional_trace_hash_remains_explicit():
    candidate = audit("candidate")
    del candidate["trace_sha256"]
    result = compare_reports(audit(), candidate)
    assert result["candidate_trace_sha256"] is None
    assert any("lacks a trace hash" in item for item in result["limitations"])


@pytest.mark.parametrize("location", ["metadata", "pricing"])
def test_snapshot_cannot_introduce_non_json_values(location):
    candidate = audit("candidate")
    candidate[location]["unexpected"] = float("nan")
    with pytest.raises(ValueError, match="JSON-safe"):
        compare_reports(audit(), candidate)


def test_small_cost_regression_does_not_vanish_in_rounded_cps():
    baseline = audit(costs=("1", "0", "0"))
    candidate = audit("candidate", costs=("1", "0", "0"))
    candidate["tasks"][1]["total_cost_usd"] = "0." + "0" * 59 + "1"
    candidate["totals"]["total_cost_usd"] = "1." + "0" * 59 + "1"
    # At the billing report's 28-digit precision both display as 0.5.
    candidate["totals"]["cost_per_successful_task_usd"] = "0.5000000000000000000000000000"
    result = compare_reports(baseline, candidate)
    assert result["gates"]["cost"]["status"] == "fail"
    assert Decimal(result["summary"]["cost_per_success_savings_pct"]) < 0


def test_actual_loaded_fixture_reports_match_comparison_contract():
    from pathlib import Path
    from kvpilot.context import audit_costs, load_pricing, load_trace

    folder = Path(__file__).resolve().parents[1] / "examples" / "context-cost"
    pricing = load_pricing(folder / "pricing.json")
    baseline = audit_costs(load_trace(folder / "baseline.jsonl"), pricing)
    candidate = audit_costs(load_trace(folder / "candidate.jsonl"), pricing)
    result = compare_reports(baseline, candidate)
    assert result["passed"]
    assert result["pricing"] == baseline["pricing"]
    assert result["baseline_trace_sha256"] == baseline["trace_sha256"]
    json.dumps(result, allow_nan=False)


def test_total_delta_preserves_exact_billing_precision():
    baseline = audit(costs=("1", "0", "0"))
    candidate = audit("candidate", costs=("2", "0", "0"))
    candidate["tasks"][1]["total_cost_usd"] = "0." + "0" * 59 + "1"
    candidate["totals"]["total_cost_usd"] = "2." + "0" * 59 + "1"
    candidate["totals"]["cost_per_successful_task_usd"] = "1"
    result = compare_reports(baseline, candidate)
    assert Decimal(result["summary"]["total_cost_delta_usd"]) == Decimal("1." + "0" * 59 + "1")
