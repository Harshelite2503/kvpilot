from copy import deepcopy
from decimal import Decimal
import json

import pytest

from kvpilot.context import ContextDataError, audit_costs, load_pricing, load_trace


def capture(tmp_path, *, success=True, evidence="synthetic", rates=None):
    header = {"schema_version": 1, "type": "run", "run_id": "run", "dataset_id": "data",
              "workload_version": "1", "evaluator_id": "eval", "agent_revision": "agent",
              "provider": "anthropic", "model": "fixture-model", "evidence_type": evidence,
              "cache_protocol": "fixture-protocol"}
    rows = [header,
            {"type": "request", "request_id": "r1", "task_id": "good", "session_id": "s1",
             "sequence": 0, "kind": "inference", "latency_ms": 100,
             "usage": {"input_tokens": 100, "cache_read_input_tokens": 1000,
                       "cache_creation_input_tokens": 300, "output_tokens": 10,
                       "cache_creation": {"ephemeral_5m_input_tokens": 200, "ephemeral_1h_input_tokens": 100}}},
            {"type": "request", "request_id": "r2", "task_id": "good", "session_id": "s1",
             "sequence": 1, "kind": "compaction", "prefix_change_reason": "intentional_compaction",
             "usage": {"input_tokens": 50, "output_tokens": 5}},
            {"type": "task_end", "task_id": "good", "success": success, "duration_ms": 1234.5},
            {"type": "request", "request_id": "r3", "task_id": "failed", "session_id": "s2",
             "sequence": 0, "kind": "inference", "usage": {"input_tokens": 20, "output_tokens": 1}},
            {"type": "task_end", "task_id": "failed", "success": False},
            {"type": "run_end", "complete": True, "request_count": 3, "task_count": 2}]
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    card = {"schema_version": 1, "pricing_id": "illustrative", "currency": "USD", "source": "synthetic rates",
            "effective_date": "2026-10-29", "provider": "anthropic", "model": "fixture-model",
            "rates_usd_per_million": rates or {"input": "3", "cache_read": "0.3", "cache_write_5m": "3.75",
                                             "cache_write_1h": "6", "output": "15"}}
    price_path = tmp_path / "prices.json"
    price_path.write_text(json.dumps(card))
    return load_trace(trace_path), load_pricing(price_path)


def test_no_double_counting_includes_failed_and_compaction_work(tmp_path):
    trace, pricing = capture(tmp_path)
    report = audit_costs(trace, pricing)
    assert report["totals"]["tokens"] == {"input": 170, "cache_read": 1000,
        "cache_write_5m": 200, "cache_write_1h": 100, "output": 16}
    assert report["requests"][0]["cost_breakdown_usd"] == {"input": "0.0003", "cache_read": "0.0003",
        "cache_write_5m": "0.00075", "cache_write_1h": "0.0006", "output": "0.00015"}
    assert report["requests"][0]["total_cost_usd"] == "0.0021"
    assert report["totals"]["total_cost_usd"] == "0.0024"
    assert report["totals"]["compaction_request_cost_usd"] == "0.000225"
    assert report["totals"]["compaction_request_count"] == 1
    assert report["totals"]["cost_per_successful_task_usd"] == "0.0024"
    assert report["tasks"] == [
        {"task_id": "good", "success": True, "duration_ms": 1234.5, "total_cost_usd": "0.002325"},
        {"task_id": "failed", "success": False, "duration_ms": None, "total_cost_usd": "0.000075"}]
    assert report["totals"]["successful_tasks"] == report["totals"]["failed_tasks"] == 1


def test_zero_successes_is_undefined_not_free(tmp_path):
    trace, pricing = capture(tmp_path, success=False)
    totals = audit_costs(trace, pricing)["totals"]
    assert totals["cost_per_successful_task_usd"] is None
    assert totals["total_cost_usd"] == "0.0024" and totals["failed_tasks"] == 2


def test_observed_is_accounting_not_invoice_or_savings_evidence(tmp_path):
    trace, pricing = capture(tmp_path, evidence="observed")
    report = audit_costs(trace, pricing)
    assert report["evidence_type"] == "observed_usage_accounting"
    assert report["trace_sha256"] == trace.source_sha256
    assert report["pricing"]["source_sha256"] == pricing.source_sha256
    assert report["pricing"]["rates_usd_per_million"]["input"] == "3"
    assert any("invoice" in item for item in report["scope_exclusions"])
    assert any("causal savings" in item for item in report["scope_exclusions"])
    assert any("long-context thresholds" in item and "mixed pricing tiers" in item
               for item in report["scope_exclusions"])
    assert any("response.model" in item and "actual returned model" in item
               for item in report["scope_exclusions"])
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_synthetic_label_and_no_input_mutation(tmp_path):
    trace, pricing = capture(tmp_path)
    before = deepcopy(trace)
    report = audit_costs(trace, pricing)
    assert report["evidence_type"] == "synthetic_example" and trace == before
    report["metadata"]["model"] = "changed"
    report["requests"][0]["usage"]["input_tokens"] = 0
    assert trace == before


@pytest.mark.parametrize("key", ["provider", "model"])
def test_no_fuzzy_provider_or_model_matching(tmp_path, key):
    trace, pricing = capture(tmp_path)
    pricing.metadata[key] += "-other"
    with pytest.raises(ContextDataError, match="mismatch"):
        audit_costs(trace, pricing)


def test_decimal_rates_remain_exact_beyond_default_precision(tmp_path):
    tiny = "0.000000000000000000000000000001"
    rates = {"input": "3.000000000000000000000000000001", "cache_read": tiny,
             "cache_write_5m": "0", "cache_write_1h": "0", "output": "0"}
    trace, pricing = capture(tmp_path, rates=rates)
    report = audit_costs(trace, pricing)
    assert report["totals"]["total_cost_usd"] == "0.00051000000000000000000000000000117"
    assert report["requests"][0]["cost_breakdown_usd"]["cache_read"] == "0.000000000000000000000000000000001"


def test_zero_rate_card_and_tasks_without_requests(tmp_path):
    trace, pricing = capture(tmp_path, rates=dict.fromkeys(
        ["input", "cache_read", "cache_write_5m", "cache_write_1h", "output"], "0"))
    assert audit_costs(trace, pricing)["totals"]["total_cost_usd"] == "0"
    path = tmp_path / "empty.jsonl"
    path.write_text(json.dumps(trace.metadata) + '\n' + json.dumps(
        {"type": "task_end", "task_id": "never_called_model", "success": False}) + '\n' + json.dumps(
        {"type": "run_end", "complete": True, "request_count": 0, "task_count": 1}) + '\n')
    report = audit_costs(load_trace(path), pricing)
    assert report["totals"]["request_count"] == 0 and report["totals"]["failed_tasks"] == 1
    assert report["totals"]["cost_per_successful_task_usd"] is None
    assert report["tasks"][0]["total_cost_usd"] == "0"


def test_thinking_detail_is_not_double_billed(tmp_path):
    trace, pricing = capture(tmp_path)
    trace.requests[0]["usage"]["output_tokens_details"] = {"thinking_tokens": 7}
    assert audit_costs(trace, pricing)["totals"]["total_cost_usd"] == "0.0024"


def test_success_denominator_uses_all_spending_with_declared_decimal_rounding(tmp_path):
    trace, pricing = capture(tmp_path)
    trace.tasks.extend([{"type": "task_end", "task_id": "free2", "success": True},
                        {"type": "task_end", "task_id": "free3", "success": True},
                        {"type": "task_end", "task_id": "free4", "success": True},
                        {"type": "task_end", "task_id": "free5", "success": True},
                        {"type": "task_end", "task_id": "free6", "success": True},
                        {"type": "task_end", "task_id": "free7", "success": True}])
    report = audit_costs(trace, pricing)
    assert Decimal(report["totals"]["cost_per_successful_task_usd"]) == Decimal("0.0003428571428571428571428571429")
    assert "ROUND_HALF_EVEN" in report["cost_per_successful_task_rounding"]
