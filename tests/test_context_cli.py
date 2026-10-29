"""End-to-end checks: real CLI, supplied costs, gates, and readable artifacts."""
import json
from decimal import Decimal, localcontext
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kvpilot.context.analysis import add_findings
from kvpilot.context.billing import audit_costs
from kvpilot.context.plan import estimate_compaction
from kvpilot.context.report import render_html, render_markdown
from kvpilot.context.schema import load_pricing, load_trace


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/context-cost"


def cli(*arguments):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run([sys.executable, "-m", "kvpilot.cli", "context", *map(str, arguments)],
                          capture_output=True, text=True, env=env, cwd=ROOT)


def compare_arguments(output, candidate=None):
    return ["compare", "--baseline", EXAMPLE / "baseline.jsonl", "--candidate", candidate or EXAMPLE / "candidate.jsonl",
            "--pricing", EXAMPLE / "pricing.json", "--min-savings-pct", "10", "--output", output,
            "--fail-on-regression"]


def changed_trace(tmp_path, change):
    events = [json.loads(line) for line in (EXAMPLE / "candidate.jsonl").read_text().splitlines()]
    change(events)
    target = tmp_path / "changed.jsonl"
    target.write_text("".join(json.dumps(event) + "\n" for event in events))
    return target


def test_real_audit_outputs_all_three_formats_and_exact_fixture_cost(tmp_path):
    result = cli("audit", "--trace", EXAMPLE / "baseline.jsonl", "--pricing", EXAMPLE / "pricing.json", "--output", tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert float(report["totals"]["total_cost_usd"]) == 0.7185
    assert report["totals"]["successful_tasks"] == 2
    assert report["totals"]["failed_tasks"] == 1
    assert report["evidence_type"] == "synthetic_example"
    assert report["coverage"]["finding_counts"]["declared_prefix_changed"] == 1
    assert "not a customer saving" in (tmp_path / "report.md").read_text()
    assert "Task outcomes" in (tmp_path / "report.html").read_text()


def test_real_compare_includes_compaction_and_failed_work(tmp_path):
    result = cli(*compare_arguments(tmp_path))
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["passed"] is True
    assert report["certifies_customer_savings"] is False
    assert float(report["summary"]["candidate_total_cost_usd"]) == 0.57
    assert report["paired_outcomes"]["failed_both"] == 1
    assert any("long-context" in item for item in report["limitations"])
    assert "Regression gates" in (tmp_path / "report.html").read_text()
    assert "20.67%" in (tmp_path / "report.md").read_text()


def test_ci_rejects_lower_quality_even_if_candidate_is_cheaper(tmp_path):
    candidate = changed_trace(tmp_path, lambda events: next(e for e in events if e["type"] == "task_end" and e["task_id"] == "fix-tests").update(success=False))
    result = cli(*compare_arguments(tmp_path / "out", candidate))
    assert result.returncode == 1, result.stderr
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["gates"]["quality"]["status"] == "fail"
    assert report["paired_outcomes"]["regressions"] == ["fix-tests"]


def test_ci_rejects_incomplete_timing_instead_of_treating_it_as_zero(tmp_path):
    candidate = changed_trace(tmp_path, lambda events: next(e for e in events if e["type"] == "task_end").pop("duration_ms"))
    result = cli(*compare_arguments(tmp_path / "out", candidate))
    assert result.returncode == 1, result.stderr
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["status"] == "inconclusive"
    assert report["summary"]["candidate_p95_task_duration_ms"] is None


def test_truncated_export_is_an_error_not_a_smaller_bill(tmp_path):
    candidate = changed_trace(tmp_path, lambda events: events.pop())
    result = cli(*compare_arguments(tmp_path / "out", candidate))
    assert result.returncode == 2
    assert not (tmp_path / "out/report.json").exists()


def test_plan_cli_keeps_hypothesis_separate_from_recorded_usage(tmp_path):
    result = cli("plan", "--current-tokens", "15000", "--retained-tokens", "1000", "--future-turns", "12",
                 "--cache-read-usd-per-million", "1", "--rewrite-cost-usd", "0.12", "--summary-cost-usd", "0.04",
                 "--output", tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["evidence_type"] == "scenario_estimate"
    assert report["first_strictly_cheaper_turn"] == 12
    assert float(report["estimated_net_savings_usd"]) == 0.008
    assert report["certifies_customer_savings"] is False


def test_unknown_price_model_does_not_silently_use_a_fallback(tmp_path):
    candidate = changed_trace(tmp_path, lambda events: events[0].update(model="other-model"))
    result = cli(*compare_arguments(tmp_path / "out", candidate))
    assert result.returncode == 2


def test_reports_escape_user_controlled_labels_and_do_not_load_remote_assets():
    report = add_findings(audit_costs(load_trace(EXAMPLE / "baseline.jsonl"), load_pricing(EXAMPLE / "pricing.json")))
    report["metadata"]["run_id"] = "<img src=x onerror=alert(1)>"
    report["tasks"][0]["task_id"] = "<script>alert(1)</script>"
    for text in (render_html(report), render_markdown(report)):
        assert "<img src=x" not in text
        assert "<script>" not in text
        assert "&lt;img" in text
    assert "Content-Security-Policy" in render_html(report)


def test_fingerprint_gap_is_not_evidence_of_an_adjacent_edit():
    report = audit_costs(load_trace(EXAMPLE / "baseline.jsonl"), load_pricing(EXAMPLE / "pricing.json"))
    report["requests"][1].pop("prefix_fingerprint")
    add_findings(report)
    assert not any(f["code"] == "declared_prefix_changed" for f in report["findings"])


@pytest.mark.parametrize("field,value", [
    ("current_tokens", True), ("retained_tokens", -1), ("future_turns", 1.5),
    ("cache_read_usd_per_million", "NaN"), ("rewrite_cost_usd", "Infinity"),
    ("summary_cost_usd", "-1"), ("retained_tokens", 20000),
])
def test_plan_rejects_invalid_assumptions(field, value):
    args = dict(current_tokens=15000, retained_tokens=1000, future_turns=12,
                cache_read_usd_per_million="1", rewrite_cost_usd="0.12", summary_cost_usd="0.04")
    args[field] = value
    with pytest.raises(ValueError):
        estimate_compaction(**args)


def test_plan_requires_strict_saving_and_handles_zero_rate():
    args = dict(current_tokens=1000, retained_tokens=0, future_turns=10,
                cache_read_usd_per_million="1", rewrite_cost_usd="0.01", summary_cost_usd="0")
    result = estimate_compaction(**args)
    assert result["first_strictly_cheaper_turn"] == 11
    assert result["cheaper_at_supplied_horizon"] is False
    assert estimate_compaction(**{**args, "cache_read_usd_per_million": "0"})["first_strictly_cheaper_turn"] is None


def test_strict_break_even_preserves_tiny_savings_independent_of_decimal_context():
    with localcontext() as context:
        context.prec = 10
        result = estimate_compaction(current_tokens=1000, retained_tokens=0, future_turns=10,
                                     cache_read_usd_per_million="1",
                                     rewrite_cost_usd="0.0099999999999999999999999999999", summary_cost_usd="0")
    assert result["first_strictly_cheaper_turn"] == 10
    assert result["cheaper_at_supplied_horizon"] is True
    assert Decimal(result["estimated_net_savings_usd"]) == Decimal("1e-31")
