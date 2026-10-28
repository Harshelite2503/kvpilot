"""Frontier attribution and capture-structure checks on tiny hand-built inputs."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "audit_attribution", Path(__file__).resolve().parents[1] / "scripts/audit_attribution.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.mark.parametrize("stored, expected", [(0, 100), (5, 75), (10, 50), (15, 30), (25, 10)])
def test_frontier_is_the_lower_envelope_of_baselines(stored, expected):
    assert audit.frontier_time([(0, 100), (10, 50), (20, 10)], stored) == pytest.approx(expected)


def test_a_baseline_above_the_hull_does_not_define_the_frontier():
    # The middle baseline is beaten by splitting sessions between its neighbours.
    assert audit.frontier_time([(0, 100), (10, 90), (20, 0)], 10) == pytest.approx(50)


def rows(policy, stored, times, scenario="pressure"):
    return [{"scenario": scenario, "policy": policy, "case_id": f"c{i}", "pause_ms": t, "resume_wall_ms": 0,
             "stored_bytes_after_pause": stored} for i, t in enumerate(times)]


def baselines():
    return rows("recompute", 0, [100, 100, 100, 100]) + rows("raw_native", 100, [10, 10, 10, 10])


def report_for(candidate_rows, resamples=200, **kwargs):
    kwargs.setdefault("min_units", 1)
    return audit.frontier_report(baselines() + candidate_rows, ["recompute", "raw_native"],
                                 ["adaptive_eviction"], resamples, 1, **kwargs)["pressure"]["adaptive_eviction"]


def test_interpolated_frontier_credit_does_not_override_a_measured_same_budget_loss():
    result = report_for(rows("adaptive_eviction", 50, [20, 20, 20, 20]))
    # Below the recompute/native split at 50 bytes, but that split was never run...
    assert result["frontier"]["frontier_ms"] == pytest.approx(55)
    assert result["frontier"]["delta"] == pytest.approx(20 / 55 - 1)
    assert result["frontier"]["point"] == "interpolated"
    assert result["frontier"]["status"] == "credited"
    # ...while fixed native, run under the same configured budget, was twice as fast.
    assert result["same_budget"]["baseline"] == "raw_native"
    assert result["same_budget"]["delta"] == pytest.approx(1.0)
    assert result["same_budget"]["status"] == "regression"
    assert result["status"] == "regression"
    assert report_for(rows("adaptive_eviction", 50, [20, 20, 20, 20]), primary="frontier")["status"] == "credited"


def test_candidate_faster_than_every_baseline_is_credited_on_both_comparisons():
    result = report_for(rows("adaptive_eviction", 100, [5, 5, 5, 5]))
    assert result["frontier"]["point"] == "measured"
    assert result["frontier"]["status"] == result["same_budget"]["status"] == result["status"] == "credited"


def test_candidate_above_the_frontier_is_a_regression():
    assert report_for(rows("adaptive_eviction", 100, [20, 20, 20, 20]))["status"] == "regression"


def test_noisy_small_difference_is_inconclusive():
    assert report_for(rows("adaptive_eviction", 100, [2, 18, 3, 17]), resamples=500)["status"] == "inconclusive"


def test_too_few_units_is_never_graded():
    result = report_for(rows("adaptive_eviction", 100, [5, 5, 5, 5]), min_units=30)
    assert result["same_budget"]["status"] == "credited"
    assert result["status"] == "insufficient_units"


def test_frontier_needs_a_baseline_at_or_below_the_candidate_bytes():
    with pytest.raises(ValueError, match="zero-byte arm"):
        audit.frontier_time([(10, 50), (20, 10)], 5)
    with pytest.raises(ValueError, match="primary"):
        report_for(rows("adaptive_eviction", 100, [5, 5, 5, 5]), primary="best")


def test_arms_must_share_every_resampling_unit():
    partial = rows("adaptive_eviction", 50, [20, 20, 20])
    with pytest.raises(ValueError, match="do not share"):
        audit.frontier_report(baselines() + partial, ["recompute", "raw_native"], ["adaptive_eviction"], 10, 1)


def test_missing_baseline_arm_is_an_error_not_a_skipped_comparison():
    with pytest.raises(ValueError, match="baseline arms missing"):
        audit.frontier_report(rows("recompute", 0, [1]) + rows("adaptive_eviction", 1, [1]),
                              ["recompute", "raw_native"], ["adaptive_eviction"], 10, 1)


def capture(period=3, valid=14, block_size=4, head_dim=4):
    blocks = 4
    total = blocks * block_size
    keys = np.arange(total * head_dim, dtype=np.float32).reshape(total, head_dim) + 1
    table = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
    values = table[np.arange(total) % period]
    tokens = np.concatenate([keys, values], axis=1)
    tokens[valid:] = 0
    return tokens.reshape(blocks, block_size, 1, 2 * head_dim).transpose(0, 2, 1, 3).copy()


def test_capture_analysis_recovers_value_repetition_and_unwritten_slots():
    report = audit.analyze_capture(capture())
    assert report["valid_token_rows"] == 14
    assert report["trailing_unwritten_slots"] == 2
    assert report["value_half"]["unique_rows"] == 3
    assert report["value_half"]["best_repeat_lag"] == 3
    assert report["value_half"]["best_repeat_rate"] == 1.0
    assert report["key_half"]["unique_rows"] == 14
    assert report["key_half"]["best_repeat_rate"] == 0.0
    assert report["bf16_origin"] is True
    assert report["bf16_to_fp16"]["changed"] == 0
    assert set(report["native_bf16_zlib_ratio"]) == {"key_half", "value_half", "value_half_deduplicated",
                                                     "whole_capture"}
    # The layout test looks at each axis-1 slice separately.
    assert [(s["channels"], s["unique_rows"], s["repeat_rate_at_value_lag"]) for s in report["axis1_slices"]] == [
        ("low", 14, 0.0), ("high", 3, 1.0)]
    assert report["value_periodic_run"] == {"first_row": 0, "last_row": 13, "rows_outside_run": 0}


def test_capture_requires_a_k_v_concatenated_last_axis():
    with pytest.raises(ValueError, match="2 \\* head_dim"):
        audit.analyze_capture(np.zeros((2, 1, 4, 6), dtype=np.float32), head_dim=4)
    with pytest.raises(ValueError, match="4-D"):
        audit.analyze_capture(np.zeros((2, 4, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="no written token rows"):
        audit.analyze_capture(np.zeros((2, 1, 4, 8), dtype=np.float32))
