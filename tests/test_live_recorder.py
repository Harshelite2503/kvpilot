import pytest

from kvpilot.live.recorder import LiveRecorder
from kvpilot.policy import Candidate, Decision


def _decision(action="group_int4", stored_bytes=100, encode_ms=1.0):
    candidate = Candidate(action, stored_bytes, encode_ms, 0.1, 0.2, 5.0, 5.0, True)
    return Decision(candidate, (candidate,), "test")


def test_pause_charges_actual_execution_not_decision_estimates():
    rec = LiveRecorder(policy="adaptive")
    rec.pause(at_ms=0, session_id="s1", decision=_decision(), host_reserved_bytes=60, raw_bytes=1000,
              measured_encode_ms=20.0, transferred_bytes=60)
    result = rec.finalize()
    assert result["metrics"]["measured_encoding_ms"] == 20.0
    assert result["metrics"]["measured_transferred_bytes"] == 60
    assert result["metrics"]["peak_host_reserved_bytes"] == 60
    event = result["decisions"][0]
    assert event["decision"]["selected"]["encode_ms"] == 1.0
    assert event["decision"]["selected"]["stored_bytes"] == 100
    assert event["measured_encoding_ms"] == 20.0
    assert event["measured_transferred_bytes"] == 60


def test_recompute_pause_charges_nothing():
    rec = LiveRecorder(policy="recompute")
    rec.pause(at_ms=0, session_id="s1", decision=_decision(action="recompute", stored_bytes=0, encode_ms=0.0),
              host_reserved_bytes=0, raw_bytes=1000, measured_encode_ms=0.0, transferred_bytes=0)
    result = rec.finalize()
    assert result["metrics"]["measured_encoding_ms"] == 0.0
    assert result["metrics"]["measured_transferred_bytes"] == 0


def test_resume_cache_hit_vs_recompute():
    rec = LiveRecorder(policy="adaptive")
    rec.resume(at_ms=1, session_id="s1", used_cache=True, measured_delay_ms=5.0,
               host_reserved_bytes=0, transferred_bytes=100, decode_ms=2.0, recompute_ms=0)
    rec.resume(at_ms=2, session_id="s2", used_cache=False, measured_delay_ms=50.0,
               host_reserved_bytes=0, transferred_bytes=0, decode_ms=0, recompute_ms=30)
    result = rec.finalize()
    assert result["metrics"]["resume_count"] == 2
    assert result["metrics"]["full_recomputes"] == 1
    assert result["metrics"]["measured_transferred_bytes"] == 100
    assert result["metrics"]["measured_decoding_ms"] == 2.0
    assert result["metrics"]["measured_recompute_ms"] == 30.0
    assert result["metrics"]["measured_p50_resume_ms"] == 27.5


def test_failed_restore_work_is_not_discarded_and_resume_updates_peak():
    rec = LiveRecorder(policy="adaptive")
    rec.pause(at_ms=0, session_id="s1", decision=_decision(), host_reserved_bytes=100,
              raw_bytes=1000, measured_encode_ms=3, transferred_bytes=90)
    rec.resume(at_ms=1, session_id="s1", used_cache=False, measured_delay_ms=50,
               host_reserved_bytes=200, transferred_bytes=90, decode_ms=2, recompute_ms=30)
    rec.resume(at_ms=2, session_id="s2", used_cache=True, measured_delay_ms=5,
               host_reserved_bytes=10, transferred_bytes=10, decode_ms=1, recompute_ms=0)
    rec.evict(2)
    rec.evict(0)
    rec.cancel(at_ms=3, session_id="s3")
    metrics = rec.finalize()["metrics"]
    assert metrics["measured_transferred_bytes"] == 190
    assert metrics["measured_decoding_ms"] == 3
    assert metrics["measured_recompute_ms"] == 30
    assert metrics["peak_host_reserved_bytes"] == 200
    assert metrics["full_recomputes"] == 1
    assert metrics["resume_count"] == 2
    assert metrics["cancel_count"] == 1
    assert metrics["evicted_blocks"] == 2


@pytest.mark.parametrize("method,field,value", [
    ("pause", "measured_encode_ms", -1),
    ("pause", "measured_encode_ms", float("nan")),
    ("pause", "transferred_bytes", 1.5),
    ("pause", "transferred_bytes", True),
    ("pause", "raw_bytes", -1),
    ("pause", "host_reserved_bytes", -1),
    ("resume", "measured_delay_ms", float("inf")),
    ("resume", "decode_ms", -1),
    ("resume", "recompute_ms", float("nan")),
    ("resume", "transferred_bytes", -1),
    ("resume", "host_reserved_bytes", 0.5),
    ("resume", "used_cache", 1),
    ("cancel", "at_ms", True),
    ("cancel", "session_id", ""),
    ("evict", "count", -1),
    ("evict", "count", 1.5),
    ("evict", "count", True),
])
def test_invalid_observation_is_rejected_without_changing_counters(method, field, value):
    rec = LiveRecorder(policy="adaptive")
    arguments = {
        "pause": dict(at_ms=0, session_id="s1", decision=_decision(), host_reserved_bytes=100,
                      raw_bytes=1000, measured_encode_ms=1, transferred_bytes=100),
        "resume": dict(at_ms=1, session_id="s1", used_cache=True, measured_delay_ms=5,
                       host_reserved_bytes=0, transferred_bytes=100, decode_ms=2, recompute_ms=0),
        "cancel": dict(at_ms=2, session_id="s1"),
        "evict": dict(count=1),
    }[method]
    arguments[field] = value
    before = rec.finalize()
    with pytest.raises(ValueError):
        getattr(rec, method)(**arguments)
    assert rec.finalize() == before


def test_pause_requires_actual_measurements_even_with_a_decision():
    rec = LiveRecorder(policy="adaptive")
    with pytest.raises(TypeError, match="measured_encode_ms.*transferred_bytes"):
        rec.pause(at_ms=0, session_id="s1", decision=_decision(), host_reserved_bytes=100, raw_bytes=1000)
    assert rec.finalize()["decisions"] == []


def test_finalized_log_is_a_snapshot_independent_of_future_observations():
    rec = LiveRecorder(policy="adaptive")
    rec.cancel(at_ms=0, session_id="s1")
    snapshot = rec.finalize()
    rec.cancel(at_ms=1, session_id="s2")
    assert len(snapshot["decisions"]) == 1
    snapshot["decisions"][0]["session_id"] = "edited"
    assert rec.finalize()["decisions"][0]["session_id"] == "s1"


def test_cancel_increments_count_without_a_delay():
    rec = LiveRecorder(policy="raw_fp16")
    rec.cancel(at_ms=0, session_id="s1")
    result = rec.finalize()
    assert result["metrics"]["cancel_count"] == 1
    assert result["metrics"]["resume_count"] == 0


def test_finalize_shape_matches_replay_evidence_labeling():
    rec = LiveRecorder(policy="adaptive")
    result = rec.finalize()
    assert result["evidence_type"] == "live_measured_session"
    assert result["quality_validated"] is False
    assert set(result) == {"schema_version", "evidence_type", "policy", "quality_validated", "metrics", "decisions"}
