from kvpilot.live.recorder import LiveRecorder
from kvpilot.policy import Candidate, Decision


def _decision(action="group_int4", stored_bytes=100, encode_ms=1.0):
    candidate = Candidate(action, stored_bytes, encode_ms, 0.1, 0.2, 5.0, 5.0, True)
    return Decision(candidate, (candidate,), "test")


def test_pause_charges_real_encode_and_transfer():
    rec = LiveRecorder(policy="adaptive")
    rec.pause(at_ms=0, session_id="s1", decision=_decision(), host_reserved_bytes=100, raw_bytes=1000)
    result = rec.finalize()
    assert result["metrics"]["measured_encoding_ms"] == 1.0
    assert result["metrics"]["measured_transferred_bytes"] == 100
    assert result["metrics"]["peak_host_reserved_bytes"] == 100


def test_recompute_pause_charges_nothing():
    rec = LiveRecorder(policy="recompute")
    rec.pause(at_ms=0, session_id="s1", decision=_decision(action="recompute", stored_bytes=0, encode_ms=0.0),
              host_reserved_bytes=0, raw_bytes=1000)
    result = rec.finalize()
    assert result["metrics"]["measured_encoding_ms"] == 0.0
    assert result["metrics"]["measured_transferred_bytes"] == 0


def test_resume_cache_hit_vs_recompute():
    rec = LiveRecorder(policy="adaptive")
    rec.resume(at_ms=1, session_id="s1", used_cache=True, measured_delay_ms=5.0,
               host_reserved_bytes=0, transferred_bytes=100, decode_ms=2.0)
    rec.resume(at_ms=2, session_id="s2", used_cache=False, measured_delay_ms=50.0, host_reserved_bytes=0)
    result = rec.finalize()
    assert result["metrics"]["resume_count"] == 2
    assert result["metrics"]["full_recomputes"] == 1
    assert result["metrics"]["measured_transferred_bytes"] == 100
    assert result["metrics"]["measured_decoding_ms"] == 2.0
    assert result["metrics"]["measured_recompute_ms"] == 50.0
    assert result["metrics"]["measured_p50_resume_ms"] == 27.5


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
