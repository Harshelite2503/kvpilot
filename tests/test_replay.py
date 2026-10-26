from copy import deepcopy

import pytest

from kvpilot.policy import CodecProfile, CostModel
from kvpilot.replay import replay


RAW = CodecProfile("raw_fp16", 1, 0, 0)


def pause(sid, at=0, tenant="a", blocks=None, recompute_ms=100):
    return {"at_ms": at, "type": "pause", "session_id": sid, "pause_kind": "tool",
            "namespace": {"tenant": tenant, "model_revision": "model@sha", "adapter_revision": "none", "layout": "fp16-v1"},
            "blocks": blocks or [{"prefix_hash": "prefix-1", "raw_bytes": 100}], "recompute_ms": recompute_ms}


def end(sid, at, kind="resume"):
    return {"at_ms": at, "type": kind, "session_id": sid}


def run(events, **kwargs):
    return replay(events, CostModel(100, kwargs.pop("budget", 1000)), [RAW], policy=kwargs.pop("policy", "raw_fp16"), **kwargs)


def test_shared_prefix_is_stored_and_transferred_out_once():
    r = run([pause("a"), pause("b", 1), end("a", 2), end("b", 3)])
    assert r["metrics"]["peak_host_reserved_bytes"] == 100
    assert r["metrics"]["shared_block_hits"] == 1
    assert r["metrics"]["modeled_transferred_bytes"] == 300
    assert r["metrics"]["full_recomputes"] == 0


@pytest.mark.parametrize("field", ["tenant", "model_revision", "adapter_revision", "layout"])
def test_incompatible_namespace_never_shares(field):
    second = pause("b", 1)
    second["namespace"][field] = "different"
    r = run([pause("a"), second, end("a", 2), end("b", 3)])
    assert r["metrics"]["shared_block_hits"] == 0
    assert r["metrics"]["peak_host_reserved_bytes"] == 200


def test_cancel_one_branch_does_not_delete_siblings_prefix():
    r = run([pause("a"), pause("b", 1), end("a", 2, "cancel"), end("b", 3)])
    assert r["metrics"]["full_recomputes"] == 0
    assert r["metrics"]["cancel_count"] == 1


def test_lru_eviction_stays_within_budget_and_miss_recomputes():
    b = pause("b", 1, blocks=[{"prefix_hash": "other", "raw_bytes": 100}])
    r = run([pause("a"), b, end("a", 2), end("b", 3)], budget=100)
    assert r["metrics"]["peak_host_reserved_bytes"] <= 100
    assert r["metrics"]["evicted_blocks"] == 1
    assert r["metrics"]["full_recomputes"] == 1


def test_new_branch_does_not_evict_its_own_shared_prefix():
    b = pause("b", 1, blocks=[{"prefix_hash": "prefix-1", "raw_bytes": 100}, {"prefix_hash": "new", "raw_bytes": 100}])
    r = run([pause("a"), b, end("a", 2), end("b", 3)], budget=100)
    resumes = [e for e in r["decisions"] if e["type"] == "resume"]
    assert resumes[0]["path"] == "restore"
    assert resumes[1]["path"] == "full_recompute"
    assert r["metrics"]["peak_host_reserved_bytes"] == 100


def test_same_identity_different_size_is_rejected():
    b = pause("b", 1, blocks=[{"prefix_hash": "prefix-1", "raw_bytes": 99}])
    with pytest.raises(ValueError, match="inconsistent"):
        run([pause("a"), b])


def test_future_outcomes_cannot_change_a_past_decision():
    prefix = [pause("a"), end("a", 2), pause("b", 3)]
    r1 = run(prefix + [end("b", 4)], policy="adaptive")
    r2 = run(prefix + [end("b", 1000, "cancel")], policy="adaptive")
    assert r1["decisions"][:3] == r2["decisions"][:3]


def test_raw_source_not_ready_fast_resume_can_recompute():
    r = replay([pause("a", recompute_ms=0.1), end("a", 0.01)], CostModel(100, 1000), [RAW], policy="raw_fp16")
    assert r["decisions"][-1]["path"] == "full_recompute"


def test_unresolved_does_not_become_a_cancel():
    r = run([pause("a")])
    assert r["metrics"]["unresolved_count"] == 1
    assert r["metrics"]["cancel_count"] == 0
    assert r["metrics"]["modeled_p95_resume_ms"] is None


def test_no_gpu_or_quality_measurements_are_claimed():
    r = run([pause("a"), end("a", 10)])
    assert r["evidence_type"] == "analytical_trace_replay"
    assert r["quality_validated"] is False
    assert r["metrics"]["measured_gpu_itl_ms"] is None
    assert r["metrics"]["measured_llm_task_success"] is None


@pytest.mark.parametrize("events", [[end("a", 0)], [pause("a", 10), end("a", 9)], [pause("a"), pause("a", 1)]])
def test_malformed_lifecycle_rejected(events):
    with pytest.raises(ValueError):
        run(events)
