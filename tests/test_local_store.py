"""Real byte-store accounting and causal decisions, without inference calls."""

from copy import deepcopy

import numpy as np
import pytest

from experiments.local.state_codec import decode_state, encode_state, encoded_size
from experiments.local.store import PausedStateStore
from kvpilot.policy import CodecProfile, CostModel, MIB


def controller(policy="raw_native", capacity=100, *, lossy=True, slo=1000):
    return PausedStateStore(
        policy,
        [CodecProfile("raw_native", 1, 0, 0), CodecProfile("group_int4", 4, 1, 0, False)],
        CostModel(1000, capacity, resume_slo_ms=slo, allow_unvalidated_lossy=lossy),
    )


def pause(store, sid, time, *, size=40, int4_size=None, recompute=1000, kind="tool"):
    plan = store.decide(sid, MIB, recompute, kind, time,
                        {"raw_native": size, "group_int4": int4_size or size})
    payload = None if plan.selected.action == "recompute" else sid.encode()[:1] * plan.selected.stored_bytes
    store.commit(plan, payload)
    return plan, payload


def snapshot(store):
    return (dict(store.payloads), tuple(store.payloads), store.paused_sessions,
            store.used_bytes, store.peak_bytes, store.byte_event_sum, store.event_count,
            store.evicted_states, deepcopy(store.history.stats))


def test_actual_payload_lengths_exact_lru_and_missing_resume_are_consistent():
    store = controller()
    _, a = pause(store, "a", 0)
    _, b = pause(store, "b", 1)
    assert store.payloads["a"] == a  # A read does not alter eviction order.
    plan, c = pause(store, "c", 2)
    assert plan.victims == ("a",)
    assert tuple(store.payloads) == ("b", "c")
    assert store.paused_sessions == ("a", "b", "c")
    assert store.used_bytes == sum(map(len, store.payloads.values())) == 80
    assert store.peak_bytes == 80 and store.evicted_states == 1
    assert store.popresume("a", 3) is None
    assert store.popresume("b", 4) == b
    assert store.popresume("c", 5) == c
    assert store.used_bytes == 0 and store.paused_sessions == ()
    assert store.history.stats["tool"] == (3, 3, 9.0)


def test_byte_event_sum_samples_occupancy_after_lifecycle_events_not_time():
    store = controller()
    pause(store, "a", 0, size=40)
    pause(store, "b", 500, size=20)
    store.cancel("a", 501)
    store.popresume("b", 100000)
    assert store.byte_event_sum == 40 + 60 + 20 + 0
    assert store.event_count == 4 and store.peak_bytes == 60


def test_planning_is_read_only_including_history_recency_and_statistics():
    store = controller("adaptive_eviction")
    pause(store, "a", 0, size=70)
    before = snapshot(store)
    for time in (1, 20, 100000):
        plan = store.decide("b", MIB, 100, "new-kind", time,
                            {"raw_native": 60, "group_int4": 25})
        assert plan.forecast.resume_probability == .5
        assert plan.forecast.pause_ms == 1000
    assert snapshot(store) == before


def test_eviction_cost_changes_selection_and_exact_ablation_is_identical_otherwise():
    stores = [controller(policy) for policy in ("adaptive_exact", "adaptive_eviction")]
    plans = []
    for store in stores:
        initial, _ = pause(store, "old", 0, size=70, int4_size=70)
        assert initial.selected.action == "raw_native"
        plan = store.decide("new", MIB, 100, "tool", 10,
                            {"raw_native": 60, "group_int4": 25})
        plans.append(plan)
    exact, aware = plans
    assert exact.selected.action == "raw_native" and exact.victims == ("old",)
    assert aware.selected.action == "group_int4" and aware.victims == ()
    for a, b in zip(exact.admissions, aware.admissions):
        assert a.codec == b.codec and a.stored_bytes == b.stored_bytes and a.victims == b.victims
        assert a.estimated_eviction_loss_ms == b.estimated_eviction_loss_ms
        assert a.charged_eviction_cost_ms == 0
        assert b.charged_eviction_cost_ms == b.estimated_eviction_loss_ms
    raw = next(c for c in aware.alternatives if c.action == "raw_native")
    assert raw.eviction_cost_ms == pytest.approx(.5 * (1000 - .07))
    stores[1].commit(aware, b"q" * aware.selected.stored_bytes)
    assert tuple(stores[1].payloads) == ("old", "new")
    assert stores[1].used_bytes == 95


def test_evicted_owner_is_not_charged_again_or_marked_as_canceled():
    store = controller("adaptive_eviction")
    pause(store, "a", 0, size=100, recompute=1000)
    # A very expensive incoming recomputation makes admission worthwhile.
    pause(store, "b", 1, size=100, recompute=10000)
    assert "a" not in store.payloads and "a" in store.paused_sessions
    assert store.history.stats == {}
    plan = store.decide("c", MIB, 100000, "tool", 2,
                        {"raw_native": 100, "group_int4": 100})
    assert plan.victims == ("b",)
    assert plan.admissions[0].estimated_eviction_loss_ms == pytest.approx(.5 * (10000 - .1))
    store.cancel("a", 3)
    assert store.history.stats == {"tool": (0, 1, 3.0)}


def test_resolved_only_forecast_updates_even_for_evicted_or_recompute_owners():
    store = controller("recompute")
    pause(store, "a", 0, kind="tool")
    pause(store, "b", 1, kind="tool")
    assert store.history.stats == {} and store.used_bytes == 0
    assert store.popresume("a", 100) is None
    plan = store.decide("c", MIB, 50, "tool", 110,
                        {"raw_native": 40, "group_int4": 20})
    assert plan.forecast.resume_probability == pytest.approx(2 / 3)
    assert plan.forecast.pause_ms == 100
    store.cancel("b", 201)
    assert store.history.stats["tool"] == (1, 2, 300.0)
    other = store.decide("other", MIB, 50, "unknown", 202,
                         {"raw_native": 40, "group_int4": 20})
    assert other.forecast.resume_probability == .5 and other.forecast.pause_ms == 1000


@pytest.mark.parametrize("policy", ["raw_native", "group_int4"])
def test_fixed_codecs_execute_same_victim_order_with_exact_sizes(policy):
    store = controller(policy)
    for i in range(3):
        plan, _ = pause(store, str(i), i, size=45, int4_size=45)
    assert plan.victims == ("0",)
    assert tuple(store.payloads) == ("1", "2")
    assert store.used_bytes == 90


def test_overcapacity_fixed_codec_falls_back_without_evicting_existing_payload():
    store = controller()
    pause(store, "a", 0, size=60)
    plan, _ = pause(store, "b", 1, size=101, int4_size=20)
    assert plan.selected.action == "recompute" and plan.victims == ()
    assert store.payloads == {"a": b"a" * 60}
    assert store.popresume("b", 2) is None


def test_recompute_baseline_never_retains_or_hits_a_snapshot():
    store = controller("recompute")
    for i in range(3):
        plan, data = pause(store, str(i), i)
        assert data is None and plan.selected.stored_bytes == 0
    for i in range(3):
        assert store.popresume(str(i), i + 3) is None
    assert store.peak_bytes == store.used_bytes == store.byte_event_sum == store.evicted_states == 0


def test_admit_if_fits_uses_exact_free_bytes_and_never_evicts_or_compresses():
    store = controller("raw_admit_if_fits", capacity=100)
    first, a = pause(store, "a", 0, size=60, int4_size=10)
    second, b = pause(store, "b", 1, size=40, int4_size=10)
    assert first.selected.action == second.selected.action == "raw_native"
    assert first.victims == second.victims == ()
    assert store.used_bytes == store.peak_bytes == 100
    blocked, payload = pause(store, "c", 2, size=1, int4_size=1)
    assert blocked.selected.action == "recompute" and payload is None
    assert blocked.victims == () and store.evicted_states == 0
    assert tuple(store.payloads) == ("a", "b")
    assert store.popresume("c", 3) is None
    assert store.popresume("a", 4) == a
    # A lossy snapshot would fit, but only a native snapshot is eligible.
    blocked, payload = pause(store, "d", 5, size=61, int4_size=10)
    assert blocked.selected.action == "recompute" and payload is None
    assert blocked.victims == () and store.payloads == {"b": b}
    admitted, _ = pause(store, "e", 6, size=60, int4_size=10)
    assert admitted.selected.action == "raw_native" and admitted.victims == ()
    store.cancel("b", 7)
    admitted, _ = pause(store, "f", 8, size=40, int4_size=10)
    assert admitted.selected.action == "raw_native"
    assert store.used_bytes == store.peak_bytes == 100 and store.evicted_states == 0


def test_admit_if_fits_is_capacity_only_despite_cost_slo_and_resolved_history():
    store = PausedStateStore(
        "raw_admit_if_fits", [CodecProfile("raw_native", 100, 1000, 1000)],
        CostModel(1, 100, resume_slo_ms=.001),
    )
    for i in range(3):
        plan = store.decide(str(i), MIB, 0, "tool", i * 2, {"raw_native": 100})
        assert plan.selected.action == "raw_native"
        assert not plan.selected.within_resume_slo
        assert plan.selected.expected_cost_ms > plan.alternatives[0].expected_cost_ms
        store.commit(plan, b"x" * 100)
        store.cancel(str(i), i * 2 + 1)
    assert store.history.stats["tool"] == (0, 3, 3.0)
    assert store.evicted_states == 0


@pytest.mark.parametrize("profiles", [[], [CodecProfile("other", 1, 0, 0)],
                                     [CodecProfile("raw_native", 1, 0, 0, False)]])
def test_admit_if_fits_requires_an_explicit_lossless_native_profile(profiles):
    with pytest.raises(ValueError, match="lossless raw_native profile"):
        PausedStateStore("raw_admit_if_fits", profiles,
                         CostModel(1000, 100, allow_unvalidated_lossy=True))


def test_admit_if_fits_policy_name_cannot_shadow_a_codec():
    with pytest.raises(ValueError, match="reserved local store policy"):
        PausedStateStore("raw_admit_if_fits", [CodecProfile("raw_admit_if_fits", 1, 0, 0)],
                         CostModel(1000, 100))


def test_admit_if_fits_plans_are_read_only_and_preserve_commit_validation():
    store = controller("raw_admit_if_fits", capacity=100)
    pause(store, "a", 0, size=60)
    before = snapshot(store)
    fits = store.decide("b", MIB, 0, "tool", 1, {"raw_native": 40, "group_int4": 10})
    blocked = store.decide("c", MIB, 0, "tool", 1, {"raw_native": 41, "group_int4": 10})
    assert fits.selected.action == "raw_native" and blocked.selected.action == "recompute"
    assert snapshot(store) == before
    with pytest.raises(ValueError, match="immutable payload length"):
        store.commit(fits, b"x" * 39)
    assert snapshot(store) == before
    store.commit(fits, b"b" * 40)
    with pytest.raises(ValueError, match="stale pause plan"):
        store.commit(blocked, None)
    assert store.used_bytes == 100 and store.evicted_states == 0


def test_candidate_size_uses_full_actual_envelope_not_profile_ratio():
    store = controller(capacity=1000)
    plan = store.decide("a", MIB, 100, "tool", 0,
                        {"raw_native": 79, "group_int4": 37})
    assert {c.action: c.stored_bytes for c in plan.alternatives} == {"recompute": 0, "raw_native": 79, "group_int4": 37}
    store.commit(plan, b"a" * 79)
    assert store.used_bytes == 79


@pytest.mark.parametrize("payload", [b"short", b"x" * 41, bytearray(b"x" * 40), memoryview(b"x" * 40), None])
def test_wrong_size_or_mutable_payload_fails_before_evictions(payload):
    store = controller(capacity=60)
    pause(store, "a", 0, size=40)
    plan = store.decide("b", MIB, 1000, "tool", 1,
                        {"raw_native": 40, "group_int4": 20})
    before = snapshot(store)
    with pytest.raises(ValueError, match="immutable payload length"):
        store.commit(plan, payload)
    assert snapshot(store) == before
    store.commit(plan, b"b" * 40)  # Failed validation did not consume the plan.
    assert tuple(store.payloads) == ("b",)


def test_recompute_commit_rejects_payload_without_state_changes():
    store = controller("recompute")
    plan = store.decide("a", MIB, 10, "tool", 0,
                        {"raw_native": 40, "group_int4": 20})
    before = snapshot(store)
    with pytest.raises(ValueError, match="no payload"):
        store.commit(plan, b"unexpected")
    assert snapshot(store) == before


def test_stale_and_cross_store_plans_fail_without_mutation():
    store = controller()
    old = store.decide("a", MIB, 10, "tool", 1,
                       {"raw_native": 40, "group_int4": 20})
    pause(store, "b", 1)
    before = snapshot(store)
    with pytest.raises(ValueError, match="stale"):
        store.commit(old, b"a" * 40)
    assert snapshot(store) == before
    with pytest.raises(ValueError, match="different store"):
        controller().commit(old, b"a" * 40)


def test_resolution_invalidates_plan_and_session_ids_can_be_reused_after_resolving():
    store = controller()
    pause(store, "a", 0)
    plan = store.decide("b", MIB, 10, "tool", 1,
                        {"raw_native": 40, "group_int4": 20})
    store.cancel("a", 1)
    with pytest.raises(ValueError, match="stale"):
        store.commit(plan, b"b" * 40)
    pause(store, "a", 2)
    assert store.popresume("a", 3) == b"a" * 40


def test_commit_completion_timestamp_preserves_original_pause_duration():
    store = controller()
    plan = store.decide("a", MIB, 100, "tool", 5,
                        {"raw_native": 40, "group_int4": 20})
    with pytest.raises(ValueError, match="precede"):
        store.commit(plan, b"a" * 40, timestamp_ms=4)
    store.commit(plan, b"a" * 40, timestamp_ms=15)
    with pytest.raises(ValueError, match="nondecreasing"):
        store.popresume("a", 14)
    assert store.popresume("a", 25) == b"a" * 40
    assert store.history.stats["tool"] == (1, 1, 20.0)


def test_lossy_approval_is_enforced_and_readonly_payload_view_cannot_mutate():
    with pytest.raises(ValueError, match="approval"):
        controller("group_int4", lossy=False)
    store = controller("adaptive_eviction", lossy=False)
    plan, _ = pause(store, "a", 0)
    assert "group_int4" not in {c.action for c in plan.alternatives}
    with pytest.raises(TypeError):
        store.payloads["b"] = b"x"


def test_slo_priority_is_preserved_when_eviction_penalty_is_large():
    store = controller("adaptive_eviction", slo=.1)
    pause(store, "a", 0, size=70, recompute=1000)
    plan = store.decide("b", MIB, 100, "tool", 1,
                        {"raw_native": 60, "group_int4": 101})
    assert plan.selected.action == "raw_native"
    assert plan.selected.expected_cost_ms > 50
    assert plan.selected.within_resume_slo


@pytest.mark.parametrize("sizes", [{}, {"raw_native": 4}, {"raw_native": 4, "group_int4": 0},
                                  {"raw_native": True, "group_int4": 2},
                                  {"raw_native": 4.0, "group_int4": 2},
                                  {"raw_native": 4, "group_int4": 2, "unknown": 1}])
def test_inexact_or_incomplete_sizes_are_rejected_without_mutation(sizes):
    store = controller()
    before = snapshot(store)
    with pytest.raises(ValueError, match="size"):
        store.decide("a", MIB, 100, "tool", 0, sizes)
    assert snapshot(store) == before


@pytest.mark.parametrize("timestamp", [-1, float("nan"), float("inf"), True])
def test_invalid_timestamps_do_not_mutate_store(timestamp):
    store = controller()
    before = snapshot(store)
    with pytest.raises(ValueError, match="timestamp"):
        pause(store, "a", timestamp)
    assert snapshot(store) == before


def test_duplicate_pauses_and_unknown_resolutions_are_errors_even_after_eviction():
    store = controller(capacity=50)
    pause(store, "a", 0)
    pause(store, "b", 1)
    before = snapshot(store)
    with pytest.raises(ValueError, match="already paused"):
        pause(store, "a", 2)
    for operation in (store.popresume, store.cancel):
        with pytest.raises(ValueError, match="not paused"):
            operation("unknown", 2)
    assert snapshot(store) == before


def test_real_snapshot_roundtrip_retains_native_recurrence_and_counts_all_metadata():
    tensors = {"k": np.linspace(-2, 3, 257, dtype=np.float16),
               "recurrent": np.array([1e10, 1.0000001], dtype=np.float32)}
    roles = {"k": "attention", "recurrent": "recurrent"}
    metadata = {"offset": 257, "model_revision": "test-fixture"}
    sizes = {codec: encoded_size(tensors, roles, metadata, codec)
             for codec in ("raw_native", "group_int4")}
    store = controller("group_int4", capacity=sizes["group_int4"])
    plan = store.decide("session", sum(x.nbytes for x in tensors.values()), 100,
                        "tool", 0, sizes)
    payload = encode_state(tensors, roles, metadata, plan.selected.action)
    store.commit(plan, payload)
    assert store.used_bytes == store.peak_bytes == len(payload) == sizes["group_int4"]
    restored, restored_roles, restored_meta = decode_state(store.popresume("session", 10))
    assert restored_roles == roles and restored_meta == metadata
    assert restored["recurrent"].tobytes() == tensors["recurrent"].tobytes()
    assert np.max(np.abs(restored["k"] - tensors["k"])) < .25
    assert store.used_bytes == 0
