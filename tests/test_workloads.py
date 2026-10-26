"""Validate synthetic inputs without running any policy or held-out seed."""

from collections import Counter
from copy import deepcopy
import json
import math

import pytest

from kvpilot.policy import CodecProfile, CostModel
from kvpilot.workloads import DEFAULT_PROTOCOL, generate_workload, load_protocol, scenario_config


PROTOCOL = load_protocol()
SCENARIOS = tuple(PROTOCOL["scenarios"])
DEVELOPMENT_SEEDS = (101, 102, 103)


def _pauses(events):
    return [event for event in events if event["type"] == "pause"]


def _block_key(event, block):
    namespace = event["namespace"]
    return tuple(namespace[key] for key in ("tenant", "model_revision", "adapter_revision", "layout")) + (block["prefix_hash"],)


def _peak_physical_bytes_without_eviction(events, ratio=1.0):
    """Input concurrency diagnostic, not a cache-policy replay or result."""
    owners = Counter()
    paused = {}
    sizes = {}
    current = peak = 0
    for event in events:
        sid = event["session_id"]
        if event["type"] == "pause":
            keys = []
            for block in event["blocks"]:
                key = _block_key(event, block)
                keys.append(key)
                sizes[key] = math.ceil(block["raw_bytes"] / ratio)
                if not owners[key]:
                    current += sizes[key]
                owners[key] += 1
            paused[sid] = keys
            peak = max(peak, current)
        else:
            for key in paused.pop(sid):
                owners[key] -= 1
                if not owners[key]:
                    current -= sizes[key]
    assert not paused and current == 0
    return peak


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("seed", DEVELOPMENT_SEEDS)
def test_development_traces_have_complete_ordered_compatible_lifecycles(scenario, seed):
    events = generate_workload(scenario, seed, protocol=PROTOCOL)
    assert len(events) == 2 * PROTOCOL["sessions_per_run"]
    assert events == sorted(events, key=lambda e: (e["at_ms"], e["type"] == "pause", e["session_id"]))
    paused = {}
    seen = set()
    known_sizes = {}
    outcomes = Counter()
    for event in events:
        assert math.isfinite(event["at_ms"]) and event["at_ms"] >= 0
        sid = event["session_id"]
        assert isinstance(sid, str) and sid
        if event["type"] == "pause":
            assert sid not in seen
            seen.add(sid)
            paused[sid] = event["at_ms"]
            assert set(event) == {"at_ms", "type", "session_id", "pause_kind", "namespace", "blocks", "recompute_ms"}
            assert set(event["namespace"]) == {"tenant", "model_revision", "adapter_revision", "layout"}
            assert all(isinstance(value, str) and value for value in event["namespace"].values())
            assert event["pause_kind"] in {"tool", "human"}
            assert math.isfinite(event["recompute_ms"]) and event["recompute_ms"] > 0
            assert event["blocks"]
            keys = []
            for block in event["blocks"]:
                assert set(block) == {"prefix_hash", "raw_bytes"}
                assert isinstance(block["prefix_hash"], str) and block["prefix_hash"]
                assert isinstance(block["raw_bytes"], int) and block["raw_bytes"] > 0
                key = _block_key(event, block)
                assert known_sizes.setdefault(key, block["raw_bytes"]) == block["raw_bytes"]
                keys.append(key)
            assert len(keys) == len(set(keys))
        else:
            assert set(event) == {"at_ms", "type", "session_id"}
            assert event["type"] in {"resume", "cancel"}
            assert event["at_ms"] > paused.pop(sid)
            outcomes[event["type"]] += 1
    assert not paused
    assert len(seen) == PROTOCOL["sessions_per_run"]
    assert outcomes["resume"] > 0


def test_seed_split_and_endpoints_are_predeclared():
    assert PROTOCOL["development_seeds"] == list(DEVELOPMENT_SEEDS)
    assert PROTOCOL["evaluation_seeds"] == list(range(1101, 1121))
    assert not set(PROTOCOL["development_seeds"]) & set(PROTOCOL["evaluation_seeds"])
    assert PROTOCOL["sessions_per_run"] == 240
    assert PROTOCOL["primary_metrics"] == ["modeled_p95_resume_ms"]
    assert PROTOCOL["baseline_policy"] == "group_int4"
    assert PROTOCOL["candidate_policy"] == "adaptive_eviction"
    assert {"adaptive", "adaptive_exact", "adaptive_eviction"} <= set(PROTOCOL["policies"])
    assert PROTOCOL["evidence_type"] == "analytical_assumptions"


def test_holdout_guard_rejects_before_any_generation(monkeypatch):
    def forbidden_rng(*args, **kwargs):
        raise AssertionError("must not instantiate RNG for sealed evaluation")
    monkeypatch.setattr("kvpilot.workloads._rng", forbidden_rng)
    # This exercises only the lock, never creates an evaluation trace or outcome.
    for seed in PROTOCOL["evaluation_seeds"]:
        with pytest.raises(ValueError, match="sealed"):
            generate_workload("sustained_pressure", seed, protocol=PROTOCOL)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_costs_are_valid_and_preserve_existing_codec_assumptions(scenario):
    config = scenario_config(PROTOCOL, scenario)
    assert config["evidence_type"] == "analytical_assumptions"
    model = CostModel(**config["cost_model"])
    profiles = [CodecProfile(**entry) for entry in config["codecs"]]
    assert model.allow_unvalidated_lossy is True
    assert {profile.name for profile in profiles} == {"raw_fp16", "zlib_fp16", "group_int8", "group_int4"}
    existing = json.loads((DEFAULT_PROTOCOL.parent / "assumed-costs.json").read_text())
    assert config["codecs"] == existing["codecs"]
    assert config["default_pause_ms"] == existing["default_pause_ms"]


def test_config_and_generated_events_do_not_mutate_protocol():
    original = deepcopy(PROTOCOL)
    config = scenario_config(PROTOCOL, "heterogeneous")
    config["codecs"][0]["ratio"] = 123
    config["cost_model"]["host_budget_bytes"] = 1
    generate_workload("branching_pressure", 101, protocol=PROTOCOL, sessions=12)
    assert PROTOCOL == original


def test_deterministic_rng_is_independent_of_process_state():
    import random
    first = generate_workload("heterogeneous", 101, protocol=PROTOCOL)
    random.seed(12345)
    for _ in range(100):
        random.random()
    assert generate_workload("heterogeneous", 101, protocol=PROTOCOL) == first
    assert generate_workload("heterogeneous", 102, protocol=PROTOCOL) != first


def test_future_outcome_settings_do_not_change_pause_observations():
    changed = deepcopy(PROTOCOL)
    workload = changed["scenarios"]["branching_pressure"]["workload"]
    workload["resume_probability_by_kind"] = {"tool": 0.0, "human": 0.0}
    workload["wait_ms_by_kind"] = {"tool": [10000], "human": [30000]}
    workload["cancel_wait_scale"] = 0.001
    original_events = generate_workload("branching_pressure", 101, protocol=PROTOCOL)
    changed_events = generate_workload("branching_pressure", 101, protocol=changed)
    assert _pauses(original_events) == _pauses(changed_events)
    assert original_events != changed_events


def test_pressure_and_low_pressure_are_input_properties():
    for scenario in ("sustained_pressure", "branching_pressure"):
        events = generate_workload(scenario, 101, protocol=PROTOCOL)
        budget = scenario_config(PROTOCOL, scenario)["cost_model"]["host_budget_bytes"]
        assert _peak_physical_bytes_without_eviction(events, ratio=3.76) > budget
    events = generate_workload("low_pressure", 101, protocol=PROTOCOL)
    budget = scenario_config(PROTOCOL, "low_pressure")["cost_model"]["host_budget_bytes"]
    assert _peak_physical_bytes_without_eviction(events) < budget


def test_branch_prefixes_are_physical_shared_pages_with_private_suffixes():
    events = generate_workload("branching_pressure", 101, protocol=PROTOCOL)
    pauses = _pauses(events)
    counts = Counter(_block_key(event, block) for event in pauses for block in event["blocks"])
    assert set(counts.values()) == {1, 4}
    logical_bytes = sum(block["raw_bytes"] for event in pauses for block in event["blocks"])
    physical = {}
    for event in pauses:
        for block in event["blocks"]:
            physical[_block_key(event, block)] = block["raw_bytes"]
    assert sum(physical.values()) < logical_bytes
    active = {}
    canceled_with_sibling = False
    for event in events:
        sid = event["session_id"]
        if event["type"] == "pause":
            active[sid] = {_block_key(event, block) for block in event["blocks"]}
        else:
            keys = active.pop(sid)
            if event["type"] == "cancel" and any(keys & other for other in active.values()):
                canceled_with_sibling = True
    assert canceled_with_sibling


def test_fast_returns_cancellations_heterogeneity_and_shift_are_present():
    fast = generate_workload("fast_return", 101, protocol=PROTOCOL)
    starts = {event["session_id"]: event["at_ms"] for event in _pauses(fast)}
    assert max(event["at_ms"] - starts[event["session_id"]] for event in fast if event["type"] != "pause") < 3
    canceled = generate_workload("cancel_heavy", 101, protocol=PROTOCOL)
    assert sum(event["type"] == "cancel" for event in canceled) > PROTOCOL["sessions_per_run"] / 2
    heterogeneous = _pauses(generate_workload("heterogeneous", 101, protocol=PROTOCOL))
    sizes = [sum(block["raw_bytes"] for block in event["blocks"]) for event in heterogeneous]
    assert max(sizes) / min(sizes) >= 16
    rates = [event["recompute_ms"] / size for event, size in zip(heterogeneous, sizes)]
    assert max(rates) / min(rates) > 20
    shifted = _pauses(generate_workload("shifted_pause_mix", 101, protocol=PROTOCOL))
    middle = len(shifted) // 2
    assert sum(event["pause_kind"] == "human" for event in shifted[:middle]) < middle / 3
    assert sum(event["pause_kind"] == "human" for event in shifted[middle:]) > middle / 2


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_session_count_rejected(count):
    with pytest.raises(ValueError, match="positive integer"):
        generate_workload("low_pressure", 101, protocol=PROTOCOL, sessions=count)


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError, match="unknown workload scenario"):
        generate_workload("absent", 101, protocol=PROTOCOL)
