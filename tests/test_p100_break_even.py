"""Accounting/causality tests for an explicitly counterfactual cost screen."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random
import shutil

import pytest

from experiments.local.store import PausedStateStore
from experiments.torch import break_even as sweep
from kvpilot.policy import CodecProfile, CostModel, MIB


def rates():
    return {"raw_native": {"ratio": 1.0, "export_ms": 5., "encode_ms": 2., "decode_ms": 1., "import_ms": 3.},
            "group_int8": {"ratio": 2.0, "export_ms": 5., "encode_ms": 15., "decode_ms": 9., "import_ms": 3.},
            "group_int4": {"ratio": 4.0, "export_ms": 5., "encode_ms": 20., "decode_ms": 10., "import_ms": 3.}}


def row(name, resolution="resume"):
    result = {"case_id": name, "family": "lookup", "raw_tensor_bytes": MIB,
              "exact_candidate_bytes": {"raw_native": MIB + 40, "group_int8": MIB // 2 + 40, "group_int4": MIB // 4 + 40},
              "initial_prefill_ms": 100., "pause_ms": 2., "export_ms": 0., "encode_ms": 0.,
              "resolution": resolution}
    if resolution == "resume":
        result.update(recompute_ms=90., decode_ms=0., import_ms=0., resume_wall_ms=113., generation={"wall_ms": 20.})
    else:
        result["cancel_ms"] = 4.
    return result


def replay(policy="group_int4", scale=1., schedule=None, rows=None, budget=10 * MIB):
    rows = rows or {name: row(name) for name in ("a", "b")}
    schedule = schedule or [("pause", "a"), ("pause", "b"), ("resume", "a"), ("resume", "b")]
    return sweep.replay_block(context=256, scenario="pressure", repeat=0, policy=policy,
                             scale=scale, budget=budget, schedule=schedule,
                             control_rows=rows, overhead_rows=rows, rates=rates())


def test_zero_codec_cost_preserves_copy_floor_and_native_serialization():
    full_profiles, full = sweep.scaled_profiles(rates(), 1)
    zero_profiles, zero = sweep.scaled_profiles(rates(), 0)
    assert zero["raw_native"] == full["raw_native"]
    for codec in ("group_int8", "group_int4"):
        assert zero[codec] == {"export_ms": 5., "encode_ms": 0., "decode_ms": 0., "import_ms": 3.}
    assert zero_profiles[0] == full_profiles[0]
    assert zero_profiles[-1].encode_ms_per_mib == 5.
    assert zero_profiles[-1].decode_ms_per_mib == 3.


@pytest.mark.parametrize("policy", sweep.POLICIES)
@pytest.mark.parametrize("budget", [41, 300, 1000])
def test_metadata_store_matches_real_payload_store(policy, budget):
    profiles = [CodecProfile("raw_native", 1, .5, .4), CodecProfile("group_int8", 2, 1, .6, False),
                CodecProfile("group_int4", 4, 2, 1, False)]
    model = CostModel(1e30, budget, allow_unvalidated_lossy=True)
    real = PausedStateStore(policy, profiles, model)
    metadata = sweep.MetadataStore(policy, profiles, model)
    rng = random.Random(103)
    clock = 0.
    for group in range(4):
        names = [f"{group}-{i}" for i in range(5)]
        for name in names:
            clock += 100
            raw = rng.randint(20, 200)
            sizes = {"raw_native": raw + 10, "group_int8": raw // 2 + 10, "group_int4": raw // 4 + 10}
            actual = real.decide(name, raw, 10, "lookup", clock, sizes)
            modeled = metadata.decide(name, raw, 10, "lookup", clock, sizes)
            assert actual.selected == modeled.selected
            assert actual.victims == modeled.victims
            assert actual.forecast == modeled.forecast
            count = actual.selected.stored_bytes
            real.commit(actual, bytes(count) if count else None, timestamp_ms=clock + 1)
            metadata.commit(modeled, sweep.PayloadLength(count) if count else None, timestamp_ms=clock + 1)
        rng.shuffle(names)
        for i, name in enumerate(names):
            clock += 100
            if i % 3 == 0:
                real.cancel(name, clock)
                metadata.cancel(name, clock)
            else:
                a, b = real.popresume(name, clock), metadata.popresume(name, clock)
                assert (None if a is None else len(a)) == (None if b is None else len(b))
        for field in ("used_bytes", "peak_bytes", "event_count", "evicted_states", "byte_event_sum", "paused_sessions"):
            assert getattr(real, field) == getattr(metadata, field)
        assert real.history.predict("lookup") == metadata.history.predict("lookup")


def test_large_payload_sentinel_does_not_allocate_cache_bytes():
    payload = sweep.PayloadLength(10 ** 12)
    assert len(payload) == 10 ** 12
    assert not isinstance(payload, bytes)


def test_canceled_admissions_pay_export_encode_and_control_work():
    rows = {"a": row("a", "cancel"), "b": row("b", "cancel")}
    block, events = replay(rows=rows, schedule=[("pause", "a"), ("pause", "b"), ("cancel", "a"), ("cancel", "b")])
    costs = block["components_modeled_ms"]
    assert costs["export_ms"] == 10.
    assert costs["encode_ms"] == 40.
    assert costs["decode_ms"] == costs["import_ms"] == costs["recompute_ms"] == 0.
    assert costs["lifecycle_overhead_ms"] == 12.
    assert block["total_modeled_ms"] == 262.
    assert block["cancellations"] == 2 and block["hits"] == 0
    assert events[-1]["used_host_payload_bytes"] == 0


def test_evicted_payload_cost_is_not_erased_and_miss_pays_recompute():
    block, events = replay(budget=MIB // 4 + 40)
    costs = block["components_modeled_ms"]
    assert costs["export_ms"] == 10. and costs["encode_ms"] == 40.
    assert costs["decode_ms"] == 10. and costs["import_ms"] == 3.
    assert costs["recompute_ms"] == 90.
    assert block["hits"] == block["misses"] == block["evictions"] == 1
    assert events[1]["victims"] == ["a"]


def test_simple_admission_retains_first_owner_and_never_evicts():
    block, events = replay(policy="raw_admit_if_fits", budget=MIB + 40)
    assert [e["action"] for e in events if e["event"] == "pause"] == ["raw_native", "recompute"]
    assert block["hits"] == 1 and block["evictions"] == 0
    assert block["components_modeled_ms"]["export_ms"] == 5.


def test_future_resolution_and_recompute_cost_do_not_leak_into_admission():
    base = {"a": row("a"), "b": row("b")}
    changed = deepcopy(base)
    changed["a"] = row("a", "cancel")
    changed["b"]["recompute_ms"] = 100000.
    changed["b"]["resume_wall_ms"] = 100023.
    for policy in ("adaptive_exact", "adaptive_eviction"):
        _, events = replay(policy=policy, rows=base, budget=MIB + 40)
        _, altered = replay(policy=policy, rows=changed, budget=MIB + 40,
                            schedule=[("pause", "a"), ("pause", "b"), ("cancel", "a"), ("resume", "b")])
        assert events[:2] == altered[:2]


def test_each_scale_reruns_policy_decisions():
    # Compressing both owners is worthwhile at zero codec cost, while their
    # initial prefill is too cheap to justify expensive codecs at scale one.
    rows = {name: row(name) for name in ("a", "b")}
    for record in rows.values():
        record["initial_prefill_ms"] = 20.
    low, _ = replay(policy="adaptive_exact", scale=0, budget=MIB // 2 + 80, rows=rows)
    high, _ = replay(policy="adaptive_exact", scale=1, budget=MIB // 2 + 80, rows=rows)
    assert low["actions"] != high["actions"]
    assert low["actions"].get("group_int4", 0) + low["actions"].get("group_int8", 0) > 0
    assert high["actions"] == {"recompute": 2}


def test_break_even_reports_bracket_and_nonmonotonic_crossings_without_interpolation():
    points = [{"codec_scale": scale, "total_modeled_ms": cost, "recompute_total_modeled_ms": 10.}
              for scale, cost in ((0, 9), (.2, 11), (.4, 8), (1, 10))]
    result = sweep.first_break_even_bracket(points)
    assert result["status"] == "mixed_grid"
    assert result["observed_transition_count"] == 3
    assert result["first_transition_bracket"] == {"lower_scale": 0, "upper_scale": .2,
                                                "lower_scale_wins": True, "upper_scale_wins": False}
    assert result["wins_by_scale"][-1]["wins"] is False  # A tie is not a win.
    assert sweep.first_break_even_bracket(points[:1])["status"] == "wins_entire_grid"
    assert sweep.first_break_even_bracket(points[1:2])["status"] == "never_wins_on_grid"


def test_component_rates_validate_and_keep_four_components_separate():
    calibration = {"256": {"samples": [dict(codec=codec, raw_tensor_bytes=MIB, stored_bytes=MIB // ratio,
                                            export_ms=5., encode_ms=20., decode_ms=10., import_ms=3.)
                                       for codec, ratio in (("raw_native", 1), ("group_int8", 2), ("group_int4", 4))]}}
    actual = sweep.component_rates(calibration)["256"]["group_int4"]
    assert actual == dict(ratio=4., export_ms=5., encode_ms=20., decode_ms=10., import_ms=3.)
    calibration["256"]["samples"][0]["export_ms"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        sweep.component_rates(calibration)


@pytest.mark.parametrize("scale", [-1., 1.01, float("nan"), float("inf"), True])
def test_invalid_scales_rejected(scale):
    with pytest.raises(ValueError):
        sweep.scaled_profiles(rates(), scale)


SOURCE = Path(__file__).resolve().parents[1] / "results/p100-readiness-mac-smoke-v1"


def test_audited_schema_one_archive_runs_without_tensor_engine_and_preserves_provenance(tmp_path):
    output = tmp_path / "sweep"
    result = sweep.run(SOURCE, output, [0, .5, 1])
    assert result["evidence_type"] == "counterfactual_owned_cache_codec_cost_sweep"
    assert not result["measured_performance"] and not result["quality_certified"]
    assert result["input_timing_mode"] == "free_generation"
    assert len(result["blocks"]) == 2 * 6 * 3
    assert result["provenance"]["input_validation"]["valid"]
    assert result["overhead_fallbacks"]
    for filename, digest in result["provenance"]["input_artifact_sha256"].items():
        assert sha256((SOURCE / filename).read_bytes()).hexdigest() == digest
    for filename, digest in json.loads((output / "artifact-sha256.json").read_text()).items():
        assert sha256((output / filename).read_bytes()).hexdigest() == digest
    # The same source produces the same model; no local timing is smuggled in.
    repeated = sweep.run(SOURCE, tmp_path / "repeat", [0, .5, 1])
    assert result == repeated
    with pytest.raises(ValueError, match="already exists"):
        sweep.run(SOURCE, output)


def test_rejects_tampered_archive_before_output_creation(tmp_path):
    source = tmp_path / "mutated"
    shutil.copytree(SOURCE, source)
    with (source / "rows.jsonl").open("a") as handle:
        handle.write("{}\n")
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="hash mismatch"):
        sweep.run(source, output)
    assert not output.exists()


def test_negative_lifecycle_residual_not_silently_treated_as_free_work():
    rows = {name: row(name) for name in ("a", "b")}
    rows["a"]["pause_ms"] = 1.
    rows["a"]["encode_ms"] = 2.
    with pytest.raises(ValueError, match="negative measured lifecycle residual"):
        replay(rows=rows)
