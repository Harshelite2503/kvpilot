"""Pure benchmark accounting/protocol tests; never import MLX or load a model."""

from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark_local_under_test", ROOT / "scripts/benchmark_local_kvpilot.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def _row(case_id, *, policy="raw_native", repeat=0, scenario="roomy", correct=True, tokens=None):
    return {"case_id": case_id, "policy": policy, "repeat": repeat, "scenario": scenario,
            "correct": correct, "hit": policy != "recompute", "selected": {"action": policy},
            "generation": {"tokens": [1] if tokens is None else tokens},
            "initial_prefill_ms": 100, "pause_ms": 3, "resume_wall_ms": 20,
            "resume_ttft_ms": 10}


def _block(*, policy="raw_native", scenario="roomy", peak=100, byte_sum=400, events=8, evicted=0):
    return {"scenario": scenario, "policy": policy, "peak_paused_payload_bytes": peak,
            "byte_event_sum": byte_sum, "event_count": events, "evicted_states": evicted,
            "engine_final": {"mlx_peak_active_bytes": 1234}}


def test_state_specs_count_hybrid_state_and_native_byte_slope():
    specs, roles = benchmark.state_specs(1500)
    assert len(specs) == 64
    assert sum(role == "attention" for role in roles.values()) == 16
    assert sum(role == "recurrent" for role in roles.values()) == 48
    total = sum(np.prod(shape) * dtype.itemsize for shape, dtype in specs.values())
    recurrent_floor = 24 * (32 * 128 * 128 * 4 + 3 * 8192 * 2)
    assert total == recurrent_floor + 32768 * 1500
    assert recurrent_floor == 51_511_296
    assert specs["3.k"] == ((1, 4, 1500, 256), np.dtype("float16"))
    assert specs["0.recurrent"] == ((1, 32, 128, 128), np.dtype("float32"))


def test_candidate_sizes_include_unchanged_recurrent_floor_without_export():
    class State:
        offset = 1500

    class Runner:
        def state_metadata(self, state):
            return {"offset": state.offset, "model": "fixture", "prefix_hash": "a" * 64}

        def export_state(self, state):
            pytest.fail("candidate planning must not export or encode tensors")

    sizes = benchmark.size_candidates(Runner(), State())
    assert set(sizes) == set(benchmark.CODECS)
    assert sizes["raw_native"] > sizes["group_int8"] > sizes["group_int4"] > 51_511_296
    # Compressing only attention cannot compress the whole hybrid state fourfold.
    assert sizes["raw_native"] / sizes["group_int4"] < 2


def test_summary_pairs_same_case_repeat_scenario_and_keeps_full_quality_denominator():
    rows = [
        _row("a", policy="recompute", tokens=[1]),
        _row("a", correct=False, tokens=[2]),
        _row("unpaired", tokens=[1]),
        _row("a", repeat=1, policy="recompute", tokens=[3]),
        _row("a", repeat=1, tokens=[3]),
        _row("a", policy="recompute", scenario="pressure", tokens=[9]),
        _row("a", scenario="pressure", tokens=[9]),
    ]
    blocks = [_block(policy="recompute"), _block(peak=100, byte_sum=100, events=4),
              _block(peak=200, byte_sum=300, events=6, evicted=1),
              _block(policy="recompute", scenario="pressure"), _block(scenario="pressure")]
    summary = {(r["scenario"], r["policy"]): r for r in benchmark.summarize(rows, blocks)}
    roomy = summary["roomy", "raw_native"]
    assert roomy["responses"] == 3 and roomy["correct"] == 2
    assert roomy["quality_pairs"] == 2 and roomy["token_identical_to_recompute"] == 1
    assert roomy["peak_paused_payload_bytes"] == 200
    assert roomy["mean_paused_payload_bytes_after_event"] == 40  # (100+300)/(4+6)
    assert roomy["pause_plus_resume_ms"] == 69
    assert roomy["total_including_initial_prefill_ms"] == 369
    assert roomy["evicted_states"] == 1
    assert summary["pressure", "raw_native"]["token_identical_to_recompute"] == 1


def test_normalization_preserves_case_punctuation_and_explanations():
    assert benchmark.normalized(" \tBeta\n  12  ") == "Beta 12"
    assert benchmark.normalized("beta") != benchmark.normalized("Beta")
    assert benchmark.normalized('"Beta"') != benchmark.normalized("Beta")
    assert benchmark.normalized("Answer: Beta") != benchmark.normalized("Beta")


@pytest.mark.parametrize(
    "answer,truncated,correct",
    [("  Beta\n12 ", False, True), ("beta 12", False, False),
     ("Beta 12 because", False, False), ("Beta 12", True, False)],
)
def test_resume_counts_truncated_or_changed_answers_as_failures_and_includes_restore_time(
    monkeypatch, answer, truncated, correct
):
    released = []

    class Live:
        token_digest = "prefix"

        def release(self):
            released.append(True)

    class Runner:
        def prefill(self, tokens, *, chunk_size):
            assert tokens == [1, 2]
            return object(), Live()

        def generate(self, tokens, *, state, max_tokens, chunk_size):
            assert tokens == [3] and state.token_digest == "prefix"
            return {"answer": answer, "tokens": [7], "truncated": truncated,
                    "ttft_ms": 5, "wall_ms": 10}, state

    store = SimpleNamespace(popresume=lambda *args: None, used_bytes=0)
    engine = SimpleNamespace(get_active_memory=lambda: 0, get_cache_memory=lambda: 0,
                             get_peak_memory=lambda: 0)
    times = iter([100, 110, 120, 130, 140])
    monkeypatch.setattr(benchmark, "now_ms", lambda: next(times))
    row = benchmark.resume_session(
        Runner(), engine, store, {"id": "case", "expected_answer": "Beta 12"},
        {"prefix": [1, 2], "suffix": [3]}, {"prefix_hash": "prefix"}, 128, 32
    )
    assert row["correct"] is correct
    assert not row["hit"] and row["recompute_ms"] == 10
    assert row["restore_or_recompute_ms"] == 30
    assert row["resume_ttft_ms"] == 35 and row["resume_wall_ms"] == 40
    assert released == [True]


@pytest.fixture
def controlled_main(tmp_path, monkeypatch):
    """Run orchestration with fake math/runner, preserving real JSON evidence."""
    cases = [{"id": f"case-{i}", "context": "Fixture context", "question": f"Question {i}",
              "expected_answer": str(i)} for i in range(5)]
    fixture = tmp_path / "cases.json"
    fixture.write_text(json.dumps({"cases": cases}))
    output = tmp_path / "run"
    events = []

    @dataclass
    class Live:
        offset: int = 0

        def release(self):
            pass

    class Runner:
        provenance = {"fixture": True}

        def prompt_tokens(self, text):
            return list(range(80))

        def generate(self, tokens, **kwargs):
            return {"answer": "4"}, Live()

    class Store:
        def __init__(self, policy, *args, **kwargs):
            self.policy = policy
            self.paused_sessions = set()
            self.used_bytes = self.peak_bytes = self.byte_event_sum = 0
            self.event_count = self.evicted_states = 0

        def event(self):
            self.used_bytes = len(self.paused_sessions) * 8
            self.peak_bytes = max(self.peak_bytes, self.used_bytes)
            self.byte_event_sum += self.used_bytes
            self.event_count += 1

    engine = ModuleType("mlx.core")
    engine.clear_cache = lambda: None
    engine.reset_peak_memory = lambda: None
    engine.get_active_memory = lambda: 0
    engine.get_cache_memory = lambda: 0
    engine.get_peak_memory = lambda: 0
    package = ModuleType("mlx")
    package.core = engine
    model_module = ModuleType("experiments.local.qwen_runner")
    model_module.QwenRunner, model_module.LiveState = Runner, Live
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", engine)
    monkeypatch.setitem(sys.modules, "experiments.local.qwen_runner", model_module)
    monkeypatch.setattr(benchmark, "source_hashes", lambda: {"fixture.py": "frozen"})
    monkeypatch.setattr(benchmark.importlib.metadata, "version", lambda name: "fixture-version")
    monkeypatch.setattr(benchmark, "calibrate", lambda *args: ([], {"integrity_gate": {"test": True}}))
    monkeypatch.setattr(benchmark, "size_candidates", lambda *args: {"raw_native": 20, "group_int8": 12, "group_int4": 8})
    monkeypatch.setattr(benchmark, "PausedStateStore", Store)

    def pause(runner, mx, store, case, prepared, chunk_size):
        events.append(("pause", case["id"]))
        assert len(prepared["prefix"]) == 16 and len(prepared["suffix"]) == 64
        store.paused_sessions.add(case["id"])
        store.event()
        return _row(case["id"], policy=store.policy)

    def resume(runner, mx, store, case, prepared, row, chunk_size, max_tokens):
        events.append(("resume", case["id"]))
        assert case["id"] in store.paused_sessions
        store.paused_sessions.remove(case["id"])
        store.event()
        return row

    monkeypatch.setattr(benchmark, "pause_session", pause)
    monkeypatch.setattr(benchmark, "resume_session", resume)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--cases", str(fixture), "--output", str(output),
                                    "--scenarios", "roomy", "--policies", "raw_native", "--repeats", "1"])
    return fixture, output, events


def test_controlled_schedule_pauses_group_before_resuming_and_preserves_partial_group(controlled_main):
    fixture, output, events = controlled_main
    original = fixture.read_bytes()
    benchmark.main()
    assert events == [("pause", f"case-{i}") for i in range(4)] + [
        ("resume", f"case-{i}") for i in (0, 2, 1, 3)
    ] + [("pause", "case-4"), ("resume", "case-4")]
    assert (output / "fixture.json").read_bytes() == original
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["fixture_sha256"] == sha256(original).hexdigest()
    assert protocol["source_sha256"] == {"fixture.py": "frozen"}
    assert protocol["resume_order_per_group"] == [0, 2, 1, 3]
    assert len((output / "rows.jsonl").read_text().splitlines()) == 5
    summary = json.loads((output / "summary.json").read_text())
    assert summary["valid_source_freeze"] and summary["summaries"][0]["responses"] == 5


def test_source_change_during_calibration_aborts_before_policy_measurements(controlled_main, monkeypatch):
    _, output, events = controlled_main
    calls = iter([{"source": "before"}, {"source": "after"}])
    monkeypatch.setattr(benchmark, "source_hashes", lambda: next(calls))
    with pytest.raises(RuntimeError, match="during setup/calibration"):
        benchmark.main()
    assert events == []
    assert not (output / "protocol.json").exists()
    assert not (output / "summary.json").exists()


def test_fixture_change_after_freeze_preserves_rows_but_never_validates_summary(controlled_main, monkeypatch):
    fixture, output, events = controlled_main
    original_resume = benchmark.resume_session

    def altered(*args, **kwargs):
        row = original_resume(*args, **kwargs)
        fixture.write_bytes(fixture.read_bytes() + b" ")
        return row

    monkeypatch.setattr(benchmark, "resume_session", altered)
    with pytest.raises(RuntimeError, match="changed after freeze"):
        benchmark.main()
    assert events
    assert (output / "protocol.json").exists()
    assert (output / "rows.jsonl").exists()
    assert not (output / "summary.json").exists()
