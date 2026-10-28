"""Real tiny tensor execution for lifecycle/evidence, never model-quality claims."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("psutil")
from experiments.torch import benchmark
from experiments.torch import qwen_runner
from experiments.torch.weights import tensor_shapes

SPEC = importlib.util.spec_from_file_location("independent_p100_audit", Path(__file__).resolve().parents[1] / "scripts/validate_p100_results.py")
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class Tokenizer:
    def encode(self, text, **kwargs):
        return SimpleNamespace(ids=[2 + sum(word.encode()) % 29 for word in text.split()])

    def decode(self, tokens):
        return " ".join(str(x) for x in tokens)

    def token_to_id(self, text):
        return 0 if text == "<|im_end|>" else None


def tiny_runner():
    config = {"model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
              "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
              "vocab_size": 32, "max_position_embeddings": 512, "tie_word_embeddings": True,
              "eos_token_id": 0, "rms_norm_eps": 1e-6, "rope_theta": 10000.0}
    generator = torch.Generator().manual_seed(181)
    weights = {name: torch.randn(shape, generator=generator) * .05 for name, shape in tensor_shapes(config).items()}
    for name in weights:
        if "norm.weight" in name:
            weights[name] = torch.ones_like(weights[name])
    return qwen_runner.QwenRunner.from_components(config, weights, Tokenizer())


@pytest.fixture(scope="module", params=["free_generation", "equal_work"])
def archive(tmp_path_factory, request):
    directory = tmp_path_factory.mktemp("torch-records") / "run"
    runner = tiny_runner()
    args = benchmark.parse_args(["--model-dir", "unused", "--output", str(directory), "--device", "cpu",
        "--contexts", "128", "--cases", "4", "--repeats", "1", "--group-size", "4", "--max-tokens", "2",
        "--chunk-size", "32", "--timing-mode", request.param, "--allow-unvalidated-lossy"])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(qwen_runner, "QwenRunner", lambda *args, **kwargs: runner)
        try:
            benchmark.run(args)
        finally:
            torch.set_num_threads(previous_threads)
    return directory


def test_tiny_real_runner_completes_all_arms_and_independent_audit(archive):
    result = validator.audit(archive)
    assert result["valid"] and result["sessions_checked"] == 72 and result["blocks_checked"] == 18
    rows = [json.loads(line) for line in (archive / "rows.jsonl").read_text().splitlines()]
    assert sum(r["resolution"] == "cancel" for r in rows) == 6
    assert all(r["original_live_cache_released"] for r in rows)
    calibration = json.loads((archive / "calibration.json").read_text())["128"]
    assert calibration["integrity_gate"]["raw_restore_logits_bit_exact"]
    for row in rows:
        if row["selected_prediction"]["action"] == "group_int4":
            assert row["exported_tensor_bytes"] > row["serialized_payload_bytes"]


def rewrite_hash(folder, name):
    manifest = json.loads((folder / "artifact-sha256.json").read_text())
    manifest[name] = sha256((folder / name).read_bytes()).hexdigest()
    (folder / "artifact-sha256.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("tamper", ["score", "summary", "bytes", "ttft", "missing_row", "missing_event", "incomplete", "hash"])
def test_validator_rejects_tampering_even_with_refreshed_hashes(archive, tmp_path, tamper):
    import shutil
    run = tmp_path / "mutated"
    shutil.copytree(archive, run)
    if tamper in ("score", "bytes", "ttft", "missing_row", "hash"):
        name = "rows.jsonl"
        rows = [json.loads(line) for line in (run / name).read_text().splitlines()]
        row = next(r for r in rows if r["resolution"] == "resume")
        if tamper in ("score", "hash"):
            row["correct"] = not row["correct"]
        elif tamper == "bytes":
            row["exported_tensor_bytes"] += 1
        elif tamper == "ttft":
            row["resume_ttft_ms"] = 0.0
        else:
            rows.pop()
        (run / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    elif tamper == "summary":
        name = "summary.json"
        data = json.loads((run / name).read_text())
        data["summaries"][0]["mean_total_ms_per_started_session"] /= 2
        (run / name).write_text(json.dumps(data))
    elif tamper == "missing_event":
        name = "events.jsonl"
        lines = (run / name).read_text().splitlines()
        (run / name).write_text("\n".join(lines[:-1]) + "\n")
    else:
        name = "status.json"
        data = json.loads((run / name).read_text())
        data["completed"] = False
        (run / name).write_text(json.dumps(data))
    if tamper != "hash":
        rewrite_hash(run, name)
    with pytest.raises(ValueError):
        validator.audit(run)


def test_validator_rejects_dropping_a_whole_arm_from_self_consistent_records(archive, tmp_path):
    import shutil
    run = tmp_path / "missing-arm"
    shutil.copytree(archive, run)
    protocol = json.loads((run / "protocol.json").read_text())
    protocol["jobs"] = [j for j in protocol["jobs"] if j["policy"] != "adaptive_eviction"]
    (run / "protocol.json").write_text(json.dumps(protocol))
    for name in ("rows.jsonl", "events.jsonl"):
        records = [json.loads(line) for line in (run / name).read_text().splitlines()]
        (run / name).write_text("\n".join(json.dumps(r) for r in records if r["policy"] != "adaptive_eviction") + "\n")
    blocks = json.loads((run / "blocks.json").read_text())
    blocks = [b for b in blocks if b["policy"] != "adaptive_eviction"]
    (run / "blocks.json").write_text(json.dumps(blocks))
    summary = json.loads((run / "summary.json").read_text())
    summary["summaries"] = [s for s in summary["summaries"] if s["policy"] != "adaptive_eviction"]
    (run / "summary.json").write_text(json.dumps(summary))
    status = json.loads((run / "status.json").read_text())
    status["completed_blocks"] = len(blocks)
    (run / "status.json").write_text(json.dumps(status))
    for name in ("protocol.json", "rows.jsonl", "events.jsonl", "blocks.json", "summary.json", "status.json"):
        rewrite_hash(run, name)
    with pytest.raises(ValueError, match="comparison jobs"):
        validator.audit(run)


def test_cancellation_work_remains_in_summary_denominator(archive):
    summaries = json.loads((archive / "summary.json").read_text())["summaries"]
    rows = [json.loads(line) for line in (archive / "rows.jsonl").read_text().splitlines()]
    for summary in summaries:
        if summary["scenario"] != "cancellation":
            continue
        selected = [r for r in rows if r["scenario"] == "cancellation" and r["policy"] == summary["policy"]]
        assert summary["sessions"] == 4 and summary["responses"] == 3 and summary["cancellations"] == 1
        canceled = next(r for r in selected if r["resolution"] == "cancel")
        resumed_cost = sum(r["initial_prefill_ms"] + r["pause_ms"] + r["resume_wall_ms"] for r in selected if r["resolution"] == "resume")
        assert summary["total_ms_including_initial_prefill_and_cancellations"] == pytest.approx(
            resumed_cost + canceled["initial_prefill_ms"] + canceled["pause_ms"] + canceled["cancel_ms"])


def test_fixture_preserves_facts_question_and_deterministic_unseen_seed():
    runner = tiny_runner()
    cases = benchmark.make_cases(8, 182)
    assert cases == benchmark.make_cases(8, 182)
    assert cases != benchmark.make_cases(8, 183)
    prepared = benchmark.prepare_case(runner, cases[0], 128)
    assert cases[0]["facts"] in prepared["text"] and cases[0]["question"] in prepared["text"]
    assert len(prepared["tokens"]) <= 128
    assert prepared["tokens"] == prepared["prefix"] + prepared["suffix"]
    events = benchmark.schedule(cases, 4, "cancellation", 10)
    assert len(events) == 16 and sum(event == "cancel" for event, _ in events) == 2


def test_invalid_options_fail_before_model_load(tmp_path):
    base = ["--model-dir", "unused", "--output", str(tmp_path / "new")]
    with pytest.raises(SystemExit):
        benchmark.parse_args(base)
    with pytest.raises(SystemExit):
        benchmark.parse_args(base + ["--allow-unvalidated-lossy", "--contexts", "512", "512"])


def test_source_freeze_includes_nested_live_and_integration_modules():
    paths = benchmark.source_hashes()
    assert "src/kvpilot/live/recorder.py" in paths
    assert "src/kvpilot/integrations/vllm_connector.py" in paths
    assert "experiments/torch/qwen_runner.py" in paths


def test_changed_answer_is_not_hidden_by_unchanged_accuracy(archive):
    rows = [json.loads(line) for line in (archive / "rows.jsonl").read_text().splitlines()]
    blocks = json.loads((archive / "blocks.json").read_text())
    if rows[0]["timing_mode"] == "equal_work":
        pytest.skip("equal-work inputs are not generated answers")
    altered = deepcopy(rows)
    target = next(r for r in altered if r["policy"] == "group_int4" and r["scenario"] == "roomy")
    target["generation"]["tokens"] = [99999]
    summary = next(s for s in benchmark.summarize(altered, blocks) if s["policy"] == "group_int4" and s["scenario"] == "roomy")
    assert summary["token_identical_to_recompute"] < summary["responses"]


def test_equal_work_never_scores_forced_inputs_as_answers(archive):
    protocol = json.loads((archive / "protocol.json").read_text())
    if protocol["timing_mode"] != "equal_work":
        pytest.skip("fixed-work mode only")
    fixture = json.loads((archive / "fixture.json").read_text())
    cases = {c["id"]: c for c in fixture["cases_by_context"]["128"]}
    rows = [json.loads(line) for line in (archive / "rows.jsonl").read_text().splitlines()]
    for row in rows:
        if row["resolution"] != "resume":
            continue
        generation = row["generation"]
        assert generation["tokens"] == cases[row["case_id"]]["fixed_continuation"]
        assert len(generation["predicted_tokens"]) == len(generation["tokens"]) == 2
        assert row["correct"] is None and row["expected_answer"] is None
        assert "answer" not in generation
    for summary in json.loads((archive / "summary.json").read_text())["summaries"]:
        assert summary["correct"] is summary["output_tokens"] is summary["token_identical_to_recompute"] is None
        assert summary["forced_token_steps"] == summary["responses"] * 2


@pytest.mark.parametrize("tamper", ["forced_token", "prediction_length", "fake_answer", "timing_alias", "fake_quality", "mode"])
def test_equal_work_auditor_rejects_relabeling_and_work_changes(archive, tmp_path, tamper):
    import shutil
    if json.loads((archive / "protocol.json").read_text())["timing_mode"] != "equal_work":
        pytest.skip("fixed-work mode only")
    run = tmp_path / "mutated-fixed"
    shutil.copytree(archive, run)
    name = "rows.jsonl"
    rows = [json.loads(line) for line in (run / name).read_text().splitlines()]
    row = next(r for r in rows if r["resolution"] == "resume")
    if tamper == "forced_token":
        row["generation"]["tokens"][0] = (row["generation"]["tokens"][0] + 1) % 32
    elif tamper == "prediction_length":
        row["generation"]["predicted_tokens"].pop()
    elif tamper == "fake_answer":
        row["generation"]["answer"] = "42"
    elif tamper == "timing_alias":
        row["generation"]["total_ms"] += 1
    elif tamper == "fake_quality":
        row["correct"] = True
    else:
        row["timing_mode"] = "free_generation"
    (run / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    rewrite_hash(run, name)
    with pytest.raises(ValueError):
        validator.audit(run)


def test_legacy_five_arm_archive_still_audits():
    folder = Path(__file__).resolve().parents[1] / "results/p100-readiness-mac-smoke-v1"
    result = validator.audit(folder)
    assert result["valid"] and result["sessions_checked"] == 40


def test_continuation_freeze_keeps_exact_step_count_and_releases_cache():
    runner = tiny_runner()
    case = benchmark.prepare_case(runner, benchmark.make_cases(4, 20261030)[0], 128)
    first = benchmark.freeze_continuation(runner, case, 3, 32)
    second = benchmark.freeze_continuation(runner, case, 3, 32)
    assert first["fixed_continuation"] == second["fixed_continuation"]
    assert len(first["fixed_continuation"]) == 3
    assert "fixed_continuation" not in case
    assert first["continuation_preparation_ms"] >= 0


def test_sweep_consumes_schema_two_without_borrowed_baseline_costs(archive, tmp_path):
    from experiments.torch import break_even
    result = break_even.run(archive, tmp_path / "sweep", [0, 1])
    protocol = json.loads((archive / "protocol.json").read_text())
    assert result["input_timing_mode"] == protocol["timing_mode"]
    assert result["overhead_fallbacks"] == []
    assert not result["measured_performance"] and not result["quality_certified"]
    assert len(result["blocks"]) == 3 * 6 * 2
