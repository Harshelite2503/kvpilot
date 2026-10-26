"""Evaluation integrity checks, using tiny development fixtures only."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("evaluate_eviction", Path(__file__).resolve().parents[1] / "scripts" / "evaluate_eviction.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def protocol():
    return {
        "development_seeds": [1, 2], "evaluation_seeds": [9001, 9002], "sessions_per_run": 2,
        "scenarios": {"fixture": {}}, "policies": ["raw_fp16", "recompute"],
        "baseline_policy": "raw_fp16", "baseline_version": "fixture-v1", "candidate_policy": "recompute",
        "primary_metrics": ["modeled_p95_resume_ms"],
        "secondary_metrics": ["modeled_mean_resume_ms", "modeled_total_service_work_ms", "host_reserved_byte_ms"],
        "bootstrap": {"resamples": 100, "confidence": 0.95, "seed": 5},
    }


def fixture_root(tmp_path, config):
    for relative in ("src/kvpilot/policy.py", "src/kvpilot/replay.py", "src/kvpilot/workloads.py", "scripts/evaluate_eviction.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture source\n")
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(config))
    return path


def test_bootstrap_preserves_pairing_and_uses_runs_not_requests():
    # Baseline magnitudes differ, but every independent run has exactly 50% reduction.
    result = evaluation.paired_bootstrap([(1, 10, 5), (2, 100, 50), (3, 1000, 500)],
                                         resamples=500, confidence=0.95, seed=7)
    assert result["run_pairs"] == 3
    assert result["wins"] == 3
    assert result["improvement_fraction"] == 0.5
    assert result["improvement_fraction_ci"] == [0.5, 0.5]
    assert result["improvement_delta_ci"][0] > 0


def test_bootstrap_counts_ties_losses_and_undefined_zero_baseline():
    tied = evaluation.paired_bootstrap([(1, 0, 0), (2, 0, 0)], resamples=50, confidence=0.95, seed=1)
    assert tied["ties"] == 2
    assert tied["improvement_fraction"] is None
    assert tied["improvement_fraction_ci"] is None
    mixed = evaluation.paired_bootstrap([(1, 10, 5), (2, 10, 20), (3, 10, 10)], resamples=50, confidence=0.95, seed=1)
    assert (mixed["wins"], mixed["ties"], mixed["losses"]) == (1, 1, 1)


def test_missing_pairs_are_exposed_and_do_not_get_confirmatory_ci():
    result = evaluation.paired_bootstrap([(1, 10, 5), (2, None, None)], resamples=20, confidence=0.95, seed=1)
    assert result["unavailable_seeds"] == [2]
    assert result["status"] == "incomplete_pairs"
    assert result["improvement_delta_ci"] is None
    with pytest.raises(ValueError, match="unique"):
        evaluation.paired_bootstrap([(1, 10, 5), (1, 10, 5)], resamples=20, confidence=0.95, seed=1)


def test_freeze_requires_approval_and_detects_source_change(tmp_path):
    config = protocol()
    path = fixture_root(tmp_path, config)
    freeze = evaluation.build_freeze_manifest(path, config, root=tmp_path)
    with pytest.raises(ValueError, match="approved_for_holdout"):
        evaluation.verify_freeze(freeze, path, config, root=tmp_path)
    freeze.update(status="approved_for_holdout", approval_note="Fixture test approval only")
    evaluation.verify_freeze(freeze, path, config, root=tmp_path)
    (tmp_path / "src/kvpilot/policy.py").write_text("# changed policy\n")
    with pytest.raises(ValueError, match="component_sha256"):
        evaluation.verify_freeze(freeze, path, config, root=tmp_path)


def test_freeze_binds_protocol_baseline_version_and_predeclared_metrics(tmp_path):
    config = protocol()
    path = fixture_root(tmp_path, config)
    freeze = evaluation.build_freeze_manifest(path, config, root=tmp_path)
    freeze.update(status="approved_for_holdout", approval_note="Fixture test approval only")
    changed = deepcopy(freeze)
    changed["baseline"]["version"] = "other"
    with pytest.raises(ValueError, match="baseline"):
        evaluation.verify_freeze(changed, path, config, root=tmp_path)
    changed = deepcopy(freeze)
    changed["predeclared_metrics"]["secondary"] = []
    with pytest.raises(ValueError, match="predeclared_metrics"):
        evaluation.verify_freeze(changed, path, config, root=tmp_path)
    path.write_text(json.dumps({**config, "sessions_per_run": 999}))
    with pytest.raises(ValueError, match="component_sha256"):
        evaluation.verify_freeze(freeze, path, config, root=tmp_path)


def test_holdout_is_blocked_before_workload_generation(tmp_path):
    config = protocol()
    path = fixture_root(tmp_path, config)
    generated = []
    with pytest.raises(ValueError, match="approved freeze manifest"):
        evaluation.run_experiment(path, split="holdout", root=tmp_path,
                                  workload_factory=lambda *a, **kw: generated.append(kw))
    assert generated == []


def scenario_config(config, scenario):
    return {"evidence_type": "analytical_assumptions", "default_pause_ms": 10,
            "cost_model": {"bandwidth_bytes_per_ms": 100, "host_budget_bytes": 1000},
            "codecs": [{"name": "raw_fp16", "ratio": 1, "encode_ms_per_mib": 0, "decode_ms_per_mib": 0}]}


def workload(scenario, seed, *, protocol, sessions, allow_evaluation):
    assert not allow_evaluation, "tests must not generate holdout traces"
    assert seed in protocol["development_seeds"]
    return [{"at_ms": 0, "type": "pause", "session_id": "a", "pause_kind": "tool",
             "namespace": {"tenant": "a", "model_revision": "m1", "adapter_revision": "none", "layout": "fp16"},
             "blocks": [{"prefix_hash": "x", "raw_bytes": 100}], "recompute_ms": 10 + seed},
            {"at_ms": 100, "type": "resume", "session_id": "a"}]


def test_development_pairs_identical_inputs_and_archives_no_decision_logs(tmp_path):
    config = protocol()
    path = fixture_root(tmp_path, config)
    manifest, rows, report = evaluation.run_experiment(path, split="development", root=tmp_path,
                                                       workload_factory=workload, config_factory=scenario_config)
    assert len(rows) == 4
    assert len(manifest["input_traces"]) == 2
    for seed in config["development_seeds"]:
        matching = [row for row in rows if row["seed"] == seed]
        assert len({row["trace_sha256"] for row in matching}) == 1
        assert all("decisions" not in row for row in matching)
    raw = next(row for row in rows if row["policy"] == "raw_fp16")
    assert raw["metrics"]["modeled_total_service_work_ms"] == 2.0  # 100B out + 100B in, at 100B/ms.
    assert raw["metrics"]["modeled_mean_resume_ms"] == 1.0
    assert raw["metrics"]["time_weighted_host_reserved_bytes"] == 100.0
    assert report["quality_validated"] is False and report["overall_winner"] is None
    archive = tmp_path / "archive"
    evaluation.archive_results(archive, manifest, rows, report)
    assert {p.name for p in archive.iterdir()} == {"manifest.json", "runs.jsonl", "summary.json"}
    assert len((archive / "runs.jsonl").read_text().splitlines()) == 4
    with pytest.raises(FileExistsError):
        evaluation.archive_results(archive, manifest, rows, report)


def test_pairing_rejects_different_input_hashes():
    config = protocol()
    rows = [{"scenario": "fixture", "seed": 1, "policy": p, "trace_sha256": p,
             "metrics": {metric: 1 for metric in config["primary_metrics"] + config["secondary_metrics"]}}
            for p in config["policies"]]
    with pytest.raises(ValueError, match="different input traces"):
        evaluation.summarize_runs(rows, config)


def test_attribution_is_separate_from_primary_baseline_and_bound_by_freeze(tmp_path):
    config = protocol()
    config["policies"].append("adaptive_exact")
    config["attribution_baseline_policy"] = "adaptive_exact"
    path = fixture_root(tmp_path, config)
    freeze = evaluation.build_freeze_manifest(path, config, root=tmp_path)
    assert freeze["baseline"]["policy"] == "raw_fp16"
    assert freeze["attribution_baseline_policy"] == "adaptive_exact"
    _, _, report = evaluation.run_experiment(path, split="development", root=tmp_path,
                                             workload_factory=workload, config_factory=scenario_config)
    assert all(item["baseline_policy"] == "raw_fp16" for item in report["comparisons"])
    attribution = report["attribution_comparison"]
    assert len(attribution["comparisons"]) == 1
    assert attribution["comparisons"][0]["baseline_policy"] == "adaptive_exact"
    assert attribution["comparisons"][0]["policy"] == "recompute"
    assert report["predeclared_metrics"]["primary"] == ["modeled_p95_resume_ms"]
    freeze.update(status="approved_for_holdout", approval_note="Fixture test approval only")
    freeze["attribution_baseline_policy"] = "raw_fp16"
    with pytest.raises(ValueError, match="attribution_baseline_policy"):
        evaluation.verify_freeze(freeze, path, config, root=tmp_path)


def test_inconsistent_replay_work_accounting_is_rejected():
    result = {"metrics": {"modeled_transferred_bytes": 100, "modeled_encoding_ms": 1,
                           "modeled_decoding_ms": 2, "modeled_recompute_ms": 3,
                           "modeled_total_service_ms": 999}}
    with pytest.raises(ValueError, match="disagrees"):
        evaluation.extract_metrics(result, evaluation.CostModel(100, 1000), [])
