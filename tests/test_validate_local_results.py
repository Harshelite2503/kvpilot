"""Tamper checks for the independent validator; tiny JSON fixtures only."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("validate_local_results", Path(__file__).resolve().parents[1] / "scripts/validate_local_results.py")
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def provenance():
    return {"manifest_sha256": "0" * 64, "config": {"digest": "sha256:" + "1" * 64},
            "tokenizer": {"digest": "sha256:" + "2" * 64}}


def tiny_archive(tmp_path):
    root, archive = tmp_path / "repo", tmp_path / "archive"
    archive.mkdir()
    fixture = {"cases": [{"id": "a", "context": "fixture", "question": "first?", "expected_answer": "Alpha"},
                         {"id": "b", "context": "fixture", "question": "second?", "expected_answer": "Beta"}]}
    for name in ("src/kvpilot/policy.py", "experiments/local/store.py", "scripts/benchmark_local_kvpilot.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Frozen fixture source\n")
    source_fixture = root / "examples/local_kvpilot_cases.json"
    source_fixture.parent.mkdir(parents=True)
    write_json(source_fixture, fixture)
    (archive / "fixture.json").write_bytes(source_fixture.read_bytes())
    profs = [{"name": name, "ratio": 1, "encode_ms_per_mib": 1, "decode_ms_per_mib": 1,
              "lossless": name == "raw_native"} for name in validator.CODECS]
    raw_bytes, sizes = validator.candidate_sizes(128, "a" * 64, provenance())
    budget = sizes["raw_native"] + 1
    jobs = [[0, "pressure", policy] for policy in ("recompute", "raw_native", "adaptive_exact")]
    gate = {"raw_restore_logits_bit_exact": True, "max_abs_logit_difference": 0.0}
    protocol = {"source_sha256": {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                                  for path in validator.expected_source_paths(root)},
                "fixture_sha256": sha256(source_fixture.read_bytes()).hexdigest(),
                "cases": ["a", "b"], "jobs": jobs, "budgets_bytes": {"pressure": budget},
                "profiles": profs, "max_tokens": 32, "suffix_tokens": 64, "group_size": 4,
                "resume_order_per_group": [0, 2, 1, 3], "all_sessions_resume": True,
                "integrity_gate": gate}
    rows, blocks, summaries = [], [], []
    for repeat, scenario, policy in jobs:
        action = "group_int4" if policy == "adaptive_exact" else policy
        amount = 0 if action == "recompute" else sizes[action]
        for index, case in enumerate(fixture["cases"]):
            alternatives = []
            for candidate_action, encode, restore in [("recompute", 0, 10), ("raw_native", 1, 1),
                                                       ("group_int8", 2, 1), ("group_int4", 0.1, 1)]:
                alternatives.append({"action": candidate_action,
                                     "stored_bytes": 0 if candidate_action == "recompute" else sizes[candidate_action],
                                     "encode_ms": encode, "offload_ms": 0, "restore_ms": restore,
                                     "predicted_resume_ms": restore, "expected_cost_ms": encode + 0.5 * restore,
                                     "within_resume_slo": True, "eviction_cost_ms": 0})
            hit = bool(amount) and index == 1
            restore_time = 1 if hit else 10
            rows.append({"repeat": repeat, "scenario": scenario, "policy": policy, "case_id": case["id"],
                         "prefix_tokens": 128, "suffix_tokens": 64, "prefix_hash": case["id"] * 64,
                         "raw_tensor_bytes": raw_bytes, "exact_candidate_bytes": sizes,
                         "initial_prefill_ms": 10, "pause_ms": 3 if amount else 0.2, "planning_ms": 0.1,
                         "export_ms": 1 if amount else 0, "encode_ms": 1 if amount else 0,
                         "decode_ms": 0.4 if hit else 0, "import_ms": 0.4 if hit else 0,
                         "recompute_ms": 0 if hit else 9, "restore_or_recompute_ms": restore_time,
                         "resume_ttft_ms": restore_time + 2, "resume_wall_ms": restore_time + 4,
                         "selected": deepcopy(next(c for c in alternatives if c["action"] == action)),
                         "alternatives": alternatives, "forecast": {"resume_probability": 0.5, "pause_ms": 30000},
                         "victims": ["a"] if amount and index == 1 else [],
                         "stored_bytes_after_pause": amount,
                         "stored_bytes_after_resume": amount if index == 0 else 0,
                         "hit": hit, "generation": {"answer": case["expected_answer"], "tokens": [index + 1],
                                                     "truncated": False, "ttft_ms": 2, "wall_ms": 3},
                         "expected_answer": case["expected_answer"], "correct": True})
        blocks.append({"repeat": repeat, "scenario": scenario, "policy": policy, "block_wall_ms": 100,
                       "peak_paused_payload_bytes": amount, "byte_event_sum": amount * 3,
                       "event_count": 4, "evicted_states": int(bool(amount)),
                       "engine_final": {"mlx_peak_active_bytes": 123}})
        summaries.append({"scenario": scenario, "policy": policy, "responses": 2, "correct": 2,
                          "token_identical_to_recompute": 2, "quality_pairs": 2, "hits": int(bool(amount)),
                          "actions": {action: 2}, "total_pause_ms": 6 if amount else 0.4,
                          "total_resume_wall_ms": 19 if amount else 28,
                          "pause_plus_resume_ms": 25 if amount else 28.4,
                          "total_including_initial_prefill_ms": 45 if amount else 48.4,
                          "median_resume_ttft_ms": 7.5 if amount else 12,
                          "peak_paused_payload_bytes": amount,
                          "mean_paused_payload_bytes_after_event": amount * 3 / 4,
                          "evicted_states": int(bool(amount)), "peak_mlx_active_bytes": 123})
    write_json(archive / "protocol.json", protocol)
    write_json(archive / "model-provenance.json", provenance())
    write_json(archive / "calibration.json", {"profiles": profs, "integrity_gate": gate})
    write_json(archive / "blocks.json", blocks)
    write_json(archive / "summary.json", {"valid_source_freeze": True, "summaries": summaries})
    write_rows(archive, rows)
    return root, archive


def write_rows(archive, rows):
    (archive / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_valid_archive_recomputes_every_row_and_denominator(tmp_path):
    root, archive = tiny_archive(tmp_path)
    report = validator.validate_archive(archive, root=root)
    assert report["valid"] is True, report["checks"]
    assert report["row_count"] == report["expected_row_count"] == 6
    assert report["block_count"] == 3 and report["rows_filtered"] == 0
    assert report["model_executed"] is False and len(report["recomputed_summaries"]) == 3
    assert all(cell["quality_pairs"] == cell["responses"] == 2 for cell in report["recomputed_summaries"])


def test_exact_byte_formula_matches_independently_recorded_1456_token_layout():
    raw, sizes = validator.candidate_sizes(1456, "a" * 64, provenance())
    assert raw == 99_221_504
    assert sizes == {"raw_native": 99_230_231, "group_int8": 76_122_181, "group_int4": 64_194_613}


def test_selected_policies_without_recompute_report_zero_control_pairs(tmp_path):
    root, archive = tiny_archive(tmp_path)
    protocol = json.loads((archive / "protocol.json").read_text())
    protocol["jobs"] = [job for job in protocol["jobs"] if job[2] != "recompute"]
    write_json(archive / "protocol.json", protocol)
    blocks = json.loads((archive / "blocks.json").read_text())
    write_json(archive / "blocks.json", [block for block in blocks if block["policy"] != "recompute"])
    rows = [json.loads(line) for line in (archive / "rows.jsonl").read_text().splitlines()]
    write_rows(archive, [row for row in rows if row["policy"] != "recompute"])
    summary = json.loads((archive / "summary.json").read_text())
    summary["summaries"] = [cell for cell in summary["summaries"] if cell["policy"] != "recompute"]
    for cell in summary["summaries"]:
        cell["quality_pairs"] = cell["token_identical_to_recompute"] = 0
    write_json(archive / "summary.json", summary)
    report = validator.validate_archive(archive, root=root)
    assert report["valid"] is True, report["checks"]
    assert len(report["jobs_without_recompute_control"]) == 2
    assert all(cell["quality_pairs"] == cell["token_identical_to_recompute"] == 0
               for cell in report["recomputed_summaries"])


@pytest.mark.parametrize("tamper", ["score", "token_pair", "negative_time", "candidate_bytes", "unrecorded_action",
                                   "victims", "duplicate_row", "hit", "forecast"])
def test_response_tampering_is_rejected_without_filtering(tmp_path, tamper):
    root, archive = tiny_archive(tmp_path)
    rows = [json.loads(line) for line in (archive / "rows.jsonl").read_text().splitlines()]
    if tamper == "score": rows[0]["correct"] = False
    elif tamper == "token_pair": rows[-1]["generation"]["tokens"] = [999]
    elif tamper == "negative_time": rows[0]["pause_ms"] = -1
    elif tamper == "candidate_bytes": rows[0]["exact_candidate_bytes"]["group_int4"] += 1
    elif tamper == "unrecorded_action": rows[-1]["selected"]["action"] = "magic_codec"
    elif tamper == "victims": rows[-1]["victims"] = []
    elif tamper == "duplicate_row": rows.append(deepcopy(rows[0]))
    elif tamper == "hit": rows[-1]["hit"] = False
    elif tamper == "forecast": rows[-1]["forecast"]["resume_probability"] = 0.9
    write_rows(archive, rows)
    report = validator.validate_archive(archive, root=root)
    assert report["status"] == "invalid" and report["valid"] is False
    assert report["rows_filtered"] == 0 and report["row_count"] == len(rows)
    assert report["checks"][-1]["passed"] is False


@pytest.mark.parametrize("target", ["source", "fixture", "summary", "capacity", "event_denominator", "calibration_time"])
def test_source_fixture_summary_and_capacity_tampering_is_rejected(tmp_path, target):
    root, archive = tiny_archive(tmp_path)
    if target == "source":
        (root / "src/kvpilot/policy.py").write_text("# changed after freeze\n")
    elif target == "fixture":
        data = json.loads((archive / "fixture.json").read_text())
        data["cases"][0]["expected_answer"] = "Wrong"
        write_json(archive / "fixture.json", data)
    elif target == "summary":
        data = json.loads((archive / "summary.json").read_text())
        data["summaries"][0]["quality_pairs"] = 1
        write_json(archive / "summary.json", data)
    elif target == "calibration_time":
        data = json.loads((archive / "calibration.json").read_text())
        data["prefill_ms"] = -1
        write_json(archive / "calibration.json", data)
    else:
        data = json.loads((archive / "blocks.json").read_text())
        data[1]["peak_paused_payload_bytes" if target == "capacity" else "event_count"] += 100_000_000
        write_json(archive / "blocks.json", data)
    report = validator.validate_archive(archive, root=root)
    assert report["valid"] is False and report["checks"][-1]["passed"] is False


def test_partial_archive_is_incomplete_not_a_validated_small_sample(tmp_path):
    root, archive = tiny_archive(tmp_path)
    (archive / "summary.json").unlink()
    report = validator.validate_archive(archive, root=root)
    assert report["status"] == "incomplete" and report["valid"] is False
    assert "summary.json" in report["missing_files"]
    assert "recomputed_summaries" not in report
