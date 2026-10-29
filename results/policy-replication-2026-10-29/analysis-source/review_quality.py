#!/usr/bin/env python3
"""Read-only P100 versus RTX attempt-3 artifact/quality comparison."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import types

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "outputs/kvpilot"
INPUTS = {
    "p100": Path("/Users/harshelite/Downloads/local-p100-admission-20261028T180805Z"),
    "rtx_attempt_3": ROOT / "outputs/rtx5060ti-20261029/extracted/local-rtx5060ti-replication",
}
OUT = ROOT / "outputs/p100-vs-rtx-20261029/independent-quality-review.json"
PINNED = "ca6f6feaaf7faef4b942bd37d78ee29ded99a27a"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def norm(s):
    return " ".join(s.split())


def row_key(row):
    return row["context_target"], row["scenario"], row["repeat"], row["case_id"]


def has_value(row):
    expected = row["expected_answer"]
    return bool(expected.isdecimal() and re.search(r"(?<!\w)" + re.escape(expected) + r"(?!\w)", row["generation"]["answer"]))


def quality(rows):
    resumed = [r for r in rows if r["resolution"] == "resume"]
    groups = defaultdict(list)
    for row in resumed:
        groups[row["context_target"], row["scenario"], row["case_id"]].append(row)
    return {
        "sessions": len(rows), "responses": len(resumed), "cancellations": len(rows) - len(resumed),
        "exact_match_correct": sum(r["correct"] for r in resumed),
        "truncated": sum(r["generation"]["truncated"] for r in resumed),
        "output_tokens": sum(len(r["generation"]["tokens"]) for r in resumed),
        "distinct_base_cases_answered": len({r["case_id"] for r in resumed}),
        "distinct_context_case_pairs_answered": len({(r["context_target"], r["case_id"]) for r in resumed}),
        "repeats": sorted({r["repeat"] for r in rows}),
        "actions": dict(sorted(Counter(r["selected_prediction"]["action"] for r in rows).items())),
        "hits": sum(r["hit"] for r in resumed),
        "eviction_events": sum(len(r["victims"]) for r in rows),
        "context_scenario_cases_with_tokens_changing_across_repeats": sum(
            len({tuple(r["generation"]["tokens"]) for r in group}) > 1 for group in groups.values()),
    }


def compare(candidate, baseline, include_differences=False):
    a = {row_key(r): r for r in candidate if r["resolution"] == "resume"}
    b = {row_key(r): r for r in baseline if r["resolution"] == "resume"}
    assert set(a) == set(b)
    regressions = [a[k] for k in a if b[k]["correct"] and not a[k]["correct"]]
    result = {
        "paired_responses": len(a),
        "token_identical": sum(a[k]["generation"]["tokens"] == b[k]["generation"]["tokens"] for k in a),
        "normalized_answer_identical": sum(norm(a[k]["generation"]["answer"]) == norm(b[k]["generation"]["answer"]) for k in a),
        "exact_match_regressions": len(regressions),
        "exact_match_improvements": sum(a[k]["correct"] and not b[k]["correct"] for k in a),
        "regressions_with_expected_standalone_numeric_value_present": sum(has_value(r) for r in regressions),
        "regressions_without_expected_standalone_numeric_value_present": sum(not has_value(r) for r in regressions),
        "distinct_context_case_pairs_with_regressions": [list(k) for k in sorted({(r["context_target"], r["case_id"]) for r in regressions})],
        "same_cache_hit_outcomes": sum(a[k]["hit"] == b[k]["hit"] for k in a),
    }
    if include_differences:
        grouped = defaultdict(list)
        for key in a:
            if a[key]["generation"]["tokens"] != b[key]["generation"]["tokens"]:
                grouped[key[0], key[1], key[3]].append(key)
        result["differences"] = [{"context_target": context, "scenario": scenario, "case_id": case_id,
            "repeats": sorted(key[2] for key in keys),
            "candidate_answer_variants": sorted({a[key]["generation"]["answer"] for key in keys}),
            "baseline_answer_variants": sorted({b[key]["generation"]["answer"] for key in keys}),
            "expected_answer": a[keys[0]]["expected_answer"]}
            for (context, scenario, case_id), keys in sorted(grouped.items())]
    return result


def fixture_cases(fixture):
    return {(int(context), case["id"]): case for context, cases in fixture["cases_by_context"].items() for case in cases}


def compare_case_fields(left, right, ignored=()):
    assert set(left) == set(right)
    differences = []
    for key in sorted(left):
        names = sorted(name for name in set(left[key]) | set(right[key])
                       if name not in ignored and left[key].get(name) != right[key].get(name))
        if names:
            differences.append({"context_target": key[0], "case_id": key[1], "different_fields": names})
    return {"paired_context_cases": len(left), "fully_matching_context_cases": len(left) - len(differences), "differences": differences}


def main():
    source = subprocess.check_output(["git", "-C", str(REPO), "show", f"{PINNED}:scripts/validate_p100_results.py"])
    validator = types.ModuleType("pinned_cross_device_quality_validator")
    exec(compile(source, f"{PINNED}:scripts/validate_p100_results.py", "exec"), validator.__dict__)
    datasets, audit = {}, {}
    for name, root in INPUTS.items():
        modes = {}
        for mode in ("equal_work", "free_generation"):
            folder = root / f"full-{mode}"
            audit[f"{name}/{mode}"] = {**validator.audit(folder),
                "artifact_manifest_sha256": digest(folder / "artifact-sha256.json")}
            modes[mode] = {"protocol": read(folder / "protocol.json"), "fixture": read(folder / "fixture.json"),
                           "provenance": read(folder / "model-provenance.json"),
                           "rows": [json.loads(line) for line in (folder / "rows.jsonl").read_text().splitlines()]}
            assert modes[mode]["protocol"]["timing_mode"] == mode
        fixture = fixture_cases(modes["free_generation"]["fixture"])
        for row in modes["free_generation"]["rows"]:
            if row["resolution"] == "resume":
                expected = fixture[row["context_target"], row["case_id"]]["expected_answer"]
                assert row["expected_answer"] == expected
                assert row["correct"] is (not row["generation"]["truncated"] and norm(row["generation"]["answer"]) == norm(expected))
        datasets[name] = modes

    device_reports = {}
    for name, modes in datasets.items():
        rows = modes["free_generation"]["rows"]
        policies = sorted({r["policy"] for r in rows})
        policy_rows = {policy: [r for r in rows if r["policy"] == policy] for policy in policies}
        cells = []
        for context, scenario in sorted({(r["context_target"], r["scenario"]) for r in rows}):
            subsets = {policy: [r for r in subset if (r["context_target"], r["scenario"]) == (context, scenario)] for policy, subset in policy_rows.items()}
            cells.append({"timing_mode": "free_generation", "context_target": context, "scenario": scenario,
                "policies": {policy: quality(subset) for policy, subset in subsets.items()},
                "each_policy_vs_recompute": {policy: compare(subset, subsets["recompute"]) for policy, subset in subsets.items()},
                "adaptive_vs_int4": compare(subsets["adaptive_eviction"], subsets["group_int4"]),
                "adaptive_vs_simple_admission": compare(subsets["adaptive_eviction"], subsets["raw_admit_if_fits"])})
        source_commit = (INPUTS[name] / "source-commit.txt").read_text().strip()
        source_map = modes["free_generation"]["protocol"]["source_sha256"]
        git_matches = {}
        for commit in (source_commit, PINNED):
            git_matches[commit] = {path: sha256(subprocess.check_output(["git", "-C", str(REPO), "show", f"{commit}:{path}"])).hexdigest() == expected
                                   for path, expected in source_map.items()}
        device_reports[name] = {
            "input_directory": str(INPUTS[name]), "declared_source_commit": source_commit,
            "recorded_source_hashes_match_git": git_matches,
            "git_status_record": (INPUTS[name] / "git-status.txt").read_text(),
            "preflight": read(INPUTS[name] / "preflight.json"),
            "runtime_capture_present": (INPUTS[name] / "runtime.json").exists(),
            "outer_suite_manifest_present": (INPUTS[name] / "suite-artifact-sha256.json").exists(),
            "protocol_versions": modes["free_generation"]["protocol"]["versions"],
            "model_identity": modes["free_generation"]["protocol"]["model_identity"],
            "cases": len(fixture_cases(modes["free_generation"]["fixture"])),
            "distinct_underlying_questions": len({case["question"] for case in fixture_cases(modes["free_generation"]["fixture"]).values()}),
            "observed_mix": {"timing_mode": "free_generation", "policies": {policy: quality(subset) for policy, subset in policy_rows.items()},
                "each_policy_vs_recompute": {policy: compare(subset, policy_rows["recompute"]) for policy, subset in policy_rows.items()},
                "adaptive_vs_int4": compare(policy_rows["adaptive_eviction"], policy_rows["group_int4"]),
                "adaptive_vs_simple_admission": compare(policy_rows["adaptive_eviction"], policy_rows["raw_admit_if_fits"])},
            "cells": cells,
        }

    control_comparisons = {}
    for mode in ("equal_work", "free_generation"):
        p, r = datasets["p100"][mode], datasets["rtx_attempt_3"][mode]
        pp, rp = p["protocol"], r["protocol"]
        args = sorted(set(pp["args"]) | set(rp["args"]))
        control_comparisons[mode] = {
            "source_hash_maps_identical": pp["source_sha256"] == rp["source_sha256"],
            "recorded_source_files": len(pp["source_sha256"]),
            "argument_differences": {k: {"p100": pp["args"].get(k), "rtx": rp["args"].get(k)} for k in args if pp["args"].get(k) != rp["args"].get(k)},
            "benchmark_arguments_identical_except_paths": all(pp["args"].get(k) == rp["args"].get(k) for k in args if k not in ("model_dir", "output")),
            "jobs_identical": pp["jobs"] == rp["jobs"], "schedules_identical": pp["schedules"] == rp["schedules"],
            "host_payload_budgets_identical": pp["budgets_bytes"] == rp["budgets_bytes"],
            "recorded_model_file_hashes_identical": p["provenance"]["file_sha256"] == r["provenance"]["file_sha256"],
            "recorded_model_digests_identical": p["provenance"]["model_digest"] == r["provenance"]["model_digest"],
            "fixture_content_comparison_excluding_continuation_preparation_ms": compare_case_fields(fixture_cases(p["fixture"]), fixture_cases(r["fixture"]), ("continuation_preparation_ms",)),
        }

    p_cases = fixture_cases(datasets["p100"]["equal_work"]["fixture"])
    r_cases = fixture_cases(datasets["rtx_attempt_3"]["equal_work"]["fixture"])
    continuation_differences = []
    for k in sorted(p_cases):
        p_tokens, r_tokens = p_cases[k]["fixed_continuation"], r_cases[k]["fixed_continuation"]
        if p_tokens != r_tokens:
            continuation_differences.append({"context_target": k[0], "case_id": k[1],
                "p100_token_ids": p_tokens, "rtx_token_ids": r_tokens,
                "first_differing_zero_based_index": next((i for i, (a, b) in enumerate(zip(p_tokens, r_tokens)) if a != b), min(len(p_tokens), len(r_tokens)))})
    cross_device = {}
    p_rows, r_rows = (datasets[name]["free_generation"]["rows"] for name in ("p100", "rtx_attempt_3"))
    for policy in sorted({r["policy"] for r in p_rows}):
        p = [r for r in p_rows if r["policy"] == policy]
        r = [r for r in r_rows if r["policy"] == policy]
        a, b = {row_key(x): x for x in p}, {row_key(x): x for x in r}
        assert set(a) == set(b)
        cross_device[policy] = {"timing_mode": "free_generation", **compare(p, r, include_differences=True), "candidate": "p100", "baseline": "rtx_attempt_3",
            "paired_started_sessions": len(a), "same_selected_action": sum(a[k]["selected_prediction"]["action"] == b[k]["selected_prediction"]["action"] for k in a),
            "same_victims": sum(a[k]["victims"] == b[k]["victims"] for k in a)}

    result = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_type": "independent_p100_vs_rtx_attempt_3_quality_and_input_review",
        "pinned_validator_commit": PINNED, "pinned_validator_source_sha256": sha256(source).hexdigest(),
        "computation_source": str(Path(__file__).resolve().relative_to(ROOT)), "computation_source_sha256": digest(Path(__file__)),
        "audit": audit, "devices": device_reports, "cross_device_controls": control_comparisons,
        "cross_device_quality_same_policy": cross_device,
        "equal_work_adaptive_actions_for_scope_contrast": {
            name: dict(sorted(Counter(row["selected_prediction"]["action"] for row in modes["equal_work"]["rows"]
                                      if row["policy"] == "adaptive_eviction").items()))
            for name, modes in datasets.items()},
        "equal_work_continuations": {"paired_context_cases": len(p_cases), "identical": len(p_cases) - len(continuation_differences),
            "all_have_24_steps_on_both_devices": all(len(p_cases[k]["fixed_continuation"]) == len(r_cases[k]["fixed_continuation"]) == 24 for k in p_cases),
            "differences": continuation_differences,
            "scope": "Each device freezes its own native continuation. Equal step count and tensor shapes do not establish identical token inputs across devices."},
        "scope": {
            "new_inference": False, "remote_actions": False, "input_or_repository_modified": False,
            "first_rtx_attempt_pooled": False, "gpu_only_causal_timing_claim": False, "general_quality_certification": False,
            "scoring": "Predefined whitespace-normalized exact answer, with truncation scored incorrect, independently recomputed from stored answers and fixture truth.",
            "format_diagnostic": "Expected numeric occurrence is a literal diagnostic only. Some exact-match failures contain the correct value with added text; no semantic rescoring is applied.",
            "repeat_caveat": "24 underlying generated questions, 48 context variants, three repeats and repeated scenarios; response totals are not independent quality samples.",
            "device_caveat": "P100 CUDA 12.6 / Torch cu126 and RTX CUDA 12.8 / Torch cu128 run on different host systems. P100 lacks the matching runtime CPU/thread/TF32/cgroup snapshot; these controls cannot be asserted identical.",
            "source_caveat": "Declared Git commits differ, but all 25 recorded execution source files match both commits and each other. This does not assert identical entire repositories or environments.",
            "integrity_caveat": "Per-run hashes/accounting/scores are independently audited. This is not hardware attestation. P100 has no supplied outer suite manifest.",
            "free_generation_timing": "Free-generation lengths differ under INT4; they must not be used as equal-work speedup evidence.",
            "action_identity_scope": "Cross-device action, victim and hit identity counts in this quality review refer only to free_generation. Equal_work choices differ; their separately computed adaptive action counts are supplied only to make this scope distinction explicit.",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": digest(OUT), "bytes": OUT.stat().st_size,
                      "continuations_identical": result["equal_work_continuations"]["identical"],
                      "recompute_cross_device_tokens_identical": cross_device["recompute"]["token_identical"]}))


if __name__ == "__main__":
    main()
