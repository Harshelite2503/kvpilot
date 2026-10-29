#!/usr/bin/env python3
"""Audit and summarize an extracted owned-cache suite without running inference."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import statistics


DEFAULT_REPOSITORY = Path(__file__).resolve().parents[1]
BASELINES = ("group_int4", "raw_admit_if_fits", "raw_native", "recompute")
MODES = ("equal_work", "free_generation")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest_file(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator not in (None, 0) else None


def normalized(value):
    return " ".join(value.split())


def total_ms(row):
    return row["initial_prefill_ms"] + row["pause_ms"] + row.get("resume_wall_ms", row.get("cancel_ms", 0))


def audit_outer(folder):
    manifest_path = folder / "suite-artifact-sha256.json"
    if not manifest_path.is_file():
        return {"present": False, "valid": None, "scope": "Per-run audit still required; no outer manifest supplied."}
    manifest = read(manifest_path)
    actual_names = {str(path.relative_to(folder)) for path in folder.rglob("*")
                    if path.is_file() and path != manifest_path}
    if set(manifest) != actual_names:
        raise ValueError("Suite manifest membership differs from supplied files")
    for name, expected in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe suite manifest entry: {name}")
        path = folder / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError(f"Invalid suite artifact: {name}")
        if digest_file(path) != expected:
            raise ValueError(f"Suite artifact hash mismatch: {name}")
    return {"present": True, "valid": True, "files_checked": len(manifest),
            "manifest_sha256": digest_file(manifest_path),
            "scope": "Consistency with supplied hashes; not an independent signature or proof of physical execution."}


def compare_manifest(folder, historical_path):
    current_path = folder / "model-manifest.json"
    if not current_path.is_file() or not historical_path.is_file():
        return {"available": False, "same_pinned_files": None,
                "scope": "Model-input comparison only; no historical timing or quality result is imported."}
    current, historical = read(current_path), read(historical_path)
    names = sorted(set(current["files"]) | set(historical["files"]))
    matches = {name: current["files"].get(name) == historical["files"].get(name) for name in names}
    same_revision = current.get("repo") == historical.get("repo") and current.get("revision") == historical.get("revision")
    return {"available": True, "same_repository_and_revision": same_revision,
            "same_pinned_files": same_revision and all(matches.values()),
            "repo": current.get("repo"), "revision": current.get("revision"),
            "file_hash_and_size_matches": matches,
            "current_manifest_sha256": digest_file(current_path),
            "historical_manifest_sha256": digest_file(historical_path),
            "scope": "Compares recorded file SHA256/size pairs to the historical P100 model manifest only. Does not compare hardware performance, certify independent signatures, or establish identical kernels."}


def pair_rows(candidate, reference):
    key = lambda row: (row["repeat"], row["case_id"])
    index = {key(row): row for row in reference}
    if len(index) != len(reference) or {key(row) for row in candidate} != set(index):
        raise ValueError("Candidate and baseline do not contain identical started sessions")
    pairs = [(row, index[key(row)]) for row in candidate]
    if any(left["resolution"] != right["resolution"] for left, right in pairs):
        raise ValueError("Candidate and baseline have different session resolutions")
    return pairs


def quality_pair(pairs, mode):
    resumed = [(left, right) for left, right in pairs if left["resolution"] == "resume"]
    if mode == "equal_work":
        return {"answer_quality_evaluated": False,
                "paired_responses": len(resumed),
                "same_forced_inputs_for_all_paired_responses": all(
                    left["generation"]["tokens"] == right["generation"]["tokens"] for left, right in resumed),
                "predicted_token_sequences_identical": sum(
                    left["generation"]["predicted_tokens"] == right["generation"]["predicted_tokens"] for left, right in resumed),
                "scope": "Predicted-token agreement is diagnostic only; forced inputs are not generated answers."}
    return {"answer_quality_evaluated": True,
            "paired_response_observations": len(resumed),
            "distinct_case_ids": len({left["case_id"] for left, _ in resumed}),
            "token_identical_response_observations": sum(
                left["generation"]["tokens"] == right["generation"]["tokens"] for left, right in resumed),
            "normalized_answer_identical_response_observations": sum(
                normalized(left["generation"]["answer"]) == normalized(right["generation"]["answer"]) for left, right in resumed),
            "correctness_regression_observations": sum(bool(right["correct"] and not left["correct"]) for left, right in resumed),
            "correctness_improvement_observations": sum(bool(left["correct"] and not right["correct"]) for left, right in resumed),
            "case_ids_with_any_changed_tokens": sorted({left["case_id"] for left, right in resumed
                if left["generation"]["tokens"] != right["generation"]["tokens"]}),
            "case_ids_with_any_correctness_regression": sorted({left["case_id"] for left, right in resumed
                if right["correct"] and not left["correct"]}),
            "scope": "Repeated observations of the same cases are not independent quality samples."}


def policy_entry(summary, rows, blocks, mode):
    entry = {key: summary.get(key) for key in (
        "policy", "sessions", "responses", "cancellations", "distinct_answered_cases",
        "mean_total_ms_per_started_session", "p95_resume_ttft_ms_descriptive",
        "peak_paused_payload_bytes", "peak_cuda_allocated_bytes", "sampled_peak_process_rss_bytes",
        "hits", "actions", "exported_tensor_bytes", "imported_tensor_bytes")}
    reserved = [block["memory"]["after"].get("device_peak_reserved_bytes") for block in blocks]
    entry["peak_cuda_reserved_bytes"] = max(reserved) if reserved and all(value is not None for value in reserved) else None
    entry["mean_components_ms_per_started_session"] = {
        key: sum(row.get(key, 0) for row in rows) / len(rows)
        for key in ("initial_prefill_ms", "pause_ms", "planning_ms", "export_ms", "encode_ms",
                    "decode_ms", "import_ms", "recompute_ms", "resume_wall_ms", "cancel_ms")}
    if mode == "equal_work":
        entry["answer_quality"] = None
        entry["forced_token_steps"] = summary["forced_token_steps"]
        entry["predicted_tokens_identical_to_recompute"] = summary["predicted_tokens_identical_to_recompute"]
    else:
        entry["answer_quality"] = {key: summary[key] for key in (
            "correct", "responses", "distinct_answered_cases", "token_identical_to_recompute",
            "quality_regressions_vs_recompute", "quality_improvements_vs_recompute", "truncated", "output_tokens")}
        case_groups = defaultdict(list)
        for row in rows:
            if row["resolution"] == "resume":
                case_groups[row["case_id"]].append(row)
        entry["answer_quality"]["distinct_cases_correct_in_every_repeat"] = sum(
            all(row["correct"] for row in values) for values in case_groups.values())
        entry["answer_quality"]["case_ids_with_mixed_correctness_across_repeats"] = sorted(
            name for name, values in case_groups.items() if len({row["correct"] for row in values}) > 1)
    return entry


def analyze_mode(folder, mode, auditor):
    validation = auditor.audit(folder)
    protocol, summary, fixture = read(folder / "protocol.json"), read(folder / "summary.json"), read(folder / "fixture.json")
    if protocol.get("timing_mode", "free_generation") != mode:
        raise ValueError(f"Run directory disagrees with timing mode: {folder}")
    rows = [json.loads(line) for line in (folder / "rows.jsonl").read_text().splitlines()]
    blocks = read(folder / "blocks.json")
    row_groups, block_groups = defaultdict(list), defaultdict(list)
    for row in rows:
        row_groups[row["context_target"], row["scenario"], row["policy"]].append(row)
    for block in blocks:
        block_groups[block["context_target"], block["scenario"], block["policy"]].append(block)
    indexed = {(value["context_target"], value["scenario"], value["policy"]): value for value in summary["summaries"]}
    cells = []
    for context, scenario in sorted({key[:2] for key in indexed}):
        policies = sorted(policy for target, name, policy in indexed if (target, name) == (context, scenario))
        entries = {policy: policy_entry(indexed[context, scenario, policy], row_groups[context, scenario, policy],
                    block_groups[context, scenario, policy], mode) for policy in policies}
        candidate = indexed[context, scenario, "adaptive_eviction"]
        candidate_rows = row_groups[context, scenario, "adaptive_eviction"]
        comparisons = {}
        for baseline in BASELINES:
            if (context, scenario, baseline) not in indexed:
                raise ValueError(f"Required baseline missing: {baseline}")
            reference = indexed[context, scenario, baseline]
            pairs = pair_rows(candidate_rows, row_groups[context, scenario, baseline])
            candidate_ms = candidate["mean_total_ms_per_started_session"]
            baseline_ms = reference["mean_total_ms_per_started_session"]
            by_repeat = defaultdict(list)
            for left, right in pairs:
                by_repeat[left["repeat"]].append((left, right))
            repeat_ratios = [{"repeat": repeat, "speed_ratio_baseline_over_adaptive": ratio(
                sum(total_ms(right) for _, right in values), sum(total_ms(left) for left, _ in values))}
                for repeat, values in sorted(by_repeat.items())]
            comparisons[baseline] = {
                "speed_ratio_baseline_over_adaptive": ratio(baseline_ms, candidate_ms),
                "adaptive_full_session_latency_reduction_percent": 100 * (1 - candidate_ms / baseline_ms) if baseline_ms else None,
                "per_repeat_speed_ratios": repeat_ratios,
                "adaptive_payload_to_baseline_ratio": ratio(candidate["peak_paused_payload_bytes"], reference["peak_paused_payload_bytes"]),
                "adaptive_payload_minus_baseline_bytes": candidate["peak_paused_payload_bytes"] - reference["peak_paused_payload_bytes"],
                "adaptive_cuda_allocation_to_baseline_ratio": ratio(candidate["peak_cuda_allocated_bytes"], reference["peak_cuda_allocated_bytes"]),
                "paired_quality_or_input_diagnostic": quality_pair(pairs, mode),
            }
        cells.append({"context_target": context, "scenario": scenario, "policies": entries,
                      "adaptive_eviction_vs": comparisons})
    report = {"run_directory": str(folder), "validation": validation,
              "input_artifact_manifest_sha256": digest_file(folder / "artifact-sha256.json"),
              "protocol_created_utc": protocol.get("created_utc"),
              "device": protocol["device"], "device_name": protocol.get("device_name"),
              "cuda_runtime": protocol.get("cuda_version"), "versions": protocol["versions"],
              "model_identity": protocol["model_identity"], "arguments": protocol["args"],
              "sessions": len(rows), "responses": sum(row["resolution"] == "resume" for row in rows),
              "blocks": len(blocks), "distinct_base_case_ids": len(protocol["cases"]),
              "summed_measured_block_wall_seconds": sum(block["wall_ms"] for block in blocks) / 1000,
              "cells": cells}
    return report, protocol, fixture


def analyze(folder, repository, historical_path):
    validator_path = repository / "scripts" / "validate_p100_results.py"
    spec = importlib.util.spec_from_file_location("replication_independent_audit", validator_path)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    outer = audit_outer(folder)
    status = read(folder / "suite-status.json") if (folder / "suite-status.json").is_file() else None
    if status is not None and status.get("completed") is not True:
        raise ValueError("Suite is marked incomplete; preserve partial evidence and do not report a completed benchmark")
    mode_reports, protocols, fixtures = {}, {}, {}
    for mode in MODES:
        mode_reports[mode], protocols[mode], fixtures[mode] = analyze_mode(folder / f"full-{mode}", mode, auditor)
    equal, free = (protocols[mode] for mode in MODES)
    inputs = lambda fixture: {context: [{key: case[key] for key in ("id", "tokens", "prefix", "suffix", "expected_answer")}
                                       for case in cases] for context, cases in fixture["cases_by_context"].items()}
    controlled_args = lambda protocol: {key: value for key, value in protocol["args"].items()
                                        if key not in ("timing_mode", "output")}
    controls = {
        "source_hashes_match": equal["source_sha256"] == free["source_sha256"],
        "model_identity_matches": equal["model_identity"] == free["model_identity"],
        "device_type_matches": equal["device"] == free["device"],
        "device_name_matches": equal.get("device_name") == free.get("device_name"),
        "cuda_runtime_matches": equal.get("cuda_version") == free.get("cuda_version"),
        "package_versions_match": equal["versions"] == free["versions"],
        "platform_matches": equal.get("platform") == free.get("platform"),
        "arguments_match_except_mode_and_output": controlled_args(equal) == controlled_args(free),
        "schedules_match": equal["schedules"] == free["schedules"],
        "budgets_match": equal["budgets_bytes"] == free["budgets_bytes"],
        "prompt_token_inputs_match": inputs(fixtures["equal_work"]) == inputs(fixtures["free_generation"]),
        "scope": "Within this machine only. Timings across modes are not equal computational work; across-hardware continuation identity remains unchecked.",
    }
    if not all(value for key, value in controls.items() if key != "scope"):
        raise ValueError("Controls differ between the supplied full modes")
    manifest_path = folder / "model-manifest.json"
    model_manifest = read(manifest_path) if manifest_path.is_file() else None
    return {"evidence_type": "descriptive_analysis_of_measured_owned_cache_replication",
            "created_utc": datetime.now(timezone.utc).isoformat(), "suite_directory": str(folder),
            "source_commit": (folder / "source-commit.txt").read_text().strip() if (folder / "source-commit.txt").is_file() else None,
            "outer_integrity": outer, "suite_status": status,
            "model_manifest_identity": None if model_manifest is None else {
                "repo": model_manifest.get("repo"), "revision": model_manifest.get("revision"),
                "files": len(model_manifest.get("files", {})), "manifest_sha256": digest_file(manifest_path)},
            "historical_p100_model_inputs": compare_manifest(folder, historical_path),
            "cross_mode_controls": controls, "modes": mode_reports,
            "interpretation": {
                "speed_ratio": "Baseline full-session time divided by adaptive time. Above 1 favors adaptive; includes original prefill and canceled work.",
                "memory": "Serialized paused-state bytes are host-cache storage, distinct from CUDA allocation/reservation. RSS sampling can miss short peaks.",
                "quality": "Free-generation scores are exact answers on generated lookup, two-hop and arithmetic tasks. Repeats and contexts are not new independent questions. Equal-work results do not score answers.",
                "components": "Component means overlap: pause includes planning/export/encode; resume includes restore/recompute/generation. Do not add component columns together.",
                "statistics": "Descriptive run and per-repeat ratios; no significance or confidence interval is claimed.",
                "cross_mode_decisions": "Each mode calibrates costs independently. The policy algorithm and workload controls match, but equal-work and free-generation modes can choose different cache actions; quality evidence is specific to the free-generation actions actually exercised.",
                "historical_comparison": "Only model-manifest file hashes are compared with historical P100 records. No old-protocol speed or quality results are pooled.",
                "scope": "Sequential reference runner, CPU codecs after native export; not serving throughput, capacity, dollar savings or a pure GPU-only causal comparison.",
            }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--historical-manifest", type=Path)
    args = parser.parse_args()
    historical = args.historical_manifest or args.repository / "results/p100-2026-10-28-analysis/model-download-manifest.json"
    if args.output.exists():
        parser.error("Output exists; choose a new analysis file")
    result = analyze(args.suite_directory.resolve(), args.repository.resolve(), historical.resolve())
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"analysis": str(args.output), "mode_cells": {
        mode: len(record["cells"]) for mode, record in result["modes"].items()},
        "historical_model_files_match": result["historical_p100_model_inputs"].get("same_pinned_files")}, indent=2))


if __name__ == "__main__":
    main()
