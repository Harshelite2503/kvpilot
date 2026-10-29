#!/usr/bin/env python3
"""Independently audit and compare recorded equal-work P100 and RTX measurements."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import types

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "outputs/kvpilot"
INPUTS = {
    "p100": Path("/Users/harshelite/Downloads/local-p100-admission-20261028T180805Z"),
    "rtx5060ti": ROOT / "outputs/rtx5060ti-20261029/extracted/local-rtx5060ti-replication",
}
OUTPUT = ROOT / "outputs/p100-vs-rtx-20261029/independent-timing-review.json"
BASELINES = ("group_int4", "raw_admit_if_fits", "raw_native", "recompute")
COMPONENTS = ("initial_prefill_ms", "pause_ms", "planning_ms", "export_ms", "encode_ms",
              "decode_ms", "import_ms", "recompute_ms", "resume_wall_ms", "cancel_ms")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def total(row):
    return row["initial_prefill_ms"] + row["pause_ms"] + row.get("resume_wall_ms", row.get("cancel_ms", 0))


def mean(rows):
    return sum(map(total, rows)) / len(rows)


def reduction(candidate, reference):
    return 100 * (1 - candidate / reference)


def state(row):
    return {"action": row["selected_prediction"]["action"], "resolution": row["resolution"],
            "hit": row.get("hit"), "victims": row["victims"]}


def policy_stats(rows, blocks):
    components = {name: sum(row.get(name, 0) for row in rows) / len(rows) for name in COMPONENTS}
    components["generation_wall_ms"] = sum(row.get("generation", {}).get("wall_ms", 0) for row in rows) / len(rows)
    return {
        "started_sessions": len(rows), "resumes": sum(row["resolution"] == "resume" for row in rows),
        "cancellations": sum(row["resolution"] == "cancel" for row in rows),
        "actions": dict(sorted(Counter(row["selected_prediction"]["action"] for row in rows).items())),
        "hits": sum(row.get("hit", False) for row in rows), "evictions": sum(len(row["victims"]) for row in rows),
        "admissions_not_resumed_from_cache": sum(row["selected_prediction"]["action"] != "recompute"
                                                and not row.get("hit", False) for row in rows),
        "total_full_session_ms": sum(map(total, rows)), "mean_full_session_ms": mean(rows),
        "mean_components_ms_per_started_session": components,
        "peak_serialized_host_payload_bytes": max(block["peak_paused_payload_bytes"] for block in blocks),
        "peak_cuda_allocated_bytes": max(block["memory"]["after"]["device_peak_allocated_bytes"] for block in blocks),
        "peak_cuda_reserved_bytes": max(block["memory"]["after"]["device_peak_reserved_bytes"] for block in blocks),
        "sampled_peak_rss_bytes": max(block["memory"]["sampled_peak_process_rss_bytes"] for block in blocks),
        "exported_tensor_bytes": sum(row["exported_tensor_bytes"] for row in rows),
        "imported_tensor_bytes": sum(row.get("imported_tensor_bytes", 0) for row in rows),
    }


def comparisons(groups):
    adaptive = groups["adaptive_eviction"]
    result = {}
    for baseline in BASELINES:
        reference = groups[baseline]
        adaptive_keys = {(row["context_target"], row["scenario"], row["repeat"], row["case_id"]) for row in adaptive}
        reference_keys = {(row["context_target"], row["scenario"], row["repeat"], row["case_id"]) for row in reference}
        assert len(adaptive) == len(reference) == len(adaptive_keys) and adaptive_keys == reference_keys
        repeats = []
        for repeat in sorted({row["repeat"] for row in adaptive}):
            left = [row for row in adaptive if row["repeat"] == repeat]
            right = [row for row in reference if row["repeat"] == repeat]
            repeats.append({"repeat": repeat, "adaptive_mean_ms": mean(left), "baseline_mean_ms": mean(right),
                            "time_reduction_percent": reduction(mean(left), mean(right))})
        result[baseline] = {
            "baseline_over_adaptive_time_ratio": mean(reference) / mean(adaptive),
            "adaptive_time_reduction_percent": reduction(mean(adaptive), mean(reference)),
            "per_repeat": repeats,
            "adaptive_faster_in_every_repeat": all(row["time_reduction_percent"] > 0 for row in repeats),
        }
    return result


def decision_comparison(adaptive, simple):
    index = {(row["repeat"], row["case_id"]): row for row in simple}
    adaptive_index = {(row["repeat"], row["case_id"]): row for row in adaptive}
    assert len(index) == len(adaptive_index) == len(simple) == len(adaptive)
    pairs = [(row, index[row["repeat"], row["case_id"]]) for row in adaptive]
    assert all(left["resolution"] == right["resolution"] for left, right in pairs)
    assert all(left["generation"]["tokens"] == right["generation"]["tokens"]
               for left, right in pairs if left["resolution"] == "resume")
    differences, replacements = [], []
    for left, right in pairs:
        if state(left) != state(right):
            differences.append({"repeat": left["repeat"], "case_id": left["case_id"],
                                "adaptive": state(left), "simple": state(right)})
        for victim in left["victims"]:
            replacements.append({"repeat": left["repeat"], "admitted_case_id": left["case_id"],
                                 "victim_case_id": victim, "admitted_adaptive": state(left),
                                 "admitted_simple": state(right),
                                 "victim_adaptive": state(adaptive_index[left["repeat"], victim]),
                                 "victim_simple": state(index[left["repeat"], victim])})
    return {
        "paired_started_sessions": len(pairs),
        "same_action": sum(state(left)["action"] == state(right)["action"] for left, right in pairs),
        "same_action_hit_resolution_and_victims": sum(state(left) == state(right) for left, right in pairs),
        "same_hit_outcome_among_resumes": sum(left["hit"] == right["hit"] for left, right in pairs if left["resolution"] == "resume"),
        "paired_resumes": sum(left["resolution"] == "resume" for left, _ in pairs),
        "differences": sorted(differences, key=lambda row: (row["repeat"], row["case_id"])),
        "adaptive_replacement_events": replacements,
        "replacement_admissions_that_cancel": sum(row["admitted_adaptive"]["resolution"] == "cancel" for row in replacements),
        "replacements_of_simple_hits_by_admissions_that_cancel": sum(
            row["admitted_adaptive"]["resolution"] == "cancel" and row["victim_simple"]["hit"] is True for row in replacements),
    }


def analyze(label, suite):
    folder = suite / "full-equal_work"
    commit = (suite / "source-commit.txt").read_text().strip()
    source = subprocess.check_output(["git", "-C", str(REPO), "show", f"{commit}:scripts/validate_p100_results.py"])
    validator = types.ModuleType("independent_timing_validator_" + label)
    exec(compile(source, f"{commit}:scripts/validate_p100_results.py", "exec"), validator.__dict__)
    validation = validator.audit(folder)
    assert validation["valid"] and validation["sessions_checked"] == 2592 and validation["blocks_checked"] == 108
    rows = [json.loads(line) for line in (folder / "rows.jsonl").read_text().splitlines()]
    protocol, blocks, fixture = read(folder / "protocol.json"), read(folder / "blocks.json"), read(folder / "fixture.json")
    assert protocol["timing_mode"] == "equal_work"
    grouped, block_groups = defaultdict(list), defaultdict(list)
    for row in rows:
        grouped[row["context_target"], row["scenario"], row["policy"]].append(row)
    for block in blocks:
        block_groups[block["context_target"], block["scenario"], block["policy"]].append(block)
    cells = []
    for context, scenario in sorted({key[:2] for key in grouped}):
        groups = {policy: grouped[context, scenario, policy] for policy in protocol["policies"]}
        stats = {policy: policy_stats(values, block_groups[context, scenario, policy]) for policy, values in groups.items()}
        fastest = min(BASELINES, key=lambda policy: stats[policy]["mean_full_session_ms"])
        comparison = comparisons(groups)
        component_deltas = {}
        for baseline in BASELINES:
            components = stats[baseline]["mean_components_ms_per_started_session"]
            candidate = stats["adaptive_eviction"]["mean_components_ms_per_started_session"]
            component_deltas[baseline] = {
                "adaptive_minus_baseline_full_session_ms": stats["adaptive_eviction"]["mean_full_session_ms"] - stats[baseline]["mean_full_session_ms"],
                "adaptive_minus_baseline_components_ms": {name: candidate[name] - components[name] for name in candidate},
                "baseline_minus_adaptive_cpu_encode_decode_ms": components["encode_ms"] + components["decode_ms"] - candidate["encode_ms"] - candidate["decode_ms"],
            }
        cells.append({"context_target": context, "scenario": scenario, "policies": stats,
                      "adaptive_vs_baselines": comparison, "fastest_observed_baseline": fastest,
                      "adaptive_vs_fastest_baseline": comparison[fastest],
                      "adaptive_beats_all_baselines_in_every_repeat": all(row["adaptive_faster_in_every_repeat"] for row in comparison.values()),
                      "cuda_allocated_peaks_identical_across_policies": len({row["peak_cuda_allocated_bytes"] for row in stats.values()}) == 1,
                      "adaptive_host_payload_to_int4_ratio": stats["adaptive_eviction"]["peak_serialized_host_payload_bytes"] / stats["group_int4"]["peak_serialized_host_payload_bytes"],
                      "paired_decisions_vs_simple": decision_comparison(groups["adaptive_eviction"], groups["raw_admit_if_fits"]),
                      "component_differences": component_deltas})
    all_groups = {policy: [row for row in rows if row["policy"] == policy] for policy in protocol["policies"]}
    aggregate = {policy: policy_stats(values, [block for block in blocks if block["policy"] == policy]) for policy, values in all_groups.items()}
    aggregate_comparisons = comparisons(all_groups)
    return {
        "source_directory": str(folder), "recorded_commit": commit,
        "artifact_manifest_sha256": digest(folder / "artifact-sha256.json"),
        "pinned_validator_source_sha256": sha256(source).hexdigest(), "validation": validation,
        "device_name": protocol["device_name"], "cuda_version": protocol["cuda_version"],
        "versions": protocol["versions"], "balanced_observed_mix": {
            "weighting": "Raw total session time / started sessions. Each cell contributes 72 started sessions per policy; ratios are calculated after summing times, not averaged.",
            "policies": aggregate, "adaptive_vs_baselines": aggregate_comparisons,
            "fastest_observed_baseline": min(BASELINES, key=lambda policy: aggregate[policy]["mean_full_session_ms"]),
        },
        "cells": cells,
        "record_based_interpretation": {
            "adaptive_lossy_admissions": sum(aggregate["adaptive_eviction"]["actions"].get(name, 0) for name in ("group_int4", "group_int8")),
            "adaptive_minus_simple_native_admissions": aggregate["adaptive_eviction"]["actions"].get("raw_native", 0) - aggregate["raw_admit_if_fits"]["actions"].get("raw_native", 0),
            "adaptive_minus_simple_hits": aggregate["adaptive_eviction"]["hits"] - aggregate["raw_admit_if_fits"]["hits"],
            "cells_with_identical_cuda_allocated_peaks": sum(cell["cuda_allocated_peaks_identical_across_policies"] for cell in cells),
            "roomy_decisions_identical_to_simple_in_every_case_repeat": all(
                cell["paired_decisions_vs_simple"]["same_action_hit_resolution_and_victims"] == cell["paired_decisions_vs_simple"]["paired_started_sessions"]
                for cell in cells if cell["scenario"] == "roomy"),
            "notes": [
                "More admissions need not produce more hits: replacement records show which retained owners are evicted and whether new owners cancel. These are observed outcomes, not information available to admission in advance.",
                "Roomy CPU encode/decode differences can be compared with the full-session difference. This diagnoses the reference-codec overhead; it does not establish an advantage over simple native admission or optimized GPU quantization.",
                "Initial-prefill and generation timers can differ despite identical logical work. Their deltas and per-repeat results are retained so total-time changes are not attributed entirely to policy decisions.",
                "These are equal-work decisions only. Do not reuse their admissions or hit counts as free-generation quality-run decisions.",
            ],
        },
        "cell_counts": {
            "adaptive_mean_faster_than_baseline": {baseline: sum(cell["adaptive_vs_baselines"][baseline]["adaptive_time_reduction_percent"] > 0 for cell in cells) for baseline in BASELINES},
            "adaptive_mean_faster_than_all_baselines": sum(all(cmp["adaptive_time_reduction_percent"] > 0 for cmp in cell["adaptive_vs_baselines"].values()) for cell in cells),
            "adaptive_faster_than_all_baselines_every_repeat": sum(cell["adaptive_beats_all_baselines_in_every_repeat"] for cell in cells),
        },
    }, protocol, fixture


def main():
    reports, protocols, fixtures = {}, {}, {}
    for label, suite in INPUTS.items():
        reports[label], protocols[label], fixtures[label] = analyze(label, suite)
    p100, rtx = protocols["p100"], protocols["rtx5060ti"]
    contexts = []
    for context in fixtures["p100"]["cases_by_context"]:
        left = {row["id"]: row for row in fixtures["p100"]["cases_by_context"][context]}
        right = {row["id"]: row for row in fixtures["rtx5060ti"]["cases_by_context"][context]}
        assert set(left) == set(right)
        contexts.append({"context_target": int(context), "cases": len(left),
                         "identical_prompt_token_arrays": sum(left[name]["tokens"] == right[name]["tokens"] for name in left),
                         "identical_fixed_continuations": sum(left[name]["fixed_continuation"] == right[name]["fixed_continuation"] for name in left),
                         "different_continuation_case_ids": sorted(name for name in left if left[name]["fixed_continuation"] != right[name]["fixed_continuation"])})
    args = lambda protocol: {key: value for key, value in protocol["args"].items() if key not in ("model_dir", "output")}
    result = {
        "evidence_type": "independent_two_system_equal_work_timing_review",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "computation_source": str(Path(__file__).resolve()), "computation_source_sha256": digest(Path(__file__)),
        "scope": {
            "measured": True, "answer_quality_evaluated": False, "inference_rerun": False,
            "timing_definition": "Initial prefill + pause + resume wall time or cancellation work, averaged over all started sessions. Equal-work runs force 24 token steps regardless of EOS.",
            "memory_definition": "Host serialized cache payload, CUDA allocator peaks, and sampled RSS are separate metrics. The pressure budget limits retained host payload, not GPU VRAM.",
            "component_warning": "Components overlap; generation, restore and recomputation are within resume, and export/encode are within pause. Deltas identify recorded overhead, not isolated causal effects.",
            "cross_system_warning": "GPU, CUDA wheel, operating system and recorded commit differ. Host CPU/quota/thread settings are not established as matched. Frozen continuation IDs also differ for one case. This is not a pure GPU comparison and no cross-GPU speedup is inferred.",
            "statistics_warning": "Three repeats reuse 24 generated questions. Balanced mixture is descriptive and not a deployment-weighted claim, significance test, serving throughput, or dollars saved.",
            "quality_warning": "This artifact contains timing-mode action choices only; free-generation choices and answer scoring require their own audit.",
            "integrity_warning": "Each run is independently audited with validator code loaded from its recorded commit. Artifact consistency is not physical-execution attestation.",
        },
        "recorded_controls": {
            "commits_identical": reports["p100"]["recorded_commit"] == reports["rtx5060ti"]["recorded_commit"],
            "recorded_source_hashes_identical": p100["source_sha256"] == rtx["source_sha256"],
            "validator_source_identical": reports["p100"]["pinned_validator_source_sha256"] == reports["rtx5060ti"]["pinned_validator_source_sha256"],
            "workload_arguments_identical_excluding_paths": args(p100) == args(rtx),
            "model_digest_identical": p100["model_identity"]["model_digest"] == rtx["model_identity"]["model_digest"],
            "schedules_identical": p100["schedules"] == rtx["schedules"], "host_payload_budgets_identical": p100["budgets_bytes"] == rtx["budgets_bytes"],
            "fixture_contexts": contexts,
        },
        "systems": reports,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(OUTPUT), "sha256": digest(OUTPUT), "audits": {label: report["validation"] for label, report in reports.items()}}))


if __name__ == "__main__":
    main()
