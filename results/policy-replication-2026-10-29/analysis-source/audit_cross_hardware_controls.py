"""Read-only source, recorded artifact and cross-hardware input audit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import types

WORKSPACE = Path(__file__).resolve().parents[2]
REPO = WORKSPACE / "outputs/kvpilot"
P100 = Path("/Users/harshelite/Downloads/local-p100-admission-20261028T180805Z")
RTX = WORKSPACE / "outputs/rtx5060ti-20261029/extracted/local-rtx5060ti-replication"
OUTPUT = WORKSPACE / "outputs/p100-vs-rtx-20261029/comparability-review.json"
COMMIT = "ca6f6feaaf7faef4b942bd37d78ee29ded99a27a"
POLICIES = ["recompute", "raw_native", "raw_admit_if_fits", "group_int4", "adaptive_exact", "adaptive_eviction"]
COMMON_VERSIONS = {"numpy": "2.2.6", "tokenizers": "0.21.2", "safetensors": "0.5.3", "psutil": "7.0.0"}
MODEL_DIGEST = "c3b57e4a3bcb257287b2b3d5b368115727de1174bcdd4314c548104815769ff0"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True).stdout


def blob(commit, name):
    return git("show", f"{commit}:{name}")


def jobs(contexts, scenarios, repeats, mode):
    values = [{"repeat": repeat, "context_target": target, "scenario": scenario,
               "policy": policy, "timing_mode": mode}
              for repeat in range(repeats) for target in contexts for scenario in scenarios for policy in POLICIES]
    random.Random(20261104).shuffle(values)
    for index, value in enumerate(values):
        value["block_id"] = f"block-{index:04d}"
    return values


def schedule(case_ids, group_size, scenario):
    generator = random.Random(20261103)
    events = []
    for start in range(0, len(case_ids), group_size):
        group = case_ids[start:start + group_size]
        events.extend([["pause", name] for name in group])
        generator.shuffle(group)
        events.extend([["cancel" if scenario == "cancellation" and index % 4 == 0 else "resume", name]
                       for index, name in enumerate(group)])
    return events


def audit():
    assert git("rev-parse", "HEAD").decode().strip() == COMMIT
    all_paths = git("ls-tree", "-r", "--name-only", COMMIT).decode().splitlines()
    source_paths = [name for name in all_paths if
                    (name.endswith(".py") and name.startswith(("src/kvpilot/", "experiments/"))) or
                    name in ("scripts/benchmark_p100_kvpilot.py", "scripts/validate_p100_results.py")]
    sources = {name: hashlib.sha256(blob(COMMIT, name)).hexdigest() for name in source_paths}
    assert len(sources) == 25
    pinned_model_manifest = json.loads(blob(COMMIT, "results/p100-2026-10-28-analysis/model-download-manifest.json"))
    assert pinned_model_manifest["repo"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert pinned_model_manifest["revision"] == "7ae557604adf67be50417f59c2c2f167def9a775"
    # Execute the validator from the pinned Git blob, never from either result directory.
    validator = types.ModuleType("pinned_standard_library_result_validator")
    exec(compile(blob(COMMIT, "scripts/validate_p100_results.py"), "pinned:validate_p100_results.py", "exec"),
         validator.__dict__)
    commits = {name: (folder / "source-commit.txt").read_text().strip() for name, folder in (("p100", P100), ("rtx", RTX))}
    changed = git("diff", "--name-only", commits["p100"], commits["rtx"]).decode().splitlines()
    source_report = {"pinned_reference_commit": COMMIT, "recorded_commits": commits,
                     "commit_labels_identical": len(set(commits.values())) == 1,
                     "recorded_source_file_count": len(sources),
                     "changed_paths_between_commits": changed,
                     "changed_recorded_runner_sources": sorted(set(changed) & set(source_paths)),
                     "pinned_source_sha256": sources,
                     "unrecorded_but_relevant_git_files": {name: {
                         "p100_sha256": hashlib.sha256(blob(commits["p100"], name)).hexdigest(),
                         "rtx_sha256": hashlib.sha256(blob(commits["rtx"], name)).hexdigest(),
                         "identical_git_blobs": blob(commits["p100"], name) == blob(commits["rtx"], name)}
                         for name in ("scripts/run_p100_admission_suite.sh", "scripts/sweep_p100_break_even.py",
                                      "experiments/torch/requirements.txt", "experiments/torch/requirements-p100.txt")},
                     "git_status": {name: (folder / "git-status.txt").read_text().splitlines()
                                    for name, folder in (("p100", P100), ("rtx", RTX))}}
    runs = {}
    source_matches = []
    metadata_packages = {}
    for kind in ("smoke", "full"):
        for mode in ("equal_work", "free_generation"):
            name = f"{kind}-{mode}"
            contexts = [256] if kind == "smoke" else [512, 2048]
            scenarios = ["roomy", "pressure"] if kind == "smoke" else ["roomy", "pressure", "cancellation"]
            cases, repeats, group_size = (4, 1, 4) if kind == "smoke" else (24, 3, 8)
            expected_args = {"device": "cuda", "timing_mode": mode, "contexts": contexts, "cases": cases,
                             "repeats": repeats, "group_size": group_size, "scenarios": scenarios,
                             "max_tokens": 24, "chunk_size": 128, "seed": 20261102,
                             "smoke": kind == "smoke", "allow_unvalidated_lossy": True}
            protocols, fixtures, provenance, validations, per_side = {}, {}, {}, {}, {}
            for side, folder in (("p100", P100), ("rtx", RTX)):
                run = folder / name
                validations[side] = validator.audit(run)
                assert validations[side]["valid"] is True
                assert validations[side] == read(folder / f"{name}-validation.json")
                artifact_manifest = read(run / "artifact-sha256.json")
                membership = {p.name for p in run.iterdir() if p.is_file() and p.name != "artifact-sha256.json"}
                assert membership == set(artifact_manifest), (name, side, "manifest membership")
                assert not any(p.is_symlink() for p in run.rglob("*"))
                protocol = protocols[side] = read(run / "protocol.json")
                fixtures[side] = read(run / "fixture.json")
                provenance[side] = read(run / "model-provenance.json")
                loaded_names = set(provenance[side]["file_sha256"])
                assert loaded_names == set(pinned_model_manifest["files"]) - {"LICENSE"}
                assert provenance[side]["file_sha256"] == {
                    path: pinned_model_manifest["files"][path]["sha256"] for path in loaded_names}
                assert provenance[side]["source_file_bytes"] == {
                    path: pinned_model_manifest["files"][path]["bytes"] for path in loaded_names}
                source_ok = protocol["source_sha256"] == sources
                source_matches.append(source_ok)
                assert source_ok, (name, side, "pinned source hashes")
                args = {key: value for key, value in protocol["args"].items() if key not in ("model_dir", "output")}
                assert args == expected_args
                case_ids = [f"case-{index:04d}" for index in range(cases)]
                assert protocol["cases"] == case_ids and protocol["policies"] == POLICIES
                assert protocol["jobs"] == jobs(contexts, scenarios, repeats, mode)
                assert protocol["schedules"] == {scenario: schedule(case_ids, group_size, scenario) for scenario in scenarios}
                expected_versions = dict(COMMON_VERSIONS, torch="2.7.1+cu126" if side == "p100" else "2.7.1+cu128")
                assert protocol["versions"] == expected_versions
                metadata_packages[side] = expected_versions
                assert protocol["model_identity"]["model_digest"] == MODEL_DIGEST
                per_side[side] = {"validator": validations[side], "artifact_manifest_sha256": digest(run / "artifact-sha256.json"),
                                  "protocol_sha256": digest(run / "protocol.json"), "fixture_sha256": digest(run / "fixture.json"),
                                  "source_matches_pinned_commit": source_ok, "arguments_match_expected": True,
                                  "jobs_and_schedules_match_expected": True}
            a, b = protocols["p100"], protocols["rtx"]
            model_files_equal = provenance["p100"]["file_sha256"] == provenance["rtx"]["file_sha256"]
            assert model_files_equal
            assert provenance["p100"]["source_file_bytes"] == provenance["rtx"]["source_file_bytes"]
            model = {key: value for key, value in a["model_identity"].items() if key != "torch_version"}
            assert model == {key: value for key, value in b["model_identity"].items() if key != "torch_version"}
            by_context = {}
            for context in map(str, contexts):
                left = fixtures["p100"]["cases_by_context"][context]
                right = fixtures["rtx"]["cases_by_context"][context]
                assert [case["id"] for case in left] == [case["id"] for case in right]
                differences, input_differences = [], []
                for x, y in zip(left, right):
                    omitted = {"fixed_continuation", "continuation_preparation_ms"}
                    if {k: v for k, v in x.items() if k not in omitted} != {k: v for k, v in y.items() if k not in omitted}:
                        input_differences.append(x["id"])
                    if x.get("fixed_continuation") != y.get("fixed_continuation"):
                        xt, yt = x.get("fixed_continuation", []), y.get("fixed_continuation", [])
                        differences.append({"case_id": x["id"], "p100_tokens": xt, "rtx_tokens": yt,
                                            "p100_length": len(xt), "rtx_length": len(yt),
                                            "different_zero_based_positions": [i for i in range(max(len(xt), len(yt)))
                                                if (xt[i] if i < len(xt) else None) != (yt[i] if i < len(yt) else None)]})
                assert not input_differences
                by_context[context] = {"case_count": len(left), "all_prompt_text_and_token_inputs_identical": True,
                                       "case_order_identical": True,
                                       "fixed_continuations_applicable": mode == "equal_work",
                                       "identical_fixed_continuations": len(left) - len(differences) if mode == "equal_work" else None,
                                       "fixed_continuation_differences": differences}
            runs[name] = {"sides": per_side, "canonical_arguments": expected_args,
                          "args_identical_except_model_and_output_paths": True,
                          "model_file_hashes_and_sizes_identical": model_files_equal,
                          "model_identity_identical_except_torch_cuda_build": True,
                          "budgets_identical": a["budgets_bytes"] == b["budgets_bytes"],
                          "job_order_and_event_schedules_identical": a["jobs"] == b["jobs"] and a["schedules"] == b["schedules"],
                          "protocol_fields_differing": [key for key in a if a[key] != b.get(key)],
                          "fixture_comparison_by_context": by_context}
    source_report["all_eight_runs_match_pinned_source_hashes"] = all(source_matches)
    source_report["p100_commit_runner_blobs_match_reference"] = all(
        hashlib.sha256(blob(commits["p100"], name)).hexdigest() == value for name, value in sources.items())
    outer = {}
    for side, folder in (("p100", P100), ("rtx", RTX)):
        manifest_path = folder / "suite-artifact-sha256.json"
        if manifest_path.exists():
            manifest = read(manifest_path)
            actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p != manifest_path}
            assert actual == set(manifest)
            assert all(digest(folder / name) == value for name, value in manifest.items())
            outer[side] = {"present": True, "valid": True, "files_checked": len(manifest)}
        else:
            outer[side] = {"present": False, "valid": None,
                           "limitation": "Per-run artifacts validate; suite-level logs, test XML and source-commit label lack an enclosing supplied SHA256 manifest."}
    preflights = {name: read(folder / "preflight.json") for name, folder in (("p100", P100), ("rtx", RTX))}
    rtx_runtime = read(RTX / "runtime.json")
    rtx_legacy = read(RTX / "runtime-legacy-cgroup.json")
    review = {"evidence_type": "independent_cross_hardware_recorded_controls_review",
              "reviewed_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": {"p100": str(P100), "rtx": str(RTX)},
              "all_eight_pinned_artifact_audits_passed": True, "source": source_report,
              "outer_integrity": outer, "runs": runs,
              "model": {"model_digest": MODEL_DIGEST, "loaded_file_sha256": provenance["p100"]["file_sha256"],
                        "loaded_file_bytes": provenance["p100"]["source_file_bytes"],
                        "loaded_model_files_identical_on_both_systems": True,
                        "loaded_file_hashes_and_sizes_match_pinned_repository_manifest": True,
                        "pinned_model_repo": pinned_model_manifest["repo"],
                        "pinned_model_revision": pinned_model_manifest["revision"],
                        "expected_hashes_verified_field": {side: provenance[side]["expected_hashes_verified"] for side in provenance},
                        "note": "Loader reports expected_hashes_verified=false; separate offline model-check logs report manifest verification. The recorded five actually loaded file hashes match across systems."},
              "runtime": {"preflight": preflights, "recorded_direct_package_versions": metadata_packages,
                          "python_3_11_18_on_both": all(v["python"] == "3.11.18" for v in preflights.values()),
                          "p100_runtime_sidecar_supplied": (P100 / "runtime.json").exists(),
                          "p100_missing_comparability_metadata": ["CPU model and effective allocation", "Torch intra/inter-op thread counts",
                              "CPU affinity", "cgroup CPU quota", "TF32 and deterministic/precision defaults",
                              "allocator environment", "GPU driver version", "measured PCIe copy bandwidth",
                              "pytest version and complete transitive dependency snapshot"],
                          "rtx_observed_runtime": rtx_runtime, "rtx_observed_legacy_cgroup": rtx_legacy,
                          "test_log_summaries": {side: (folder / "tests.log").read_text().splitlines()[-1]
                                                 for side, folder in (("p100", P100), ("rtx", RTX))}},
              "conclusions": {
                  "benchmark_implementation_matches": True,
                  "prompt_inputs_policies_schedules_and_requested_work_match": True,
                  "all_frozen_token_sequences_identical": False,
                  "full_equal_work_identical_context_case_sequences": 47,
                  "full_equal_work_total_context_case_sequences": 48,
                  "cross_system_descriptive_comparison_supported_with_caveats": True,
                  "pure_gpu_causal_speedup_supported": False,
                  "within_machine_policy_comparisons_remain_controlled": True,
                  "recommended_claim": "Same recorded benchmark implementation and inputs, with one frozen continuation difference; observed performance compares complete P100 and RTX host/runtime setups, not GPU hardware alone.",
                  "limits": ["One context-512 case has different frozen continuation token IDs, although step counts match.",
                             "Python/direct dependency bases match; CUDA wheels and host platforms differ.",
                             "P100 thread, CPU allocation, precision-default and driver metadata were not captured in the supplied folder.",
                             "Artifact audits verify accounting and consistency, not independent physical execution, quality generalization, statistical significance or serving economics."]}}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(review, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "all_eight_audits_passed": True,
                      "source_hashes_matched": len(sources), "conclusions": review["conclusions"]}, indent=2))


if __name__ == "__main__":
    audit()
