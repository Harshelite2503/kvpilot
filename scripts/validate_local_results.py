"""Independently validate a completed local KVPilot JSON result archive.

Uses only the Python standard library. Does not import the frozen harness,
model, codecs, cache manager, NumPy or MLX, and never runs model inference.
No rows are filtered. Missing final summary means incomplete, not validated.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("recompute", "raw_native", "group_int4", "adaptive_exact", "adaptive_eviction")
CODECS = ("raw_native", "group_int8", "group_int4")
FILES = ("protocol.json", "fixture.json", "model-provenance.json", "calibration.json",
         "rows.jsonl", "blocks.json", "summary.json")


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"{name}: expected finite nonnegative number")
    return value


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"{name}: expected integer >= {minimum}")
    return value


def close(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6),
            f"{label}: recorded {actual!r} != independently calculated {expected!r}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def strict_json(raw):
    def nonfinite(value):
        raise ValidationError(f"nonfinite JSON value: {value}")
    return json.loads(raw, parse_constant=nonfinite)


def normalize(answer):
    return " ".join(answer.split())


def check_duration_fields(value, path="archive"):
    if isinstance(value, dict):
        for key, item in value.items():
            label = path + "/" + key
            if key.endswith("_ms") or "_ms_per_" in key:
                number(item, label)
            else:
                check_duration_fields(item, label)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            check_duration_fields(item, f"{path}/{index}")


def expected_source_paths(root):
    return sorted({*root.glob("src/kvpilot/*.py"), *root.glob("experiments/**/*.py"),
                   root / "scripts/benchmark_local_kvpilot.py",
                   root / "examples/local_kvpilot_cases.json"})


def candidate_sizes(prefix_tokens, prefix_hash, provenance):
    """Independent byte accounting for the frozen single-sequence Qwen layout.

    Rebuilds native tensor lengths and both JSON envelopes from the archived
    identities. No arrays or encoded payloads are constructed.
    """
    integer(prefix_tokens, "prefix_tokens", 1)
    require(isinstance(prefix_hash, str) and re.fullmatch(r"[0-9a-f]{64}", prefix_hash), "invalid token-prefix digest")
    identity = {"manifest_sha256": provenance["manifest_sha256"],
                "config_digest": provenance["config"]["digest"],
                "tokenizer_digest": provenance["tokenizer"]["digest"],
                "execution": "kvpilot_qwen35_fp16_v1"}
    metadata = {"offset": prefix_tokens, "model": identity, "prefix_hash": prefix_hash,
                "activation_dtype": "float16", "recurrent_dtype": "float32",
                "layout": "kvpilot_qwen35_single_sequence_v1"}
    shapes = {}
    for layer in range(32):
        if layer % 4 == 3:
            for kind in ("k", "v"):
                shapes[f"{layer}.{kind}"] = ([1, 4, prefix_tokens, 256], "<f2", "attention")
        else:
            shapes[f"{layer}.conv"] = ([1, 3, 8192], "<f2", "recurrent")
            shapes[f"{layer}.recurrent"] = ([1, 32, 128, 128], "<f4", "recurrent")
    native_bytes = sum(math.prod(shape) * (4 if dtype == "<f4" else 2) for shape, dtype, _ in shapes.values())
    sizes = {}
    for codec in CODECS:
        records, offset = [], 0
        for name in sorted(shapes):
            shape, dtype, role = shapes[name]
            count = math.prod(shape)
            storage = codec if role == "attention" and codec != "raw_native" else "native"
            if storage == "native":
                length = count * (4 if dtype == "<f4" else 2)
            else:
                inner = {"version": 1, "codec": codec, "dtype": "<f2", "shape": shape, "group_size": 64}
                scale_bytes = math.prod(shape[:-1]) * ((shape[-1] + 63) // 64) * 2
                packed_bytes = count if codec == "group_int8" else (count + 1) // 2
                length = 8 + len(canonical(inner)) + scale_bytes + packed_bytes
            records.append({"name": name, "role": role, "storage": storage, "dtype": dtype,
                            "shape": shape, "offset": offset, "length": length})
            offset += length
        outer = {"version": 1, "codec": codec, "metadata": metadata, "tensors": records}
        sizes[codec] = 8 + len(canonical(outer)) + offset
    return native_bytes, sizes


def independent_summaries(rows, blocks):
    """Aggregate every response once, independently of the benchmark function."""
    controls = {(r["scenario"], r["repeat"], r["case_id"]): r for r in rows if r["policy"] == "recompute"}
    summaries = []
    for scenario, policy in sorted({(r["scenario"], r["policy"]) for r in rows}):
        group = [r for r in rows if (r["scenario"], r["policy"]) == (scenario, policy)]
        matching = [b for b in blocks if (b["scenario"], b["policy"]) == (scenario, policy)]
        paired = [(r, controls[(scenario, r["repeat"], r["case_id"])]) for r in group
                  if (scenario, r["repeat"], r["case_id"]) in controls]
        events = sum(b["event_count"] for b in matching)
        require(events > 0, "zero occupancy-event denominator")
        summaries.append({
            "scenario": scenario, "policy": policy, "responses": len(group),
            "correct": sum(r["correct"] for r in group),
            "token_identical_to_recompute": sum(r["generation"]["tokens"] == control["generation"]["tokens"] for r, control in paired),
            "quality_pairs": len(paired), "hits": sum(r["hit"] for r in group),
            "actions": dict(Counter(r["selected"]["action"] for r in group)),
            "total_pause_ms": sum(r["pause_ms"] for r in group),
            "total_resume_wall_ms": sum(r["resume_wall_ms"] for r in group),
            "pause_plus_resume_ms": sum(r["pause_ms"] + r["resume_wall_ms"] for r in group),
            "total_including_initial_prefill_ms": sum(r["initial_prefill_ms"] + r["pause_ms"] + r["resume_wall_ms"] for r in group),
            "median_resume_ttft_ms": statistics.median(r["resume_ttft_ms"] for r in group),
            "peak_paused_payload_bytes": max(b["peak_paused_payload_bytes"] for b in matching),
            "mean_paused_payload_bytes_after_event": sum(b["byte_event_sum"] for b in matching) / events,
            "evicted_states": sum(b["evicted_states"] for b in matching),
            "peak_mlx_active_bytes": max(b["engine_final"]["mlx_peak_active_bytes"] for b in matching),
        })
    return summaries


def validate_archive(archive: str | Path, *, root: str | Path = ROOT) -> dict:
    archive, root = Path(archive).resolve(), Path(root).resolve()
    report = {"schema_version": 1, "validator": "independent_standard_library_archive_checks_v1",
              "validated_at_utc": datetime.now(timezone.utc).isoformat(), "archive": str(archive),
              "status": "invalid", "valid": False, "checks": [], "rows_filtered": 0,
              "model_executed": False,
              "limitations": [
                  "This checks archived records, exact framing arithmetic and lifecycle consistency; payload contents were not archived and cannot be independently re-decoded here.",
                  "Current source and archived-file hashes establish consistency, not authenticated provenance or a re-execution of measured timings.",
                  "Eviction penalties and pause forecasts lack original event timestamps for full independent numerical replay; allowed actions, selection from recorded costs and resolved-only probability counts are checked.",
                  "Correlated synthetic cases and repeated timings do not establish general quality, statistical noninferiority or production performance.",
              ]}
    missing = [name for name in FILES if not (archive / name).is_file()]
    if missing:
        report.update(status="incomplete", missing_files=missing)
        report["checks"].append({"name": "completed_archive", "passed": False, "detail": "Final summary and all archive files are required; partial results were not validated."})
        return report
    stage = "parse_archive"
    try:
        raw = {name: (archive / name).read_bytes() for name in FILES}
        report["archive_sha256"] = {name: sha256(value).hexdigest() for name, value in raw.items()}
        data = {name: strict_json(value) for name, value in raw.items() if name != "rows.jsonl"}
        lines = raw["rows.jsonl"].splitlines()
        require(all(line.strip() for line in lines), "blank response row")
        rows = [strict_json(line) for line in lines]
        protocol, fixture, blocks = data["protocol.json"], data["fixture.json"], data["blocks.json"]
        summary, provenance = data["summary.json"], data["model-provenance.json"]
        cases = {c["id"]: c for c in fixture["cases"]}
        case_ids, jobs = protocol["cases"], [tuple(job) for job in protocol["jobs"]]
        report.update(row_count=len(rows), block_count=len(blocks), selected_case_count=len(case_ids),
                      expected_row_count=len(jobs) * len(case_ids), expected_block_count=len(jobs))
        report["checks"].append({"name": stage, "passed": True})

        stage = "source_and_fixture_hashes"
        frozen = protocol["source_sha256"]
        current = {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in expected_source_paths(root)}
        require(frozen == current, "current source set or SHA256 differs from frozen protocol")
        require(sha256(raw["fixture.json"]).hexdigest() == protocol["fixture_sha256"], "archived fixture SHA256 mismatch")
        require(len(cases) == len(fixture["cases"]) and len(case_ids) == len(set(case_ids)), "duplicate fixture or selected case IDs")
        require(case_ids and all(case_id in cases for case_id in case_ids), "selected case is absent from archived fixture")
        report["selected_fixture_sha256"] = sha256(canonical([cases[c] for c in case_ids])).hexdigest()
        report["checks"].append({"name": stage, "passed": True, "source_files": len(frozen)})

        stage = "complete_rows_blocks_and_integrity_gate"
        require(summary.get("valid_source_freeze") is True, "final source-freeze completion marker absent")
        require(len(jobs) == len(set(jobs)) and jobs, "duplicate or empty protocol jobs")
        require(all(len(job) == 3 and job[2] in POLICIES for job in jobs), "unknown job policy")
        for repeat, scenario, policy in jobs:
            integer(repeat, "repeat")
            integer(protocol["budgets_bytes"][scenario], "budget", 1)
        report["jobs_without_recompute_control"] = [list(job) for job in jobs if (job[0], job[1], "recompute") not in jobs]
        require(protocol["group_size"] == 4 and protocol["resume_order_per_group"] == [0, 2, 1, 3]
                and protocol["all_sessions_resume"] is True, "unsupported lifecycle protocol")
        expected_order = []
        for job in jobs:
            for start in range(0, len(case_ids), 4):
                group = case_ids[start:start + 4]
                expected_order += [(*job, group[i]) for i in [0, 2, 1, 3] if i < len(group)]
        actual_order = [(r["repeat"], r["scenario"], r["policy"], r["case_id"]) for r in rows]
        require(actual_order == expected_order, "response rows missing, duplicated, reordered or outside declared jobs")
        require([(b["repeat"], b["scenario"], b["policy"]) for b in blocks] == jobs, "block rows differ from declared jobs")
        gate = protocol["integrity_gate"]
        require(gate == data["calibration.json"]["integrity_gate"] and gate["raw_restore_logits_bit_exact"] is True
                and gate["max_abs_logit_difference"] == 0, "raw restore integrity gate failed or disagrees")
        require(protocol["profiles"] == data["calibration.json"]["profiles"], "calibration/profile mismatch")
        require([p["name"] for p in protocol["profiles"]] == list(CODECS), "unexpected codec profile set/order")
        report["checks"].append({"name": stage, "passed": True})

        stage = "quality_timings_candidate_bytes_and_actions"
        for name in ("protocol.json", "calibration.json", "blocks.json", "summary.json"):
            check_duration_fields(data[name], name)
        check_duration_fields(rows, "rows.jsonl")
        prefix_by_case = {}
        for row in rows:
            label = f"{row['repeat']}/{row['scenario']}/{row['policy']}/{row['case_id']}"
            generation = row["generation"]
            require(isinstance(generation["answer"], str) and type(generation["truncated"]) is bool, f"{label}: invalid generation")
            require(type(row["correct"]) is bool and type(row["hit"]) is bool, f"{label}: invalid outcome flags")
            expected_answer = cases[row["case_id"]]["expected_answer"]
            require(row["expected_answer"] == expected_answer, f"{label}: ground truth changed")
            correct = not generation["truncated"] and normalize(generation["answer"]) == normalize(expected_answer)
            require(row["correct"] == correct, f"{label}: incorrect exact-answer score")
            require(isinstance(generation["tokens"], list) and all(type(t) is int and t >= 0 for t in generation["tokens"]), f"{label}: invalid token list")
            require(len(generation["tokens"]) <= protocol["max_tokens"], f"{label}: token limit exceeded")
            for key in ("initial_prefill_ms", "pause_ms", "planning_ms", "export_ms", "encode_ms", "decode_ms",
                        "import_ms", "recompute_ms", "restore_or_recompute_ms", "resume_ttft_ms", "resume_wall_ms"):
                number(row[key], f"{label}/{key}")
            for key in ("ttft_ms", "wall_ms"):
                number(generation[key], f"{label}/generation/{key}")
            close(row["resume_ttft_ms"], row["restore_or_recompute_ms"] + generation["ttft_ms"], label + "/TTFT decomposition")
            require(row["resume_wall_ms"] + 1e-6 >= row["resume_ttft_ms"], label + ": wall time shorter than TTFT")
            require(row["pause_ms"] + 1e-6 >= row["planning_ms"] + row["export_ms"] + row["encode_ms"], label + ": pause shorter than its components")
            require(row["restore_or_recompute_ms"] + 1e-6 >= row["decode_ms"] + row["import_ms"] + row["recompute_ms"], label + ": restore shorter than its components")
            require(row["suffix_tokens"] == protocol["suffix_tokens"], label + ": suffix length changed")
            identity = (row["prefix_tokens"], row["suffix_tokens"], row["prefix_hash"])
            require(prefix_by_case.setdefault(row["case_id"], identity) == identity, label + ": prefix differs across arms")
            raw_bytes, sizes = candidate_sizes(row["prefix_tokens"], row["prefix_hash"], provenance)
            require(row["raw_tensor_bytes"] == raw_bytes and row["exact_candidate_bytes"] == sizes, label + ": state/candidate framing bytes disagree")
            budget = protocol["budgets_bytes"][row["scenario"]]
            alternatives = row["alternatives"]
            allowed = ["recompute"] + [codec for codec in CODECS if sizes[codec] <= budget]
            require([c["action"] for c in alternatives] == allowed, label + ": alternatives omitted, duplicated or unsupported")
            require(row["selected"] in alternatives, label + ": selected action not recorded among alternatives")
            probability = number(row["forecast"]["resume_probability"], label + "/resume probability")
            require(probability <= 1, label + ": invalid resume probability")
            number(row["forecast"]["pause_ms"], label + "/pause forecast")
            for candidate in alternatives:
                action = candidate["action"]
                required_bytes = 0 if action == "recompute" else sizes[action]
                require(type(candidate["stored_bytes"]) is int and candidate["stored_bytes"] == required_bytes, label + ": selected/alternative bytes mismatch")
                for key in ("encode_ms", "offload_ms", "restore_ms", "predicted_resume_ms", "expected_cost_ms", "eviction_cost_ms"):
                    number(candidate[key], label + "/candidate/" + key)
                require(type(candidate["within_resume_slo"]) is bool, label + ": invalid SLO flag")
                if row["policy"] != "adaptive_eviction":
                    require(candidate["eviction_cost_ms"] == 0, label + ": uncharged arm has eviction penalty")
                close(candidate["expected_cost_ms"], candidate["encode_ms"] + candidate["offload_ms"]
                      + probability * candidate["predicted_resume_ms"] + candidate["eviction_cost_ms"], label + "/recorded cost decomposition")
            action = row["selected"]["action"]
            if row["policy"].startswith("adaptive_"):
                feasible = [c for c in alternatives if c["within_resume_slo"]]
                chosen = min(feasible, key=lambda c: c["expected_cost_ms"]) if feasible else min(alternatives, key=lambda c: c["predicted_resume_ms"])
                require(row["selected"] == chosen, label + ": adaptive selection disagrees with recorded costs")
            else:
                require(action == (row["policy"] if row["policy"] in allowed else "recompute"), label + ": fixed baseline action changed")
            if action == "recompute":
                require(row["export_ms"] == row["encode_ms"] == 0, label + ": recompute-only action encoded state")
            if row["hit"]:
                require(row["recompute_ms"] == 0, label + ": hit recomputed state")
            else:
                require(row["decode_ms"] == row["import_ms"] == 0, label + ": miss decoded/imported a payload")
        report["checks"].append({"name": stage, "passed": True, "responses_checked": len(rows)})

        stage = "independent_lifecycle_capacity_and_denominators"
        indexed = {(r["repeat"], r["scenario"], r["policy"], r["case_id"]): r for r in rows}
        for job, block in zip(jobs, blocks):
            resident = OrderedDict()
            used = peak = byte_sum = events = evictions = resolved = 0
            budget = protocol["budgets_bytes"][job[1]]
            for start in range(0, len(case_ids), 4):
                group = case_ids[start:start + 4]
                for case_id in group:
                    row = indexed[(*job, case_id)]
                    close(row["forecast"]["resume_probability"], (resolved + 1) / (resolved + 2), f"{job}/{case_id}/resolved-only prior")
                    amount = row["selected"]["stored_bytes"]
                    victims = []
                    while amount and used + amount > budget and resident:
                        victim, length = resident.popitem(last=False)
                        used -= length
                        victims.append(victim)
                    require(row["victims"] == victims, f"{job}/{case_id}: victims disagree with independent LRU/capacity calculation")
                    evictions += len(victims)
                    if amount:
                        resident[case_id] = amount
                        used += amount
                    require(0 <= used <= budget and row["stored_bytes_after_pause"] == used, f"{job}/{case_id}: pause capacity mismatch")
                    peak, byte_sum, events = max(peak, used), byte_sum + used, events + 1
                for i in [0, 2, 1, 3]:
                    if i >= len(group):
                        continue
                    case_id = group[i]
                    row = indexed[(*job, case_id)]
                    require(row["hit"] == (case_id in resident), f"{job}/{case_id}: hit disagrees with residency")
                    used -= resident.pop(case_id, 0)
                    require(row["stored_bytes_after_resume"] == used, f"{job}/{case_id}: resume capacity mismatch")
                    peak, byte_sum, events, resolved = max(peak, used), byte_sum + used, events + 1, resolved + 1
            require(not resident and used == 0, f"{job}: unresolved final payloads")
            for key, expected in {"peak_paused_payload_bytes": peak, "byte_event_sum": byte_sum,
                                  "event_count": events, "evicted_states": evictions}.items():
                require(type(block[key]) is int and block[key] == expected, f"{job}: block {key} mismatch")
            number(block["block_wall_ms"], "block wall time")
            for key, value in block["engine_final"].items():
                integer(value, f"engine_final/{key}")
        report["checks"].append({"name": stage, "passed": True, "lifecycle_events": 2 * len(rows)})

        stage = "independent_summary_and_control_pairing"
        recomputed = independent_summaries(rows, blocks)
        recorded = summary["summaries"]
        require(len(recorded) == len(recomputed), "summary-cell count mismatch")
        by_cell = {(s["scenario"], s["policy"]): s for s in recorded}
        require(len(by_cell) == len(recorded), "duplicate summary cells")
        for expected in recomputed:
            cell = (expected["scenario"], expected["policy"])
            require(cell in by_cell and set(by_cell[cell]) == set(expected), f"summary fields differ: {cell}")
            for key, value in expected.items():
                if type(value) in (int, float):
                    close(by_cell[cell][key], value, f"summary/{cell}/{key}")
                else:
                    require(by_cell[cell][key] == value, f"summary/{cell}/{key} mismatch")
        report["recomputed_summaries"] = recomputed
        report["checks"].append({"name": stage, "passed": True, "summary_cells": len(recomputed),
                                 "quality_pairs": sum(cell["quality_pairs"] for cell in recomputed)})
        stage = "archive_stable_during_validation"
        require(all((archive / name).read_bytes() == raw[name] for name in FILES), "archive changed during validation")
        require({str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in expected_source_paths(root)} == frozen,
                "frozen source changed during validation")
        report["checks"].append({"name": stage, "passed": True})
        report.update(status="valid", valid=True)
    except (ValidationError, KeyError, TypeError, ValueError, OSError, IndexError) as exc:
        report["checks"].append({"name": stage, "passed": False, "detail": f"{type(exc).__name__}: {exc}"})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="optional new validation JSON file; existing files are refused")
    args = parser.parse_args(argv)
    report = validate_archive(args.archive, root=args.repo_root)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as output:
            output.write(rendered)
    else:
        print(rendered, end="")
    return 0 if report["valid"] else (2 if report["status"] == "incomplete" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
