"""Independent standard-library audit of a completed owned-cache Torch archive.

Checks frozen artifact hashes, complete planned sessions/events, actual payload
residency/evictions, answer scoring, paired baselines and every summary metric.
This validates record consistency, not physical execution or model correctness.
No benchmark, tensor engine, codec, cache manager or policy imports are used.
"""
from __future__ import annotations
import argparse
from collections import Counter, OrderedDict
from hashlib import sha256
import json
import math
import struct
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def finite(value, name):
    require(type(value) in (float, int) and math.isfinite(value) and value >= 0, f"invalid measured {name}")


def count(value, name):
    require(type(value) is int and value >= 0, f"invalid byte/count {name}")


def tokens(value, name, vocab_size=2 ** 32):
    require(isinstance(value, list) and all(type(token) is int and 0 <= token < vocab_size for token in value),
            f"invalid token sequence: {name}")


def equal(actual, expected, location):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"keys differ: {location}")
        for key in expected:
            equal(actual[key], expected[key], f"{location}.{key}")
    elif type(expected) is float:
        require(type(actual) in (float, int) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-7), f"value differs: {location}")
    else:
        require(type(actual) is type(expected) and actual == expected, f"value differs: {location}")


def audit(folder):
    folder = Path(folder)
    manifest = read(folder / "artifact-sha256.json")
    required = {"protocol.json", "fixture.json", "model-provenance.json", "calibration.json",
                "rows.jsonl", "events.jsonl", "blocks.json", "summary.json", "status.json"}
    require(required <= set(manifest), "missing required artifact hashes")
    for name, digest in manifest.items():
        require(Path(name).name == name and name not in ("", ".", ".."), "unsafe artifact filename")
        require(sha256((folder / name).read_bytes()).hexdigest() == digest, f"artifact hash mismatch: {name}")
    protocol, fixture = read(folder / "protocol.json"), read(folder / "fixture.json")
    summary, status = read(folder / "summary.json"), read(folder / "status.json")
    calibration = read(folder / "calibration.json")
    rows = [json.loads(line) for line in (folder / "rows.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    blocks = read(folder / "blocks.json")
    require(status["completed"] is True and status["stage"] == "complete", "incomplete run")
    require(summary["complete"] is True and summary["valid_source_freeze"] is True, "invalid completion/source freeze")
    require(summary["quality_validated"] is False, "archive must not certify general answer quality")
    require(sha256((folder / "fixture.json").read_bytes()).hexdigest() == protocol["fixture_sha256"], "fixture not frozen")
    schema = protocol["schema_version"]
    require(type(schema) is int and schema in (1, 2), "unsupported protocol schema")
    arguments = protocol["args"]
    mode = protocol.get("timing_mode", "free_generation")
    require(mode in ("free_generation", "equal_work"), "unknown timing mode")
    vocab_size = 2 ** 32
    policies = ["recompute", "raw_native", "group_int4", "adaptive_exact", "adaptive_eviction"]
    if schema == 2:
        policies.insert(2, "raw_admit_if_fits")
        require(protocol["timing_mode"] == arguments["timing_mode"] == summary["timing_mode"],
                "protocol/args/summary timing mode mismatch")
        require(fixture["timing_mode"] == mode, "fixture timing mode mismatch")
        source = ("fresh native recompute greedy tokens, EOS ignored, frozen before calibration and policy measurements"
                  if mode == "equal_work" else None)
        require(protocol["continuation_source"] == fixture["continuation_source"] == source,
                "fixed continuation provenance mismatch")
        vocab_size = protocol["vocab_size"]
        require(type(vocab_size) is int and 0 < vocab_size <= 2 ** 32, "invalid vocabulary size")
        for record in protocol["jobs"] + rows + events + blocks:
            require(record["timing_mode"] == mode, "record timing mode mismatch")
    else:
        require(mode == "free_generation", "legacy schema permits only free generation")
        for record in [arguments, summary] + protocol["jobs"] + rows + events + blocks:
            require(record.get("timing_mode", "free_generation") == mode, "legacy timing mode mismatch")
    require(protocol["policies"] == policies, "incomplete comparison arms")
    require(set(calibration) == set(fixture["cases_by_context"]), "calibration context mismatch")
    for context, value in calibration.items():
        require(value["integrity_gate"] == {"raw_restore_logits_bit_exact": True, "max_abs_logit_difference": 0.0,
                                             "original_live_cache_released": True}, "native continuation integrity gate failed")
        profiles = {p["name"]: p for p in value["profiles"]}
        require(set(profiles) == {"raw_native", "group_int8", "group_int4"}, "missing calibration profiles")
        for name, profile in profiles.items():
            require(profile["lossless"] is (name == "raw_native"), "invalid lossless claim")
    jobs = {j["block_id"]: j for j in protocol["jobs"]}
    expected_jobs = {(repeat, context, scenario, policy) for repeat in range(arguments["repeats"])
                     for context in arguments["contexts"] for scenario in arguments["scenarios"]
                     for policy in protocol["policies"]}
    actual_jobs = [(j["repeat"], j["context_target"], j["scenario"], j["policy"]) for j in protocol["jobs"]]
    require(len(actual_jobs) == len(set(actual_jobs)) and set(actual_jobs) == expected_jobs, "missing, duplicate or extra comparison jobs")
    require(len(protocol["cases"]) == arguments["cases"] == len(set(protocol["cases"])), "case count mismatch")
    require(set(fixture["cases_by_context"]) == {str(c) for c in arguments["contexts"]}, "fixture context mismatch")
    for case_list in fixture["cases_by_context"].values():
        require([c["id"] for c in case_list] == protocol["cases"], "fixture case IDs mismatch")
        if schema == 2:
            for case in case_list:
                for key in ("prefix", "suffix", "tokens"):
                    tokens(case[key], f"fixture.{key}", vocab_size)
        if mode == "equal_work":
            require(type(arguments["max_tokens"]) is int and arguments["max_tokens"] > 0,
                    "invalid fixed continuation length")
            for case in case_list:
                tokens(case["fixed_continuation"], "fixed_continuation", vocab_size)
                require(len(case["fixed_continuation"]) == arguments["max_tokens"],
                        "fixed continuation length mismatch")
                finite(case["continuation_preparation_ms"], "continuation_preparation_ms")
    require(len(jobs) == len(protocol["jobs"]) == len(blocks) == status["completed_blocks"], "block count mismatch")
    require(len({b["block_id"] for b in blocks}) == len(blocks), "duplicate blocks")
    row_map = {(r["block_id"], r["case_id"]): r for r in rows}
    require(len(row_map) == len(rows), "duplicate response/session rows")
    require(set(row_map) == {(j, c) for j in jobs for c in protocol["cases"]}, "missing or extra session rows")
    require(set(e["block_id"] for e in events) == set(jobs), "event block mismatch")
    for block in blocks:
        job = jobs[block["block_id"]]
        for key, value in job.items():
            require(block[key] == value, "block/job identity mismatch")
        context, scenario = str(job["context_target"]), job["scenario"]
        budget = protocol["budgets_bytes"][context][scenario]
        require(block["budget_bytes"] == budget, "budget mismatch")
        planned = protocol["schedules"][scenario]
        actual = [e for e in events if e["block_id"] == job["block_id"]]
        require([(e["event"], e["case_id"]) for e in actual] == [tuple(p) for p in planned], "schedule mismatch")
        residents, unresolved = OrderedDict(), set()
        peak = event_sum = evicted = 0
        for index, e in enumerate(actual):
            require(e["sequence"] == index, "event sequence mismatch")
            for key, value in job.items():
                require(e[key] == value, "event/job identity mismatch")
            row = row_map[(job["block_id"], e["case_id"])]
            for key, value in job.items():
                require(row[key] == value, "row/job identity mismatch")
            name = e["case_id"]
            if e["event"] == "pause":
                require(name not in unresolved, "duplicate pause")
                unresolved.add(name)
                action = row["selected_prediction"]["action"]
                size = row["serialized_payload_bytes"]
                count(size, "serialized_payload_bytes")
                require(e["action"] == action and e["payload_bytes"] == size, "pause payload/action mismatch")
                if row["policy"] == "raw_admit_if_fits":
                    native_size = row["exact_candidate_bytes"]["raw_native"]
                    count(native_size, "exact native candidate bytes")
                    require(native_size > 0, "empty native candidate")
                    expected_action = "raw_native" if native_size <= budget - sum(residents.values()) else "recompute"
                    require(action == expected_action, "admit-if-fits did not follow currently unused capacity")
                    require(e["victims"] == row["victims"] == [], "admit-if-fits evicted an owner")
                if action == "recompute":
                    require(size == 0 and e["victims"] == row["victims"] == [], "recompute retained payload or evicted cache")
                else:
                    require(action in ("raw_native", "group_int8", "group_int4"), "unknown action")
                    require(0 < size <= budget and size == row["exact_candidate_bytes"][action], "incorrect actual encoded length")
                    expected_victims = []
                    while sum(residents.values()) + size > budget:
                        victim, _ = residents.popitem(last=False)
                        expected_victims.append(victim)
                    require(e["victims"] == row["victims"] == expected_victims, "incorrect eviction accounting")
                    evicted += len(expected_victims)
                    residents[name] = size
                require(row["payload_bytes_after_pause"] == sum(residents.values()), "pause byte count mismatch")
            else:
                require(name in unresolved, "resolve without pause")
                unresolved.remove(name)
                size = residents.pop(name, 0)
                require(row["resolution"] == e["event"], "resolution mismatch")
                if e["event"] == "resume":
                    require(e["hit"] is (size > 0) and row["hit"] is (size > 0), "incorrect cache hit")
                    require(e["payload_bytes"] == row["restored_payload_bytes"] == size, "restore byte mismatch")
                require(row["payload_bytes_after_resolution"] == sum(residents.values()), "resolution byte mismatch")
            used = sum(residents.values())
            require(e["used_payload_bytes"] == used and used <= budget, "residency budget violation")
            peak, event_sum = max(peak, used), event_sum + used
        require(not residents and not unresolved, "unresolved cache at block end")
        require((block["peak_paused_payload_bytes"], block["byte_event_sum"], block["event_count"], block["evicted_states"])
                == (peak, event_sum, len(actual), evicted), "block accounting mismatch")
    for row in rows:
        case = next(c for c in fixture["cases_by_context"][str(row["context_target"])] if c["id"] == row["case_id"])
        require(row["prefix_tokens"] == len(case["prefix"]) and row["suffix_tokens"] == len(case["suffix"]), "token length mismatch")
        require(case["tokens"] == case["prefix"] + case["suffix"], "fixture token split mismatch")
        digest = sha256(b"kvpilot-token-v1").hexdigest()
        for token in case["prefix"]:
            require(type(token) is int and 0 <= token < 2 ** 32, "invalid fixture token")
            digest = sha256(bytes.fromhex(digest) + struct.pack("<I", token)).hexdigest()
        require(row["prefix_hash"] == digest, "prefix identity mismatch")
        require(row["original_live_cache_released"] is True, "original cache retained")
        for key in ("raw_tensor_bytes", "exported_tensor_bytes", "serialized_payload_bytes", "payload_bytes_after_pause", "payload_bytes_after_resolution"):
            count(row[key], key)
        require(row["raw_tensor_bytes"] > 0, "empty model cache")
        for key in ("initial_prefill_ms", "pause_ms", "planning_ms", "export_ms", "encode_ms"):
            finite(row[key], key)
        require(row["planning_ms"] + row["export_ms"] + row["encode_ms"] <= row["pause_ms"] + 1e-6, "pause omits measured work")
        action = row["selected_prediction"]["action"]
        require(row["exported_tensor_bytes"] == (0 if action == "recompute" else row["raw_tensor_bytes"]), "export bytes confused with compressed payload")
        if row["policy"] in ("recompute", "raw_native", "group_int4"):
            require(action in (row["policy"], "recompute"), "fixed policy used different codec")
        if row["resolution"] == "resume":
            for key in ("imported_tensor_bytes", "restored_payload_bytes"):
                count(row[key], key)
            for key in ("decode_ms", "import_ms", "recompute_ms", "restore_or_recompute_ms", "resume_ttft_ms", "resume_wall_ms"):
                finite(row[key], key)
            require(row["decode_ms"] + row["import_ms"] + row["recompute_ms"] <= row["restore_or_recompute_ms"] + 1e-6, "restore omits measured work")
            require(row["resume_ttft_ms"] <= row["resume_wall_ms"] + 1e-6, "TTFT exceeds resume duration")
            require(row["imported_tensor_bytes"] == (row["raw_tensor_bytes"] if row["hit"] else 0), "incorrect native import byte count")
            generation = row["generation"]
            if schema == 2:
                tokens(generation["tokens"], "generation.tokens", vocab_size)
            finite(generation["ttft_ms"], "generation.ttft_ms")
            finite(generation["wall_ms"], "generation.wall_ms")
            require(math.isclose(row["resume_ttft_ms"], row["restore_or_recompute_ms"] + generation["ttft_ms"], rel_tol=1e-9, abs_tol=1e-7), "TTFT omits restore/recompute time")
            require(row["restore_or_recompute_ms"] + generation["wall_ms"] <= row["resume_wall_ms"] + 1e-6, "resume omits generation time")
            if mode == "equal_work":
                require(generation["mode"] == mode, "generation timing mode mismatch")
                tokens(generation["predicted_tokens"], "generation.predicted_tokens", vocab_size)
                require(generation["tokens"] == case["fixed_continuation"], "fixed continuation changed during measurement")
                require(len(generation["predicted_tokens"]) == len(case["fixed_continuation"]),
                        "predicted token count differs from fixed work")
                require(isinstance(generation["per_token_ms"], list) and
                        len(generation["per_token_ms"]) == len(case["fixed_continuation"]),
                        "per-token timing count differs from fixed work")
                for elapsed in generation["per_token_ms"]:
                    finite(elapsed, "per_token_ms")
                finite(generation["first_token_ms"], "first_token_ms")
                finite(generation["total_ms"], "total_ms")
                equal(generation["first_token_ms"], float(generation["ttft_ms"]), "first-token timing alias")
                equal(generation["total_ms"], float(generation["wall_ms"]), "total timing alias")
                equal(generation["per_token_ms"][0], float(generation["first_token_ms"]), "first-token timing")
                equal(sum(generation["per_token_ms"]), float(generation["total_ms"]), "per-token timing sum")
                require(generation["truncated"] is False, "fixed work was truncated")
                require(row["expected_answer"] is None and row["correct"] is None and "answer" not in generation,
                        "equal-work timing fabricated an answer-quality observation")
            else:
                require(generation.get("mode", "free_generation") == mode, "generation timing mode mismatch")
                require(row["expected_answer"] == case["expected_answer"], "ground truth changed")
                correct = not generation["truncated"] and " ".join(generation["answer"].split()) == " ".join(case["expected_answer"].split())
                require(row["correct"] is correct, "incorrect answer scoring")
        else:
            finite(row["cancel_ms"], "cancel_ms")
            require("correct" not in row and "generation" not in row, "cancellation fabricated an answer")
    expected = []
    controls = {(r["context_target"], r["scenario"], r["repeat"], r["case_id"]): r for r in rows
                if r["policy"] == "recompute" and r["resolution"] == "resume"}
    for context, scenario, policy in sorted({(r["context_target"], r["scenario"], r["policy"]) for r in rows}):
        subset = [r for r in rows if (r["context_target"], r["scenario"], r["policy"]) == (context, scenario, policy)]
        answered = [r for r in subset if r["resolution"] == "resume"]
        pairs = [(r, controls[(context, scenario, r["repeat"], r["case_id"])]) for r in answered]
        selected_blocks = [b for b in blocks if (b["context_target"], b["scenario"], b["policy"]) == (context, scenario, policy)]
        total = sum(r["initial_prefill_ms"] + r["pause_ms"] + (r["resume_wall_ms"] if r["resolution"] == "resume" else r["cancel_ms"]) for r in subset)
        service = total - sum(r["initial_prefill_ms"] for r in subset)
        ttft = sorted(r["resume_ttft_ms"] for r in answered)
        allocated = [b["memory"]["after"].get("device_peak_allocated_bytes") for b in selected_blocks]
        expected.append({"context_target": context, "scenario": scenario, "policy": policy,
            "sessions": len(subset), "responses": len(answered), "cancellations": len(subset) - len(answered),
            "distinct_answered_cases": len({r["case_id"] for r in answered}),
            "correct": sum(r["correct"] for r in answered) if mode == "free_generation" else None,
            "token_identical_to_recompute": sum(r["generation"]["tokens"] == c["generation"]["tokens"] for r, c in pairs) if mode == "free_generation" else None,
            "quality_regressions_vs_recompute": sum(c["correct"] and not r["correct"] for r, c in pairs) if mode == "free_generation" else None,
            "quality_improvements_vs_recompute": sum(r["correct"] and not c["correct"] for r, c in pairs) if mode == "free_generation" else None,
            "truncated": sum(r["generation"]["truncated"] for r in answered) if mode == "free_generation" else None,
            "hits": sum(r["hit"] for r in answered),
            "actions": dict(Counter(r["selected_prediction"]["action"] for r in subset)),
            "total_service_ms_including_canceled_pauses": service,
            "total_ms_including_initial_prefill_and_cancellations": total, "mean_total_ms_per_started_session": total / len(subset),
            "median_resume_ttft_ms": statistics.median(ttft) if ttft else None,
            "p95_resume_ttft_ms_descriptive": ttft[math.ceil(.95 * len(ttft)) - 1] if ttft else None,
            "peak_paused_payload_bytes": max(b["peak_paused_payload_bytes"] for b in selected_blocks),
            "sampled_peak_process_rss_bytes": max(b["memory"]["sampled_peak_process_rss_bytes"] for b in selected_blocks),
            "peak_cuda_allocated_bytes": max(allocated) if all(a is not None for a in allocated) else None,
            "exported_tensor_bytes": sum(r["exported_tensor_bytes"] for r in subset),
            "imported_tensor_bytes": sum(r.get("imported_tensor_bytes", 0) for r in subset),
            "output_tokens": sum(len(r["generation"]["tokens"]) for r in answered) if mode == "free_generation" else None})
        if mode == "equal_work":
            expected[-1].update({
                "forced_token_steps": sum(len(r["generation"]["tokens"]) for r in answered),
                "predicted_tokens_identical_to_recompute": sum(
                    r["generation"]["predicted_tokens"] == c["generation"]["predicted_tokens"] for r, c in pairs),
            })
    require(len(summary["summaries"]) == len(expected), "summary count mismatch")
    for actual, wanted in zip(summary["summaries"], expected):
        equal(actual, wanted, f"summary {wanted['context_target']}/{wanted['scenario']}/{wanted['policy']}")
    return {"valid": True, "sessions_checked": len(rows), "events_checked": len(events), "blocks_checked": len(blocks),
            "scope": "independent artifact/accounting/score audit; does not rerun inference or prove performance/quality generalization"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.directory)
    except (KeyError, TypeError, ValueError, OSError, StopIteration) as exc:
        result = {"valid": False, "error": str(exc)}
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
