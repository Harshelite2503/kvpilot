"""Real local Qwen inference with KVPilot-owned paused state; no Ollama API.

Run from the repo with .venv-local/bin/python scripts/benchmark_local_kvpilot.py.
Requires the existing local qwen3.5:4b tensor-format weights and Apple Metal.
No model downloads, native cache managers, simulated time sleeps or network calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
from kvpilot.policy import CodecProfile, CostModel, MIB
from experiments.local.state_codec import encode_state, decode_state, encoded_size_from_specs
from experiments.local.store import PausedStateStore

CODECS = ("raw_native", "group_int8", "group_int4")
POLICIES = ("recompute", "raw_native", "group_int4", "adaptive_exact", "adaptive_eviction")


def now_ms():
    return time.perf_counter() * 1000


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def normalized(text):
    return " ".join(text.split())


def source_hashes():
    files = [*ROOT.glob("src/kvpilot/*.py"), *ROOT.glob("experiments/**/*.py"),
             Path(__file__), ROOT / "examples/local_kvpilot_cases.json"]
    return {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def prompt(case):
    return ("Use only the repository below. Answer with only the requested literal value; "
            "no explanation, Markdown, quotes, labels or extra units.\n<repository>\n" +
            case["context"] + "\n</repository>\nQuestion: " + case["question"] + "\nAnswer:")


def state_specs(offset):
    """Shape-only accounting for the constructor-validated Qwen3.5:4b layout."""
    specs, roles = {}, {}
    for i in range(32):
        shapes = {"k": (1, 4, offset, 256), "v": (1, 4, offset, 256)} if (i + 1) % 4 == 0 else {
            "conv": (1, 3, 8192), "recurrent": (1, 32, 128, 128)}
        for kind, shape in shapes.items():
            specs[f"{i}.{kind}"] = (shape, np.dtype("float32" if kind == "recurrent" else "float16"))
            roles[f"{i}.{kind}"] = "attention" if kind in {"k", "v"} else "recurrent"
    return specs, roles


def size_candidates(runner, state):
    specs, roles = state_specs(state.offset)
    metadata = runner.state_metadata(state)
    return {codec: encoded_size_from_specs(specs, roles, metadata, codec) for codec in CODECS}


def engine_memory(mx):
    return {"mlx_active_bytes": mx.get_active_memory(), "mlx_cache_bytes": mx.get_cache_memory(),
            "mlx_peak_active_bytes": mx.get_peak_memory()}


def calibrate(runner, mx, output, chunk_size):
    # Independent text, no evaluated question/answer or policy outcome is used here.
    text = "\n".join(f"Calibration record {i}: service iris uses queue jobs.{i % 7}, port {6000+i}."
                     for i in range(64)) + "\nReturn only the first service name."
    tokens = runner.prompt_tokens(text)
    prefix, suffix = tokens[:-64], tokens[-64:]
    started = now_ms()
    logits, state = runner.prefill(prefix, chunk_size=chunk_size)
    del logits
    prefill_ms = now_ms() - started
    raw_bytes = state.nbytes()
    prefix_hash = state.token_digest
    rows = []
    for repeat in range(3):
        for codec in CODECS[repeat:] + CODECS[:repeat]:
            started = now_ms()
            tensors, roles, metadata = runner.export_state(state)
            export_ms = now_ms() - started
            started = now_ms()
            payload = encode_state(tensors, roles, metadata, codec)
            encode_ms = now_ms() - started
            sizes = size_candidates(runner, state)
            assert len(payload) == sizes[codec]
            del tensors, roles, metadata
            started = now_ms()
            decoded, roles, metadata = decode_state(payload)
            decode_ms = now_ms() - started
            started = now_ms()
            restored = runner.import_state(decoded, roles, metadata, expected_prefix_hash=prefix_hash)
            import_ms = now_ms() - started
            if codec == "raw_native":
                exact = all(np.asarray(state.tensors[k]).tobytes() == decoded[k].tobytes() for k in decoded)
                if not exact:
                    raise RuntimeError("native state failed bitwise roundtrip")
            del decoded, roles, metadata
            restored.release()
            del restored
            rows.append({"repeat": repeat, "codec": codec, "raw_tensor_bytes": raw_bytes,
                         "stored_bytes": len(payload), "export_ms": export_ms,
                         "encode_ms": encode_ms, "decode_ms": decode_ms, "import_ms": import_ms})
            del payload
    # Full-model exact-state hook gate. No original live tensor or NumPy snapshot
    # remains when the saved state is restored and its continuation is executed.
    tensors, roles, metadata = runner.export_state(state)
    raw_payload = encode_state(tensors, roles, metadata, "raw_native")
    del tensors, roles, metadata
    reference_logits = np.array(runner.forward(suffix, state))
    state.release()
    del state
    memory_without_live_state = engine_memory(mx)
    restored = runner.import_state(*decode_state(raw_payload), expected_prefix_hash=prefix_hash)
    del raw_payload
    actual_logits = np.array(runner.forward(suffix, restored))
    gate = {"prefix_tokens": len(prefix), "suffix_tokens": len(suffix),
            "raw_restore_logits_bit_exact": reference_logits.tobytes() == actual_logits.tobytes(),
            "max_abs_logit_difference": float(np.max(np.abs(reference_logits - actual_logits))),
            "memory_after_original_state_release": memory_without_live_state}
    restored.release()
    del restored
    if not gate["raw_restore_logits_bit_exact"]:
        raise RuntimeError(f"exact-state continuation gate failed: {gate}")
    profiles = []
    for codec in CODECS:
        samples = [r for r in rows if r["codec"] == codec]
        profiles.append(CodecProfile(
            codec, raw_bytes / samples[0]["stored_bytes"],
            statistics.median(r["export_ms"] + r["encode_ms"] for r in samples) / (raw_bytes / MIB),
            statistics.median(r["decode_ms"] + r["import_ms"] for r in samples) / (raw_bytes / MIB),
            codec == "raw_native"))
    result = {"prefix_tokens": len(prefix), "prefill_ms": prefill_ms,
              "profiles": [asdict(p) for p in profiles], "samples": rows, "integrity_gate": gate}
    write_json(output / "calibration.json", result)
    return profiles, result


def pause_session(runner, mx, store, case, prepared, chunk_size):
    tokens = prepared["prefix"]
    started = now_ms()
    logits, live = runner.prefill(tokens, chunk_size=chunk_size)
    del logits
    prefill_ms = now_ms() - started
    memory_live = engine_memory(mx)
    raw_bytes, prefix_hash = live.nbytes(), live.token_digest
    pause_started = now_ms()
    sizes = size_candidates(runner, live)
    plan = store.decide(case["id"], raw_bytes, prefill_ms, "repository", now_ms(), sizes)
    planning_ms = now_ms() - pause_started
    export_ms = encode_ms = 0.0
    payload = None
    if plan.selected.action != "recompute":
        started = now_ms()
        tensors, roles, metadata = runner.export_state(live)
        export_ms = now_ms() - started
        started = now_ms()
        payload = encode_state(tensors, roles, metadata, plan.selected.action)
        encode_ms = now_ms() - started
        del tensors, roles, metadata
    store.commit(plan, payload, timestamp_ms=now_ms())
    del payload
    live.release()
    assert not live.tensors and live.offset == 0
    del live
    pause_ms = now_ms() - pause_started
    return {"case_id": case["id"], "prefix_tokens": len(tokens),
            "suffix_tokens": len(prepared["suffix"]), "prefix_hash": prefix_hash,
            "raw_tensor_bytes": raw_bytes, "exact_candidate_bytes": sizes,
            "initial_prefill_ms": prefill_ms, "pause_ms": pause_ms,
            "planning_ms": planning_ms, "export_ms": export_ms, "encode_ms": encode_ms,
            "selected": asdict(plan.selected), "alternatives": [asdict(c) for c in plan.alternatives],
            "forecast": asdict(plan.forecast), "victims": list(plan.victims), "reason": plan.reason,
            "stored_bytes_after_pause": store.used_bytes,
            "engine_with_live_state": memory_live, "engine_after_release": engine_memory(mx)}


def resume_session(runner, mx, store, case, prepared, row, chunk_size, max_tokens):
    started = now_ms()
    payload = store.popresume(case["id"], started)
    hit = payload is not None
    decode_ms = import_ms = recompute_ms = 0.0
    if hit:
        before = now_ms()
        tensors, roles, metadata = decode_state(payload)
        decode_ms = now_ms() - before
        before = now_ms()
        live = runner.import_state(tensors, roles, metadata, expected_prefix_hash=row["prefix_hash"])
        import_ms = now_ms() - before
        del tensors, roles, metadata, payload
    else:
        del payload
        before = now_ms()
        logits, live = runner.prefill(prepared["prefix"], chunk_size=chunk_size)
        del logits
        recompute_ms = now_ms() - before
        if live.token_digest != row["prefix_hash"]:
            raise RuntimeError("recomputed token prefix changed")
    restore_ms = now_ms() - started
    answer, live = runner.generate(prepared["suffix"], state=live, max_tokens=max_tokens, chunk_size=chunk_size)
    memory_resumed = engine_memory(mx)
    live.release()
    del live
    row.update({"hit": hit, "decode_ms": decode_ms, "import_ms": import_ms,
                "recompute_ms": recompute_ms, "restore_or_recompute_ms": restore_ms,
                "resume_ttft_ms": restore_ms + answer["ttft_ms"],
                "resume_wall_ms": now_ms() - started, "generation": answer,
                "expected_answer": case["expected_answer"],
                "correct": not answer["truncated"] and normalized(answer["answer"]) == normalized(case["expected_answer"]),
                "stored_bytes_after_resume": store.used_bytes, "engine_resumed": memory_resumed})
    return row


def summarize(rows, blocks):
    result = []
    for scenario in sorted({r["scenario"] for r in rows}):
        for policy in POLICIES:
            subset = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            if not subset:
                continue
            matching = [b for b in blocks if b["scenario"] == scenario and b["policy"] == policy]
            controls = {(r["repeat"], r["case_id"]): r for r in rows
                        if r["scenario"] == scenario and r["policy"] == "recompute"}
            pairs = [(r, controls[(r["repeat"], r["case_id"])]) for r in subset
                     if (r["repeat"], r["case_id"]) in controls]
            result.append({"scenario": scenario, "policy": policy, "responses": len(subset),
                           "correct": sum(r["correct"] for r in subset),
                           "token_identical_to_recompute": sum(r["generation"]["tokens"] == c["generation"]["tokens"] for r, c in pairs),
                           "quality_pairs": len(pairs), "hits": sum(r["hit"] for r in subset),
                           "actions": dict(Counter(r["selected"]["action"] for r in subset)),
                           "total_pause_ms": sum(r["pause_ms"] for r in subset),
                           "total_resume_wall_ms": sum(r["resume_wall_ms"] for r in subset),
                           "pause_plus_resume_ms": sum(r["pause_ms"] + r["resume_wall_ms"] for r in subset),
                           "total_including_initial_prefill_ms": sum(r["initial_prefill_ms"] + r["pause_ms"] + r["resume_wall_ms"] for r in subset),
                           "median_resume_ttft_ms": statistics.median(r["resume_ttft_ms"] for r in subset),
                           "peak_paused_payload_bytes": max(b["peak_paused_payload_bytes"] for b in matching),
                           "mean_paused_payload_bytes_after_event": sum(b["byte_event_sum"] for b in matching) / sum(b["event_count"] for b in matching),
                           "evicted_states": sum(b["evicted_states"] for b in matching),
                           "peak_mlx_active_bytes": max(b["engine_final"]["mlx_peak_active_bytes"] for b in matching)})
    return result


def main():
    entry_hashes = source_hashes()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/qwen-kvpilot-v1")
    parser.add_argument("--cases", type=Path, default=ROOT / "examples/local_kvpilot_cases.json")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--scenarios", nargs="+", choices=("roomy", "pressure"), default=["roomy", "pressure"])
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1 or args.chunk_size < 1:
        parser.error("repeats, max-tokens and chunk-size must be positive")
    if args.case_limit is not None and args.case_limit < 1:
        parser.error("case-limit must be positive")
    if len(set(args.scenarios)) != len(args.scenarios) or len(set(args.policies)) != len(args.policies):
        parser.error("scenarios and policies must not contain duplicates")
    if args.output.exists():
        parser.error("output already exists; use a new run directory to preserve all evidence")
    fixture_bytes = args.cases.read_bytes()
    cases = json.loads(fixture_bytes)["cases"]
    if args.case_limit is not None:
        cases = cases[:args.case_limit]
    if not cases:
        parser.error("no cases")
    if len({c["id"] for c in cases}) != len(cases):
        parser.error("case ids must be unique")
    import mlx.core as mx
    from experiments.local.qwen_runner import QwenRunner, LiveState
    args.output.mkdir(parents=True)
    (args.output / "fixture.json").write_bytes(fixture_bytes)
    print("Loading existing local text weights; no inference server or model download.", flush=True)
    runner = QwenRunner()
    write_json(args.output / "model-provenance.json", runner.provenance)
    warmup, live = runner.generate(runner.prompt_tokens("What is 2 + 2? Answer only the number."), max_tokens=8)
    live.release()
    del live
    print("Warmup:", warmup["answer"], flush=True)
    profiles, calibration = calibrate(runner, mx, args.output, args.chunk_size)
    prepared = {}
    sizes = []
    for case in cases:
        tokens = runner.prompt_tokens(prompt(case))
        if len(tokens) <= 64:
            raise ValueError("each prompt must have a prefix and 64-token continuation")
        prepared[case["id"]] = {"prefix": tokens[:-64], "suffix": tokens[-64:]}
        sizes.append(size_candidates(runner, LiveState(offset=len(tokens)-64)))
    # Deliberately controlled four-session occupancy boundaries, fixed before
    # evaluating answers/timing. This is not a representative production mix.
    budgets = {"roomy": 4 * max(s["raw_native"] for s in sizes) + MIB,
               "pressure": 3 * max(s["group_int4"] for s in sizes) + MIB}
    jobs = [(repeat, scenario, policy) for repeat in range(args.repeats)
            for scenario in args.scenarios for policy in args.policies]
    random.Random(20261027).shuffle(jobs)
    if source_hashes() != entry_hashes or args.cases.read_bytes() != fixture_bytes:
        raise RuntimeError("source or fixture changed during setup/calibration")
    protocol = {"evidence": "measured_real_local_kvpilot_state", "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": entry_hashes, "fixture_sha256": sha256(fixture_bytes).hexdigest(),
                "platform": platform.platform(), "machine": platform.machine(),
                "versions": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-metal", "numpy", "tokenizers")},
                "cases": [c["id"] for c in cases], "jobs": jobs, "budgets_bytes": budgets,
                "profiles": [asdict(p) for p in profiles], "chunk_size": args.chunk_size,
                "max_tokens": args.max_tokens, "suffix_tokens": 64, "group_size": 4,
                "resume_order_per_group": [0, 2, 1, 3], "all_sessions_resume": True,
                "timing": "synchronized local wall time; no simulated delay; initial prefill reported separately",
                "cost_model": "calibrated export+encode and decode+import; transfer term effectively zero (bandwidth=1e30); no storage penalty; no binding latency target; Beta(1,1) resolved-only history",
                "memory": "paused payload byte lengths, including exact recurrent state+metadata; excludes transient CPU buffers, active tensors, weights, Python overhead; MLX active allocations separately",
                "quality": "case-sensitive whitespace-only exact match; truncation fails; greedy token agreement to recompute; selected correlated controlled development cases, not a held-out production dataset",
                "execution": "own Qwen text forward, FP16 activation/BF16 checkpoint conversion, FP32 recurrent state, NVFP4 weights; not native Ollama numerical equivalence",
                "warmup": warmup, "integrity_gate": calibration["integrity_gate"]}
    write_json(args.output / "protocol.json", protocol)
    print("Frozen protocol:", args.output / "protocol.json", flush=True)
    rows, blocks = [], []
    with (args.output / "rows.jsonl").open("w", buffering=1) as row_file:
        for repeat, scenario, policy in jobs:
            mx.clear_cache()
            mx.reset_peak_memory()
            model = CostModel(1e30, budgets[scenario], resume_slo_ms=1e9, allow_unvalidated_lossy=True)
            store = PausedStateStore(policy, profiles, model, default_pause_ms=30000)
            block_started = now_ms()
            print(f"RUN r{repeat+1} {scenario} {policy}", flush=True)
            for start in range(0, len(cases), 4):
                group = cases[start:start+4]
                paused = {c["id"]: pause_session(runner, mx, store, c, prepared[c["id"]], args.chunk_size) for c in group}
                for index in (0, 2, 1, 3):
                    if index >= len(group):
                        continue
                    case = group[index]
                    row = resume_session(runner, mx, store, case, prepared[case["id"]], paused.pop(case["id"]), args.chunk_size, args.max_tokens)
                    row.update({"repeat": repeat, "scenario": scenario, "policy": policy})
                    rows.append(row)
                    row_file.write(json.dumps(row, allow_nan=False) + "\n")
                    print(f"  {case['id']}: {row['selected']['action']} hit={row['hit']} correct={row['correct']} TTFT={row['resume_ttft_ms']:.0f}ms", flush=True)
            if store.used_bytes or store.paused_sessions:
                raise RuntimeError("finished block retained unresolved state")
            blocks.append({"repeat": repeat, "scenario": scenario, "policy": policy,
                           "block_wall_ms": now_ms()-block_started,
                           "peak_paused_payload_bytes": store.peak_bytes, "byte_event_sum": store.byte_event_sum,
                           "event_count": store.event_count, "evicted_states": store.evicted_states,
                           "engine_final": engine_memory(mx)})
            write_json(args.output / "blocks.json", blocks)
    if source_hashes() != protocol["source_sha256"] or args.cases.read_bytes() != fixture_bytes:
        raise RuntimeError("experiment source or fixture changed after freeze; preserve run as invalid")
    summary = summarize(rows, blocks)
    write_json(args.output / "summary.json", {"valid_source_freeze": True, "summaries": summary})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
