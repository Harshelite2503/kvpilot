"""Measured, sequential owned-cache experiment; never an inference service.

CPU codecs operate AFTER a raw device-to-host export. The transfer byte counters
therefore count native tensor bytes, not the smaller serialized host payload.
Timing includes synchronization and all policy/codec work, with no fake delays.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import statistics
import threading
import time

import numpy as np

from experiments.local.state_codec import encode_state, decode_state, encoded_size_from_specs
from experiments.local.store import PausedStateStore
from kvpilot.policy import CodecProfile, CostModel, MIB

ROOT = Path(__file__).resolve().parents[2]
CODECS = ("raw_native", "group_int8", "group_int4")
POLICIES = ("recompute", "raw_native", "group_int4", "adaptive_exact", "adaptive_eviction")
SCENARIOS = ("roomy", "pressure", "cancellation")


def now_ms():
    return time.perf_counter() * 1000


def write_json(path, value):
    """Replace complete JSON atomically; an interrupted run retains its records."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def normalized(text):
    return " ".join(text.split())


def source_hashes():
    paths = [*ROOT.glob("src/kvpilot/**/*.py"), *ROOT.glob("experiments/**/*.py"),
             ROOT / "scripts/benchmark_p100_kvpilot.py", ROOT / "scripts/validate_p100_results.py"]
    return {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def make_cases(count, seed):
    """Controlled generated tasks, NOT a standardized benchmark or secret holdout.

    The seed and the complete rendered fixtures are frozen before measurement.
    Changing the seed makes a new development fixture, not statistical proof of
    generalization. Expected answers are never supplied to the policy or runner.
    """
    rng = random.Random(seed)
    cases = []
    for i in range(count):
        service = f"service_{rng.randrange(10000, 99999)}"
        port, first, second = rng.randrange(3000, 9000), rng.randrange(2, 15), rng.randrange(2, 15)
        team = f"team_{rng.randrange(10000, 99999)}"
        if i % 3 == 0:
            facts = f"{service}.port = {port}\n{service}.timeout_seconds = {first}"
            question, answer, family = f"What is the port of {service}?", str(port), "lookup"
        elif i % 3 == 1:
            facts = f"{service}.owner = {team}\n{team}.pager_code = {port}"
            question, answer, family = f"What is the pager_code of the team that owns {service}?", str(port), "two_hop"
        else:
            facts = f"{service}.first_wait_seconds = {first}\n{service}.second_wait_seconds = {second}"
            question = f"What is the sum of the two wait times for {service}, in seconds?"
            answer, family = str(first + second), "addition"
        cases.append({"id": f"case-{i:04d}", "facts": facts, "question": question,
                      "expected_answer": answer, "family": family})
    return cases


def prepare_case(runner, case, target_tokens, suffix_tokens=32):
    def render(count):
        lines = [f"unrelated_{i}.port = {1000 + i % 1000}; timeout = {1 + i % 19}"
                 for i in range(count)]
        split = count // 2
        context = "\n".join([*lines[:split], case["facts"], *lines[split:]])
        return ("Use only these records. Answer with only the requested number, without words or punctuation.\n"
                "<records>\n" + context + "\n</records>\nQuestion: " + case["question"] + "\nAnswer:")

    # Select whole distractor lines; never truncate the actual facts or question.
    low, high = 0, target_tokens
    if len(runner.prompt_tokens(render(0))) > target_tokens:
        raise ValueError("context target too small for the complete task")
    while low < high:
        middle = (low + high + 1) // 2
        if len(runner.prompt_tokens(render(middle))) <= target_tokens:
            low = middle
        else:
            high = middle - 1
    text = render(low)
    tokens = runner.prompt_tokens(text)
    if len(tokens) <= suffix_tokens:
        raise ValueError("prompt must contain a nonempty prefix")
    return {**case, "text": text, "tokens": tokens, "prefix": tokens[:-suffix_tokens],
            "suffix": tokens[-suffix_tokens:], "target_tokens": target_tokens}


def schedule(cases, group_size, scenario, seed):
    rng = random.Random(seed)
    events = []
    for start in range(0, len(cases), group_size):
        ids = [c["id"] for c in cases[start:start + group_size]]
        events.extend(("pause", name) for name in ids)
        rng.shuffle(ids)
        for i, name in enumerate(ids):
            events.append(("cancel" if scenario == "cancellation" and i % 4 == 0 else "resume", name))
    return events


def size_candidates(runner, live):
    specs, roles, metadata = runner.state_specs(live)
    return {codec: encoded_size_from_specs(specs, roles, metadata, codec) for codec in CODECS}


class MemoryMeter:
    """Separate allocator counters, owned payload bytes, and sampled process RSS.

    RSS is a 10 ms sampled lower bound on peak process memory; CUDA reports its
    own allocator's exact high-water mark. MPS has no equivalent peak API here.
    """
    def __init__(self, runner):
        import psutil
        import torch
        self.runner, self.torch = runner, torch
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.rss_peak = self.process.memory_info().rss
        self.samples = 0
        self.thread = None

    def sample(self):
        rss = self.process.memory_info().rss
        self.rss_peak = max(self.rss_peak, rss)
        self.samples += 1
        data = {"process_rss_bytes": rss}
        device = str(self.runner.device)
        if device.startswith("cuda"):
            data.update(device_allocated_bytes=self.torch.cuda.memory_allocated(self.runner.device),
                        device_reserved_bytes=self.torch.cuda.memory_reserved(self.runner.device),
                        device_peak_allocated_bytes=self.torch.cuda.max_memory_allocated(self.runner.device),
                        device_peak_reserved_bytes=self.torch.cuda.max_memory_reserved(self.runner.device))
        elif device.startswith("mps"):
            data.update(device_allocated_bytes=self.torch.mps.current_allocated_memory(),
                        device_driver_allocated_bytes=self.torch.mps.driver_allocated_memory())
        return data

    def start(self):
        self.runner.synchronize()
        if str(self.runner.device).startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats(self.runner.device)
        self.before = self.sample()
        def poll():
            while not self.stop_event.wait(.01):
                self.rss_peak = max(self.rss_peak, self.process.memory_info().rss)
                self.samples += 1
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def finish(self):
        self.runner.synchronize()
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        after = self.sample()
        return {"before": self.before, "after": after,
                "sampled_peak_process_rss_bytes": self.rss_peak,
                "rss_sampling_interval_ms": 10, "rss_samples": self.samples}


def calibrate(runner, target_tokens, chunk_size, repeats=3):
    case = {"id": "calibration", "facts": "calibration_service.port = 6123",
            "question": "What is the port of calibration_service?", "expected_answer": "6123",
            "family": "calibration"}
    prepared = prepare_case(runner, case, target_tokens)
    before = now_ms()
    logits, state = runner.prefill(prepared["prefix"], chunk_size=chunk_size)
    del logits
    runner.synchronize()
    prefill_ms = now_ms() - before
    raw_bytes, digest = state.nbytes(), state.token_digest
    sizes = size_candidates(runner, state)
    samples = []
    for repeat in range(repeats):
        for codec in CODECS[repeat % 3:] + CODECS[:repeat % 3]:
            before = now_ms()
            arrays, roles, metadata = runner.export_state(state)
            runner.synchronize()
            export_ms = now_ms() - before
            before = now_ms()
            payload = encode_state(arrays, roles, metadata, codec)
            encode_ms = now_ms() - before
            if len(payload) != sizes[codec]:
                raise RuntimeError("shape-only candidate size differs from actual payload")
            before = now_ms()
            decoded, decoded_roles, decoded_metadata = decode_state(payload)
            decode_ms = now_ms() - before
            if codec == "raw_native" and any(arrays[k].tobytes() != decoded[k].tobytes() for k in arrays):
                raise RuntimeError("native bytes failed to roundtrip")
            max_error = max(float(np.max(np.abs(arrays[k].astype(np.float32) - decoded[k].astype(np.float32))))
                            for k in arrays)
            before = now_ms()
            restored = runner.import_state(decoded, decoded_roles, decoded_metadata, expected_prefix_hash=digest)
            runner.synchronize()
            import_ms = now_ms() - before
            restored.release()
            samples.append({"repeat": repeat, "codec": codec, "raw_tensor_bytes": raw_bytes,
                            "stored_bytes": len(payload), "export_ms": export_ms, "encode_ms": encode_ms,
                            "decode_ms": decode_ms, "import_ms": import_ms, "source_max_abs_error": max_error})
            del arrays, roles, metadata, decoded, decoded_roles, decoded_metadata, restored, payload
    arrays, roles, metadata = runner.export_state(state)
    payload = encode_state(arrays, roles, metadata, "raw_native")
    del arrays, roles, metadata
    reference = runner.forward(prepared["suffix"], state).detach().float().cpu().numpy().copy()
    state.release()
    if state.tensors or state.offset:
        raise RuntimeError("runner did not release original live cache")
    del state
    arrays, roles, metadata = decode_state(payload)
    restored = runner.import_state(arrays, roles, metadata, expected_prefix_hash=digest)
    del arrays, roles, metadata, payload
    actual = runner.forward(prepared["suffix"], restored).detach().float().cpu().numpy().copy()
    restored.release()
    del restored
    gate = {"raw_restore_logits_bit_exact": reference.tobytes() == actual.tobytes(),
            "max_abs_logit_difference": float(np.max(np.abs(reference - actual))),
            "original_live_cache_released": True}
    if not gate["raw_restore_logits_bit_exact"]:
        raise RuntimeError(f"native continuation integrity gate failed: {gate}")
    profiles = []
    for codec in CODECS:
        selected = [s for s in samples if s["codec"] == codec]
        profiles.append(CodecProfile(codec, raw_bytes / selected[0]["stored_bytes"],
            statistics.median(s["export_ms"] + s["encode_ms"] for s in selected) / (raw_bytes / MIB),
            statistics.median(s["decode_ms"] + s["import_ms"] for s in selected) / (raw_bytes / MIB),
            codec == "raw_native"))
    return profiles, {"target_tokens": target_tokens, "actual_prefix_tokens": len(prepared["prefix"]),
                      "prefill_ms": prefill_ms, "samples": samples, "integrity_gate": gate,
                      "profiles": [asdict(p) for p in profiles]}


def pause(runner, store, case, chunk_size, meter):
    before = now_ms()
    logits, live = runner.prefill(case["prefix"], chunk_size=chunk_size)
    del logits
    runner.synchronize()
    prefill_ms = now_ms() - before
    raw_bytes, digest = live.nbytes(), live.token_digest
    memory_live = meter.sample()
    before = now_ms()
    sizes = size_candidates(runner, live)
    plan = store.decide(case["id"], raw_bytes, prefill_ms, case["family"], now_ms(), sizes)
    planning_ms = now_ms() - before
    export_ms = encode_ms = 0.0
    exported_bytes, stored_bytes = 0, 0
    payload = None
    if plan.selected.action != "recompute":
        started = now_ms()
        arrays, roles, metadata = runner.export_state(live)
        runner.synchronize()
        export_ms = now_ms() - started
        exported_bytes = sum(a.nbytes for a in arrays.values())
        started = now_ms()
        payload = encode_state(arrays, roles, metadata, plan.selected.action)
        encode_ms = now_ms() - started
        stored_bytes = len(payload)
        del arrays, roles, metadata
    store.commit(plan, payload, timestamp_ms=now_ms())
    del payload
    live.release()
    if live.tensors or live.offset:
        raise RuntimeError("live cache remained after pause")
    del live
    runner.synchronize()
    pause_ms = now_ms() - before
    return {"case_id": case["id"], "family": case["family"], "prefix_hash": digest,
            "prefix_tokens": len(case["prefix"]), "suffix_tokens": len(case["suffix"]),
            "raw_tensor_bytes": raw_bytes, "exact_candidate_bytes": sizes,
            "initial_prefill_ms": prefill_ms, "pause_ms": pause_ms, "planning_ms": planning_ms,
            "export_ms": export_ms, "encode_ms": encode_ms, "exported_tensor_bytes": exported_bytes,
            "serialized_payload_bytes": stored_bytes, "original_live_cache_released": True,
            "selected_prediction": asdict(plan.selected), "alternatives": [asdict(x) for x in plan.alternatives],
            "forecast": asdict(plan.forecast), "victims": list(plan.victims), "reason": plan.reason,
            "payload_bytes_after_pause": store.used_bytes, "memory_with_live_state": memory_live,
            "memory_after_release": meter.sample()}


def resume(runner, store, case, row, chunk_size, max_tokens, meter):
    started = now_ms()
    payload = store.popresume(case["id"], started)
    hit = payload is not None
    restored_payload_bytes = len(payload) if hit else 0
    decode_ms = import_ms = recompute_ms = 0.0
    imported_bytes = 0
    if hit:
        before = now_ms()
        arrays, roles, metadata = decode_state(payload)
        decode_ms = now_ms() - before
        before = now_ms()
        live = runner.import_state(arrays, roles, metadata, expected_prefix_hash=row["prefix_hash"])
        runner.synchronize()
        import_ms = now_ms() - before
        imported_bytes = live.nbytes()
        del arrays, roles, metadata, payload
    else:
        del payload
        before = now_ms()
        logits, live = runner.prefill(case["prefix"], chunk_size=chunk_size)
        del logits
        runner.synchronize()
        recompute_ms = now_ms() - before
        if live.token_digest != row["prefix_hash"]:
            raise RuntimeError("recomputed prefix identity changed")
    restore_ms = now_ms() - started
    generation, live = runner.generate(case["suffix"], state=live, max_tokens=max_tokens, chunk_size=chunk_size)
    runner.synchronize()
    resumed_memory = meter.sample()
    live.release()
    del live
    runner.synchronize()
    row.update({"resolution": "resume", "hit": hit, "restored_payload_bytes": restored_payload_bytes,
                "decode_ms": decode_ms, "import_ms": import_ms, "recompute_ms": recompute_ms,
                "imported_tensor_bytes": imported_bytes, "restore_or_recompute_ms": restore_ms,
                "resume_ttft_ms": restore_ms + generation["ttft_ms"], "resume_wall_ms": now_ms() - started,
                "generation": generation, "expected_answer": case["expected_answer"],
                "correct": not generation["truncated"] and normalized(generation["answer"]) == normalized(case["expected_answer"]),
                "payload_bytes_after_resolution": store.used_bytes, "memory_resumed": resumed_memory})
    return row


def percentile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * probability) - 1)]


def summarize(rows, blocks):
    summaries = []
    groups = sorted({(r["context_target"], r["scenario"], r["policy"]) for r in rows})
    controls = {(r["context_target"], r["scenario"], r["repeat"], r["case_id"]): r
                for r in rows if r["policy"] == "recompute" and r["resolution"] == "resume"}
    for context, scenario, policy in groups:
        subset = [r for r in rows if (r["context_target"], r["scenario"], r["policy"]) == (context, scenario, policy)]
        answered = [r for r in subset if r["resolution"] == "resume"]
        paired = [(r, controls[(context, scenario, r["repeat"], r["case_id"])]) for r in answered]
        matching = [b for b in blocks if (b["context_target"], b["scenario"], b["policy"]) == (context, scenario, policy)]
        total = sum(r["initial_prefill_ms"] + r["pause_ms"] + r.get("resume_wall_ms", r.get("cancel_ms", 0)) for r in subset)
        service = sum(r["pause_ms"] + r.get("resume_wall_ms", r.get("cancel_ms", 0)) for r in subset)
        allocated = [b["memory"]["after"].get("device_peak_allocated_bytes") for b in matching]
        summaries.append({"context_target": context, "scenario": scenario, "policy": policy,
            "sessions": len(subset), "responses": len(answered), "cancellations": len(subset) - len(answered),
            "distinct_answered_cases": len({r["case_id"] for r in answered}),
            "correct": sum(r["correct"] for r in answered),
            "token_identical_to_recompute": sum(r["generation"]["tokens"] == c["generation"]["tokens"] for r, c in paired),
            "quality_regressions_vs_recompute": sum(c["correct"] and not r["correct"] for r, c in paired),
            "quality_improvements_vs_recompute": sum(r["correct"] and not c["correct"] for r, c in paired),
            "truncated": sum(r["generation"]["truncated"] for r in answered), "hits": sum(r["hit"] for r in answered),
            "actions": dict(Counter(r["selected_prediction"]["action"] for r in subset)),
            "total_service_ms_including_canceled_pauses": service,
            "total_ms_including_initial_prefill_and_cancellations": total,
            "mean_total_ms_per_started_session": total / len(subset),
            "median_resume_ttft_ms": statistics.median(r["resume_ttft_ms"] for r in answered) if answered else None,
            "p95_resume_ttft_ms_descriptive": percentile([r["resume_ttft_ms"] for r in answered], .95),
            "peak_paused_payload_bytes": max(b["peak_paused_payload_bytes"] for b in matching),
            "sampled_peak_process_rss_bytes": max(b["memory"]["sampled_peak_process_rss_bytes"] for b in matching),
            "peak_cuda_allocated_bytes": max(allocated) if all(x is not None for x in allocated) else None,
            "exported_tensor_bytes": sum(r["exported_tensor_bytes"] for r in subset),
            "imported_tensor_bytes": sum(r.get("imported_tensor_bytes", 0) for r in subset),
            "output_tokens": sum(len(r["generation"]["tokens"]) for r in answered)})
    return summaries


def render_report(summary):
    """Human-readable measured comparisons with explicit quality/sample limits."""
    lines = ["# Owned-cache Torch experiment", "",
        f"Device: `{summary['device']}`. This is a sequential reference-runner experiment, not a serving-throughput benchmark.", "",
        "Full-session time includes original prefill, pause/compression, restore or recomputation, generation and canceled-session work. "
        "A speed ratio above 1 means less full-session time than fixed INT4 in this run. Memory is peak retained serialized cache, not total GPU or process memory.", "",
        "| Context target | Scenario | Policy | Mean full session (ms) | Speed ratio vs INT4 | Peak stored MiB | Correct / responses | Tokens identical to recompute |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    fixed = {(r["context_target"], r["scenario"]): r for r in summary["summaries"] if r["policy"] == "group_int4"}
    for row in summary["summaries"]:
        baseline = fixed[row["context_target"], row["scenario"]]
        ratio = baseline["mean_total_ms_per_started_session"] / row["mean_total_ms_per_started_session"]
        lines.append(f"| {row['context_target']} | {row['scenario']} | {row['policy']} | "
                     f"{row['mean_total_ms_per_started_session']:.2f} | {ratio:.3f}× | "
                     f"{row['peak_paused_payload_bytes'] / MIB:.3f} | {row['correct']}/{row['responses']} | "
                     f"{row['token_identical_to_recompute']}/{row['responses']} |")
    lines += ["", "These are descriptive results on generated lookup, two-hop and arithmetic tasks. Repeated questions are not independent quality observations. "
              "Unchanged accuracy does not imply unchanged answers. No quality-preservation or general performance claim is certified.", "",
              "`summary.json` also reports p95 resume time, truncations, baseline-relative quality regressions, sampled process RSS and CUDA allocator peaks. "
              "RSS sampling can miss short peaks. CPU compression follows native export, so compressed stored bytes are not reduced GPU-to-host transfer bytes.", "",
              "Run `python scripts/validate_p100_results.py RUN_DIRECTORY` to independently audit archive hashes, accounting, completeness and scores.", ""]
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--contexts", nargs="+", type=int, default=[512, 2048])
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--smoke", action="store_true", help="4 cases, 256-token target, one repeat, two scenarios")
    parser.add_argument("--allow-unvalidated-lossy", action="store_true", help="required acknowledgement: INT4/INT8 answer preservation is unproven")
    args = parser.parse_args(argv)
    if args.smoke:
        args.cases, args.contexts, args.repeats, args.group_size, args.scenarios = 4, [256], 1, 4, ["roomy", "pressure"]
    if not args.allow_unvalidated_lossy:
        parser.error("this comparison intentionally runs lossy formats; pass --allow-unvalidated-lossy")
    if not 4 <= args.cases <= 256 or not 1 <= args.repeats <= 100 or not 2 <= args.group_size <= 32:
        parser.error("require 4..256 cases, 1..100 repeats, 2..32 group-size")
    if any(c < 128 or c > 8192 for c in args.contexts) or len(set(args.contexts)) != len(args.contexts):
        parser.error("context targets must be unique and in 128..8192")
    if not 1 <= args.max_tokens <= 256 or not 1 <= args.chunk_size <= 512:
        parser.error("max-tokens must be 1..256; chunk-size must be 1..512")
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("scenarios must be unique")
    if args.output.exists():
        parser.error("output already exists; preserve it and use a new directory")
    return args


def run(args):
    import torch
    from experiments.torch.qwen_runner import QwenRunner
    hashes = source_hashes()
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"completed": False, "stage": "loading", "device_requested": args.device}
    write_json(args.output / "status.json", status)
    try:
        runner = QwenRunner(args.model_dir, device=args.device)
        write_json(args.output / "model-provenance.json", runner.provenance)
        warmup, live = runner.generate(runner.prompt_tokens("What is 2 + 2? Answer only the number."), max_tokens=8)
        live.release()
        del live
        cases = make_cases(args.cases, args.seed)
        prepared = {target: [prepare_case(runner, c, target) for c in cases] for target in args.contexts}
        if any(len(c["tokens"]) + args.max_tokens > runner.max_context
               for context_cases in prepared.values() for c in context_cases):
            raise ValueError("prompt plus max-tokens exceeds runner context; reduce --contexts (e.g. 8000 rather than 8192)")
        fixture = {"seed": args.seed, "cases_by_context": prepared}
        write_json(args.output / "fixture.json", fixture)
        calibration, profiles_by_context, budgets = {}, {}, {}
        for target in args.contexts:
            print(f"CALIBRATE context={target}", flush=True)
            profiles, record = calibrate(runner, target, args.chunk_size)
            calibration[target], profiles_by_context[target] = record, profiles
            shapes = []
            for case in prepared[target]:
                specs, roles, metadata = runner.expected_state_specs(len(case["prefix"]))
                shapes.append({codec: encoded_size_from_specs(specs, roles, metadata, codec) for codec in CODECS})
            largest = max(s["raw_native"] for s in shapes)
            # Capacity derives only from shape, before outcomes are evaluated.
            budgets[target] = {"roomy": args.group_size * largest + 1024,
                               "pressure": max(1, args.group_size // 4) * largest + 1024,
                               "cancellation": max(1, args.group_size // 4) * largest + 1024}
        write_json(args.output / "calibration.json", calibration)
        schedules = {scenario: schedule(cases, args.group_size, scenario, args.seed + 1) for scenario in args.scenarios}
        jobs = [{"repeat": repeat, "context_target": target, "scenario": scenario, "policy": policy}
                for repeat in range(args.repeats) for target in args.contexts
                for scenario in args.scenarios for policy in POLICIES]
        random.Random(args.seed + 2).shuffle(jobs)
        for i, job in enumerate(jobs):
            job["block_id"] = f"block-{i:04d}"
        protocol = {"schema_version": 1, "evidence_type": "measured_owned_cache_torch",
            "created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": hashes,
            "fixture_sha256": sha256((args.output / "fixture.json").read_bytes()).hexdigest(),
            "platform": platform.platform(), "device": str(runner.device),
            "device_name": torch.cuda.get_device_name(runner.device) if args.device == "cuda" else args.device,
            "cuda_version": torch.version.cuda, "versions": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "tokenizers", "safetensors", "psutil")},
            "model_identity": runner.identity, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "cases": [c["id"] for c in cases], "jobs": jobs, "schedules": schedules, "budgets_bytes": budgets,
            "warmup": warmup, "policies": list(POLICIES), "quality_validated": False,
            "timing": "synchronized sequential wall time; no sleeps, server queues or throughput claim; full prefill and cancellation costs included",
            "memory": "owned payload lengths; CUDA allocator high-water mark separately; process RSS sampled every 10ms (lower bound on peak); MPS current counters only",
            "transfers": "native arrays exported before CPU compression and imported after CPU decoding; bytes are tensor-copy payloads, not measured PCIe bus traffic; CPU/MPS copies are not PCIe claims",
            "policy_costs": "export+encode and decode+import calibrated per context; bandwidth term effectively zero to avoid charging transfers twice; estimates never reported as measured time",
            "quality": "strict numeric answer exact match and paired greedy-token agreement; controlled generated tasks, not production quality certification; repeated cases are not independent samples",
            "cache_ownership": "own K/V tensors, codecs and PausedStateStore; no Transformers/vLLM/Ollama/cache library; CPU-side compression"}
        if source_hashes() != hashes:
            raise RuntimeError("source changed during preparation; run from a stable checkout")
        write_json(args.output / "protocol.json", protocol)
        rows, blocks = [], []
        with (args.output / "rows.jsonl").open("w", buffering=1) as rowfile, (args.output / "events.jsonl").open("w", buffering=1) as eventfile:
            for job in jobs:
                status.update(stage="measuring", current_block=job["block_id"], completed_blocks=len(blocks))
                write_json(args.output / "status.json", status)
                target, scenario, policy = job["context_target"], job["scenario"], job["policy"]
                print(f"RUN {job['block_id']} context={target} {scenario} {policy} repeat={job['repeat']}", flush=True)
                store = PausedStateStore(policy, profiles_by_context[target], CostModel(
                    1e30, budgets[target][scenario], resume_slo_ms=1e9, allow_unvalidated_lossy=True), default_pause_ms=30000)
                case_map = {c["id"]: c for c in prepared[target]}
                paused = {}
                meter = MemoryMeter(runner).start()
                block_started = now_ms()
                try:
                    for sequence, (event, name) in enumerate(schedules[scenario]):
                        case = case_map[name]
                        if event == "pause":
                            row = pause(runner, store, case, args.chunk_size, meter)
                            row.update(job)
                            paused[name] = row
                            entry = {"action": row["selected_prediction"]["action"], "payload_bytes": row["serialized_payload_bytes"], "victims": row["victims"]}
                        else:
                            row = paused.pop(name)
                            if event == "resume":
                                resume(runner, store, case, row, args.chunk_size, args.max_tokens, meter)
                                entry = {"hit": row["hit"], "payload_bytes": row["restored_payload_bytes"]}
                            else:
                                before = now_ms()
                                store.cancel(name, before)
                                row.update(resolution="cancel", cancel_ms=now_ms() - before,
                                           payload_bytes_after_resolution=store.used_bytes)
                                entry = {}
                            rows.append(row)
                            rowfile.write(json.dumps(row, allow_nan=False) + "\n")
                        eventfile.write(json.dumps({**job, "sequence": sequence, "event": event, "case_id": name,
                                                   "used_payload_bytes": store.used_bytes, **entry}, allow_nan=False) + "\n")
                finally:
                    memory = meter.finish()
                if store.used_bytes or store.paused_sessions or paused:
                    raise RuntimeError("unresolved state remained at end of block")
                blocks.append({**job, "wall_ms": now_ms() - block_started, "budget_bytes": budgets[target][scenario],
                               "peak_paused_payload_bytes": store.peak_bytes, "byte_event_sum": store.byte_event_sum,
                               "event_count": store.event_count, "evicted_states": store.evicted_states, "memory": memory})
                write_json(args.output / "blocks.json", blocks)
        if (source_hashes() != hashes or
                sha256((args.output / "fixture.json").read_bytes()).hexdigest() != protocol["fixture_sha256"]):
            raise RuntimeError("source or fixture changed during measurements; archive is incomplete")
        summary = {"evidence_type": protocol["evidence_type"], "complete": True, "quality_validated": False,
                   "device": protocol["device"], "valid_source_freeze": True, "summaries": summarize(rows, blocks)}
        write_json(args.output / "summary.json", summary)
        (args.output / "report.md").write_text(render_report(summary))
        status.update(completed=True, stage="complete", completed_blocks=len(blocks))
        write_json(args.output / "status.json", status)
        files = sorted(p for p in args.output.iterdir() if p.is_file() and p.name != "artifact-sha256.json")
        write_json(args.output / "artifact-sha256.json", {p.name: sha256(p.read_bytes()).hexdigest() for p in files})
        return summary
    except BaseException as exc:
        status.update(completed=False, stage="failed", error_type=type(exc).__name__, error=str(exc))
        write_json(args.output / "status.json", status)
        raise


def main(argv=None):
    run(parse_args(argv))
