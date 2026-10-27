"""Explain native Ollama branch-checkpoint behavior with three prompt suffixes.

This is a source-motivated diagnostic, not a KVPilot policy/compression benchmark.
Each synthetic case gets a fresh content nonce, then fixed-order A, B and C
requests with the same document: Zeta setup, Beta setup, and the target question.
No models are downloaded and no daemon or global settings are changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import uuid

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from benchmark_ollama import (Ollama, cache_fraction, check_memory, local_url, make_prompt,
                              normalized_answer, reset_and_warm, write_json)

ROOT = SCRIPT_DIR.parent
EVIDENCE = "measured_local_ollama_branch_checkpoint_diagnostic"
ARMS = ("a_prime", "b_first_divergence", "c_after_divergence")
SETUP_QUESTIONS = ("Zeta setup: reply only READY.", "Beta setup: reply only READY.")
MEMORY_STOP_BYTES = 8 * 2**30
SOURCE_REFERENCES = [
    {
        "url": "https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/pipeline.go#L153",
        "interpretation": "Schedules recurrent checkpoints at 8192-token intervals and near the prompt end (length minus four).",
    },
    {
        "url": "https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/cache/recurrent.go#L244",
        "interpretation": "Cumulative recurrent state cannot be rewound to an arbitrary shared prefix.",
    },
    {
        "url": "https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/prefix_cache.go#L108",
        "interpretation": "Restores a checkpoint no later than the match, and schedules a newly observed branch point for capture during prefill.",
    },
]


def load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text())["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("fixture must contain a nonempty cases list")
    required = ("id", "context", "question", "expected_answer")
    if any(not isinstance(c, dict) or any(not isinstance(c.get(k), str) or not c[k] for k in required)
           for c in cases):
        raise ValueError("each case needs nonempty string id/context/question/expected_answer")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("case IDs must be unique")
    if any(c["question"] in SETUP_QUESTIONS for c in cases):
        raise ValueError("target questions must differ from both setup questions")
    return cases


def make_branch_plan(cases: list[dict], run_id: str) -> list[dict]:
    plan = []
    for case in cases:
        nonce = sha256(f"{run_id}/{case['id']}/branch-diagnostic".encode()).hexdigest()
        questions = (*SETUP_QUESTIONS, case["question"])
        phases = [{"arm": arm, "question": question,
                   "prompt_sha256": sha256(make_prompt(case["context"], question, nonce).encode()).hexdigest()}
                  for arm, question in zip(ARMS, questions)]
        plan.append({"trial_id": case["id"], "case_id": case["id"], "nonce": nonce, "phases": phases})
    return plan


def source_hashes(fixture_path: Path) -> dict:
    return {name: {"path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest()}
            for name, path in (("harness", SCRIPT_DIR / "benchmark_ollama.py"),
                               ("diagnostic", Path(__file__)), ("fixture", fixture_path))}


def request_defaults(model: str, seed: int) -> dict:
    # Match benchmark_ollama defaults; this experiment does not enlarge context.
    return {"model": model, "think": False, "stream": True, "truncate": False,
            "shift": False, "keep_alive": "5m",
            "options": {"temperature": 0, "seed": seed, "num_ctx": 4096,
                        "num_predict": 64, "presence_penalty": 0, "repeat_penalty": 1}}


def _median(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def summarize(rows: list[dict], expected_case_ids: list[str]) -> dict:
    identities = [(r["trial_id"], r["arm"]) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate trial/arm rows")
    if any(r["arm"] not in ARMS or r["case_id"] not in expected_case_ids for r in rows):
        raise ValueError("row is not part of the declared diagnostic")
    arm_summaries = {}
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        allocations = [r["memory_after"].get("size") for r in group if r.get("memory_after")]
        allocations = [v for v in allocations if isinstance(v, int)]
        arm_summaries[arm] = {
            "requests": len(group), "correct": sum(r["correct"] for r in group),
            "quality_role": "target exact-answer check" if arm == ARMS[2] else "setup compliance only",
            "median_ttft_ms": _median([r["client_ttft_ms"] for r in group]),
            "median_wall_ms": _median([r["client_wall_ms"] for r in group]),
            "median_cached_fraction": _median([cache_fraction(r) for r in group]),
            "median_prompt_eval_count": _median([r["server_metrics"].get("prompt_eval_count") for r in group]),
            "median_prompt_eval_cached_count": _median([r["server_metrics"].get("prompt_eval_cached_count") for r in group]),
            "median_prompt_eval_duration_ns": _median([r["server_metrics"].get("prompt_eval_duration") for r in group]),
            "reported_allocation_bytes_min": min(allocations) if allocations else None,
            "reported_allocation_bytes_max": max(allocations) if allocations else None,
            "truncated_outputs": sum(r["done_reason"] == "length" for r in group),
        }
    observations = []
    for case_id in expected_case_ids:
        group = {r["arm"]: r for r in rows if r["case_id"] == case_id}
        fractions = {arm: cache_fraction(group[arm]) if arm in group else None for arm in ARMS}
        available = all(value is not None for value in fractions.values())
        pattern = (fractions[ARMS[0]] <= .05 and fractions[ARMS[1]] <= .05
                   and fractions[ARMS[2]] >= .8) if available else None
        loads = {arm: group[arm]["server_metrics"].get("load_duration") if arm in group else None for arm in ARMS}
        observations.append({
            "case_id": case_id, "completed_arms": [arm for arm in ARMS if arm in group],
            "cached_fractions": fractions,
            "ttft_ms": {arm: group[arm]["client_ttft_ms"] if arm in group else None for arm in ARMS},
            "source_predicted_pattern_observed": pattern,
            "load_duration_ns": loads,
            "all_loads_below_250ms": all(v is not None and v < 250_000_000 for v in loads.values()),
            "target_correct": group[ARMS[2]]["correct"] if ARMS[2] in group else None,
        })
    targets = [r for r in rows if r["arm"] == ARMS[2]]
    return {
        "schema_version": 1, "evidence_type": EVIDENCE,
        "expected_cases": len(expected_case_ids), "completed_requests": len(rows),
        "complete": len(rows) == len(expected_case_ids) * 3,
        "arms": arm_summaries, "case_observations": observations,
        "source_predicted_pattern_count": sum(o["source_predicted_pattern_observed"] is True for o in observations),
        "other_observed_pattern_count": sum(o["source_predicted_pattern_observed"] is False for o in observations),
        "unavailable_pattern_count": sum(o["source_predicted_pattern_observed"] is None for o in observations),
        "target_quality": {"scored_cases": len(targets), "correct": sum(r["correct"] for r in targets),
                           "incorrect_case_ids": [r["case_id"] for r in targets if not r["correct"]],
                           "validation": "Synthetic exact-answer checks only; no cold target-answer comparison in this diagnostic."},
        "excluded_requests": 0,
        "kvpilot_policy_tested": False, "kvpilot_codecs_tested": False,
        "compression_quality_validated": False, "general_task_quality_validated": False,
        "kv_only_memory_saved_bytes": None,
        "limitations": [
            "All requests, including misses, slow loads and setup costs, remain in this descriptive summary and raw log.",
            "Fixed A/B/C order and different questions make this a diagnostic, not an isolated causal speedup estimate.",
            "The default twelve synthetic cases share one repository and are correlated; they are not independent general coding tasks.",
            "The source hypothesis is specific to the inspected Ollama v0.40.11 MLX hybrid recurrent-state path; the installed backend and tokenization may differ.",
            "Cache fractions are server-reported observations. A matching pattern is consistent with checkpoint reuse, not instrumentation proving its internal cause.",
            "Allocation snapshots are not KV-only memory, peak usage, or separately additive CPU/GPU memory.",
            "Each triplet begins with a model unload/reload and warmup; setup costs are reported separately. This is not a multi-session pressure test.",
            "The reported 8 GiB allocation stop is checked between requests, not a guaranteed peak-memory cap.",
            "Setup A and B are real work; no new KVPilot controller, codec, memory saving or compression-quality result is measured.",
        ],
    }


def _memory_snapshot(client, model: str) -> tuple[dict | None, str | None]:
    try:
        return client.model_memory(model), None
    except Exception as exc:
        # Keep a completed generation even if its subsequent telemetry request fails.
        return None, f"{type(exc).__name__}: {exc}"


def run_diagnostic(*, output_dir: Path, cases_path: Path, model: str = "qwen3.5:4b",
                   base_url: str = "http://127.0.0.1:11434", seed: int = 20261027,
                   client=None, run_id: str | None = None) -> dict:
    base_url = local_url(base_url)
    cases = load_cases(cases_path)
    run_id = run_id or uuid.uuid4().hex
    plan = make_branch_plan(cases, run_id)
    defaults = request_defaults(model, seed)
    hashes = source_hashes(cases_path)
    # Reserve a new archive and record the intended experiment before ANY API call.
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1, "status": "preflight", "evidence_type": EVIDENCE,
        "started_at": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
        "python": platform.python_version(), "platform": platform.platform(),
        "base_url": base_url, "requested_model": model, "request_defaults": defaults,
        "reported_memory_stop_bytes": MEMORY_STOP_BYTES,
        "source_hashes": hashes, "plan": plan, "source_references": SOURCE_REFERENCES,
        "protocol": {
            "order": "One A/B/C triplet per case in fixture order; one unique leading content nonce per triplet.",
            "a_prime": SETUP_QUESTIONS[0], "b_first_divergence": SETUP_QUESTIONS[1],
            "c_after_divergence": "Original fixture question; all three suffixes differ after the common document.",
            "hypothesis": "A is cold; the first divergent B may also be cold, then C may reuse the branch checkpoint learned while prefilling B.",
            "observational_pattern": "A cached fraction <=5%, B <=5%, C >=80%; descriptive classification only, never a row exclusion rule.",
            "quality": "Use benchmark_ollama.normalized_answer: case-sensitive whitespace normalization, no substring extraction; truncated answers fail.",
            "warmup": "Unload and warm this model before every triplet using the benchmark harness helper. Save every setup cost and result in setup.jsonl. Finally unload this model again.",
            "memory_stop": "Check reported model allocation against 8 GiB between requests; stop and unload on breach. This is not a guaranteed peak-memory cap.",
            "memory": "Before/after model allocation snapshots from /api/ps; not KV-only or peak memory.",
        },
        "kvpilot_policy_tested": False, "kvpilot_codecs_tested": False,
    }
    write_json(output_dir / "manifest.json", manifest)
    rows, setups = [], []
    model_ready = False
    try:
        client = client or Ollama(base_url)
        installed = next((m for m in client.json("/api/tags").get("models", []) if m.get("name") == model), None)
        if installed is None:
            raise ValueError("model must already be installed; this diagnostic never downloads models")
        shown = client.json("/api/show", {"model": model})
        manifest.update(
            status="running", ollama=client.json("/api/version"),
            model={k: installed.get(k) for k in ("name", "digest", "size", "details")},
            show={k: shown.get(k) for k in ("details", "model_info", "capabilities", "thinking")},
        )
        manifest["memory_initial"], manifest["memory_initial_error"] = _memory_snapshot(client, model)
        # Complete model/version/settings metadata is persisted before generation.
        write_json(output_dir / "manifest.json", manifest)
        model_ready = True
        by_id = {case["id"]: case for case in cases}
        with (output_dir / "requests.jsonl").open("x") as log, (output_dir / "setup.jsonl").open("x") as setup_log:
            for trial in plan:
                setup_started = time.perf_counter()
                try:
                    setup = dict(reset_and_warm(client, defaults), trial_id=trial["trial_id"], status="complete")
                except BaseException as exc:
                    setup = {"trial_id": trial["trial_id"], "status": "failed",
                             "reset_and_warm_wall_ms": (time.perf_counter() - setup_started) * 1000,
                             "error": f"{type(exc).__name__}: {exc}"}
                    setups.append(setup)
                    setup_log.write(json.dumps(setup, allow_nan=False) + "\n")
                    setup_log.flush()
                    raise
                setups.append(setup)
                setup_log.write(json.dumps(setup, allow_nan=False) + "\n")
                setup_log.flush()
                check_memory(setup.get("memory_after"), MEMORY_STOP_BYTES)
                case = by_id[trial["case_id"]]
                for phase in trial["phases"]:
                    prompt = make_prompt(case["context"], phase["question"], trial["nonce"])
                    before, before_error = _memory_snapshot(client, model)
                    check_memory(before, MEMORY_STOP_BYTES)
                    result = client.generate(dict(defaults, prompt=prompt))
                    after, after_error = _memory_snapshot(client, model)
                    expected = case["expected_answer"] if phase["arm"] == ARMS[2] else "READY"
                    result.update(
                        trial_id=trial["trial_id"], case_id=case["id"], arm=phase["arm"],
                        nonce=trial["nonce"], prompt_sha256=phase["prompt_sha256"],
                        memory_before=before, memory_before_error=before_error,
                        memory_after=after, memory_after_error=after_error,
                        expected_answer=expected,
                        quality_role="target" if phase["arm"] == ARMS[2] else "setup",
                        correct=(normalized_answer(result["answer"]) == normalized_answer(expected)
                                 and result["done_reason"] != "length"),
                    )
                    rows.append(result)
                    log.write(json.dumps(result, allow_nan=False) + "\n")
                    log.flush()
                    print(f"{trial['case_id']} {phase['arm']}: ttft={result['client_ttft_ms']} ms "
                          f"cached={cache_fraction(result)} correct={result['correct']}", flush=True)
                    # Preserve the completed row before stopping on its memory snapshot.
                    check_memory(after, MEMORY_STOP_BYTES)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if model_ready:
            cleanup_started = time.perf_counter()
            try:
                manifest["cleanup"] = client.json("/api/generate", {
                    "model": model, "keep_alive": 0, "stream": False})
                manifest["memory_final"], manifest["memory_final_error"] = _memory_snapshot(client, model)
            except Exception as exc:
                manifest["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            manifest["cleanup_wall_ms"] = (time.perf_counter() - cleanup_started) * 1000
        summary = summarize(rows, [case["id"] for case in cases])
        summary["setup"] = {
            "attempted_triplets": len(setups),
            "completed_resets_and_warmups": sum(s["status"] == "complete" for s in setups),
            "total_reset_and_warm_wall_ms": sum(s["reset_and_warm_wall_ms"] for s in setups),
            "median_reset_and_warm_wall_ms": _median([s["reset_and_warm_wall_ms"] for s in setups]),
            "cleanup_wall_ms": manifest.get("cleanup_wall_ms"),
        }
        summary["run_status"] = manifest["status"]
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["completed_requests"] = len(rows)
        try:
            manifest["source_hashes_after"] = source_hashes(cases_path)
            manifest["sources_unchanged"] = manifest["source_hashes_after"] == hashes
        except OSError as exc:
            manifest["sources_unchanged"] = False
            manifest["source_check_error"] = f"{type(exc).__name__}: {exc}"
        summary["sources_unchanged"] = manifest["sources_unchanged"]
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "manifest.json", manifest)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new archive directory; existing paths are refused")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--cases", type=Path, default=ROOT / "examples/ollama_quality_cases.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args(argv)
    summary = run_diagnostic(output_dir=args.output_dir, cases_path=args.cases, model=args.model,
                             base_url=args.base_url, seed=args.seed)
    print(f"Native branch diagnostic: {summary['source_predicted_pattern_count']}/{summary['expected_cases']} "
          "cases matched the source-motivated cache pattern. This is not a KVPilot gain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
