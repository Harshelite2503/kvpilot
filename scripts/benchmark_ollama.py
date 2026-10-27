"""Measure native local Ollama prefix reuse; this is NOT a KVPilot integration.

No model downloads, daemon restarts, global configuration edits, or KV precision
changes. Results use synthetic prompts. Memory snapshots are reported allocation,
not KV-only memory or process peaks. Cold means a new content prefix, not an empty
engine. A unique leading nonce and returned cached-token counts check that control.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from urllib.parse import urlparse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = "measured_local_ollama_native_prefix_cache"
METRIC_KEYS = (
    "total_duration", "load_duration", "prompt_eval_count",
    "prompt_eval_cached_count", "prompt_eval_duration", "eval_count", "eval_duration",
)


def normalized_answer(answer: str) -> str:
    """Whitespace only; never extract a convenient substring from a wrong answer."""
    return " ".join(answer.split())


def local_url(value: str) -> str:
    parsed = urlparse(value)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment):
        raise ValueError("use an HTTP loopback Ollama URL with no credentials or path")
    return value.rstrip("/")


def make_prompt(context: str, question: str, nonce: str) -> str:
    return (f"Benchmark instance {nonce}. This identifier is irrelevant to the answer.\n"
            "Use only the document below. Answer the question with only the requested "
            "literal value: no explanation, markdown, quotes, or units unless requested.\n"
            f"<document>\n{context}\n</document>\nQuestion: {question}\nAnswer:")


def make_plan(cases: list[dict], repeats: int, seed: int, run_id: str) -> list[dict]:
    rng = random.Random(seed)
    plan = []
    for repetition in range(repeats):
        order = list(cases)
        rng.shuffle(order)
        for case in order:
            blocks = ["repeat_block", "branch_block"]
            rng.shuffle(blocks)
            trial_id = f"r{repetition + 1}-{case['id']}"
            # Hash the unique trial first: shared string prefixes cannot retain the document.
            nonces = {b: sha256(f"{run_id}/{trial_id}/{b}".encode()).hexdigest() for b in blocks}
            plan.append({"trial_id": trial_id, "case_id": case["id"],
                         "repetition": repetition + 1, "blocks": blocks, "nonces": nonces})
    return plan


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("redirect refused: benchmark must stay on the selected local server")


class Ollama:
    def __init__(self, base_url: str, timeout: float = 180):
        self.base_url = local_url(base_url)
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, path: str, body: dict | None = None):
        return self.opener.open(urllib.request.Request(
            self.base_url + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}), timeout=self.timeout)

    def json(self, path: str, body: dict | None = None) -> dict:
        with self.request(path, body) as response:
            return json.load(response)

    def model_memory(self, model: str) -> dict | None:
        for entry in self.json("/api/ps").get("models", []):
            if entry.get("name") == model or entry.get("model") == model:
                return {k: entry.get(k) for k in (
                    "name", "digest", "size", "size_vram", "context_length", "runner")}
        return None

    def generate(self, body: dict) -> dict:
        started = time.perf_counter()
        first_answer = None
        answer = []
        thinking_chars = 0
        final = None
        with self.request("/api/generate", body) as response:
            for line in response:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                fragment = chunk.get("response", "")
                if fragment:
                    if first_answer is None:
                        first_answer = time.perf_counter() - started
                    answer.append(fragment)
                thinking_chars += len(chunk.get("thinking", ""))
                if chunk.get("done"):
                    final = chunk
        elapsed = time.perf_counter() - started
        if final is None:
            raise RuntimeError("Ollama stream ended without a final done record")
        return {"client_wall_ms": elapsed * 1000,
                "client_ttft_ms": None if first_answer is None else first_answer * 1000,
                "answer": "".join(answer), "thinking_chars": thinking_chars,
                "done_reason": final.get("done_reason"),
                "server_metrics": {k: final.get(k) for k in METRIC_KEYS}}


def cache_fraction(row: dict) -> float | None:
    metrics = row["server_metrics"]
    total, cached = metrics.get("prompt_eval_count"), metrics.get("prompt_eval_cached_count")
    if not isinstance(total, int) or total <= 0 or not isinstance(cached, int):
        return None
    if not 0 <= cached <= total:
        return None
    return cached / total


def summarize(rows: list[dict]) -> dict:
    keys = [(r["trial_id"], r["arm"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate trial/arm rows are not a valid paired experiment")
    arms = {}
    for arm in ("fresh_prefix", "exact_repeat", "shared_prefix", "prime"):
        values = [r for r in rows if r.get("arm") == arm]
        if not values:
            continue
        ttfts = [r["client_ttft_ms"] for r in values if r["client_ttft_ms"] is not None]
        memory = [r["memory_after"]["size"] for r in values
                  if r.get("memory_after") and isinstance(r["memory_after"].get("size"), int)]
        cached = [cache_fraction(r) for r in values if cache_fraction(r) is not None]
        arms[arm] = {
            "requests": len(values), "correct": sum(r.get("correct") is True for r in values),
            "scored": sum(r.get("correct") is not None for r in values),
            "median_ttft_ms": statistics.median(ttfts) if ttfts else None,
            "median_wall_ms": statistics.median(r["client_wall_ms"] for r in values),
            "median_cached_fraction": statistics.median(cached) if cached else None,
            "reported_allocation_bytes_min": min(memory) if memory else None,
            "reported_allocation_bytes_max": max(memory) if memory else None,
            "truncated_outputs": sum(r["done_reason"] == "length" for r in values),
        }
    paired = {}
    trials = {r["trial_id"] for r in rows if "trial_id" in r}
    for warm_arm in ("exact_repeat", "shared_prefix"):
        valid = []
        all_timed = []
        excluded = []
        agreement = 0
        regressions = []
        for trial_id in sorted(trials):
            group = {r["arm"]: r for r in rows if r.get("trial_id") == trial_id}
            if "fresh_prefix" not in group or warm_arm not in group:
                continue
            cold, warm = group["fresh_prefix"], group[warm_arm]
            agreement += normalized_answer(cold["answer"]) == normalized_answer(warm["answer"])
            if cold.get("correct") and not warm.get("correct"):
                regressions.append(trial_id)
            cfrac, wfrac = cache_fraction(cold), cache_fraction(warm)
            if cold["client_ttft_ms"] and warm["client_ttft_ms"]:
                all_timed.append((cold, warm))
            if (cfrac is not None and cfrac <= .05 and wfrac is not None and wfrac >= .8
                    and cold["client_ttft_ms"] and warm["client_ttft_ms"]
                    and cold["server_metrics"].get("load_duration") is not None
                    and warm["server_metrics"].get("load_duration") is not None
                    and cold["server_metrics"]["load_duration"] < 250_000_000
                    and warm["server_metrics"]["load_duration"] < 250_000_000):
                valid.append((cold, warm))
            else:
                excluded.append({"trial_id": trial_id, "fresh_cached_fraction": cfrac,
                                 "warm_cached_fraction": wfrac,
                                 "fresh_load_ns": cold["server_metrics"].get("load_duration"),
                                 "warm_load_ns": warm["server_metrics"].get("load_duration"),
                                 "reason": "failed the predeclared cache/load/TTFT control; retained in all-arm and all-timed summaries"})
        paired[warm_arm] = {
            "valid_cache_and_load_control_pairs": len(valid),
            "total_trials": len(trials), "answer_agreement_count": agreement,
            "correct_to_incorrect_trial_ids": regressions,
            "excluded_from_verified_hit_timing": excluded,
            "all_timed_pairs": len(all_timed),
            "all_timed_pairs_median_ttft_speedup": statistics.median(
                c["client_ttft_ms"] / w["client_ttft_ms"] for c, w in all_timed) if all_timed else None,
            "median_paired_ttft_speedup": statistics.median(
                c["client_ttft_ms"] / w["client_ttft_ms"] for c, w in valid) if valid else None,
            "median_paired_ttft_reduction_percent": statistics.median(
                100 * (1 - w["client_ttft_ms"] / c["client_ttft_ms"]) for c, w in valid) if valid else None,
        }
    return {"evidence_type": EVIDENCE, "arms": arms, "paired": paired,
            "kvpilot_policy_tested": False, "kvpilot_codecs_tested": False,
            "kv_only_memory_saved_bytes": None, "compression_quality_validated": False,
            "caveat": "Descriptive native-cache microbenchmark; no controller/compression, concurrency or production-quality claim. Priming is reported separately and is not free."}


def write_json(path: Path, data: object):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--cases", type=Path, default=ROOT / "examples/ollama_quality_cases.json")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.num_ctx < 512 or args.num_predict < 1:
        parser.error("positive repeats/output and context >=512 required")
    fixture = json.loads(args.cases.read_text())
    cases = fixture["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("case IDs must be unique and nonempty")
    client = Ollama(args.base_url)
    tags = client.json("/api/tags")
    installed = next((m for m in tags["models"] if m["name"] == args.model), None)
    if installed is None:
        raise ValueError("model must already be installed; this benchmark never pulls models")
    shown = client.json("/api/show", {"model": args.model})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    plan = make_plan(cases, args.repeats, args.seed, run_id)
    base_body = {"model": args.model, "think": False, "stream": True,
                 "truncate": False, "shift": False, "keep_alive": "5m",
                 "options": {"temperature": 0, "seed": args.seed, "num_ctx": args.num_ctx,
                             "num_predict": args.num_predict, "presence_penalty": 0,
                             "repeat_penalty": 1}}
    manifest = {
        "schema_version": 1, "evidence_type": EVIDENCE, "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
        "python": platform.python_version(), "platform": platform.platform(),
        "ollama": client.json("/api/version"),
        "model": {k: installed.get(k) for k in ("name", "digest", "size", "details")},
        "show": {k: shown.get(k) for k in ("details", "model_info", "capabilities", "thinking")},
        "memory_initial": client.model_memory(args.model), "request_defaults": base_body,
        "fixture_sha256": sha256(args.cases.read_bytes()).hexdigest(),
        "harness_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "plan": plan,
        "protocol": {
            "fresh_prefix": "Unique leading content nonce; model weights warmed first. Common template may still be cached.",
            "exact_repeat": "Immediately repeat identical full prompt after fresh_prefix. Always second; an upper-bound reuse case.",
            "shared_prefix": "Use another fresh nonce; prime document with a different question, then ask target question. Block order randomized against fresh+repeat.",
            "prime_question": "For this document-loading check, reply with the literal READY.",
            "quality": "Exact match after whitespace normalization only; 12 synthetic tasks, repeated trials are not independent tasks.",
            "valid_timing_pair": "fresh cached <=5%, warm cached >=80%, both load_duration <250ms, both TTFT present; invalid pairs retained.",
            "memory": "Before/after /api/ps model size and size_vram snapshots; not KV-only, peak, or separate CPU+GPU totals.",
            "priming_cost": "Prime requests included in raw rows and separate summary. Warm latency assumes earlier useful prefill; priming from scratch costs work.",
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)
    rows = []
    try:
        warm_body = dict(base_body, prompt="Reply only READY.")
        warmup = client.generate(warm_body)
        warmup["memory_after"] = client.model_memory(args.model)
        write_json(args.output_dir / "warmup.json", warmup)
        by_id = {c["id"]: c for c in cases}
        with (args.output_dir / "requests.jsonl").open("w") as log:
            for trial in plan:
                case = by_id[trial["case_id"]]
                for block in trial["blocks"]:
                    arms = ("fresh_prefix", "exact_repeat") if block == "repeat_block" else ("prime", "shared_prefix")
                    for arm in arms:
                        question = manifest["protocol"]["prime_question"] if arm == "prime" else case["question"]
                        prompt = make_prompt(case["context"], question, trial["nonces"][block])
                        before = client.model_memory(args.model)
                        result = client.generate(dict(base_body, prompt=prompt))
                        result.update(trial_id=trial["trial_id"], case_id=case["id"], arm=arm,
                                      nonce=trial["nonces"][block], prompt_sha256=sha256(prompt.encode()).hexdigest(),
                                      memory_before=before, memory_after=client.model_memory(args.model),
                                      expected_answer=None if arm == "prime" else case["expected_answer"])
                        result["correct"] = None if arm == "prime" else (
                            normalized_answer(result["answer"]) == normalized_answer(case["expected_answer"])
                            and result["done_reason"] != "length")
                        rows.append(result)
                        log.write(json.dumps(result, allow_nan=False) + "\n")
                        log.flush()
                        print(f"{trial['trial_id']} {arm}: ttft={result['client_ttft_ms']} ms cached={cache_fraction(result)} correct={result['correct']}", flush=True)
        manifest["status"] = "complete"
        write_json(args.output_dir / "summary.json", summarize(rows))
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["completed_requests"] = len(rows)
        write_json(args.output_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
