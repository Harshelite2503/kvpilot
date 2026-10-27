"""Offline integrity checks: these tests never invoke Ollama or a model."""

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import urllib.request

import pytest


SPEC = importlib.util.spec_from_file_location(
    "benchmark_ollama", Path(__file__).resolve().parents[1] / "scripts" / "benchmark_ollama.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("offline benchmark tests must not open network connections")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", reject)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:11434", "http://localhost:11434/", "http://[::1]:11434/",
])
def test_local_url_accepts_only_supported_loopback_forms(url):
    assert benchmark.local_url(url) == url.rstrip("/")


@pytest.mark.parametrize("url", [
    "https://127.0.0.1:11434", "http://example.com:11434", "http://192.168.1.2:11434",
    "http://0.0.0.0:11434", "http://localhost.example.com", "http://127.0.0.1.example.com",
    "http://user:password@localhost:11434", "http://localhost:11434/api/generate",
    "http://localhost:11434?model=remote", "http://localhost:11434/#fragment",
    "file:///tmp/ollama.sock", "localhost:11434",
])
def test_local_url_rejects_remote_ambiguous_or_credentialed_targets(url):
    with pytest.raises(ValueError):
        benchmark.local_url(url)


def test_local_client_refuses_redirects_before_following_an_external_location():
    client = benchmark.Ollama("http://127.0.0.1:11434")
    handler = next(h for h in client.opener.handlers if isinstance(h, benchmark.NoRedirect))
    with pytest.raises(RuntimeError, match="redirect refused"):
        handler.redirect_request(
            urllib.request.Request("http://127.0.0.1:11434/api/ps"),
            None, 302, "Found", {}, "http://example.com/collect",
        )


def test_plan_is_complete_reproducible_and_nonces_isolate_blocks_and_runs():
    cases = [{"id": f"case-{i}"} for i in range(12)]
    original = deepcopy(cases)
    plan = benchmark.make_plan(cases, repeats=2, seed=20261027, run_id="run-one")
    assert cases == original
    assert plan == benchmark.make_plan(cases, repeats=2, seed=20261027, run_id="run-one")
    assert len(plan) == len({trial["trial_id"] for trial in plan}) == 24
    assert {(t["case_id"], t["repetition"]) for t in plan} == {
        (c["id"], repetition) for c in cases for repetition in (1, 2)
    }
    assert all(set(t["blocks"]) == {"repeat_block", "branch_block"} for t in plan)
    assert len({tuple(t["blocks"]) for t in plan}) == 2
    nonces = {nonce for trial in plan for nonce in trial["nonces"].values()}
    assert len(nonces) == 48
    other = benchmark.make_plan(cases, repeats=2, seed=20261027, run_id="run-two")
    assert nonces.isdisjoint(n for trial in other for n in trial["nonces"].values())


def test_prompt_keeps_exact_repeat_identical_and_branch_question_after_shared_document():
    fresh = benchmark.make_prompt("Port: 42", "Which port?", "repeat-nonce")
    repeat = benchmark.make_prompt("Port: 42", "Which port?", "repeat-nonce")
    prime = benchmark.make_prompt("Port: 42", "Reply READY.", "branch-nonce")
    shared = benchmark.make_prompt("Port: 42", "Which port?", "branch-nonce")
    assert fresh == repeat
    assert prime != shared
    assert prime.split("\nQuestion:")[0] == shared.split("\nQuestion:")[0]
    assert fresh.split("\nQuestion:")[0] != shared.split("\nQuestion:")[0]
    assert fresh.index("repeat-nonce") < fresh.index("<document>")


def test_answer_normalization_does_not_turn_wrong_explanations_into_correct_answers():
    assert benchmark.normalized_answer(" \t42\n") == "42"
    assert benchmark.normalized_answer("one\n\t two") == "one two"
    for wrong in ("The answer is 42", "42.", '"42"', "```42```", "42 ms", "042"):
        assert benchmark.normalized_answer(wrong) != "42"
    assert benchmark.normalized_answer("READY") != benchmark.normalized_answer("ready")


def row(trial, arm, *, cached=0, ttft=100, load_ns=0, correct=True, answer="42", memory=None):
    return {
        "trial_id": trial, "arm": arm, "answer": answer, "correct": correct,
        "client_ttft_ms": ttft, "client_wall_ms": 120, "done_reason": "stop",
        "memory_after": memory,
        "server_metrics": {"prompt_eval_count": 1000, "prompt_eval_cached_count": cached,
                           "load_duration": load_ns},
    }


@pytest.mark.parametrize("total,cached", [(0, 0), (1000, None), (1000, -1), (1000, 1001), (None, 0)])
def test_missing_or_impossible_cache_counters_are_not_valid_controls(total, cached):
    value = row("t", "fresh_prefix")
    value["server_metrics"].update(prompt_eval_count=total, prompt_eval_cached_count=cached)
    assert benchmark.cache_fraction(value) is None


def test_summary_uses_only_verified_pairs_without_filtering_wrong_answers_or_erasing_exclusions():
    rows = [
        row("valid-bad-answer", "fresh_prefix", cached=50, ttft=100),
        row("valid-bad-answer", "exact_repeat", cached=800, ttft=25, correct=False, answer="wrong"),
        row("not-fresh", "fresh_prefix", cached=51, ttft=100),
        row("not-fresh", "exact_repeat", cached=900, ttft=1),
        row("not-warm", "fresh_prefix", ttft=100),
        row("not-warm", "exact_repeat", cached=799, ttft=1),
        row("model-loading", "fresh_prefix", ttft=100, load_ns=250_000_000),
        row("model-loading", "exact_repeat", cached=900, ttft=1),
        row("no-answer-time", "fresh_prefix", ttft=None),
        row("no-answer-time", "exact_repeat", cached=900, ttft=1),
        row("missing-counter", "fresh_prefix", cached=None),
        row("missing-counter", "exact_repeat", cached=900, ttft=1),
        row("valid-bad-answer", "prime", correct=None, answer="READY"),
    ]
    original = deepcopy(rows)
    summary = benchmark.summarize(rows)
    assert rows == original
    comparison = summary["paired"]["exact_repeat"]
    assert comparison["total_trials"] == 6
    assert comparison["valid_cache_and_load_control_pairs"] == 1
    assert comparison["median_paired_ttft_speedup"] == 4
    assert comparison["median_paired_ttft_reduction_percent"] == 75
    assert comparison["correct_to_incorrect_trial_ids"] == ["valid-bad-answer"]
    assert comparison["answer_agreement_count"] == 5
    assert {r["trial_id"] for r in comparison["excluded_from_verified_hit_timing"]} == {
        "not-fresh", "not-warm", "model-loading", "no-answer-time", "missing-counter",
    }
    assert comparison["all_timed_pairs"] == 5
    assert comparison["all_timed_pairs_median_ttft_speedup"] == 100
    assert summary["arms"]["exact_repeat"]["requests"] == 6
    assert summary["arms"]["exact_repeat"]["scored"] == 6
    assert summary["arms"]["exact_repeat"]["correct"] == 5
    assert summary["arms"]["prime"]["requests"] == 1
    assert summary["arms"]["prime"]["scored"] == 0
    assert summary["paired"]["shared_prefix"]["median_paired_ttft_speedup"] is None
    assert summary["kv_only_memory_saved_bytes"] is None
    assert summary["compression_quality_validated"] is False
    assert summary["kvpilot_policy_tested"] is False
    assert summary["kvpilot_codecs_tested"] is False
    for arm in summary["arms"].values():
        assert arm["reported_allocation_bytes_min"] is None
        assert arm["reported_allocation_bytes_max"] is None


def test_summary_rejects_duplicate_trial_arm_instead_of_silently_overwriting():
    fresh = row("trial", "fresh_prefix")
    with pytest.raises(ValueError, match="duplicate"):
        benchmark.summarize([fresh, deepcopy(fresh), row("trial", "exact_repeat", cached=900)])


def test_streaming_times_answer_separately_from_thinking_and_requires_final_record(monkeypatch):
    class Stream:
        def __init__(self, chunks):
            self.lines = [b"\n"] + [(json.dumps(chunk) + "\n").encode() for chunk in chunks]

        def __enter__(self):
            return iter(self.lines)

        def __exit__(self, *args):
            return False

    client = benchmark.Ollama("http://127.0.0.1:11434")
    chunks = [
        {"thinking": "reasoning", "response": "", "done": False},
        {"response": "4", "done": False},
        {"response": "2", "done": False},
        {"response": "", "done": True, "done_reason": "stop", "prompt_eval_count": 100,
         "prompt_eval_cached_count": 90, "eval_count": 2},
    ]
    monkeypatch.setattr(client, "request", lambda path, body: Stream(chunks))
    clock = iter([10.0, 12.0, 13.0])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    result = client.generate({"prompt": "fixture", "stream": True})
    assert result["client_ttft_ms"] == 2000
    assert result["client_wall_ms"] == 3000
    assert result["answer"] == "42"
    assert result["thinking_chars"] == len("reasoning")
    assert result["server_metrics"]["prompt_eval_cached_count"] == 90
    assert result["server_metrics"]["load_duration"] is None

    monkeypatch.setattr(client, "request", lambda path, body: Stream(chunks[:-1]))
    clock = iter([20.0, 22.0, 23.0])
    with pytest.raises(RuntimeError, match="without a final"):
        client.generate({"prompt": "fixture", "stream": True})


def test_main_archives_all_four_arms_and_scores_truncated_exact_answer_as_wrong(tmp_path, monkeypatch):
    """Exercise real orchestration with a fake client, not an HTTP mock server."""
    fixture = tmp_path / "cases.json"
    fixture.write_text(json.dumps({"cases": [{
        "id": "port", "context": "Port: 42", "question": "Which port?", "expected_answer": "42",
    }]}))
    output = tmp_path / "result"
    calls = []
    unloads = []
    seen_prompts = set()
    seen_prefixes = set()

    class FakeOllama:
        def __init__(self, base_url):
            benchmark.local_url(base_url)

        def json(self, path, body=None):
            if path == "/api/tags":
                return {"models": [{"name": "fixture-model", "digest": "fixture-digest"}]}
            if path == "/api/show":
                return {"capabilities": ["completion"]}
            if path == "/api/version":
                return {"version": "offline-fixture"}
            if path == "/api/generate":
                assert body == {"model": "fixture-model", "keep_alive": 0, "stream": False}
                unloads.append(deepcopy(body))
                seen_prompts.clear()
                seen_prefixes.clear()
                return {"done": True, "done_reason": "unload"}
            raise AssertionError(f"unexpected endpoint {path}")

        def model_memory(self, model):
            return None

        def generate(self, body):
            calls.append(deepcopy(body))
            prompt = body["prompt"]
            prefix = prompt.split("\nQuestion:")[0]
            is_target = "Question: Which port?" in prompt
            is_repeat = prompt in seen_prompts
            result = {
                "client_wall_ms": 120, "client_ttft_ms": 25 if prefix in seen_prefixes else 100,
                "answer": " \n42\t" if is_target else "READY", "thinking_chars": 0,
                "done_reason": "length" if is_repeat else "stop",
                "server_metrics": {"prompt_eval_count": 1000,
                                   "prompt_eval_cached_count": 900 if prefix in seen_prefixes else 0,
                                   "load_duration": 0},
            }
            seen_prompts.add(prompt)
            seen_prefixes.add(prefix)
            return result

    monkeypatch.setattr(benchmark, "Ollama", FakeOllama)
    monkeypatch.setattr(benchmark.sys, "argv", [
        "benchmark_ollama", "--cases", str(fixture), "--model", "fixture-model",
        "--repeats", "1", "--output-dir", str(output),
    ])
    assert benchmark.main() == 0
    records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    setup = [json.loads(line) for line in (output / "setup.jsonl").read_text().splitlines()]
    assert len(calls) == 5  # One separately archived warmup, then four experimental arms.
    assert len(unloads) == 2  # Before the trial and finally, not between cache-paired arms.
    assert len(setup) == 1
    assert setup[0]["trial_id"] == records[0]["trial_id"]
    assert setup[0]["reset_and_warm_wall_ms"] >= 0
    assert setup[0]["unload_done_reason"] == "unload"
    assert len(records) == manifest["completed_requests"] == 4
    assert manifest["status"] == "complete"
    assert manifest["reset_between_trials"] is True
    assert manifest["cleanup"]["done_reason"] == "unload"
    assert manifest["memory_final"] is None
    assert "1 synthetic tasks" in manifest["protocol"]["quality"]
    assert manifest["fixture_sha256"] == sha256(fixture.read_bytes()).hexdigest()
    indexed = {r["arm"]: r for r in records}
    assert set(indexed) == {"fresh_prefix", "exact_repeat", "prime", "shared_prefix"}
    assert indexed["fresh_prefix"]["prompt_sha256"] == indexed["exact_repeat"]["prompt_sha256"]
    assert indexed["prime"]["nonce"] == indexed["shared_prefix"]["nonce"]
    assert indexed["fresh_prefix"]["nonce"] != indexed["shared_prefix"]["nonce"]
    assert indexed["fresh_prefix"]["correct"] is True
    assert indexed["exact_repeat"]["correct"] is False
    assert indexed["prime"]["correct"] is None
    assert indexed["shared_prefix"]["correct"] is True
    assert all(r["memory_before"] is None and r["memory_after"] is None for r in records)
    assert summary["arms"]["exact_repeat"]["truncated_outputs"] == 1
    assert summary["paired"]["exact_repeat"]["valid_cache_and_load_control_pairs"] == 1
    assert summary["paired"]["shared_prefix"]["valid_cache_and_load_control_pairs"] == 1
    for request, record in zip(calls[1:], records):
        assert sha256(request["prompt"].encode()).hexdigest() == record["prompt_sha256"]
    fresh_prompt = next(c["prompt"] for c, r in zip(calls[1:], records) if r["arm"] == "fresh_prefix")
    shared_prompt = next(c["prompt"] for c, r in zip(calls[1:], records) if r["arm"] == "shared_prefix")
    assert fresh_prompt.replace(indexed["fresh_prefix"]["nonce"], "nonce") == shared_prompt.replace(
        indexed["shared_prefix"]["nonce"], "nonce"
    )


def test_memory_stop_checks_reported_allocation_and_preserves_missingness():
    benchmark.check_memory(None, 100)
    benchmark.check_memory({"size": None}, 100)
    benchmark.check_memory({"size_vram": 999}, 100)  # A different field is not a KV-size substitute.
    benchmark.check_memory({"size": 100}, 100)
    with pytest.raises(RuntimeError, match="memory stop"):
        benchmark.check_memory({"size": 101}, 100)


def install_lifecycle_fixture(tmp_path, monkeypatch, *, retain=False, failure=None, repeats=2):
    """A stateful fake exposes reset boundaries and allocation-stop behavior."""
    fixture = tmp_path / "lifecycle-cases.json"
    fixture.write_text(json.dumps({"cases": [{
        "id": "port", "context": "Port: 42", "question": "Which port?", "expected_answer": "42",
    }]}))
    output = tmp_path / "lifecycle-result"
    state = {"events": [], "loaded": False, "size": 50, "requests": 0}

    class FakeOllama:
        def __init__(self, base_url):
            benchmark.local_url(base_url)

        def json(self, path, body=None):
            if path == "/api/tags":
                return {"models": [{"name": "fixture-model"}]}
            if path == "/api/show":
                return {}
            if path == "/api/version":
                return {"version": "offline-fixture"}
            if path == "/api/generate":
                assert body == {"model": "fixture-model", "keep_alive": 0, "stream": False}
                state["events"].append("unload")
                state["loaded"] = False
                state["size"] = 50
                return {"done": True, "done_reason": "unload"}
            raise AssertionError(f"unexpected endpoint {path}")

        def model_memory(self, model):
            return {"size": state["size"]} if state["loaded"] else None

        def generate(self, body):
            warm = body["prompt"] == "Reply only READY."
            state["events"].append("warmup" if warm else "request")
            state["loaded"] = True
            if warm and failure == "warmup":
                raise RuntimeError("fixture warmup failed")
            if not warm:
                state["requests"] += 1
                if failure == "generate":
                    raise RuntimeError("fixture generation failed")
                if failure == "memory":
                    state["size"] = 101
            return {
                "client_wall_ms": 120, "client_ttft_ms": 100,
                "answer": "42", "thinking_chars": 0, "done_reason": "stop",
                "server_metrics": {"prompt_eval_count": 1000, "prompt_eval_cached_count": 0,
                                   "load_duration": 0},
            }

    monkeypatch.setattr(benchmark, "Ollama", FakeOllama)
    args = ["benchmark_ollama", "--cases", str(fixture), "--model", "fixture-model",
            "--repeats", str(repeats), "--max-reported-gib", str(100 / 2**30),
            "--output-dir", str(output)]
    if retain:
        args.append("--retain-across-trials")
    monkeypatch.setattr(benchmark.sys, "argv", args)
    return output, state


def test_default_resets_before_each_trial_but_not_within_four_arm_comparison(tmp_path, monkeypatch):
    output, state = install_lifecycle_fixture(tmp_path, monkeypatch)
    assert benchmark.main() == 0
    assert state["events"] == ["unload", "warmup", *(["request"] * 4),
                               "unload", "warmup", *(["request"] * 4), "unload"]
    assert state["loaded"] is False
    records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    setup = [json.loads(line) for line in (output / "setup.jsonl").read_text().splitlines()]
    assert len(records) == 8
    assert len(setup) == 2
    assert {r["trial_id"] for r in records} == {s["trial_id"] for s in setup}
    assert all(s["reset_and_warm_wall_ms"] >= 0 for s in setup)


def test_retain_option_warms_once_and_explicitly_preserves_loaded_model(tmp_path, monkeypatch):
    output, state = install_lifecycle_fixture(tmp_path, monkeypatch, retain=True)
    assert benchmark.main() == 0
    assert state["events"] == ["warmup", *(["request"] * 8)]
    assert state["loaded"] is True
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["reset_between_trials"] is False
    assert "cleanup" not in manifest
    setup = [json.loads(line) for line in (output / "setup.jsonl").read_text().splitlines()]
    assert len(setup) == 1


@pytest.mark.parametrize("failure,completed", [("memory", 1), ("generate", 0), ("warmup", 0)])
def test_failure_preserves_completed_rows_and_unloads_before_recording_failed_manifest(
    tmp_path, monkeypatch, failure, completed,
):
    output, state = install_lifecycle_fixture(tmp_path, monkeypatch, failure=failure)
    with pytest.raises(RuntimeError, match="memory stop" if failure == "memory" else "fixture"):
        benchmark.main()
    assert state["events"][0] == state["events"][-1] == "unload"
    assert state["events"].count("unload") == 2
    assert state["events"].count("request") == (0 if failure == "warmup" else 1)
    assert state["loaded"] is False
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["completed_requests"] == completed
    assert manifest["cleanup"]["done_reason"] == "unload"
    assert manifest["memory_final"] is None
    assert "RuntimeError" in manifest["error"]
    assert len((output / "requests.jsonl").read_text().splitlines()) == completed
    assert not (output / "summary.json").exists()
