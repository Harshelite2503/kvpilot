"""Mock-only branch-diagnostic integrity tests; no Ollama or network calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("diagnose_ollama_branches", Path(__file__).resolve().parents[1] / "scripts/diagnose_ollama_branches.py")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def cases():
    return [{"id": name, "context": "Shared fixture: port is 8443.",
             "question": "What is the port?", "expected_answer": "8443"} for name in ("one", "two")]


def fixture_path(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": cases()}))
    return path


def result(answer="READY", cached=0, done_reason="stop"):
    return {"client_wall_ms": 30.0, "client_ttft_ms": 20.0, "answer": answer,
            "thinking_chars": 0, "done_reason": done_reason,
            "server_metrics": {"prompt_eval_count": 1000, "prompt_eval_cached_count": cached,
                               "load_duration": 1_000_000, "prompt_eval_duration": 10_000_000,
                               "eval_count": 2, "eval_duration": 10_000_000}}


class FakeClient:
    def __init__(self, archive, *, fail_generate=None, installed=True):
        self.archive = archive
        self.calls = []
        self.generations = []
        self.fail_generate = fail_generate
        self.installed = installed
        self.completed_trial_generations = 0
        self.resident = False

    def manifest(self):
        path = self.archive / "manifest.json"
        assert path.is_file(), "manifest must exist before any API call"
        return json.loads(path.read_text())

    def json(self, path, body=None):
        manifest = self.manifest()
        self.calls.append(path)
        assert manifest["source_hashes"]["diagnostic"]["sha256"]
        assert manifest["plan"]
        if path == "/api/tags":
            return {"models": [{"name": "qwen3.5:4b", "digest": "fixture-digest", "size": 123}] if self.installed else []}
        if path == "/api/show":
            return {"details": {"family": "fixture"}, "capabilities": ["completion"]}
        if path == "/api/version":
            return {"version": "0.40.11-fixture"}
        if path == "/api/generate":
            assert body == {"model": "qwen3.5:4b", "keep_alive": 0, "stream": False}
            self.resident = False
            return {"done": True, "done_reason": "unload"}
        raise AssertionError(f"unexpected API path {path}")

    def model_memory(self, model):
        self.manifest()
        self.calls.append("/api/ps")
        return {"name": model, "size": 123, "size_vram": 123} if self.resident else None

    def generate(self, body):
        manifest = self.manifest()
        assert manifest["status"] == "running"
        assert manifest["ollama"]["version"] == "0.40.11-fixture"
        assert manifest["model"]["digest"] == "fixture-digest"
        # Before each subsequent call, earlier trial rows must already be flushed.
        log = self.archive / "requests.jsonl"
        if self.generations:
            assert len(log.read_text().splitlines()) == self.completed_trial_generations
        self.generations.append(deepcopy(body))
        if len(self.generations) == self.fail_generate:
            raise RuntimeError("synthetic interrupted generation")
        self.resident = True
        if body["prompt"] == "Reply only READY.":
            return result()
        self.completed_trial_generations += 1
        target = "Question: What is the port?" in body["prompt"]
        return result("8443" if target else "READY", cached=950 if target else 0)


def test_plan_is_three_distinct_suffixes_with_one_fresh_nonce_per_case():
    plan = diagnostic.make_branch_plan(cases(), "fixed-run")
    assert len({trial["nonce"] for trial in plan}) == 2
    for trial, case in zip(plan, cases()):
        assert [p["arm"] for p in trial["phases"]] == list(diagnostic.ARMS)
        assert len({p["prompt_sha256"] for p in trial["phases"]}) == 3
        prompts = [diagnostic.make_prompt(case["context"], p["question"], trial["nonce"]) for p in trial["phases"]]
        assert len({p.split("Question:", 1)[0] for p in prompts}) == 1
    other = diagnostic.make_branch_plan(cases(), "other-run")
    assert plan[0]["nonce"] != other[0]["nonce"]


def test_run_records_preflight_and_generations_with_matching_settings(tmp_path):
    fixture = fixture_path(tmp_path)
    archive = tmp_path / "run"
    client = FakeClient(archive)
    summary = diagnostic.run_diagnostic(output_dir=archive, cases_path=fixture, client=client, run_id="test")
    assert len(client.generations) == 8  # One reset/warmup per complete triplet.
    assert client.calls.count("/api/generate") == 3  # Two resets plus final unload.
    assert all(body["options"] == diagnostic.request_defaults("qwen3.5:4b", 20261027)["options"]
               for body in client.generations)
    assert all(body["options"]["num_ctx"] == 4096 and body["think"] is False for body in client.generations)
    assert all(body["truncate"] is False and body["shift"] is False for body in client.generations)
    rows = [json.loads(line) for line in (archive / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and all(row["memory_after"]["size"] == 123 for row in rows)
    assert summary["source_predicted_pattern_count"] == 2
    assert summary["target_quality"]["correct"] == 2
    assert summary["excluded_requests"] == 0
    assert summary["kvpilot_policy_tested"] is False
    manifest = json.loads((archive / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["sources_unchanged"] is True
    assert set(manifest["source_hashes"]) == {"harness", "diagnostic", "fixture"}
    assert manifest["memory_final"] is None
    setups = [json.loads(line) for line in (archive / "setup.jsonl").read_text().splitlines()]
    assert len(setups) == 2 and all(s["status"] == "complete" for s in setups)
    assert summary["setup"]["completed_resets_and_warmups"] == 2
    assert summary["setup"]["total_reset_and_warm_wall_ms"] >= 0


def test_failure_keeps_all_completed_rows_and_partial_summary(tmp_path):
    fixture = fixture_path(tmp_path)
    archive = tmp_path / "interrupted"
    client = FakeClient(archive, fail_generate=7)
    with pytest.raises(RuntimeError, match="interrupted"):
        diagnostic.run_diagnostic(output_dir=archive, cases_path=fixture, client=client)
    rows = (archive / "requests.jsonl").read_text().splitlines()
    assert len(rows) == 4  # First complete triplet and second A.
    manifest = json.loads((archive / "manifest.json").read_text())
    summary = json.loads((archive / "summary.json").read_text())
    assert manifest["status"] == "failed" and manifest["completed_requests"] == 4
    assert summary["run_status"] == "failed" and summary["complete"] is False
    assert summary["expected_cases"] == 2 and summary["unavailable_pattern_count"] == 1
    assert client.resident is False and manifest["cleanup"]["done_reason"] == "unload"


def test_existing_archive_is_refused_before_any_api_call(tmp_path):
    fixture = fixture_path(tmp_path)
    archive = tmp_path / "existing"
    archive.mkdir()
    sentinel = archive / "sentinel"
    sentinel.write_text("keep me")
    client = FakeClient(archive)
    with pytest.raises(FileExistsError):
        diagnostic.run_diagnostic(output_dir=archive, cases_path=fixture, client=client)
    assert client.calls == [] and sentinel.read_text() == "keep me"


def test_missing_installed_model_is_archived_without_generation_or_download(tmp_path):
    fixture = fixture_path(tmp_path)
    archive = tmp_path / "missing"
    client = FakeClient(archive, installed=False)
    with pytest.raises(ValueError, match="already be installed"):
        diagnostic.run_diagnostic(output_dir=archive, cases_path=fixture, client=client)
    assert client.generations == [] and client.calls == ["/api/tags"]
    assert json.loads((archive / "manifest.json").read_text())["status"] == "failed"


def test_memory_stop_keeps_completed_request_then_unloads_without_next_request(tmp_path):
    fixture = fixture_path(tmp_path)
    archive = tmp_path / "memory-stop"
    client = FakeClient(archive)
    original_memory = client.model_memory

    def large_after_first_request(model):
        snapshot = original_memory(model)
        if snapshot and client.completed_trial_generations:
            snapshot["size"] = diagnostic.MEMORY_STOP_BYTES + 1
        return snapshot

    client.model_memory = large_after_first_request
    with pytest.raises(RuntimeError, match="memory stop"):
        diagnostic.run_diagnostic(output_dir=archive, cases_path=fixture, client=client)
    assert len(client.generations) == 2  # One warmup and A; B/C never issued.
    rows = [json.loads(line) for line in (archive / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["arm"] == diagnostic.ARMS[0]
    assert rows[0]["memory_after"]["size"] > diagnostic.MEMORY_STOP_BYTES
    manifest = json.loads((archive / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["memory_final"] is None
    assert client.calls.count("/api/generate") == 2 and client.resident is False


def test_failed_warmup_is_archived_and_cleanup_still_runs(tmp_path):
    fixture = fixture_path(tmp_path)
    archive = tmp_path / "warmup-failure"
    client = FakeClient(archive, fail_generate=1)
    with pytest.raises(RuntimeError, match="interrupted"):
        diagnostic.run_diagnostic(output_dir=archive, cases_path=fixture, client=client)
    setup = json.loads((archive / "setup.jsonl").read_text())
    assert setup["status"] == "failed" and setup["reset_and_warm_wall_ms"] >= 0
    assert (archive / "requests.jsonl").read_text() == ""
    manifest = json.loads((archive / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["completed_requests"] == 0
    assert client.calls.count("/api/generate") == 2 and client.resident is False


def test_unexpected_or_unavailable_cache_counts_are_retained():
    rows = []
    for case_id, counts in (("one", (0, 950, 950)), ("two", (0, 0, None))):
        for arm, cached in zip(diagnostic.ARMS, counts):
            row = result(cached=cached)
            row.update(trial_id=case_id, case_id=case_id, arm=arm, memory_after=None, correct=False)
            rows.append(row)
    summary = diagnostic.summarize(rows, ["one", "two"])
    assert summary["other_observed_pattern_count"] == 1
    assert summary["unavailable_pattern_count"] == 1
    assert summary["completed_requests"] == 6 and summary["excluded_requests"] == 0
    assert summary["target_quality"]["incorrect_case_ids"] == ["one", "two"]
    with pytest.raises(ValueError, match="duplicate"):
        diagnostic.summarize(rows + [rows[0]], ["one", "two"])


def test_truncated_exact_text_is_not_a_correct_answer(tmp_path):
    fixture = fixture_path(tmp_path)
    archive = tmp_path / "truncated"
    client = FakeClient(archive)
    original = client.generate

    def truncated(body):
        row = original(body)
        if "Question: What is the port?" in body["prompt"]:
            row["done_reason"] = "length"
        return row

    client.generate = truncated
    summary = diagnostic.run_diagnostic(output_dir=archive, cases_path=fixture, client=client)
    assert summary["target_quality"]["correct"] == 0
    assert summary["arms"][diagnostic.ARMS[2]]["truncated_outputs"] == 2


def test_nonlocal_server_and_duplicate_cases_are_rejected_without_api(tmp_path):
    fixture = fixture_path(tmp_path)
    archive = tmp_path / "invalid"
    client = FakeClient(archive)
    with pytest.raises(ValueError, match="loopback"):
        diagnostic.run_diagnostic(output_dir=archive, cases_path=fixture, base_url="https://example.com", client=client)
    fixture.write_text(json.dumps({"cases": [cases()[0], cases()[0]]}))
    with pytest.raises(ValueError, match="unique"):
        diagnostic.run_diagnostic(output_dir=archive, cases_path=fixture, client=client)
    assert client.calls == [] and not archive.exists()
