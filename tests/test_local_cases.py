"""Derive the controlled fixture answers from its active configuration and code."""
import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/local_kvpilot_cases.json"


def fixture():
    return json.loads(FIXTURE.read_text())


def files_from_context(context):
    blocks = re.findall(r"^## ([^\n]+)\n```(json|python)\n(.*?)\n```", context, re.M | re.S)
    return {heading.split(" — ", 1)[0]: (kind, body) for heading, kind, body in blocks}


def merge(base, override):
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def source_functions(files):
    namespace = {"__builtins__": {"range": range, "str": str}}
    for path in ("src/retry.py", "src/release.py"):
        tree = ast.parse(files[path][1], filename=path)
        assert all(isinstance(node, ast.FunctionDef) for node in tree.body)
        exec(compile(tree, path, "exec"), namespace)
    return namespace


def test_eight_distinct_new_cases_share_one_frozen_compact_context():
    data = fixture()
    cases = data["cases"]
    assert data["schema_version"] == 1 and data["metadata"]["synthetic"] is True
    assert len(cases) == data["metadata"]["case_count"] == 8
    assert len({case["id"] for case in cases}) == 8
    assert len({case["question"] for case in cases}) == 8
    assert len({case["context"] for case in cases}) == 1
    for case in cases:
        assert all(isinstance(case[key], str) and case[key] for key in
                   ("id", "context", "question", "expected_answer", "domain", "type"))
    old = json.loads((ROOT / "examples/ollama_quality_cases.json").read_text())
    assert {case["id"] for case in cases}.isdisjoint({case["id"] for case in old["cases"]})
    assert {case["question"] for case in cases}.isdisjoint({case["question"] for case in old["cases"]})
    context = cases[0]["context"]
    assert sha256(context.encode()).hexdigest() == data["metadata"]["tokenized_context_sha256"]
    assert 1200 <= data["metadata"]["context_token_count"] <= 1800
    # Answers belong in ground truth. The model context contains repository facts,
    # not a duplicate question/answer sheet or hidden grading labels.
    assert "expected_answer" not in context
    assert all(case["question"] not in context and case["id"] not in context for case in cases)


def test_all_ground_truth_answers_follow_active_files_and_stated_rules():
    data = fixture()
    cases = {case["id"]: case for case in data["cases"]}
    files = files_from_context(data["cases"][0]["context"])
    documents = {name: json.loads(body) for name, (kind, body) in files.items() if kind == "json"}
    active = merge(documents["config/base.json"], documents["config/production.json"])
    teams = documents["config/teams.json"]
    mesa = merge(active, teams["mesa"]["config_overrides"])
    orbit = merge(active, teams["orbit"]["config_overrides"])
    repos = documents["config/repos.json"]
    profiles = documents["config/delivery_profiles.json"]
    regions = documents["config/regions.json"]
    functions = source_functions(files)
    derived = {
        "relay-poll-interval": str(active["worker"]["poll_interval_s"]),
        "relay-production-parallelism": str(mesa["worker"]["parallel_jobs"]),
        "relay-team-release-ttl": str(orbit["release"]["ttl_hours"]),
        "relay-repository-delivery-queue": profiles[teams[repos["ledger-ui"]["team"]]["delivery_profile"]]["queue"],
        "relay-repository-artifact-bucket": regions[teams[repos["field-agent"]["team"]]["region"]]["artifact_bucket"],
        "relay-retry-total-wait": str(sum(functions["failure_waits"](**mesa["retry"]))),
    }
    for case_id in ("relay-hold-branch-order", "relay-release-tag"):
        call_text = re.search(r"(delivery_state\([^)]*\)|make_release_tag\([^)]*\))", cases[case_id]["question"]).group(0)
        call = ast.parse(call_text, mode="eval")
        assert isinstance(call.body, ast.Call) and isinstance(call.body.func, ast.Name)
        assert all(isinstance(value, ast.Constant) for value in call.body.args)
        assert all(isinstance(kw.value, ast.Constant) for kw in call.body.keywords)
        derived[case_id] = eval(compile(call, "fixture-question", "eval"), functions)
    assert set(derived) == set(cases)
    assert {case_id: case["expected_answer"] for case_id, case in cases.items()} == derived


def test_multihop_cases_and_inactive_distractors_are_unambiguous():
    data = fixture()
    cases = {case["id"]: case for case in data["cases"]}
    assert sum(case["type"] == "simple_multi_hop" for case in cases.values()) == 2
    files = files_from_context(data["cases"][0]["context"])
    archive = json.loads(files["archive/production-2.2.json"][1])
    test_only = json.loads(files["tests/worker_fixture.json"][1])
    assert cases["relay-poll-interval"]["expected_answer"] not in {
        str(archive["worker"]["poll_interval_s"]), str(test_only["worker"]["poll_interval_s"])}
    assert cases["relay-repository-delivery-queue"]["expected_answer"] != archive["regulated_queue"]
    assert cases["relay-repository-artifact-bucket"]["expected_answer"] not in {
        archive["us_artifact_bucket"], test_only["artifact_bucket"]}
