from copy import deepcopy
from hashlib import sha256
import json

import pytest

from kvpilot.context import ContextDataError, load_pricing, load_trace


def header():
    return {"schema_version": 1, "type": "run", "run_id": "run-1", "dataset_id": "dataset-1",
            "workload_version": "1", "evaluator_id": "eval-1", "agent_revision": "agent-1",
            "provider": "anthropic", "model": "example-model", "evidence_type": "synthetic",
            "cache_protocol": "explicit-ttl-v1"}


def request():
    return {"type": "request", "request_id": "r1", "task_id": "t1", "session_id": "s1",
            "sequence": 0, "kind": "inference", "usage": {"input_tokens": 100, "output_tokens": 20}}


def events():
    return [header(), request(), {"type": "task_end", "task_id": "t1", "success": True},
            {"type": "run_end", "complete": True, "request_count": 1, "task_count": 1}]


def write_trace(tmp_path, rows):
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def pricecard():
    return {"schema_version": 1, "pricing_id": "illustrative-v1", "currency": "USD", "source": "test rates",
            "effective_date": "2026-10-29", "provider": "anthropic", "model": "example-model",
            "rates_usd_per_million": {"input": "3", "cache_read": "0.3", "cache_write_5m": "3.75",
                                      "cache_write_1h": "6", "output": "15"}}


def write_pricing(tmp_path, value):
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps(value))
    return path


def test_load_preserves_provenance_and_normalizes_only_missing_cache_counters(tmp_path):
    rows = events()
    rows[1].update(latency_ms=2.5, prefix_fingerprint="private-hash", prefix_change_reason="unknown")
    rows[1]["usage"]["service_tier"] = "standard"
    rows[1]["usage"]["output_tokens_details"] = {"thinking_tokens": 7}
    path = write_trace(tmp_path, rows)
    trace = load_trace(path)
    assert trace.source_sha256 == sha256(path.read_bytes()).hexdigest()
    assert trace.metadata == rows[0]
    assert trace.requests[0]["prefix_fingerprint"] == "private-hash"
    assert trace.requests[0]["usage"]["input_tokens"] == 100
    assert trace.requests[0]["usage"]["cache_creation"] == {
        "ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}
    assert trace.requests[0]["usage"]["output_tokens_details"]["thinking_tokens"] == 7


@pytest.mark.parametrize("mutate", [
    lambda rows: rows.pop(),
    lambda rows: rows.pop(0),
    lambda rows: rows.insert(2, header()),
    lambda rows: rows.insert(2, dict(rows[-1])),
    lambda rows: rows[-1].update(complete=False),
    lambda rows: rows[-1].update(request_count=0),
    lambda rows: rows[-1].update(task_count=2),
    lambda rows: rows[-1].update(request_count=True),
    lambda rows: rows[0].update(schema_version=True),
    lambda rows: rows[0].update(provider="openai"),
    lambda rows: rows[0].update(dataset_id=" "),
    lambda rows: rows[0].update(evidence_type="measured"),
    lambda rows: rows[0].update(typo="not silently ignored"),
    lambda rows: rows[1].update(sequence=-1),
    lambda rows: rows[1].update(sequence=True),
    lambda rows: rows[1].update(kind="tool"),
    lambda rows: rows[1].update(prefix_change_reason="proved-cache-miss"),
    lambda rows: rows[1].update(prefix_change_reason=[]),
    lambda rows: rows[1].update(prefix_change_reason=None),
    lambda rows: rows[1].update(prefix_fingerprint=""),
    lambda rows: rows[1].update(latency_ms=True),
    lambda rows: rows[1].update(latency_ms=-1),
    lambda rows: rows[1].update(latency_ms=10 ** 400),
    lambda rows: rows[2].update(success=1),
    lambda rows: rows[2].update(duration_ms=-1),
    lambda rows: rows[2].update(task_id="unrelated"),
])
def test_incomplete_or_malformed_trace_is_rejected(tmp_path, mutate):
    rows = events()
    mutate(rows)
    with pytest.raises(ContextDataError):
        load_trace(write_trace(tmp_path, rows))


@pytest.mark.parametrize("kind", ["duplicate_request", "duplicate_task", "after_end", "same_sequence", "backward_sequence"])
def test_order_and_unique_identifiers_are_enforced(tmp_path, kind):
    rows = events()
    extra = request()
    extra.update(request_id="r2", sequence=1)
    if kind == "duplicate_task":
        rows.insert(3, deepcopy(rows[2]))
        rows[-1]["task_count"] = 2
    else:
        if kind == "duplicate_request":
            extra["request_id"] = "r1"
        if kind == "same_sequence":
            extra["sequence"] = 0
        if kind == "backward_sequence":
            rows[1]["sequence"] = 2
        rows.insert(3 if kind == "after_end" else 2, extra)
        rows[-1]["request_count"] = 2
    with pytest.raises(ContextDataError):
        load_trace(write_trace(tmp_path, rows))


def test_interleaved_sessions_have_separate_sequence_domains(tmp_path):
    rows = events()
    second = request()
    second.update(request_id="r2", session_id="s2")
    rows.insert(2, second)
    rows[-1]["request_count"] = 2
    assert len(load_trace(write_trace(tmp_path, rows)).requests) == 2


@pytest.mark.parametrize("usage", [
    {"input_tokens": 1}, {"output_tokens": 1},
    {"input_tokens": True, "output_tokens": 1},
    {"input_tokens": 1.0, "output_tokens": 1},
    {"input_tokens": -1, "output_tokens": 1},
    {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": None},
    {"input_tokens": 1, "output_tokens": 1, "cache_creation_input_tokens": 100},
    {"input_tokens": 1, "output_tokens": 1, "cache_creation_input_tokens": 100,
     "cache_creation": {"ephemeral_5m_input_tokens": 99, "ephemeral_1h_input_tokens": 0}},
    {"input_tokens": 1, "output_tokens": 1, "cache_creation_input_tokens": 100,
     "cache_creation": {"ephemeral_5m_input_tokens": 100}},
    {"input_tokens": 1, "output_tokens": 1,
     "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": True}},
    {"input_tokens": 1, "output_tokens": 1, "server_tool_use": {"web_search_requests": 1}},
    {"input_tokens": 1, "output_tokens": 1, "server_tool_use": {"web_search_requests": False}},
    {"input_tokens": 1, "output_tokens": 1, "iterations": [{"type": "compaction", "input_tokens": 20}]},
    {"input_tokens": 1, "output_tokens": 1, "input_audio_tokens": 2},
    {"input_tokens": 1, "output_tokens": 1, "input_tokens_details": {"image_tokens": 2}},
    {"input_tokens": 1, "output_tokens": 1, "output_tokens_details": {"thinking_tokens": 2}},
])
def test_ambiguous_or_unsupported_usage_is_not_billed(tmp_path, usage):
    rows = events()
    rows[1]["usage"] = usage
    with pytest.raises(ContextDataError):
        load_trace(write_trace(tmp_path, rows))


def test_empty_optional_native_usage_fields_are_allowed(tmp_path):
    rows = events()
    rows[1]["usage"].update(server_tool_use={"web_search_requests": 0, "web_fetch_requests": 0},
                            iterations=[], cache_creation=None, output_tokens_details=None)
    assert load_trace(write_trace(tmp_path, rows)).requests[0]["usage"]["cache_creation_input_tokens"] == 0


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_nonfinite_trace_numbers_are_rejected(tmp_path, bad):
    path = write_trace(tmp_path, events())
    path.write_text(path.read_text().replace('"input_tokens": 100', f'"input_tokens": {bad}'))
    with pytest.raises(ContextDataError):
        load_trace(path)


def test_duplicate_json_keys_blank_lines_and_truncated_json_are_rejected(tmp_path):
    for content in ('{"type":"run","type":"run"}\n{}\n', '\n{}\n', '{"type":\n{}\n'):
        path = tmp_path / "invalid.jsonl"
        path.write_text(content)
        with pytest.raises(ContextDataError):
            load_trace(path)


def test_pricing_keeps_numeric_decimal_precision_and_raw_file_hash(tmp_path):
    path = write_pricing(tmp_path, pricecard())
    path.write_text(path.read_text().replace('"3.75"', '3.750000000000000000000000000001'))
    pricing = load_pricing(path)
    assert str(pricing.rates_usd_per_million["cache_write_5m"]) == "3.750000000000000000000000000001"
    assert pricing.source_sha256 == sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("key,value", [("currency", "EUR"), ("provider", "bedrock"),
    ("schema_version", True), ("effective_date", "2026-02-30"), ("effective_date", "2026-1-1"),
    ("model", ""), ("source", " ")])
def test_invalid_pricecard_metadata(tmp_path, key, value):
    card = pricecard()
    card[key] = value
    with pytest.raises(ContextDataError):
        load_pricing(write_pricing(tmp_path, card))


@pytest.mark.parametrize("value", [True, -1, "NaN", "Infinity", "-Infinity", float("nan"),
                                   float("inf"), " 3 ", "1_000", None])
def test_invalid_price_rates(tmp_path, value):
    card = pricecard()
    card["rates_usd_per_million"]["input"] = value
    with pytest.raises(ContextDataError):
        load_pricing(write_pricing(tmp_path, card))


def test_rate_categories_are_complete_and_unambiguous(tmp_path):
    for change in ("missing", "extra"):
        card = pricecard()
        if change == "missing":
            del card["rates_usd_per_million"]["cache_write_1h"]
        else:
            card["rates_usd_per_million"]["cache_write"] = "2"
        with pytest.raises(ContextDataError):
            load_pricing(write_pricing(tmp_path, card))
