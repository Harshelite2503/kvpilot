"""Strict offline trace and price-card inputs for the context-cost experiment.

Anthropic ``input_tokens`` excludes cache reads and cache writes. Cache writes
need the provider's TTL breakdown; guessing a TTL from request configuration
would misprice mixed-TTL writes. See https://platform.claude.com/docs/en/build-with-claude/prompt-caching.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import math
from pathlib import Path
import re


class ContextDataError(ValueError):
    """The input is incomplete, ambiguous, or outside this accounting scope."""


@dataclass(frozen=True)
class Trace:
    metadata: dict
    requests: list[dict]
    tasks: list[dict]
    source_sha256: str


@dataclass(frozen=True)
class Pricing:
    metadata: dict
    rates_usd_per_million: dict[str, Decimal]
    source_sha256: str


RATE_KEYS = ("input", "cache_read", "cache_write_5m", "cache_write_1h", "output")
RUN_FIELDS = {"schema_version", "type", "run_id", "dataset_id", "workload_version", "evaluator_id",
              "agent_revision", "provider", "model", "evidence_type", "cache_protocol"}
REQUEST_FIELDS = {"type", "request_id", "task_id", "session_id", "sequence", "kind", "usage"}
REQUEST_OPTIONAL = {"latency_ms", "prefix_fingerprint", "prefix_change_reason"}
PREFIX_REASONS = {"intentional_compaction", "intentional_edit", "unexpected", "unknown"}
TTL_KEYS = ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")


def _require(condition, message):
    if not condition:
        raise ContextDataError(message)


def _object(value, name):
    _require(isinstance(value, dict), f"{name} must be an object")


def _fields(value, required, optional, name):
    _object(value, name)
    _require(required <= set(value), f"{name} missing fields: {sorted(required - set(value))}")
    _require(set(value) <= required | optional, f"{name} has unsupported fields: {sorted(set(value) - required - optional)}")


def _text(value, name):
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be a nonempty string")


def _count(value, name):
    _require(type(value) is int and value >= 0, f"{name} must be a nonnegative integer (not bool)")
    return value


def _duration(value, name):
    try:
        valid = type(value) in (int, float) and value >= 0 and math.isfinite(value)
    except OverflowError:
        valid = False
    _require(valid,
             f"{name} must be a finite nonnegative number (not bool)")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ContextDataError(f"non-finite JSON number: {value}")


def _parse(content, name, *, decimal=False):
    try:
        return json.loads(content, object_pairs_hook=_unique_pairs,
                          parse_constant=_invalid_constant, parse_float=Decimal if decimal else float)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise ContextDataError(f"invalid JSON in {name}: {exc}") from exc


def _check_finite_tree(value, name):
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite_tree(item, f"{name}.{key}")
    elif isinstance(value, list):
        for item in value:
            _check_finite_tree(item, name)
    elif isinstance(value, float):
        _require(math.isfinite(value), f"{name} contains a non-finite number")


def _is_zero_usage(value):
    """Zero/null native optional counters do not introduce an unpriced charge."""
    if value is None:
        return True
    if isinstance(value, dict):
        return all(_is_zero_usage(v) for v in value.values())
    if isinstance(value, list):
        return not value
    return type(value) is int and value == 0


def normalize_usage(usage):
    """Preserve native metadata and make the five priced token counts explicit."""
    _object(usage, "usage")
    _check_finite_tree(usage, "usage")
    result = deepcopy(usage)
    for key in ("input_tokens", "output_tokens"):
        _require(key in result, f"usage missing {key}")
        _count(result[key], f"usage.{key}")
    for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
        result[key] = _count(result.get(key, 0), f"usage.{key}")
    creation = result.get("cache_creation")
    if creation is None:
        _require(result["cache_creation_input_tokens"] == 0,
                 "nonzero cache writes require an explicit cache_creation TTL breakdown")
        creation = dict.fromkeys(TTL_KEYS, 0)
    else:
        _fields(creation, set(TTL_KEYS), set(), "usage.cache_creation")
        for key in TTL_KEYS:
            _count(creation[key], f"usage.cache_creation.{key}")
    _require(sum(creation.values()) == result["cache_creation_input_tokens"],
             "cache_creation TTL breakdown must sum to cache_creation_input_tokens")
    result["cache_creation"] = creation

    # These categories can contain unpriced fees or nested/multi-model usage.
    # An empty native field is harmless; a populated one is not silently ignored.
    for key in ("server_tool_use", "iterations", "server_tool_usage"):
        if key in result:
            _require(_is_zero_usage(result[key]), f"unsupported {key} billing; only ordinary text Messages usage is supported")

    def reject_unpriced(value, path="usage"):
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = key.lower()
                if any(part in lowered for part in ("audio", "image", "video", "nontext", "non_text")):
                    _require(_is_zero_usage(item), f"unsupported nontext usage billing: {path}.{key}")
                reject_unpriced(item, f"{path}.{key}")
        elif isinstance(value, list):
            for item in value:
                reject_unpriced(item, path)

    reject_unpriced(result)
    details = result.get("output_tokens_details")
    if details is not None:
        _object(details, "usage.output_tokens_details")
        if "thinking_tokens" in details:
            _count(details["thinking_tokens"], "usage.output_tokens_details.thinking_tokens")
            _require(details["thinking_tokens"] <= result["output_tokens"],
                     "thinking_tokens cannot exceed inclusive output_tokens")
    return result


def token_categories(usage):
    usage = normalize_usage(usage)
    return {"input": usage["input_tokens"], "cache_read": usage["cache_read_input_tokens"],
            "cache_write_5m": usage["cache_creation"]["ephemeral_5m_input_tokens"],
            "cache_write_1h": usage["cache_creation"]["ephemeral_1h_input_tokens"],
            "output": usage["output_tokens"]}


def load_trace(path) -> Trace:
    """Read one complete, strict JSONL run without making any API requests."""
    raw = Path(path).read_bytes()
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ContextDataError("trace must be UTF-8 JSONL") from exc
    _require(len(lines) >= 2, "trace needs a run header and terminal run_end")
    events = []
    for number, line in enumerate(lines, 1):
        _require(bool(line.strip()), f"blank JSONL line {number}")
        event = _parse(line, f"line {number}")
        _object(event, f"line {number}")
        _check_finite_tree(event, f"line {number}")
        events.append(event)
    metadata = events[0]
    _fields(metadata, RUN_FIELDS, set(), "run header")
    _require(type(metadata["schema_version"]) is int and metadata["schema_version"] == 1,
             "unsupported trace schema_version")
    _require(metadata["type"] == "run", "first event must be run")
    for key in RUN_FIELDS - {"schema_version"}:
        _text(metadata[key], f"run.{key}")
    _require(metadata["provider"] == "anthropic", "only provider anthropic is supported")
    _require(metadata["evidence_type"] in ("synthetic", "observed"), "unsupported run evidence_type")
    ending = events[-1]
    _fields(ending, {"type", "complete", "request_count", "task_count"}, set(), "run_end")
    _require(ending["type"] == "run_end" and ending["complete"] is True, "trace must end with complete=true run_end")
    for key in ("request_count", "task_count"):
        _count(ending[key], f"run_end.{key}")

    requests, tasks = [], []
    request_ids, ended_tasks, requested_tasks, sequences = set(), set(), set(), {}
    for event in events[1:-1]:
        if event.get("type") == "request":
            _fields(event, REQUEST_FIELDS, REQUEST_OPTIONAL, "request")
            for key in ("request_id", "task_id", "session_id"):
                _text(event[key], f"request.{key}")
            request_id, task_id, session_id = event["request_id"], event["task_id"], event["session_id"]
            _require(request_id not in request_ids, f"duplicate request_id: {request_id}")
            _require(task_id not in ended_tasks, f"request after task_end: {task_id}")
            sequence = _count(event["sequence"], "request.sequence")
            _require(sequence > sequences.get(session_id, -1), f"sequence must strictly increase per session: {session_id}")
            _require(event["kind"] in ("inference", "compaction"), "unsupported request kind")
            if "latency_ms" in event:
                _duration(event["latency_ms"], "request.latency_ms")
            if "prefix_fingerprint" in event:
                _text(event["prefix_fingerprint"], "request.prefix_fingerprint")
            if "prefix_change_reason" in event:
                _text(event["prefix_change_reason"], "request.prefix_change_reason")
                _require(event["prefix_change_reason"] in PREFIX_REASONS, "unsupported prefix_change_reason")
            normalized = deepcopy(event)
            normalized["usage"] = normalize_usage(event["usage"])
            requests.append(normalized)
            sequences[session_id] = sequence
            request_ids.add(request_id)
            requested_tasks.add(task_id)
        elif event.get("type") == "task_end":
            _fields(event, {"type", "task_id", "success"}, {"duration_ms"}, "task_end")
            _text(event["task_id"], "task_end.task_id")
            _require(type(event["success"]) is bool, "task_end.success must be a bool")
            _require(event["task_id"] not in ended_tasks, f"duplicate task_end: {event['task_id']}")
            if "duration_ms" in event:
                _duration(event["duration_ms"], "task_end.duration_ms")
            tasks.append(deepcopy(event))
            ended_tasks.add(event["task_id"])
        else:
            raise ContextDataError("only request and task_end events may appear between run and run_end")
    _require(requested_tasks <= ended_tasks, "request task missing terminal task_end")
    _require(ending["request_count"] == len(requests), "run_end request_count does not match trace")
    _require(ending["task_count"] == len(tasks), "run_end task_count does not match trace")
    return Trace(deepcopy(metadata), requests, tasks, sha256(raw).hexdigest())


def load_pricing(path) -> Pricing:
    """Load user-supplied flat rates, retaining exact decimal input values."""
    raw = Path(path).read_bytes()
    value = _parse(raw, "pricing", decimal=True)
    required = {"schema_version", "pricing_id", "currency", "source", "effective_date", "provider", "model", "rates_usd_per_million"}
    _fields(value, required, set(), "pricing")
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1, "unsupported pricing schema_version")
    for key in required - {"schema_version", "rates_usd_per_million"}:
        _text(value[key], f"pricing.{key}")
    _require(value["currency"] == "USD", "only USD price cards are supported")
    _require(value["provider"] == "anthropic", "only anthropic price cards are supported")
    _require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value["effective_date"]), "effective_date must be YYYY-MM-DD")
    try:
        date.fromisoformat(value["effective_date"])
    except ValueError as exc:
        raise ContextDataError("invalid effective_date") from exc
    _fields(value["rates_usd_per_million"], set(RATE_KEYS), set(), "rates_usd_per_million")
    rates = {}
    for key, number in value["rates_usd_per_million"].items():
        _require(type(number) in (str, int, Decimal), f"rate {key} must be a decimal string or finite number (not bool)")
        if isinstance(number, str):
            _require(bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", number)), f"invalid decimal rate: {key}")
        try:
            rate = Decimal(number)
        except (InvalidOperation, ValueError) as exc:
            raise ContextDataError(f"invalid decimal rate: {key}") from exc
        _require(rate.is_finite() and rate >= 0, f"rate {key} must be finite and nonnegative")
        rates[key] = rate
    metadata = {key: item for key, item in value.items() if key != "rates_usd_per_million"}
    return Pricing(metadata, rates, sha256(raw).hexdigest())
