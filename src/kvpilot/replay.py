"""Causal analytical trace replay. No tensors, LLMs, PCIe or GPU are executed.

Finite paused-only host store; shared blocks are counted once. Offloads start at
pause and overlap the wait. No codec/PCIe queue or active decode is modeled. A
resume with any missing block uses full recomputation (conservative fallback).
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

from .policy import CodecProfile, CostModel, HistoricalForecast, choose_action, positive


@dataclass(frozen=True)
class BlockKey:
    tenant: str
    model_revision: str
    adapter_revision: str
    layout: str
    prefix_hash: str


@dataclass
class Entry:
    raw_bytes: int
    stored_bytes: int
    codec: str
    ready_ms: float
    decode_ms: float


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    items = sorted(values)
    position = (len(items) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return items[lo] + (items[hi] - items[lo]) * (position - lo)


def load_config(path: str | Path):
    config = json.loads(Path(path).read_text())
    if config.get("evidence_type") != "analytical_assumptions":
        raise ValueError("cost config must explicitly declare evidence_type=analytical_assumptions")
    return CostModel(**config["cost_model"]), [CodecProfile(**p) for p in config["codecs"]], config


def load_trace(path: str | Path):
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {number}") from exc


def replay(events, model: CostModel, profiles: list[CodecProfile], *, policy: str = "adaptive", default_pause_ms: float = 1000.0):
    history = HistoricalForecast(default_pause_ms)
    store: OrderedDict[BlockKey, Entry] = OrderedDict()
    paused: dict[str, dict] = {}
    known_sizes: dict[BlockKey, int] = {}
    refs: dict[BlockKey, set[str]] = {}
    log = []
    delays = []
    used = peak = shared = evictions = recomputes = canceled = 0
    transferred = 0
    encode_ms = decode_ms = recompute_total_ms = wasted_encode_upper_bound = 0.0
    last_time = -1.0
    profile_map = {p.name: p for p in profiles}
    if len(profile_map) != len(profiles):
        raise ValueError("duplicate codec profile")
    if policy not in {"adaptive", "recompute", *profile_map}:
        raise ValueError(f"unknown policy {policy}")
    host_byte_ms = 0.0

    for event in events:
        timestamp = float(event["at_ms"])
        positive(timestamp, "event at_ms", zero=True)
        if timestamp < last_time:
            raise ValueError("trace must be ordered by at_ms")
        if last_time >= 0:
            host_byte_ms += used * (timestamp - last_time)
        last_time = timestamp
        event_type, sid = event["type"], event["session_id"]
        if not isinstance(sid, str) or not sid:
            raise ValueError("session_id must be nonempty")
        if event_type == "pause":
            if sid in paused:
                raise ValueError(f"session {sid} is already paused")
            namespace = event["namespace"]
            names = [namespace[k] for k in ("tenant", "model_revision", "adapter_revision", "layout")]
            if any(not isinstance(v, str) or not v for v in names):
                raise ValueError("all namespace fields must be nonempty strings")
            blocks = event["blocks"]
            if not blocks:
                raise ValueError("a pause needs at least one block")
            keys = []
            for block in blocks:
                size, digest = block["raw_bytes"], block["prefix_hash"]
                if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                    raise ValueError("block raw_bytes must be a positive integer")
                if not isinstance(digest, str) or not digest:
                    raise ValueError("prefix_hash must be nonempty")
                key = BlockKey(*names, digest)
                if key in keys:
                    raise ValueError("duplicate block within a pause")
                if key in known_sizes and known_sizes[key] != size:
                    raise ValueError("same compatible block identity has inconsistent byte size")
                known_sizes[key] = size
                keys.append(key)
            recompute_ms = float(event["recompute_ms"])
            positive(recompute_ms, "recompute_ms", zero=True)
            raw_total = sum(known_sizes[k] for k in keys)
            kind = str(event.get("pause_kind", "unknown"))
            forecast = history.predict(kind)
            pause_encoding = 0.0
            actions = []
            # Shared resident blocks are never recompressed by a second branch.
            missing = [k for k in keys if k not in store]
            for key in keys:
                refs.setdefault(key, set()).add(sid)
                if key in store:
                    shared += 1
                    store.move_to_end(key)
            if missing:
                missing_bytes = sum(known_sizes[k] for k in missing)
                protected = {k for k in keys if k in store}
                protected_bytes = sum(store[k].stored_bytes for k in protected)
                available = model.host_budget_bytes - protected_bytes
                decision_model = replace(model, host_budget_bytes=max(1, available))
                shared_restore = sum(store[k].stored_bytes / model.bandwidth_bytes_per_ms + store[k].decode_ms for k in protected)
                shared_wait = max([max(0, store[k].ready_ms - timestamp - forecast.pause_ms) for k in protected], default=0)
                decision = choose_action(missing_bytes, recompute_ms, profiles, decision_model, forecast,
                                         force_codec=None if policy == "adaptive" else policy,
                                         shared_restore_ms=shared_restore, shared_wait_ms=shared_wait)
                chosen = decision.selected
                actions.append(asdict(decision))
                if chosen.action != "recompute":
                    cp = profile_map[chosen.action]
                    sizes = {k: math.ceil(known_sizes[k] / cp.ratio) for k in missing}
                    required = sum(sizes.values())
                    # Round each physical page separately; total can exceed rounded aggregate.
                    if required <= available:
                        while store and used + required > model.host_budget_bytes:
                            old_key = next(k for k in store if k not in protected)
                            old = store.pop(old_key)
                            used -= old.stored_bytes
                            evictions += 1
                        ready = timestamp + chosen.encode_ms + required / model.bandwidth_bytes_per_ms
                        for key in missing:
                            size = sizes[key]
                            store[key] = Entry(known_sizes[key], size, cp.name, ready,
                                               known_sizes[key] / (1024 * 1024) * cp.decode_ms_per_mib)
                            used += size
                        transferred += required
                        encode_ms += chosen.encode_ms
                        pause_encoding = chosen.encode_ms
                        peak = max(peak, used)
                    else:
                        actions.append({"reason": "per-page rounding exceeds budget; recompute fallback"})
            paused[sid] = {"keys": keys, "at_ms": timestamp, "kind": kind,
                           "recompute_ms": recompute_ms, "encoding_ms": pause_encoding}
            log.append({"at_ms": timestamp, "type": "pause", "session_id": sid,
                        "forecast": asdict(forecast), "actions": actions,
                        "host_reserved_bytes": used, "raw_bytes": raw_total})
        elif event_type in {"resume", "cancel"}:
            if sid not in paused:
                raise ValueError(f"{event_type} without a matching pause: {sid}")
            pause = paused.pop(sid)
            keys = pause["keys"]
            history.observe(pause["kind"], event_type == "resume", timestamp - pause["at_ms"])
            if event_type == "resume":
                complete = all(k in store for k in keys)
                entries = [store[k] for k in keys] if complete else []
                wait_ms = max([max(0, e.ready_ms - timestamp) for e in entries], default=0)
                load_bytes = sum(e.stored_bytes for e in entries)
                decode = sum(e.decode_ms for e in entries)
                restore = wait_ms + load_bytes / model.bandwidth_bytes_per_ms + decode
                # Compare against full recompute even on a cache hit. Source/raw tensors
                # retained during offload are NOT a separately modeled fast path.
                use_cache = complete and restore <= pause["recompute_ms"]
                delay = restore if use_cache else pause["recompute_ms"]
                if use_cache:
                    transferred += load_bytes
                    decode_ms += decode
                else:
                    recomputes += 1
                    recompute_total_ms += delay
                delays.append(delay)
                log.append({"at_ms": timestamp, "type": "resume", "session_id": sid,
                            "path": "restore" if use_cache else "full_recompute",
                            "modeled_resume_delay_ms": delay, "host_reserved_bytes": used,
                            "within_resume_slo": delay <= model.resume_slo_ms})
            else:
                canceled += 1
                wasted_encode_upper_bound += pause["encoding_ms"]
                log.append({"at_ms": timestamp, "type": "cancel", "session_id": sid})
            for key in keys:
                refs[key].discard(sid)
                if not refs[key]:
                    refs.pop(key)
                    if key in store:
                        used -= store.pop(key).stored_bytes
        else:
            raise ValueError(f"unknown event type {event_type!r}")
    return {
        "schema_version": 1, "evidence_type": "analytical_trace_replay",
        "policy": policy, "quality_validated": False,
        "cost_model": asdict(model), "codec_profiles": [asdict(p) for p in profiles],
        "limitations": ["No live LLM, GPU, PCIe, or LMCache execution", "No active-decode latency or concurrent service queues",
                        "Supplied ratios/times are assumptions; prediction uses resolved history only",
                        "Paused-only store; complete-cache restore or full recompute; no partial-prefill execution",
                        "Encoding is charged once in full, including canceled/evicted jobs; no preemption savings",
                        "Host reservation starts at pause; raw GPU staging capacity is not modeled",
                        "Paused-only host accounting ends at resume/cancel and excludes in-flight restore buffers",
                        "Fixed codec policies and simple historical expected-cost policy are reference baselines, not production schedulers"],
        "metrics": {"resume_count": len(delays), "cancel_count": canceled, "unresolved_count": len(paused),
                    "modeled_p50_resume_ms": percentile(delays, 0.5), "modeled_p95_resume_ms": percentile(delays, 0.95),
                    "modeled_slo_violations": sum(d > model.resume_slo_ms for d in delays),
                    "modeled_transferred_bytes": transferred, "peak_host_reserved_bytes": peak,
                    "host_reserved_byte_ms": host_byte_ms, "shared_block_hits": shared, "evicted_blocks": evictions,
                    "full_recomputes": recomputes, "modeled_encoding_ms": encode_ms,
                    "modeled_decoding_ms": decode_ms, "modeled_recompute_ms": recompute_total_ms,
                    "canceled_owner_encoding_ms_upper_bound": wasted_encode_upper_bound,
                    "measured_gpu_itl_ms": None, "measured_llm_task_success": None},
        "decisions": log,
    }
