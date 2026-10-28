"""Counterfactual codec-cost screening from an audited owned-runner archive.

No tensor engine, inference, transfer or real cache payload is executed here.
The sequential schedule and physical byte sizes come from the source run. Only
lossy encode/decode time is scaled; native framing and every export/import stay
measured-cost estimates. This is an opportunity screen, never a speed benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics

from experiments.local.store import PausedStateStore, PausePlan
from kvpilot.policy import CodecProfile, CostModel, MIB
from kvpilot.replay import Entry, plan_lru_evictions
from scripts.validate_p100_results import audit

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ("recompute", "raw_native", "raw_admit_if_fits", "group_int4",
            "adaptive_exact", "adaptive_eviction")
DEFAULT_SCALES = (0.0, .01, .03, .1, .25, .5, .75, 1.0)
COMPONENTS = ("export_ms", "encode_ms", "decode_ms", "import_ms")
LIMITATIONS = [
    "All reported sweep times are modeled counterfactuals, not measured speedups or dollars.",
    "No inference is rerun: the sweep cannot certify answer quality, latency SLOs, or throughput.",
    "Peak stored bytes count complete serialized host payloads, not active KV or GPU memory.",
    "Only lossy codec encode/decode costs scale; native framing and all measured export/import costs remain.",
    "Per-context component medians are normalized by native tensor MiB; scaling them linearly across case sizes is an assumption.",
    "Recompute-arm initial prefill and generation are held identical across policies. Free-generation inputs do not become a live equal-work experiment.",
    "Policy lifecycle overhead is reused from that policy's measured rows; missing raw_admit_if_fits overhead uses raw_native rows. Counterfactual overhead can differ.",
    "Every admitted payload pays export/encode, including later evicted or canceled payloads. Misses pay actual recompute-arm recomputation. Predicted eviction penalties are not charged twice.",
    "Events are synchronous and ordered as recorded. The modeled clock includes all work; no production idle intervals, queues, overlap or contention are inferred.",
    "Future resolution and recompute timings are used only when their events occur; admission receives current initial-prefill cost, shape, family and completed history.",
    "A grid crossing is only a bracket. Policy switches can make results nonmonotonic; no interpolated break-even cost is claimed.",
    "Scale one need not reproduce the live run: calibration costs, common generation work and modeled event times replace actual per-operation timing.",
]


def _finite(value, name):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class PayloadLength:
    """A byte count only; deliberately carries no cache bytes."""
    size: int

    def __post_init__(self):
        _positive_integer(self.size, "payload length")

    def __len__(self):
        return self.size


class MetadataStore(PausedStateStore):
    """Use the actual policy/history/LRU code without allocating fake tensors.

    Only commit's bytes-specific validation is replaced with a length sentinel.
    decide(), resolve(), accounting and eviction helpers are inherited unchanged.
    Differential tests cover this private-interface coupling against real bytes.
    """
    def commit(self, plan, payload, *, timestamp_ms=None):
        if not isinstance(plan, PausePlan) or plan._store_token is not self._token:
            raise ValueError("pause plan belongs to a different store")
        if plan._version != self._version or plan.session_id in self._paused:
            raise ValueError("stale pause plan; decide again before encoding")
        timestamp = self._time(plan.timestamp_ms if timestamp_ms is None else timestamp_ms)
        if timestamp < plan.timestamp_ms:
            raise ValueError("commit cannot precede the pause decision")
        selected = plan.selected
        if selected.action == "recompute":
            if payload is not None or selected.stored_bytes != 0 or plan.victims:
                raise ValueError("recompute must commit no payload and evict no owners")
        else:
            if type(payload) is not PayloadLength or len(payload) != selected.stored_bytes:
                raise ValueError("metadata length must match exact declared size")
            victims = plan_lru_evictions(self._entries, self.used_bytes,
                                        self.cost_model.host_budget_bytes, len(payload), set())
            if victims is None or victims != plan.victims:
                raise ValueError("eviction plan no longer matches actual admission")
        for victim in plan.victims:
            self._remove_payload(victim)
        self._evicted_states += len(plan.victims)
        if payload is not None:
            self._payloads[plan.session_id] = payload
            self._entries[plan.session_id] = Entry(
                plan.raw_bytes, len(payload), selected.action, timestamp,
                plan.raw_bytes / MIB * self._profile_map[selected.action].decode_ms_per_mib)
            self._used_bytes += len(payload)
        self._paused[plan.session_id] = {
            "keys": (plan.session_id,), "kind": plan.kind,
            "at_ms": plan.timestamp_ms, "recompute_ms": plan.recompute_ms}
        self._record_event(timestamp)


def component_rates(calibration):
    """Separate median ms/native-MiB components, never fake zero-copy costs."""
    rates = {}
    for context, record in calibration.items():
        samples = record["samples"]
        rates[context] = {}
        for codec in ("raw_native", "group_int8", "group_int4"):
            chosen = [s for s in samples if s["codec"] == codec]
            if not chosen:
                raise ValueError(f"missing {codec} calibration samples for context {context}")
            values = {component: [] for component in COMPONENTS}
            ratios = []
            for sample in chosen:
                raw = _positive_integer(sample["raw_tensor_bytes"], "raw tensor bytes")
                stored = _positive_integer(sample["stored_bytes"], "stored bytes")
                ratios.append(raw / stored)
                for component in COMPONENTS:
                    values[component].append(_finite(sample[component], component) / (raw / MIB))
            rates[context][codec] = {component: statistics.median(values[component]) for component in COMPONENTS}
            rates[context][codec]["ratio"] = statistics.median(ratios)
    return rates


def scaled_profiles(rates, scale):
    _finite(scale, "scale")
    if scale > 1:
        raise ValueError("codec scale must be in [0, 1]")
    profiles, scaled = [], {}
    for codec, rate in rates.items():
        multiplier = 1.0 if codec == "raw_native" else scale
        parts = {part: rate[part] * (multiplier if part in ("encode_ms", "decode_ms") else 1.0)
                 for part in COMPONENTS}
        scaled[codec] = parts
        profiles.append(CodecProfile(codec, rate["ratio"], parts["export_ms"] + parts["encode_ms"],
                                     parts["decode_ms"] + parts["import_ms"], codec == "raw_native"))
    return profiles, scaled


def _residual(total, parts, name):
    # Small rounding differences are not negative work. Larger inconsistencies
    # must be rejected even if the archive's original validator did not check it.
    total = _finite(total, name)
    difference = total - sum(_finite(part, name) for part in parts)
    if difference < -1e-5:
        raise ValueError(f"negative measured lifecycle residual: {name}")
    return max(0.0, difference)


def replay_block(*, context, scenario, repeat, policy, scale, budget, schedule,
                 control_rows, overhead_rows, rates):
    """Causal policy replay. Only current/past event data reaches decide()."""
    profiles, scaled = scaled_profiles(rates, scale)
    store = MetadataStore(policy, profiles, CostModel(
        1e30, budget, resume_slo_ms=1e9, allow_unvalidated_lossy=True), default_pause_ms=30000)
    identity = dict(context_target=context, scenario=scenario, repeat=repeat, policy=policy, codec_scale=scale)
    totals = Counter({name: 0.0 for name in (*COMPONENTS, "initial_prefill_ms", "recompute_ms", "generation_ms", "lifecycle_overhead_ms")})
    actions, paused, events = Counter(), {}, []
    hits = misses = cancellations = 0
    clock = 0.0
    for sequence, (event, name) in enumerate(schedule):
        row, overhead = control_rows[name], overhead_rows[name]
        event_record = {**identity, "sequence": sequence, "event": event, "case_id": name}
        if event == "pause":
            raw_bytes = _positive_integer(row["raw_tensor_bytes"], "raw tensor bytes")
            prefill = _finite(row["initial_prefill_ms"], "initial prefill")
            totals["initial_prefill_ms"] += prefill
            clock += prefill
            plan = store.decide(name, raw_bytes, prefill, row["family"], clock, row["exact_candidate_bytes"])
            action = plan.selected.action
            actions[action] += 1
            work = _residual(overhead["pause_ms"], [overhead["export_ms"], overhead["encode_ms"]], "pause overhead")
            totals["lifecycle_overhead_ms"] += work
            for component in ("export_ms", "encode_ms"):
                value = 0.0 if action == "recompute" else raw_bytes / MIB * scaled[action][component]
                totals[component] += value
                work += value
            clock += work
            payload = None if action == "recompute" else PayloadLength(plan.selected.stored_bytes)
            store.commit(plan, payload, timestamp_ms=clock)
            paused[name] = (action, raw_bytes)
            event_record.update(action=action, payload_bytes=plan.selected.stored_bytes,
                                victims=list(plan.victims), forecast=asdict(plan.forecast),
                                selected_prediction=asdict(plan.selected), modeled_pause_ms=work)
        elif event in ("resume", "cancel"):
            action, raw_bytes = paused.pop(name)
            if event != row["resolution"]:
                raise ValueError("control row resolution differs from schedule")
            if event == "cancel":
                store.cancel(name, clock)
                cancellations += 1
                work = _finite(overhead["cancel_ms"], "cancel overhead")
                totals["lifecycle_overhead_ms"] += work
                clock += work
            else:
                payload = store.popresume(name, clock)
                hit = payload is not None
                restore = 0.0
                if hit:
                    hits += 1
                    for component in ("decode_ms", "import_ms"):
                        value = raw_bytes / MIB * scaled[action][component]
                        totals[component] += value
                        restore += value
                else:
                    misses += 1
                    restore = _finite(row["recompute_ms"], "recompute")
                    totals["recompute_ms"] += restore
                generation = _finite(row["generation"]["wall_ms"], "common generation")
                totals["generation_ms"] += generation
                work = _residual(overhead["resume_wall_ms"], [overhead["recompute_ms"], overhead["decode_ms"],
                                  overhead["import_ms"], overhead["generation"]["wall_ms"]], "resume overhead")
                totals["lifecycle_overhead_ms"] += work
                clock += restore + generation + work
                event_record.update(hit=hit, modeled_restore_or_recompute_ms=restore,
                                    common_generation_ms=generation, lifecycle_overhead_ms=work)
        else:
            raise ValueError(f"unsupported event: {event}")
        event_record.update(used_host_payload_bytes=store.used_bytes, modeled_clock_ms=clock)
        events.append(event_record)
    if paused or store.used_bytes or store.paused_sessions:
        raise ValueError("schedule left unresolved sessions")
    total = sum(totals.values())
    if not math.isclose(total, clock, rel_tol=1e-10, abs_tol=1e-7):
        raise RuntimeError("modeled clock/component accounting diverged")
    return {**identity, "sessions": sum(actions.values()), "responses": hits + misses, "cancellations": cancellations,
            "actions": dict(actions), "hits": hits, "misses": misses, "evictions": store.evicted_states,
            "peak_host_payload_bytes": store.peak_bytes, "byte_event_sum": store.byte_event_sum,
            "event_count": store.event_count, "components_modeled_ms": dict(totals),
            "total_modeled_ms": total, "mean_modeled_ms_per_started_session": total / sum(actions.values()),
            "modeled_cache_lifecycle_ms": total - totals["initial_prefill_ms"] - totals["generation_ms"]}, events


def first_break_even_bracket(points):
    """Find an observed adjacent crossing only, without assuming monotonicity."""
    ordered = sorted(points, key=lambda point: point["codec_scale"])
    wins = [p["total_modeled_ms"] < p["recompute_total_modeled_ms"] - 1e-7 for p in ordered]
    crossings = [i for i in range(1, len(ordered)) if wins[i] != wins[i - 1]]
    bracket = None
    if crossings:
        i = crossings[0]
        bracket = {"lower_scale": ordered[i - 1]["codec_scale"], "upper_scale": ordered[i]["codec_scale"],
                   "lower_scale_wins": wins[i - 1], "upper_scale_wins": wins[i]}
    return {"status": "wins_entire_grid" if all(wins) else "never_wins_on_grid" if not any(wins) else "mixed_grid",
            "first_transition_bracket": bracket, "observed_transition_count": len(crossings),
            "wins_by_scale": [{"codec_scale": p["codec_scale"], "wins": win} for p, win in zip(ordered, wins)],
            "comparison": "strictly lower modeled mean full-session time than recompute; ties are not wins"}


def build_sweep(protocol, calibration, rows, scales=DEFAULT_SCALES):
    scales = list(scales)
    if not scales or len(scales) != len(set(scales)):
        raise ValueError("scales must be a nonempty set of unique values")
    for scale in scales:
        if _finite(scale, "scale") > 1:
            raise ValueError("codec scales must be in [0, 1]")
    scales.sort()
    rates = component_rates(calibration)
    index = defaultdict(dict)
    for row in rows:
        key = (row["context_target"], row["scenario"], row["repeat"], row["policy"])
        if row["case_id"] in index[key]:
            raise ValueError("duplicate source row")
        index[key][row["case_id"]] = row
    blocks, event_log, overhead_fallbacks = [], [], set()
    for context, scenario, repeat in sorted({key[:3] for key in index}):
        control = index[(context, scenario, repeat, "recompute")]
        if set(control) != set(protocol["cases"]):
            raise ValueError("incomplete recompute control")
        for policy in POLICIES:
            overhead_policy = policy
            if (context, scenario, repeat, policy) not in index:
                if policy != "raw_admit_if_fits":
                    raise ValueError(f"missing measured overhead arm: {policy}")
                overhead_policy = "raw_native"
                overhead_fallbacks.add("raw_admit_if_fits uses measured raw_native lifecycle overhead")
            overhead = index[(context, scenario, repeat, overhead_policy)]
            if set(overhead) != set(control):
                raise ValueError("incomplete lifecycle overhead rows")
            for scale in scales:
                block, events = replay_block(context=context, scenario=scenario, repeat=repeat,
                    policy=policy, scale=scale, budget=protocol["budgets_bytes"][str(context)][scenario],
                    schedule=protocol["schedules"][scenario], control_rows=control,
                    overhead_rows=overhead, rates=rates[str(context)])
                blocks.append(block)
                event_log.extend(events)
    groups = defaultdict(list)
    for block in blocks:
        groups[(block["context_target"], block["scenario"], block["policy"], block["codec_scale"])].append(block)
    summaries = []
    for (context, scenario, policy, scale), group in sorted(groups.items()):
        total, sessions = sum(b["total_modeled_ms"] for b in group), sum(b["sessions"] for b in group)
        actions = Counter()
        for block in group:
            actions.update(block["actions"])
        summaries.append({"context_target": context, "scenario": scenario, "policy": policy, "codec_scale": scale,
                          "sessions": sessions, "total_modeled_ms": total,
                          "mean_modeled_ms_per_started_session": total / sessions,
                          "peak_host_payload_bytes": max(b["peak_host_payload_bytes"] for b in group),
                          "hits": sum(b["hits"] for b in group), "evictions": sum(b["evictions"] for b in group),
                          "actions": dict(actions)})
    references = {(s["context_target"], s["scenario"], s["codec_scale"]): s["total_modeled_ms"]
                  for s in summaries if s["policy"] == "recompute"}
    for summary in summaries:
        reference = references[(summary["context_target"], summary["scenario"], summary["codec_scale"])]
        summary["recompute_total_modeled_ms"] = reference
        summary["modeled_time_reduction_vs_recompute_fraction"] = (reference - summary["total_modeled_ms"]) / reference if reference else None
    brackets = []
    for context, scenario, policy in sorted({(s["context_target"], s["scenario"], s["policy"]) for s in summaries if s["policy"] != "recompute"}):
        subset = [s for s in summaries if (s["context_target"], s["scenario"], s["policy"]) == (context, scenario, policy)]
        result = {"context_target": context, "scenario": scenario, "policy": policy, **first_break_even_bracket(subset)}
        bracket = result["first_transition_bracket"]
        result["codec_component_brackets_ms_per_native_mib"] = None if bracket is None else {
            codec: {component: [rates[str(context)][codec][component] * bracket[endpoint]
                                for endpoint in ("lower_scale", "upper_scale")]
                    for component in ("encode_ms", "decode_ms")}
            for codec in ("group_int8", "group_int4")}
        brackets.append(result)
    scaled_rates = {context: [{"codec_scale": scale, "components_ms_per_native_mib": scaled_profiles(rate, scale)[1]}
                              for scale in scales] for context, rate in rates.items()}
    return {"schema_version": 1, "evidence_type": "counterfactual_owned_cache_codec_cost_sweep",
            "measured_performance": False, "quality_certified": False, "dollar_savings_estimated": False,
            "input_timing_mode": protocol.get("timing_mode", protocol.get("args", {}).get("timing_mode", "free_generation")),
            "codec_scales": scales, "limitations": LIMITATIONS, "overhead_fallbacks": sorted(overhead_fallbacks),
            "base_components_ms_per_native_mib": rates, "scaled_components_ms_per_native_mib": scaled_rates,
            "blocks": blocks, "summaries": summaries, "break_even_brackets": brackets}, event_log


def render_report(result):
    lines = ["# Counterfactual codec-cost sweep", "", "**Modeled opportunity screen. No measured speedup, GPU-memory saving, quality guarantee or dollar saving is reported.**", "",
             f"Input timing mode: `{result['input_timing_mode']}`. Scales apply only to lossy encode/decode; native framing and all export/import costs remain.", "",
             "All admitted cache work is charged even if its payload is later canceled or evicted. Recompute-arm generation work is held common. Full component costs and per-event decisions are in the JSON artifacts.", "",
             "| Context | Scenario | Policy | Grid result vs recompute | First observed transition bracket |", "| ---: | --- | --- | --- | --- |"]
    for row in result["break_even_brackets"]:
        bracket = row["first_transition_bracket"]
        label = "none observed" if bracket is None else f"[{bracket['lower_scale']:g}, {bracket['upper_scale']:g}] ({'win' if bracket['lower_scale_wins'] else 'no win'} → {'win' if bracket['upper_scale_wins'] else 'no win'})"
        lines.append(f"| {row['context_target']} | {row['scenario']} | {row['policy']} | {row['status']} | {label} |")
    lines += ["", "Brackets are scale ranges, not interpolated thresholds. Consult `scaled_components_ms_per_native_mib` for encode/decode ms/MiB at each endpoint. No monotonicity is assumed.", "", "Assumptions and limits:", ""]
    lines.extend(f"- {limitation}" for limitation in result["limitations"])
    return "\n".join(lines) + "\n"


def _read(path):
    return json.loads(path.read_text())


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(run_dir, output, scales=DEFAULT_SCALES):
    run_dir, output = Path(run_dir).resolve(), Path(output)
    if output.exists():
        raise ValueError("output already exists; preserve it and use a new directory")
    # Audit first: an invalid input must not leave an apparently usable output.
    manifest_bytes = (run_dir / "artifact-sha256.json").read_bytes()
    manifest_digest = sha256(manifest_bytes).hexdigest()
    validation = audit(run_dir)
    if sha256((run_dir / "artifact-sha256.json").read_bytes()).hexdigest() != manifest_digest:
        raise ValueError("input manifest changed during audit")
    manifest = json.loads(manifest_bytes)
    input_hashes = {name: sha256((run_dir / name).read_bytes()).hexdigest() for name in manifest}
    if input_hashes != manifest:
        raise ValueError("input artifacts changed after audit")
    input_hashes["artifact-sha256.json"] = manifest_digest
    source_paths = ["experiments/torch/break_even.py", "experiments/local/store.py", "src/kvpilot/policy.py",
                    "src/kvpilot/replay.py", "scripts/validate_p100_results.py", "scripts/sweep_p100_break_even.py"]
    source_hashes = {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths}
    protocol = _read(run_dir / "protocol.json")
    modes = {r.get("timing_mode", protocol.get("timing_mode", protocol.get("args", {}).get("timing_mode", "free_generation")))
             for r in protocol["jobs"]}
    if len(modes) != 1:
        raise ValueError("mixed timing modes: sweep separate single-mode run directories")
    rows = [json.loads(line) for line in (run_dir / "rows.jsonl").read_text().splitlines()]
    result, events = build_sweep(protocol, _read(run_dir / "calibration.json"), rows, scales)
    result["provenance"] = {"input_run_directory": str(run_dir), "input_artifact_sha256": input_hashes,
                            "input_source_sha256": protocol["source_sha256"], "sweep_source_sha256": source_hashes,
                            "input_validation": validation, "source_device": protocol["device"],
                            "source_model_identity": protocol["model_identity"]}
    if any(sha256((run_dir / name).read_bytes()).hexdigest() != digest for name, digest in input_hashes.items()):
        raise ValueError("input changed during sweep")
    if any(sha256((ROOT / name).read_bytes()).hexdigest() != digest for name, digest in source_hashes.items()):
        raise ValueError("sweep source changed during execution")
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "sweep.json", result)
    (output / "events.jsonl").write_text("".join(json.dumps(event, allow_nan=False) + "\n" for event in events))
    (output / "report.md").write_text(render_report(result))
    _write(output / "artifact-sha256.json", {name: sha256((output / name).read_bytes()).hexdigest()
                                            for name in ("sweep.json", "events.jsonl", "report.md")})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="completed, independently auditable owned-runner directory")
    parser.add_argument("--output", type=Path, required=True, help="new directory; never overwrites results")
    parser.add_argument("--scales", type=float, nargs="+", default=list(DEFAULT_SCALES), help="unique lossy codec cost multipliers in [0, 1]")
    args = parser.parse_args(argv)
    try:
        result = run(args.run_dir, args.output, args.scales)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(1, f"Break-even sweep rejected: {error}\n")
    print(json.dumps({"evidence_type": result["evidence_type"], "measured_performance": False,
                      "output": str(args.output), "modeled_blocks": len(result["blocks"]),
                      "grid_points": len(result["codec_scales"])}))
