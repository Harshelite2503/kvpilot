"""Predeclared synthetic lifecycle workloads, not production traffic or GPU data.

Current observations and future outcomes use separate deterministic RNG streams.
Only current observations are attached to pause events. A full trace necessarily
contains future terminal events, but a causal replay sees them only in time order.
Evaluation seeds are guarded to avoid generating the held-out traces by accident.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

from .policy import CodecProfile, CostModel, MIB


DEFAULT_PROTOCOL = Path(__file__).resolve().parents[2] / "examples" / "holdout-protocol.json"


def load_protocol(path: str | Path = DEFAULT_PROTOCOL) -> dict:
    """Load declared assumptions without generating any workload or outcome."""
    protocol = json.loads(Path(path).read_text())
    if protocol.get("evidence_type") != "analytical_assumptions":
        raise ValueError("workload protocol must declare analytical_assumptions")
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported workload protocol schema")
    if not protocol.get("scenarios"):
        raise ValueError("protocol must declare scenarios")
    development = protocol["development_seeds"]
    evaluation = protocol["evaluation_seeds"]
    if set(development) & set(evaluation):
        raise ValueError("development and evaluation seeds must be disjoint")
    for name in protocol["scenarios"]:
        scenario_config(protocol, name)
    return protocol


def scenario_config(protocol: dict, scenario_name: str) -> dict:
    """Return a fresh, complete replay.load_config-compatible cost dictionary."""
    try:
        scenario = protocol["scenarios"][scenario_name]
    except KeyError as exc:
        raise ValueError(f"unknown workload scenario {scenario_name!r}") from exc
    config = deepcopy(protocol["base_cost_config"])
    config["cost_model"].update(scenario.get("cost_model_overrides", {}))
    # Instantiate for schema validation; unknown CostModel fields are rejected.
    CostModel(**config["cost_model"])
    for profile in config["codecs"]:
        CodecProfile(**profile)
    return config


def _rng(seed: int, scenario_name: str, stream: str) -> random.Random:
    # Python's process-randomized hash() would not be a reproducible seed.
    key = f"kvpilot-workloads-v1:{seed}:{scenario_name}:{stream}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))


def _settings(protocol: dict, scenario_name: str) -> dict:
    try:
        scenario = protocol["scenarios"][scenario_name]
    except KeyError as exc:
        raise ValueError(f"unknown workload scenario {scenario_name!r}") from exc
    settings = deepcopy(protocol["workload_defaults"])
    settings.update(deepcopy(scenario["workload"]))
    return settings


def _phase_settings(settings: dict, fraction: float) -> dict:
    result = dict(settings)
    for phase in settings.get("phases", []):
        if fraction >= phase["start_fraction"]:
            result.update(phase)
    return result


def generate_workload(
    scenario_name: str,
    seed: int,
    *,
    protocol: dict | None = None,
    sessions: int | None = None,
    allow_evaluation: bool = False,
) -> list[dict]:
    """Generate a deterministic, fully resolved trace using the current schema.

    ``allow_evaluation`` is an explicit lock, not a cryptographic seal. The caller
    must separately freeze/hash policy and evaluator artifacts before unlocking.
    Session-count overrides are useful for development; the declared evaluation
    always uses ``sessions_per_run``. IDs identify siblings, never future labels.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    protocol = load_protocol() if protocol is None else protocol
    if seed in protocol["evaluation_seeds"] and not allow_evaluation:
        raise ValueError("evaluation seeds are sealed until an explicit freeze/unlock")
    count = protocol["sessions_per_run"] if sessions is None else sessions
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("sessions must be a positive integer")
    if seed in protocol["evaluation_seeds"] and count != protocol["sessions_per_run"]:
        raise ValueError("evaluation session count must match the frozen protocol")
    settings = _settings(protocol, scenario_name)
    family_size = settings["family_size"]
    if not isinstance(family_size, int) or isinstance(family_size, bool) or family_size <= 0:
        raise ValueError("family_size must be a positive integer")
    observations = _rng(seed, scenario_name, "observations")
    outcomes = _rng(seed, scenario_name, "outcomes")
    events: list[dict] = []
    family_at_ms = 0.0
    emitted = 0

    for family in range(math.ceil(count / family_size)):
        current = _phase_settings(settings, emitted / count)
        siblings = min(family_size, count - emitted)
        family_mib = observations.choice(current["raw_mib_values"])
        family_bytes = int(family_mib * MIB)
        shared_bytes = int(family_bytes * current["shared_fraction"])
        tail_bytes = family_bytes - shared_bytes
        family_id = f"{scenario_name}-s{seed}-f{family:04d}"
        namespace = {
            "tenant": f"synthetic-tenant-{family % 3}",
            "model_revision": "synthetic-model@workloads-v1",
            "adapter_revision": "none",
            "layout": "synthetic-fp16-kv-v1",
        }
        # A common latent draw induces family correlation without exposing it.
        family_cancel_draw = outcomes.random()
        family_wait_draw = outcomes.random()
        for branch in range(siblings):
            sid = f"{family_id}-b{branch}"
            at_ms = round(family_at_ms + branch * current["branch_gap_ms"], 6)
            kind = "human" if observations.random() < current["human_fraction"] else "tool"
            recompute_rate = observations.choice(current["recompute_ms_per_mib_values"])
            recompute_ms = family_mib * recompute_rate * observations.uniform(0.85, 1.15)
            blocks = []
            if shared_bytes:
                # Two shared physical prefix pages plus a private suffix.
                first = shared_bytes // 2
                blocks.extend([
                    {"prefix_hash": f"{family_id}-shared-0", "raw_bytes": first},
                    {"prefix_hash": f"{family_id}-shared-1", "raw_bytes": shared_bytes - first},
                ])
            if tail_bytes:
                blocks.append({"prefix_hash": f"{sid}-suffix", "raw_bytes": tail_bytes})
            events.append({
                "at_ms": at_ms,
                "type": "pause",
                "session_id": sid,
                "pause_kind": kind,
                "namespace": dict(namespace),
                "blocks": blocks,
                "recompute_ms": round(recompute_ms, 6),
            })

            cancel_probability = 1.0 - current["resume_probability_by_kind"][kind]
            correlation = current["family_outcome_correlation"]
            common_cancel_probability = cancel_probability * correlation
            individual_cancel_probability = (
                (cancel_probability - common_cancel_probability) / (1.0 - common_cancel_probability)
                if common_cancel_probability < 1.0 else 0.0
            )
            individual_cancel_draw = outcomes.random()
            canceled = (family_cancel_draw < common_cancel_probability
                        or individual_cancel_draw < individual_cancel_probability)
            # A common family draw selects the wait bin for correlated siblings.
            individual_wait_draw = outcomes.random()
            wait_draw = family_wait_draw if family_size > 1 else individual_wait_draw
            waits = current["wait_ms_by_kind"][kind]
            wait = waits[min(int(wait_draw * len(waits)), len(waits) - 1)]
            jitter = current["time_jitter_fraction"]
            wait *= outcomes.uniform(1.0 - jitter, 1.0 + jitter)
            wait *= current.get("wait_scale", 1.0)
            if canceled:
                wait *= current["cancel_wait_scale"]
            events.append({
                "at_ms": round(at_ms + max(wait, 0.000001), 6),
                "type": "cancel" if canceled else "resume",
                "session_id": sid,
            })
            emitted += 1
        family_at_ms += current["interarrival_ms"] * siblings * observations.uniform(0.8, 1.2)

    # Resolve an earlier session before admitting a new one at an exact time tie.
    events.sort(key=lambda event: (event["at_ms"], event["type"] == "pause", event["session_id"]))
    return events
