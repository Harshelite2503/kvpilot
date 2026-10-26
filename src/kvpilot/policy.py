"""Transparent expected-cost screening policy. All times are supplied estimates.

This deliberately avoids claiming a scheduling, quality, or GPU performance model.
Profiles must be calibrated on the intended hardware before deployment.
"""

from dataclasses import dataclass
import math
from typing import Mapping

MIB = 1024 * 1024


def positive(value: float, name: str, *, zero: bool = False) -> None:
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


@dataclass(frozen=True)
class CodecProfile:
    name: str
    ratio: float
    encode_ms_per_mib: float
    decode_ms_per_mib: float
    lossless: bool = True

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or self.name in {"recompute", "adaptive", "adaptive_exact", "adaptive_eviction"}:
            raise ValueError("codec name must be a nonempty string and not a reserved policy name")
        if not isinstance(self.lossless, bool):
            raise ValueError("lossless must be a Boolean")
        positive(self.ratio, "ratio")
        positive(self.encode_ms_per_mib, "encode_ms_per_mib", zero=True)
        positive(self.decode_ms_per_mib, "decode_ms_per_mib", zero=True)


@dataclass(frozen=True)
class Forecast:
    resume_probability: float = 0.5
    pause_ms: float = 1000.0

    def __post_init__(self):
        if not math.isfinite(self.resume_probability) or not 0 <= self.resume_probability <= 1:
            raise ValueError("resume probability must be in [0, 1]")
        positive(self.pause_ms, "pause_ms", zero=True)


@dataclass(frozen=True)
class CostModel:
    bandwidth_bytes_per_ms: float
    host_budget_bytes: int
    storage_penalty_ms_per_mib_second: float = 0.0
    resume_slo_ms: float = 1000.0
    allow_unvalidated_lossy: bool = False

    def __post_init__(self):
        positive(self.bandwidth_bytes_per_ms, "bandwidth_bytes_per_ms")
        if not isinstance(self.host_budget_bytes, int) or isinstance(self.host_budget_bytes, bool) or self.host_budget_bytes <= 0:
            raise ValueError("host_budget_bytes must be a positive integer")
        if not isinstance(self.allow_unvalidated_lossy, bool):
            raise ValueError("allow_unvalidated_lossy must be a Boolean")
        positive(self.storage_penalty_ms_per_mib_second, "storage penalty", zero=True)
        positive(self.resume_slo_ms, "resume_slo_ms")


@dataclass(frozen=True)
class Candidate:
    action: str
    stored_bytes: int
    encode_ms: float
    offload_ms: float
    restore_ms: float
    predicted_resume_ms: float
    expected_cost_ms: float
    within_resume_slo: bool
    eviction_cost_ms: float = 0.0


@dataclass(frozen=True)
class AdmissionImpact:
    """Exact physical size and estimated external cost of this admission.

    The caller plans against current residency without changing it. The external
    cost is an expected lost-restoration benefit for affected existing owners.
    """
    stored_bytes: int
    eviction_cost_ms: float = 0.0
    feasible: bool = True

    def __post_init__(self):
        if not isinstance(self.stored_bytes, int) or isinstance(self.stored_bytes, bool) or self.stored_bytes <= 0:
            raise ValueError("stored_bytes must be a positive integer")
        positive(self.eviction_cost_ms, "eviction_cost_ms", zero=True)
        if not isinstance(self.feasible, bool):
            raise ValueError("admission feasibility must be a Boolean")


@dataclass(frozen=True)
class Decision:
    selected: Candidate
    alternatives: tuple[Candidate, ...]
    reason: str


def choose_action(raw_bytes: int, recompute_ms: float, profiles: list[CodecProfile],
                  model: CostModel, forecast: Forecast, *, force_codec: str | None = None,
                  shared_restore_ms: float = 0.0, shared_wait_ms: float = 0.0,
                  admission_impacts: Mapping[str, AdmissionImpact] | None = None) -> Decision:
    if not isinstance(raw_bytes, int) or raw_bytes <= 0:
        raise ValueError("raw_bytes must be a positive integer")
    positive(recompute_ms, "recompute_ms", zero=True)
    positive(shared_restore_ms, "shared_restore_ms", zero=True)
    positive(shared_wait_ms, "shared_wait_ms", zero=True)
    if len({p.name for p in profiles}) != len(profiles):
        raise ValueError("duplicate codec profile")
    if admission_impacts is not None and set(admission_impacts) - {p.name for p in profiles}:
        raise ValueError("admission impact names must match codec profiles")
    p = forecast.resume_probability
    candidates = [Candidate("recompute", 0, 0, 0, recompute_ms, recompute_ms,
                            p * recompute_ms, recompute_ms <= model.resume_slo_ms)]
    for profile in profiles:
        if not profile.lossless and not model.allow_unvalidated_lossy:
            continue
        impact = admission_impacts.get(profile.name) if admission_impacts is not None else None
        if impact is not None and not impact.feasible:
            continue
        size = impact.stored_bytes if impact is not None else math.ceil(raw_bytes / profile.ratio)
        eviction_cost = impact.eviction_cost_ms if impact is not None else 0.0
        if size > model.host_budget_bytes:
            continue
        encode = raw_bytes / MIB * profile.encode_ms_per_mib
        offload = size / model.bandwidth_bytes_per_ms
        restore = size / model.bandwidth_bytes_per_ms + raw_bytes / MIB * profile.decode_ms_per_mib
        # A pause estimate is a prior/history statistic, never this session's future.
        early_wait = max(0.0, encode + offload - forecast.pause_ms)
        predicted_resume = shared_restore_ms + restore + max(early_wait, shared_wait_ms)
        residency_cost = size / MIB * forecast.pause_ms / 1000 * model.storage_penalty_ms_per_mib_second
        score = encode + offload + p * predicted_resume + residency_cost + eviction_cost
        candidates.append(Candidate(profile.name, size, encode, offload, restore, predicted_resume, score,
                                    predicted_resume <= model.resume_slo_ms, eviction_cost))
    if force_codec == "recompute":
        selected = candidates[0]
        reason = "fixed recompute baseline"
    elif force_codec is not None:
        matching_profiles = [p for p in profiles if p.name == force_codec]
        if not matching_profiles:
            raise ValueError(f"unknown codec {force_codec!r}")
        if not matching_profiles[0].lossless and not model.allow_unvalidated_lossy:
            raise ValueError(f"lossy codec {force_codec!r} has no quality approval; exploration must be explicit")
        matches = [c for c in candidates if c.action == force_codec]
        if not matches:
            selected, reason = candidates[0], "fixed codec cannot fit budget; recompute fallback"
        else:
            selected, reason = matches[0], "fixed codec baseline (no quality guarantee)"
    else:
        feasible = [c for c in candidates if c.within_resume_slo]
        selected = min(feasible, key=lambda c: c.expected_cost_ms) if feasible else min(candidates, key=lambda c: c.predicted_resume_ms)
        reason = "minimum estimated expected cost within resume target" if feasible else "no action satisfies estimated resume target; fastest fallback"
        if admission_impacts is not None:
            reason += "; includes estimated loss imposed on evicted owners"
    return Decision(selected, tuple(candidates), reason)


class HistoricalForecast:
    """Beta(1,1) resume prior + mean pause among already resolved sessions.

    This is a deliberately simple baseline, not a survival model. Right-censored
    outstanding sessions do not count as cancellations. An experiment must compare
    this with fixed priors/TTL and proper survival estimators before deployment.
    """
    def __init__(self, default_pause_ms: float = 1000.0):
        positive(default_pause_ms, "default_pause_ms", zero=True)
        self.default_pause_ms = default_pause_ms
        self.stats: dict[str, tuple[int, int, float]] = {}

    def predict(self, kind: str) -> Forecast:
        resumed, resolved, duration = self.stats.get(kind, (0, 0, 0.0))
        return Forecast((resumed + 1) / (resolved + 2), duration / resolved if resolved else self.default_pause_ms)

    def observe(self, kind: str, resumed: bool, duration_ms: float) -> None:
        positive(duration_ms, "duration_ms", zero=True)
        r, n, duration = self.stats.get(kind, (0, 0, 0.0))
        self.stats[kind] = (r + int(resumed), n + 1, duration + duration_ms)
