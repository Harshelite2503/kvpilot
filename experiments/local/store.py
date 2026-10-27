"""Owned synchronous paused-state store for the local tensor experiment.

Each session owns one complete immutable byte payload; there is no prefix
sharing, partial restoration, asynchronous offload or external cache manager.
The existing policy and eviction estimator rank candidates using calibrated
cost estimates. Actual encoding, decoding and model execution belong to the
caller and must be timed there. This store does not simulate transfer delays.

Capacity and counters count serialized payload bytes, including metadata. They
exclude Python objects, active model state, temporary codec arrays, the incoming
payload before commit, and returned payloads after a resume. ``byte_event_sum`` sums occupancy after lifecycle events;
it is a byte-event statistic, never a wall-time memory integral or process RSS.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import math
from numbers import Real
from types import MappingProxyType
from typing import Mapping

from kvpilot.policy import (
    AdmissionImpact, Candidate, CodecProfile, CostModel, Decision, Forecast,
    HistoricalForecast, MIB, choose_action,
)
from kvpilot.replay import Entry, estimate_eviction_cost, plan_lru_evictions


@dataclass(frozen=True)
class AdmissionPlan:
    codec: str
    stored_bytes: int
    victims: tuple[str, ...] | None
    estimated_eviction_loss_ms: float
    charged_eviction_cost_ms: float


@dataclass(frozen=True)
class PausePlan:
    session_id: str
    raw_bytes: int
    recompute_ms: float
    kind: str
    timestamp_ms: float
    forecast: Forecast
    decision: Decision
    victims: tuple[str, ...]
    admissions: tuple[AdmissionPlan, ...]
    _version: int = field(repr=False)
    _store_token: object = field(repr=False, compare=False)

    @property
    def selected(self) -> Candidate:
        return self.decision.selected

    @property
    def alternatives(self) -> tuple[Candidate, ...]:
        return self.decision.alternatives

    @property
    def reason(self) -> str:
        return self.decision.reason


def _name(value, field_name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string")


def _nonnegative(value, field_name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and nonnegative")
    return float(value)


class PausedStateStore:
    """Plan without mutation, encode one selected action, then commit exactly.

    ``adaptive_exact`` and ``adaptive_eviction`` use identical exact sizes and
    LRU victim order. Only the latter charges the predicted cost imposed on
    existing owners. A codec name selects a fixed codec; ``recompute`` retains
    no payload. Unsupported or unapproved lossy profiles are never silently used.

    ``raw_bytes`` is the calibration denominator, normally native tensor nbytes.
    ``exact_sizes`` must include complete serialized lengths for every supplied
    profile. Session ids must be unique until resolved; model/offset compatibility
    validation remains the caller's responsibility when decoding returned bytes.
    """

    def __init__(self, policy: str, profiles: list[CodecProfile], cost_model: CostModel,
                 *, default_pause_ms: float = 1000.0):
        self.profiles = tuple(profiles)
        self._profile_map = {profile.name: profile for profile in self.profiles}
        if len(self._profile_map) != len(self.profiles):
            raise ValueError("duplicate codec profile")
        if policy not in {"adaptive_exact", "adaptive_eviction", "recompute", *self._profile_map}:
            raise ValueError(f"unsupported local store policy: {policy!r}")
        if policy in self._profile_map and not self._profile_map[policy].lossless and not cost_model.allow_unvalidated_lossy:
            raise ValueError("fixed lossy codec requires explicit exploration approval")
        self.policy, self.cost_model = policy, cost_model
        self.history = HistoricalForecast(default_pause_ms)
        self._payloads: OrderedDict[str, bytes] = OrderedDict()
        self._entries: OrderedDict[str, Entry] = OrderedDict()
        self._paused: dict[str, dict] = {}
        self._used_bytes = self._peak_bytes = self._byte_event_sum = 0
        self._event_count = self._evicted_states = 0
        self._version, self._last_event_ms = 0, -1.0
        self._token = object()

    @property
    def payloads(self) -> Mapping[str, bytes]:
        """Read-only view of immutable payloads, oldest resident first."""
        return MappingProxyType(self._payloads)

    @property
    def paused_sessions(self) -> tuple[str, ...]:
        """Includes evicted and recompute-only owners until they resolve."""
        return tuple(self._paused)

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    @property
    def peak_bytes(self) -> int:
        return self._peak_bytes

    @property
    def byte_event_sum(self) -> int:
        return self._byte_event_sum

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def evicted_states(self) -> int:
        return self._evicted_states

    def _time(self, timestamp_ms):
        timestamp = _nonnegative(timestamp_ms, "timestamp_ms")
        if timestamp < self._last_event_ms:
            raise ValueError("lifecycle events must have nondecreasing timestamps")
        return timestamp

    def decide(self, session_id: str, raw_bytes: int, recompute_ms: float, kind: str,
               timestamp_ms: float, exact_sizes: Mapping[str, int]) -> PausePlan:
        """Read-only decision using only completed history and current residency."""
        _name(session_id, "session_id")
        _name(kind, "kind")
        if session_id in self._paused:
            raise ValueError(f"session {session_id!r} is already paused")
        if type(raw_bytes) is not int or raw_bytes <= 0:
            raise ValueError("raw_bytes must be a positive integer")
        recompute = _nonnegative(recompute_ms, "recompute_ms")
        timestamp = self._time(timestamp_ms)
        if not isinstance(exact_sizes, Mapping) or set(exact_sizes) != set(self._profile_map):
            raise ValueError("exact_sizes must include exactly every codec profile")
        if any(type(size) is not int or size <= 0 for size in exact_sizes.values()):
            raise ValueError("exact serialized sizes must be positive integers")
        forecast = self.history.predict(kind)
        impacts, plans = {}, []
        for profile in self.profiles:
            if not profile.lossless and not self.cost_model.allow_unvalidated_lossy:
                continue
            required = exact_sizes[profile.name]
            victims = plan_lru_evictions(self._entries, self.used_bytes,
                                        self.cost_model.host_budget_bytes, required, set())
            loss = 0.0
            if victims is not None:
                loss, _ = estimate_eviction_cost(victims, (session_id,), self._entries,
                                                self._paused, self.history, self.cost_model, timestamp)
            charged = loss if self.policy == "adaptive_eviction" else 0.0
            impacts[profile.name] = AdmissionImpact(required, charged, victims is not None)
            plans.append(AdmissionPlan(profile.name, required, victims, loss, charged))
        decision = choose_action(
            raw_bytes, recompute, self.profiles, self.cost_model, forecast,
            force_codec=None if self.policy in {"adaptive_exact", "adaptive_eviction"} else self.policy,
            admission_impacts=impacts,
        )
        victims = () if decision.selected.action == "recompute" else next(
            plan.victims for plan in plans if plan.codec == decision.selected.action
        )
        if victims is None:
            raise RuntimeError("selected action has no feasible eviction plan")
        return PausePlan(session_id, raw_bytes, recompute, kind, timestamp, forecast,
                         decision, victims, tuple(plans), self._version, self._token)

    def commit(self, plan: PausePlan, payload: bytes | None, *, timestamp_ms: float | None = None) -> None:
        """Commit one selected payload atomically after checking exact framing size.

        An optional completion timestamp records when synchronous encoding ended.
        No other lifecycle event may intervene between planning and committing.
        Failed validation leaves payloads, history and counters unchanged.
        """
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
            if type(payload) is not bytes or len(payload) != selected.stored_bytes:
                raise ValueError("immutable payload length must match the exact declared size")
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
                plan.raw_bytes / MIB * self._profile_map[selected.action].decode_ms_per_mib,
            )
            self._used_bytes += len(payload)
        self._paused[plan.session_id] = {
            "keys": (plan.session_id,), "kind": plan.kind,
            "at_ms": plan.timestamp_ms, "recompute_ms": plan.recompute_ms,
        }
        self._record_event(timestamp)

    def _remove_payload(self, session_id):
        payload = self._payloads.pop(session_id, None)
        entry = self._entries.pop(session_id, None)
        if payload is not None:
            if entry is None or entry.stored_bytes != len(payload):
                raise RuntimeError("payload and entry accounting diverged")
            self._used_bytes -= len(payload)
        elif entry is not None:
            raise RuntimeError("entry exists without a payload")
        return payload

    def _record_event(self, timestamp):
        if self._used_bytes != sum(map(len, self._payloads.values())) or tuple(self._payloads) != tuple(self._entries):
            raise RuntimeError("payload accounting or eviction order diverged")
        if not 0 <= self._used_bytes <= self.cost_model.host_budget_bytes:
            raise RuntimeError("paused payload capacity was exceeded")
        self._peak_bytes = max(self._peak_bytes, self._used_bytes)
        self._byte_event_sum += self._used_bytes
        self._event_count += 1
        self._version += 1
        self._last_event_ms = timestamp

    def _resolve(self, session_id, timestamp_ms, resumed):
        _name(session_id, "session_id")
        timestamp = self._time(timestamp_ms)
        if session_id not in self._paused:
            raise ValueError(f"session {session_id!r} is not paused")
        session = self._paused.pop(session_id)
        payload = self._remove_payload(session_id)
        self.history.observe(session["kind"], resumed, timestamp - session["at_ms"])
        self._record_event(timestamp)
        return payload

    def popresume(self, session_id: str, timestamp_ms: float) -> bytes | None:
        """Remove and return the complete payload, or None for full recomputation.

        Eviction is not a cancellation. A later resume counts as a resume in the
        forecast even when its payload was evicted or deliberately not retained.
        The caller must recompute the complete state when None is returned.
        """
        return self._resolve(session_id, timestamp_ms, True)

    def cancel(self, session_id: str, timestamp_ms: float) -> None:
        self._resolve(session_id, timestamp_ms, False)
