"""Bookkeeping for a REAL pause/resume/cancel session, shaped like replay()'s output.

Every delay, byte count and transfer recorded here is passed in by the caller
as something that actually happened (a measured wall-clock time, a real cache
hit/miss, a real encode). This module holds no network, GPU or vLLM code and
fabricates nothing; it only accumulates and formats. The output uses
``measured_*`` metric names and ``evidence_type: live_measured_session`` so it
is never confused with kvpilot.replay's ``modeled_*`` analytical output, even
though both share the same shape for easy side-by-side comparison.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from numbers import Integral, Real

from ..policy import Decision
from ..replay import percentile


def _duration(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _event(at_ms: float, session_id: str) -> float:
    timestamp = _duration(at_ms, "at_ms")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a nonempty string")
    return timestamp


@dataclass
class LiveRecorder:
    """Accumulate observations; the caller owns session lifecycle and timing.

    Decision fields remain predictions in the log. Every cost counter requires
    an explicit measured execution value, including zero when no work occurred.
    Resume delay is the whole observed interval; decode/recompute times are
    separately measured components and need not sum to it (work may overlap).
    Host reservations are observations at event boundaries, not a claim about
    unobserved allocation peaks between events.
    """

    policy: str
    log: list = field(default_factory=list, init=False)
    delays: list = field(default_factory=list, init=False)
    transferred: int = field(default=0, init=False)
    peak_host_reserved_bytes: int = field(default=0, init=False)
    evicted_blocks: int = field(default=0, init=False)
    full_recomputes: int = field(default=0, init=False)
    canceled: int = field(default=0, init=False)
    encode_ms: float = field(default=0.0, init=False)
    decode_ms: float = field(default=0.0, init=False)
    recompute_ms: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.policy, str) or not self.policy:
            raise ValueError("policy must be a nonempty string")

    def pause(self, *, at_ms: float, session_id: str, decision: Decision,
              host_reserved_bytes: int, raw_bytes: int,
              measured_encode_ms: float, transferred_bytes: int) -> None:
        at_ms = _event(at_ms, session_id)
        host_reserved_bytes = _count(host_reserved_bytes, "host_reserved_bytes")
        raw_bytes = _count(raw_bytes, "raw_bytes")
        measured_encode_ms = _duration(measured_encode_ms, "measured_encode_ms")
        transferred_bytes = _count(transferred_bytes, "transferred_bytes")
        encoding_total = _duration(self.encode_ms + measured_encode_ms, "encoding total")
        decision_log = asdict(decision)
        self.encode_ms = encoding_total
        self.transferred += transferred_bytes
        self.peak_host_reserved_bytes = max(self.peak_host_reserved_bytes, host_reserved_bytes)
        self.log.append({"at_ms": at_ms, "type": "pause", "session_id": session_id,
                         "decision": decision_log, "host_reserved_bytes": host_reserved_bytes,
                         "raw_bytes": raw_bytes, "measured_encoding_ms": measured_encode_ms,
                         "measured_transferred_bytes": transferred_bytes})

    def resume(self, *, at_ms: float, session_id: str, used_cache: bool, measured_delay_ms: float,
               host_reserved_bytes: int, transferred_bytes: int, decode_ms: float,
               recompute_ms: float) -> None:
        at_ms = _event(at_ms, session_id)
        if type(used_cache) is not bool:
            raise ValueError("used_cache must be a Boolean")
        measured_delay_ms = _duration(measured_delay_ms, "measured_delay_ms")
        host_reserved_bytes = _count(host_reserved_bytes, "host_reserved_bytes")
        transferred_bytes = _count(transferred_bytes, "transferred_bytes")
        decode_ms = _duration(decode_ms, "decode_ms")
        recompute_ms = _duration(recompute_ms, "recompute_ms")
        decoding_total = _duration(self.decode_ms + decode_ms, "decoding total")
        recompute_total = _duration(self.recompute_ms + recompute_ms, "recompute total")
        # Failed restoration can incur transfer and decode work before fallback.
        self.transferred += transferred_bytes
        self.decode_ms = decoding_total
        self.recompute_ms = recompute_total
        self.full_recomputes += int(not used_cache)
        self.peak_host_reserved_bytes = max(self.peak_host_reserved_bytes, host_reserved_bytes)
        self.delays.append(measured_delay_ms)
        self.log.append({"at_ms": at_ms, "type": "resume", "session_id": session_id,
                         "path": "restore" if used_cache else "full_recompute",
                         "measured_resume_delay_ms": measured_delay_ms,
                         "host_reserved_bytes": host_reserved_bytes,
                         "measured_transferred_bytes": transferred_bytes,
                         "measured_decoding_ms": decode_ms,
                         "measured_recompute_ms": recompute_ms})

    def cancel(self, *, at_ms: float, session_id: str) -> None:
        at_ms = _event(at_ms, session_id)
        self.canceled += 1
        self.log.append({"at_ms": at_ms, "type": "cancel", "session_id": session_id})

    def evict(self, count: int = 1) -> None:
        self.evicted_blocks += _count(count, "count")

    def finalize(self) -> dict:
        return {
            "schema_version": 1, "evidence_type": "live_measured_session",
            "policy": self.policy, "quality_validated": False,
            "metrics": {
                "resume_count": len(self.delays), "cancel_count": self.canceled,
                "measured_mean_resume_ms": math.fsum(value / len(self.delays) for value in self.delays) if self.delays else None,
                "measured_p50_resume_ms": percentile(self.delays, 0.5),
                "measured_p95_resume_ms": percentile(self.delays, 0.95),
                "measured_transferred_bytes": self.transferred,
                "peak_host_reserved_bytes": self.peak_host_reserved_bytes,
                "evicted_blocks": self.evicted_blocks, "full_recomputes": self.full_recomputes,
                "measured_encoding_ms": self.encode_ms, "measured_decoding_ms": self.decode_ms,
                "measured_recompute_ms": self.recompute_ms,
            },
            "decisions": deepcopy(self.log),
        }
