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

from dataclasses import asdict, dataclass, field

from ..policy import Decision
from ..replay import percentile


@dataclass
class LiveRecorder:
    policy: str
    log: list = field(default_factory=list)
    delays: list = field(default_factory=list)
    transferred: int = 0
    peak_host_reserved_bytes: int = 0
    evicted_blocks: int = 0
    full_recomputes: int = 0
    canceled: int = 0
    encode_ms: float = 0.0
    decode_ms: float = 0.0
    recompute_ms: float = 0.0

    def pause(self, *, at_ms: float, session_id: str, decision: Decision,
              host_reserved_bytes: int, raw_bytes: int) -> None:
        if decision.selected.action != "recompute":
            self.encode_ms += decision.selected.encode_ms
            self.transferred += decision.selected.stored_bytes
        self.peak_host_reserved_bytes = max(self.peak_host_reserved_bytes, host_reserved_bytes)
        self.log.append({"at_ms": at_ms, "type": "pause", "session_id": session_id,
                         "decision": asdict(decision), "host_reserved_bytes": host_reserved_bytes,
                         "raw_bytes": raw_bytes})

    def resume(self, *, at_ms: float, session_id: str, used_cache: bool, measured_delay_ms: float,
               host_reserved_bytes: int, transferred_bytes: int = 0, decode_ms: float = 0.0) -> None:
        if used_cache:
            self.transferred += transferred_bytes
            self.decode_ms += decode_ms
        else:
            self.full_recomputes += 1
            self.recompute_ms += measured_delay_ms
        self.delays.append(measured_delay_ms)
        self.log.append({"at_ms": at_ms, "type": "resume", "session_id": session_id,
                         "path": "restore" if used_cache else "full_recompute",
                         "measured_resume_delay_ms": measured_delay_ms,
                         "host_reserved_bytes": host_reserved_bytes})

    def cancel(self, *, at_ms: float, session_id: str) -> None:
        self.canceled += 1
        self.log.append({"at_ms": at_ms, "type": "cancel", "session_id": session_id})

    def evict(self, count: int = 1) -> None:
        self.evicted_blocks += count

    def finalize(self) -> dict:
        return {
            "schema_version": 1, "evidence_type": "live_measured_session",
            "policy": self.policy, "quality_validated": False,
            "metrics": {
                "resume_count": len(self.delays), "cancel_count": self.canceled,
                "measured_mean_resume_ms": sum(self.delays) / len(self.delays) if self.delays else None,
                "measured_p50_resume_ms": percentile(self.delays, 0.5),
                "measured_p95_resume_ms": percentile(self.delays, 0.95),
                "measured_transferred_bytes": self.transferred,
                "peak_host_reserved_bytes": self.peak_host_reserved_bytes,
                "evicted_blocks": self.evicted_blocks, "full_recomputes": self.full_recomputes,
                "measured_encoding_ms": self.encode_ms, "measured_decoding_ms": self.decode_ms,
                "measured_recompute_ms": self.recompute_ms,
            },
            "decisions": self.log,
        }
