"""Real host-side timing of kvpilot's reference codecs on a real array.

This measures actual wall-clock encode/decode time and actual serialized size
on whatever device the array already lives on (e.g. a KV block already copied
to host memory). It is not a GPU benchmark and does not claim PCIe or device
timing. Feed it a real captured KV tensor to replace the illustrative numbers
in examples/assumed-costs.json with measured ones.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..codecs import decode, encode
from ..policy import CodecProfile, MIB


@dataclass(frozen=True)
class CodecMeasurement:
    codec: str
    raw_bytes: int
    stored_bytes: int
    encode_ms: float
    decode_ms: float
    max_absolute_error: float

    def to_profile(self, name: str | None = None) -> CodecProfile:
        """Build a policy.CodecProfile from this measurement, not an assumption."""
        mib = self.raw_bytes / MIB
        return CodecProfile(
            name or self.codec,
            ratio=self.raw_bytes / self.stored_bytes,
            encode_ms_per_mib=self.encode_ms / mib,
            decode_ms_per_mib=self.decode_ms / mib,
            lossless=self.max_absolute_error == 0.0,
        )


def measure_codec(array: np.ndarray, codec_name: str, *, group_size: int = 64, repeats: int = 5) -> CodecMeasurement:
    """Time ``repeats`` real encode/decode round trips of ``codec_name``; report the median.

    Raises on an empty array: there is no meaningful per-MiB rate for zero bytes.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    reference = np.asarray(array, dtype="<f2")
    if reference.size == 0:
        raise ValueError("array must not be empty")
    encode_times_ms = []
    decode_times_ms = []
    payload = b""
    restored = reference
    for _ in range(repeats):
        start = time.perf_counter()
        payload = encode(array, codec_name, group_size=group_size)
        encode_times_ms.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        restored = decode(payload)
        decode_times_ms.append((time.perf_counter() - start) * 1000)
    error = float(np.max(np.abs(restored.astype(np.float32) - reference.astype(np.float32))))
    return CodecMeasurement(
        codec=codec_name,
        raw_bytes=reference.nbytes,
        stored_bytes=len(payload),
        encode_ms=float(np.median(encode_times_ms)),
        decode_ms=float(np.median(decode_times_ms)),
        max_absolute_error=error,
    )
