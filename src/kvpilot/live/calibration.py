"""Real host-side timing of kvpilot's reference codecs on a real array.

This measures host wall-clock encode/decode time and serialized size for a
NumPy array already in host memory. It is not a GPU or transfer benchmark.
All reference codecs convert to FP16: raw_bytes and max_absolute_error are
relative to that FP16 reference, and a lossless profile means lossless *after*
this conversion. The original dtype and conversion error are reported
separately; zero error on one sample never makes a quantizing codec lossless.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..codecs import CODECS, decode, encode, fp16_reference
from ..policy import CodecProfile, MIB


_LOSSLESS_FP16_CODECS = frozenset({"raw_fp16", "zlib_fp16"})


@dataclass(frozen=True)
class CodecMeasurement:
    codec: str
    raw_bytes: int
    stored_bytes: int
    encode_ms: float
    decode_ms: float
    max_absolute_error: float
    reference_dtype: str = field(default="<f2", init=False)
    source_dtype: str | None = None
    source_conversion_max_absolute_error: float | None = None

    def to_profile(self, name: str | None = None) -> CodecProfile:
        """Build measured costs with the codec's FP16-reference losslessness.

        The profile's lossless flag describes the codec for every valid input,
        not whether this particular calibration sample happened to round-trip.
        """
        if self.codec not in CODECS:
            raise ValueError(f"unknown reference codec {self.codec!r}")
        mib = self.raw_bytes / MIB
        return CodecProfile(
            name or self.codec,
            ratio=self.raw_bytes / self.stored_bytes,
            encode_ms_per_mib=self.encode_ms / mib,
            decode_ms_per_mib=self.decode_ms / mib,
            lossless=self.codec in _LOSSLESS_FP16_CODECS,
        )


def measure_codec(array: np.ndarray, codec_name: str, *, group_size: int = 64, repeats: int = 5) -> CodecMeasurement:
    """Time ``repeats`` real encode/decode round trips of ``codec_name``; report the median.

    Raises on an empty array: there is no meaningful per-MiB rate for zero bytes.
    """
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    source = np.asarray(array)
    reference = fp16_reference(source)
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
        source_dtype=str(source.dtype),
        source_conversion_max_absolute_error=float(np.max(np.abs(
            source.astype(np.float64) - reference.astype(np.float64)))),
    )
