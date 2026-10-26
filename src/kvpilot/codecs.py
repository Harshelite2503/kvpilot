"""Small CPU reference codecs, not implementations of published KV codecs.

All codecs first convert their input to FP16. Quantization is symmetric, with
one FP16 scale per group along the last axis (never across different rows).
The serialized size includes the JSON envelope, shape, scales, and padding.
Payloads are untrusted: decode bounds dimensions, allocations and decompression.
"""

from __future__ import annotations

import json
import math
import struct
import zlib

import numpy as np


CODECS = ("raw_fp16", "zlib_fp16", "group_int8", "group_int4")
MAX_ELEMENTS = 16_777_216
MAX_GROUP_SIZE = 65_536
_MAGIC = b"KVP\x00"
_PREFIX = struct.Struct("<4sI")
_MAX_METADATA_BYTES = 16_384
_FP16_MAX = float(np.finfo(np.float16).max)
_MIN_SCALE = np.nextafter(np.float16(0), np.float16(1))


class CodecError(ValueError):
    """Input or serialized codec data is invalid or exceeds reference limits."""


def _shape_info(shape: object) -> tuple[tuple[int, ...], int, int, int]:
    if not isinstance(shape, (list, tuple)) or len(shape) > 32:
        raise CodecError("shape must be a list of at most 32 dimensions")
    if any(type(x) is not int or x < 0 or x > MAX_ELEMENTS for x in shape):
        raise CodecError("shape dimensions must be bounded nonnegative integers")
    count = math.prod(shape)
    if count > MAX_ELEMENTS:
        raise CodecError(f"array exceeds {MAX_ELEMENTS} elements")
    # Bound partial products too: a zero-sized axis must not hide huge strides.
    nonzero_product = math.prod(max(x, 1) for x in shape)
    if nonzero_product > MAX_ELEMENTS:
        raise CodecError("shape strides exceed the reference allocation limit")
    last = shape[-1] if shape else 1
    rows = math.prod(shape[:-1]) if shape else 1
    return tuple(shape), count, rows, last


def _group_size(value: object) -> int:
    if type(value) is not int or not 1 <= value <= MAX_GROUP_SIZE:
        raise CodecError(f"group_size must be an integer in [1, {MAX_GROUP_SIZE}]")
    return value


def fp16_reference(array: np.ndarray) -> np.ndarray:
    """Validate and convert to the common finite FP16 reference representation."""
    try:
        values = np.asarray(array)
    except (TypeError, ValueError) as exc:
        raise CodecError("input must be a real numeric array") from exc
    if values.dtype.kind not in "fiu":
        raise CodecError("input must be a real numeric array")
    _shape_info(values.shape)
    if not np.isfinite(values).all():
        raise CodecError("NaN and infinity are not supported")
    if values.size and (np.any(values > _FP16_MAX) or np.any(values < -_FP16_MAX)):
        raise CodecError("input is outside the finite FP16 range")
    # astype preserves scalar shape, unlike ascontiguousarray on a scalar.
    return values.astype("<f2", order="C", copy=True)


def _envelope(metadata: dict, body: bytes) -> bytes:
    header = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _PREFIX.pack(_MAGIC, len(header)) + header + body


def encode(array: np.ndarray, codec_name: str, group_size: int = 64) -> bytes:
    """Encode a finite real array; output includes every byte needed to decode.

    Quantization uses [-127, 127] or [-7, 7]. Int4 puts the first value in the
    low nibble and pads an odd final high nibble with zero. Subnormal scales
    are clamped to FP16's smallest positive value, avoiding division by zero.
    """
    if codec_name not in CODECS:
        raise CodecError(f"unknown codec {codec_name!r}; choose from {CODECS}")
    size = _group_size(group_size)
    reference = fp16_reference(array)
    metadata = {"version": 1, "codec": codec_name, "dtype": "<f2", "shape": list(reference.shape)}
    if codec_name in ("raw_fp16", "zlib_fp16"):
        body = reference.tobytes(order="C")
        if codec_name == "zlib_fp16":
            body = zlib.compress(body, level=6)
        return _envelope(metadata, body)

    metadata["group_size"] = size
    _, _, rows, last = _shape_info(reference.shape)
    groups = (last + size - 1) // size
    matrix = reference.reshape(rows, last).astype(np.float32)
    scales = np.empty((rows, groups), dtype="<f2")
    quantized = np.empty((rows, last), dtype=np.int8)
    limit = 127 if codec_name == "group_int8" else 7
    for group in range(groups):
        start, stop = group * size, min((group + 1) * size, last)
        block = matrix[:, start:stop]
        maximum = np.max(np.abs(block), axis=1)
        scale = np.maximum(maximum / limit, float(_MIN_SCALE)).astype("<f2")
        # A conventional unit scale makes all-zero groups unambiguous.
        scale[maximum == 0] = 1
        scales[:, group] = scale
        quantized[:, start:stop] = np.clip(np.rint(block / scale.astype(np.float32)[:, None]), -limit, limit).astype(np.int8)
    if codec_name == "group_int8":
        data = quantized.tobytes()
    else:
        nibbles = quantized.reshape(-1).astype(np.uint8) & 15
        packed = np.zeros((nibbles.size + 1) // 2, dtype=np.uint8)
        packed[:] = nibbles[::2]
        packed[: nibbles.size // 2] |= nibbles[1::2] << 4
        data = packed.tobytes()
    return _envelope(metadata, scales.tobytes() + data)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CodecError("duplicate metadata key")
        result[key] = value
    return result


def decode(payload: bytes) -> np.ndarray:
    """Decode to finite FP16, rejecting invalid lengths, metadata and values.

    This is a local benchmark format, not an authenticated network protocol.
    Quantized values reconstructed outside FP16 range are saturated to its
    finite endpoints; FP16 rounding is part of the codec's measured error.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise CodecError("payload must be bytes-like")
    try:
        payload = memoryview(payload).cast("B")
    except (TypeError, ValueError) as exc:
        raise CodecError("payload must be a contiguous byte buffer") from exc
    if len(payload) < _PREFIX.size or len(payload) > 4 * MAX_ELEMENTS + _MAX_METADATA_BYTES:
        raise CodecError("invalid payload length")
    magic, header_length = _PREFIX.unpack_from(payload)
    if magic != _MAGIC:
        raise CodecError("invalid codec magic")
    if not 1 <= header_length <= _MAX_METADATA_BYTES or header_length > len(payload) - _PREFIX.size:
        raise CodecError("invalid metadata length")
    try:
        metadata = json.loads(bytes(payload[_PREFIX.size:_PREFIX.size + header_length]), object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise CodecError("invalid metadata JSON") from exc
    if not isinstance(metadata, dict):
        raise CodecError("metadata must be an object")
    if type(metadata.get("version")) is not int or metadata["version"] != 1:
        raise CodecError("unsupported codec format version")
    codec = metadata.get("codec")
    if not isinstance(codec, str) or codec not in CODECS or metadata.get("dtype") != "<f2":
        raise CodecError("unsupported codec or dtype")
    expected_keys = {"version", "codec", "dtype", "shape"}
    if codec.startswith("group_"):
        expected_keys.add("group_size")
    if set(metadata) != expected_keys:
        raise CodecError("unexpected or missing metadata fields")
    shape, count, rows, last = _shape_info(metadata["shape"])
    body = payload[_PREFIX.size + header_length:]
    if codec in ("raw_fp16", "zlib_fp16"):
        if codec == "zlib_fp16":
            try:
                inflater = zlib.decompressobj()
                raw = inflater.decompress(body, count * 2 + 1)
            except zlib.error as exc:
                raise CodecError("invalid compressed stream") from exc
            if len(raw) != count * 2 or not inflater.eof or inflater.unconsumed_tail or inflater.unused_data:
                raise CodecError("compressed length mismatch or trailing data")
        else:
            raw = body
        if len(raw) != count * 2:
            raise CodecError("FP16 data length does not match shape")
        result = np.frombuffer(raw, dtype="<f2").copy().reshape(shape)
        if not np.isfinite(result).all():
            raise CodecError("payload contains nonfinite values")
        return result

    size = _group_size(metadata["group_size"])
    groups = (last + size - 1) // size
    scale_count = rows * groups
    scale_bytes = scale_count * 2
    data_bytes = count if codec == "group_int8" else (count + 1) // 2
    if len(body) != scale_bytes + data_bytes:
        raise CodecError("quantized data length does not match shape")
    scales = np.frombuffer(body[:scale_bytes], dtype="<f2").reshape(rows, groups)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise CodecError("scales must be finite and positive")
    if codec == "group_int8":
        quantized = np.frombuffer(body[scale_bytes:], dtype=np.int8)
        if np.any(quantized == -128):
            raise CodecError("invalid symmetric int8 value")
    else:
        packed = np.frombuffer(body[scale_bytes:], dtype=np.uint8)
        if count % 2 and packed[-1] >> 4:
            raise CodecError("nonzero int4 padding")
        nibbles = np.empty(count, dtype=np.uint8)
        nibbles[::2] = packed & 15
        nibbles[1::2] = packed[:count // 2] >> 4
        if np.any(nibbles == 8):
            raise CodecError("invalid symmetric int4 value")
        quantized = nibbles.astype(np.int8)
        quantized[quantized >= 8] -= 16
    matrix = quantized.reshape(rows, last)
    result = np.empty((rows, last), dtype="<f2")
    for group in range(groups):
        start, stop = group * size, min((group + 1) * size, last)
        block = matrix[:, start:stop].astype(np.float32) * scales[:, group].astype(np.float32)[:, None]
        result[:, start:stop] = np.clip(block, -_FP16_MAX, _FP16_MAX).astype("<f2")
    return result.reshape(shape)
