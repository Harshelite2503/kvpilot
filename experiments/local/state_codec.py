"""Owned, bounded serialization for a complete local model-state snapshot.

``raw_native`` preserves each supported tensor's dtype and C-order bytes. Other
formats apply the existing FP16 reference codecs ONLY to role ``attention``;
recurrent/convolution/position state remains native-exact. FP32 attention is
therefore rounded through FP16 even with ``raw_fp16``. This is not a claim that
the model's complete state is lossless or that restored answers are unchanged.

UINT16 may carry opaque BF16 bits, described by caller metadata. Native storage
preserves those bits; compressed attention rejects integer arrays rather than
silently treating BF16 bit patterns as numeric integers. Compatibility, offsets,
and the completeness of a model snapshot must be checked by the caller.
"""

from __future__ import annotations

import json
import math
import struct

import numpy as np

from kvpilot.codecs import CodecError, decode as decode_array, encode as encode_array


CODECS = ("raw_native", "raw_fp16", "group_int8", "group_int4")
MAX_TENSORS = 1024
MAX_ELEMENTS = 16_777_216
MAX_STATE_BYTES = 256 * 1024 * 1024
MAX_HEADER_BYTES = 1024 * 1024
_MAGIC = b"KVS\x00"
_PREFIX = struct.Struct("<4sI")
_RECORD_KEYS = {"name", "role", "storage", "dtype", "shape", "offset", "length"}


class StateCodecError(ValueError):
    """Invalid state data, metadata, framing or unsupported representation."""


def _json_value(value, *, depth=0, budget=None):
    if budget is None:
        budget = [65_536]
    budget[0] -= 1
    if depth > 32 or budget[0] < 0:
        raise StateCodecError("metadata exceeds JSON depth or item limit")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise StateCodecError("metadata numbers must be finite")
        return
    if type(value) is list:
        for item in value:
            _json_value(item, depth=depth + 1, budget=budget)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_value(item, depth=depth + 1, budget=budget)
        return
    raise StateCodecError("metadata must contain only JSON values and string keys")


def _dtype(value) -> np.dtype:
    if not isinstance(value, (str, np.dtype)):
        raise StateCodecError("tensor dtype must be a dtype string")
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        raise StateCodecError("invalid tensor dtype") from exc
    if (dtype.kind, dtype.itemsize) not in {("f", 2), ("f", 4), ("i", 4), ("u", 2)}:
        raise StateCodecError("supported native dtypes are FP16, FP32, INT32 and UINT16")
    return dtype


def _shape(value) -> tuple[tuple[int, ...], int]:
    if not isinstance(value, (list, tuple)) or len(value) > 32:
        raise StateCodecError("tensor shape must have at most 32 dimensions")
    if any(type(n) is not int or n < 0 or n > MAX_ELEMENTS for n in value):
        raise StateCodecError("invalid or unbounded tensor dimension")
    if math.prod(max(1, n) for n in value) > MAX_ELEMENTS:
        raise StateCodecError("tensor exceeds element or stride limit")
    return tuple(value), math.prod(value)


def _name_and_role(name, role):
    if not isinstance(name, str) or not name or len(name) > 1024:
        raise StateCodecError("tensor names must be nonempty bounded strings")
    if not isinstance(role, str) or not role or len(role) > 128:
        raise StateCodecError("tensor roles must be nonempty bounded strings")


def _codec(value):
    if value not in CODECS:
        raise StateCodecError(f"unsupported state codec: {value!r}")


def _finite(array):
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise StateCodecError("floating state tensors must be finite")


def _json_bytes(value) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise StateCodecError("metadata cannot be serialized as JSON") from exc


def _attention_size(shape, codec):
    """Fixed-rate core format, including its complete inner envelope."""
    count = math.prod(shape)
    inner = {"version": 1, "codec": codec, "dtype": "<f2", "shape": list(shape)}
    if codec == "raw_fp16":
        body = count * 2
    else:
        inner["group_size"] = 64
        rows = math.prod(shape[:-1]) if shape else 1
        last = shape[-1] if shape else 1
        body = rows * ((last + 63) // 64) * 2
        body += count if codec == "group_int8" else (count + 1) // 2
    return 8 + len(_json_bytes(inner)) + body


def _layout_from_specs(specs, roles, metadata, codec):
    """Plan framing entirely from shapes, dtypes and caller metadata."""
    _codec(codec)
    if type(specs) is not dict or type(roles) is not dict or set(specs) != set(roles):
        raise StateCodecError("tensor specs and roles must be dictionaries with identical keys")
    if len(specs) > MAX_TENSORS:
        raise StateCodecError("too many tensors in state")
    if type(metadata) is not dict:
        raise StateCodecError("metadata must be a JSON object")
    _json_value(metadata)
    # Validate total decoded allocation before constructing the framing table.
    native_bytes, normalized = 0, {}
    for name, spec in specs.items():
        _name_and_role(name, roles[name])
        if type(spec) is not tuple or len(spec) != 2:
            raise StateCodecError("tensor spec must be a (shape, dtype) tuple")
        shape, count = _shape(spec[0])
        dtype = _dtype(spec[1])
        native_bytes += count * dtype.itemsize
        if native_bytes > MAX_STATE_BYTES:
            raise StateCodecError("decoded state exceeds byte limit")
        if codec != "raw_native" and roles[name] == "attention" and dtype.kind != "f":
            raise StateCodecError("compressed attention requires floating arrays; opaque BF16 bits are unsupported")
        normalized[name] = (shape, count, dtype)
    records, offset = [], 0
    for name in sorted(normalized):
        shape, count, dtype = normalized[name]
        role = roles[name]
        storage = codec if role == "attention" and codec != "raw_native" else "native"
        length = count * dtype.itemsize if storage == "native" else _attention_size(shape, storage)
        records.append({"name": name, "role": role, "storage": storage,
                        "dtype": dtype.str, "shape": list(shape),
                        "offset": offset, "length": length})
        offset += length
        if offset > MAX_STATE_BYTES:
            raise StateCodecError("encoded tensor segments exceed byte limit")
    header = {"version": 1, "codec": codec, "metadata": metadata, "tensors": records}
    encoded_header = _json_bytes(header)
    if len(encoded_header) > MAX_HEADER_BYTES:
        raise StateCodecError("state header exceeds byte limit")
    if _PREFIX.size + len(encoded_header) + offset > MAX_STATE_BYTES:
        raise StateCodecError("complete encoded state exceeds byte limit")
    return encoded_header, records, _PREFIX.size + len(encoded_header) + offset


def _layout(tensors, roles, metadata, codec):
    if type(tensors) is not dict or any(not isinstance(array, np.ndarray) for array in tensors.values()):
        raise StateCodecError("state tensors must be a dictionary of NumPy arrays")
    specs = {name: (array.shape, array.dtype) for name, array in tensors.items()}
    layout = _layout_from_specs(specs, roles, metadata, codec)
    for name, array in tensors.items():
        _finite(array)
        if codec != "raw_native" and roles[name] == "attention" and array.size:
            if np.any(array > 65504) or np.any(array < -65504):
                raise StateCodecError("attention tensor is outside the finite FP16 range")
    return layout


def encoded_size_from_specs(specs: dict[str, tuple[tuple[int, ...], np.dtype]],
                            roles: dict[str, str], metadata: dict, codec: str) -> int:
    """Exact framing size from shape/dtype descriptions, without touching data.

    Use this before deciding whether to export a live tensor-engine snapshot.
    Specs contain ``name: (shape, numpy_dtype)``. No array is constructed or
    scanned; finite-value and FP16-range checks still run in ``encode_state``
    after selection. A valid size is not proof that tensor values are encodable.
    """
    return _layout_from_specs(specs, roles, metadata, codec)[2]


def encoded_size(tensors: dict[str, np.ndarray], roles: dict[str, str],
                 metadata: dict, codec: str) -> int:
    """Exact bytes for a fixed-rate snapshot, without encoding any candidate.

    This includes both envelopes, tensor shapes, per-group scales, odd-nibble
    padding and all native recurrent state. It validates array values but does
    not create FP16 copies, quantized arrays, tensor byte strings or payloads.
    Planning work still takes CPU time and should be measured by the caller.
    """
    return _layout(tensors, roles, metadata, codec)[2]


def encode_state(tensors: dict[str, np.ndarray], roles: dict[str, str],
                 metadata: dict, codec: str) -> bytes:
    """Return one complete frame; ``len(result)`` includes every stored byte.

    Array shapes and dtypes are restored, though attention values are approximate
    for non-native codecs. The returned payload owns its bytes and does not alias
    live model arrays. No pickle, engine cache object, or borrowed tensor is stored.
    """
    encoded_header, records, expected_size = _layout(tensors, roles, metadata, codec)
    segments = []
    for record in records:
        name, storage = record["name"], record["storage"]
        try:
            segment = tensors[name].tobytes(order="C") if storage == "native" else encode_array(tensors[name], storage, group_size=64)
        except CodecError as exc:
            raise StateCodecError(f"cannot encode attention tensor {name!r}: {exc}") from exc
        if len(segment) != record["length"]:
            raise StateCodecError("core codec framing changed; exact size calculation must be updated")
        segments.append(segment)
    result = _PREFIX.pack(_MAGIC, len(encoded_header)) + encoded_header + b"".join(segments)
    if len(result) != expected_size:
        raise StateCodecError("snapshot framing disagrees with planned size")
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StateCodecError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise StateCodecError(f"nonfinite JSON number: {value}")


def decode_state(payload: bytes) -> tuple[dict[str, np.ndarray], dict[str, str], dict]:
    """Validate and decode an owned snapshot, retaining caller metadata unchanged.

    Every segment must be contiguous and exactly consume the framed body. Shape,
    dtype and aggregate allocation limits are checked before allocating arrays.
    This format validates structure; it is not authenticated or checksummed.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise StateCodecError("payload must be bytes-like")
    try:
        data = memoryview(payload).cast("B")
    except (TypeError, ValueError) as exc:
        raise StateCodecError("payload must be contiguous bytes") from exc
    if not _PREFIX.size <= len(data) <= MAX_STATE_BYTES:
        raise StateCodecError("invalid encoded state length")
    magic, header_length = _PREFIX.unpack_from(data)
    if magic != _MAGIC or not 1 <= header_length <= MAX_HEADER_BYTES:
        raise StateCodecError("invalid state magic or header length")
    if _PREFIX.size + header_length > len(data):
        raise StateCodecError("truncated state header")
    try:
        header = json.loads(bytes(data[_PREFIX.size:_PREFIX.size + header_length]),
                            object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise StateCodecError(f"invalid state JSON: {exc}") from exc
    if type(header) is not dict or set(header) != {"version", "codec", "metadata", "tensors"}:
        raise StateCodecError("invalid state envelope fields")
    if type(header["version"]) is not int or header["version"] != 1:
        raise StateCodecError("unsupported state format version")
    _codec(header["codec"])
    if type(header["metadata"]) is not dict:
        raise StateCodecError("metadata must be a JSON object")
    _json_value(header["metadata"])
    records = header["tensors"]
    if type(records) is not list or len(records) > MAX_TENSORS:
        raise StateCodecError("invalid tensor table")
    body = data[_PREFIX.size + header_length:]
    plans, names, offset, decoded_bytes = [], set(), 0, 0
    for record in records:
        if type(record) is not dict or set(record) != _RECORD_KEYS:
            raise StateCodecError("invalid tensor record fields")
        name, role = record["name"], record["role"]
        _name_and_role(name, role)
        if name in names:
            raise StateCodecError("duplicate tensor name")
        names.add(name)
        dtype = _dtype(record["dtype"])
        shape, count = _shape(record["shape"])
        native_length = count * dtype.itemsize
        decoded_bytes += native_length
        if decoded_bytes > MAX_STATE_BYTES:
            raise StateCodecError("decoded state exceeds byte limit")
        expected_storage = header["codec"] if role == "attention" and header["codec"] != "raw_native" else "native"
        if record["storage"] != expected_storage:
            raise StateCodecError("tensor storage disagrees with state codec and role")
        start, length = record["offset"], record["length"]
        if type(start) is not int or type(length) is not int or start != offset or length < 0:
            raise StateCodecError("tensor segments must be contiguous and nonnegative")
        if start + length > len(body):
            raise StateCodecError("truncated tensor segment")
        if expected_storage == "native" and length != native_length:
            raise StateCodecError("native tensor byte length disagrees with shape and dtype")
        if expected_storage != "native" and dtype.kind != "f":
            raise StateCodecError("compressed attention requires floating arrays")
        plans.append((name, role, expected_storage, dtype, shape, start, length))
        offset += length
    if offset != len(body):
        raise StateCodecError("trailing bytes outside tensor segments")
    tensors, roles = {}, {}
    for name, role, storage, dtype, shape, start, length in plans:
        segment = body[start:start + length]
        if storage == "native":
            array = np.frombuffer(segment, dtype=dtype).reshape(shape).copy(order="C")
        else:
            try:
                array = decode_array(segment)
            except CodecError as exc:
                raise StateCodecError(f"invalid attention tensor {name!r}: {exc}") from exc
            # The core decoder validates its own envelope, including unique JSON
            # keys. Also bind that representation to this outer tensor record.
            inner_length = struct.unpack_from("<I", segment, 4)[0]
            inner = json.loads(bytes(segment[8:8 + inner_length]))
            if inner["codec"] != storage:
                raise StateCodecError("attention tensor codec disagrees with outer metadata")
            if array.shape != shape:
                raise StateCodecError("attention tensor shape disagrees with outer metadata")
            array = array.astype(dtype, order="C", copy=True)
        _finite(array)
        tensors[name], roles[name] = array, role
    return tensors, roles, header["metadata"]
