"""Owned snapshot serialization invariants; no inference or cache library used."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "local_state_codec", Path(__file__).resolve().parents[1] / "experiments/local/state_codec.py"
)
state = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(state)


def parts(payload):
    n = struct.unpack_from("<I", payload, 4)[0]
    return json.loads(payload[8:8 + n]), payload[8 + n:]


def frame(header, body):
    encoded = json.dumps(header).encode()
    return b"KVS\x00" + struct.pack("<I", len(encoded)) + encoded + body


def snapshot():
    return ({
        "layer3.k": np.array([[[[-7., 0., 1.3, 7., -.01]]]], dtype=np.float16),
        "layer3.v": np.array([[[[.1, -2., 0., 4., 1.]]]], dtype=np.float16),
        "layer0.recurrent": np.array([1.0000001, -0., 1e12, -1e-25], dtype=np.float32),
        "layer0.conv": np.array([0.00000013, 2.0000002], dtype=np.float32),
        "position": np.array(27, dtype=np.int32),
    }, {"layer3.k": "attention", "layer3.v": "attention", "layer0.recurrent": "recurrent",
        "layer0.conv": "convolution", "position": "position"},
        {"offset": 27, "model_revision": "fixture@1", "logical_dtypes": {"position": "int32"}})


def test_native_roundtrip_is_bit_exact_preserves_all_metadata_and_does_not_alias():
    tensors, roles, metadata = snapshot()
    before = {name: array.tobytes() for name, array in tensors.items()}
    payload = state.encode_state(tensors, roles, metadata, "raw_native")
    decoded, restored_roles, restored_metadata = state.decode_state(payload)
    assert restored_roles == roles and restored_metadata == metadata
    for name, array in tensors.items():
        assert decoded[name].dtype == array.dtype and decoded[name].shape == array.shape
        assert decoded[name].tobytes() == before[name]
        assert decoded[name].flags.writeable
        assert not np.shares_memory(decoded[name], array)
    tensors["layer3.k"].fill(99)
    assert state.decode_state(payload)[0]["layer3.k"].tobytes() == before["layer3.k"]


@pytest.mark.parametrize("codec", state.CODECS)
def test_only_attention_can_change_and_every_native_byte_is_counted(codec):
    tensors, roles, metadata = snapshot()
    payload = state.encode_state(tensors, roles, metadata, codec)
    decoded, restored_roles, restored_metadata = state.decode_state(payload)
    header, body = parts(payload)
    assert len(payload) == 8 + struct.unpack_from("<I", payload, 4)[0] + sum(r["length"] for r in header["tensors"])
    assert len(body) == sum(r["length"] for r in header["tensors"])
    assert restored_roles == roles and restored_metadata == metadata
    for name in tensors:
        assert decoded[name].shape == tensors[name].shape
        assert decoded[name].dtype == tensors[name].dtype
        assert np.isfinite(decoded[name]).all()
        if roles[name] != "attention" or codec in ("raw_native", "raw_fp16"):
            assert decoded[name].tobytes() == tensors[name].tobytes()
    if codec == "group_int4":
        assert any(decoded[n].tobytes() != tensors[n].tobytes() for n in tensors if roles[n] == "attention")


def test_raw_fp16_exposes_attention_casting_loss_without_damaging_fp32_recurrence():
    value = np.array([1.0000001, 1.0001, -.000000003], dtype=np.float32)
    tensors = {"k": value, "s": value.copy()}
    payload = state.encode_state(tensors, {"k": "attention", "s": "recurrent"}, {}, "raw_fp16")
    restored, _, _ = state.decode_state(payload)
    assert restored["k"].dtype == np.float32
    np.testing.assert_array_equal(restored["k"], value.astype(np.float16).astype(np.float32))
    assert restored["k"].tobytes() != value.tobytes()
    assert restored["s"].tobytes() == value.tobytes()


def test_uint16_bf16_bits_are_exact_native_and_never_implicitly_quantized():
    bits = np.array([0x3F80, 0x8000, 0x3F81, 0x7F7F], dtype=np.uint16)
    metadata = {"logical_dtypes": {"k": "bfloat16"}}
    payload = state.encode_state({"k": bits}, {"k": "attention"}, metadata, "raw_native")
    decoded, _, restored_metadata = state.decode_state(payload)
    assert decoded["k"].tobytes() == bits.tobytes() and restored_metadata == metadata
    for codec in ("raw_fp16", "group_int8", "group_int4"):
        with pytest.raises(state.StateCodecError, match="floating"):
            state.encode_state({"k": bits}, {"k": "attention"}, metadata, codec)
        payload = state.encode_state({"s": bits}, {"s": "recurrent"}, metadata, codec)
        assert state.decode_state(payload)[0]["s"].tobytes() == bits.tobytes()


@pytest.mark.parametrize("codec", state.CODECS)
def test_scalar_empty_noncontiguous_and_endian_arrays(codec):
    tensors = {"k": np.arange(12, dtype=np.float16).reshape(3, 4)[:, ::2],
               "empty": np.zeros((2, 0), dtype=np.float32),
               "scalar": np.array(-0., dtype=">f4"), "counter": np.array([1, -9], dtype=">i4")}
    roles = {"k": "attention", "empty": "attention", "scalar": "recurrent", "counter": "position"}
    restored, _, _ = state.decode_state(state.encode_state(tensors, roles, {}, codec))
    for name, array in tensors.items():
        assert restored[name].shape == array.shape and restored[name].dtype == array.dtype
        if roles[name] != "attention" or codec in ("raw_native", "raw_fp16"):
            assert restored[name].tobytes() == array.tobytes()


@pytest.mark.parametrize("mutation", [
    lambda h: h.update(version=True),
    lambda h: h.update(version=2),
    lambda h: h.update(codec="unknown"),
    lambda h: h["tensors"][0].update(offset=1),
    lambda h: h["tensors"][0].update(length=0),
    lambda h: h["tensors"][0].update(dtype="object"),
    lambda h: h["tensors"][0].update(shape=[state.MAX_ELEMENTS, 2]),
    lambda h: h["tensors"][0].update(shape=[0, state.MAX_ELEMENTS, state.MAX_ELEMENTS]),
    lambda h: h["tensors"][0].update(storage="group_int4"),
    lambda h: h["tensors"].append(deepcopy(h["tensors"][0])),
])
def test_malformed_outer_metadata_is_rejected_before_restore(mutation):
    header, body = parts(state.encode_state(*snapshot(), "raw_native"))
    mutation(header)
    with pytest.raises(state.StateCodecError):
        state.decode_state(frame(header, body))


def test_compressed_inner_shape_must_match_outer_shape():
    header, body = parts(state.encode_state({"k": np.ones((2, 3), dtype=np.float16)},
                                           {"k": "attention"}, {}, "group_int4"))
    header["tensors"][0]["shape"] = [3, 2]
    with pytest.raises(state.StateCodecError, match="shape"):
        state.decode_state(frame(header, body))


def test_compressed_inner_codec_must_match_outer_codec():
    header, body = parts(state.encode_state({"k": np.ones((2, 3), dtype=np.float16)},
                                           {"k": "attention"}, {}, "raw_fp16"))
    header["codec"] = "group_int4"
    header["tensors"][0]["storage"] = "group_int4"
    with pytest.raises(state.StateCodecError, match="codec"):
        state.decode_state(frame(header, body))


def test_corrupted_native_float_nonfinite_bytes_are_rejected():
    header, _ = parts(state.encode_state({"s": np.ones(1, dtype=np.float32)},
                                        {"s": "recurrent"}, {}, "raw_native"))
    with pytest.raises(state.StateCodecError, match="finite"):
        state.decode_state(frame(header, np.array([np.nan], dtype=np.float32).tobytes()))


def test_truncation_trailing_bytes_duplicate_json_and_nonfinite_state_are_rejected():
    payload = state.encode_state(*snapshot(), "raw_native")
    for malformed in (payload[:-1], payload + b"extra", b"bad", b"BAD!" + payload[4:]):
        with pytest.raises(state.StateCodecError):
            state.decode_state(malformed)
    duplicate = b'{"version":1,"version":1,"codec":"raw_native","metadata":{},"tensors":[]}'
    with pytest.raises(state.StateCodecError, match="duplicate"):
        state.decode_state(b"KVS\0" + struct.pack("<I", len(duplicate)) + duplicate)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(state.StateCodecError, match="finite"):
            state.encode_state({"s": np.array([bad], dtype=np.float32)}, {"s": "recurrent"}, {}, "raw_native")
        with pytest.raises(state.StateCodecError, match="finite"):
            state.encode_state({}, {}, {"bad": bad}, "raw_native")


def test_total_allocation_and_frame_sizes_are_bounded(monkeypatch):
    array = np.zeros(8, dtype=np.float32)
    payload = state.encode_state({"s": array}, {"s": "recurrent"}, {}, "raw_native")
    monkeypatch.setattr(state, "MAX_STATE_BYTES", 31)
    with pytest.raises(state.StateCodecError, match="byte limit"):
        state.encode_state({"s": array}, {"s": "recurrent"}, {}, "raw_native")
    with pytest.raises(state.StateCodecError, match="length"):
        state.decode_state(payload)


def test_empty_snapshot_and_json_metadata_roundtrip_without_input_mutation():
    metadata = {"offset": 0, "nested": [None, True, {"revision": "test"}], "scale": .25}
    saved = deepcopy(metadata)
    assert state.decode_state(state.encode_state({}, {}, metadata, "raw_native")) == ({}, {}, metadata)
    assert metadata == saved
    with pytest.raises(state.StateCodecError, match="identical keys"):
        state.encode_state({"s": np.array([1], dtype=np.int32)}, {}, {}, "raw_native")
    with pytest.raises(state.StateCodecError, match="JSON"):
        state.encode_state({}, {}, {"not_json": (1, 2)}, "raw_native")


@pytest.mark.parametrize("codec", state.CODECS)
@pytest.mark.parametrize("shape", [(), (0,), (0, 65), (2, 0), (3,), (2, 65), (2, 3, 129)])
def test_exact_size_includes_headers_scales_padding_native_and_unicode(codec, shape):
    tensors = {"κ": np.arange(np.prod(shape, dtype=int), dtype=np.float32).reshape(shape),
               "recurrent": np.array([1e10, 1.0000001], dtype=">f4")}
    roles = {"κ": "attention", "recurrent": "recurrent"}
    metadata = {"offset": 1001, "model": "测试-model", "nested": [True, {"x": 1.0}]}
    assert state.encoded_size(tensors, roles, metadata, codec) == len(state.encode_state(tensors, roles, metadata, codec))


def test_size_planning_does_not_invoke_array_encoder(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("candidate encoding would be uncharged work")
    monkeypatch.setattr(state, "encode_array", forbidden)
    for codec in state.CODECS:
        assert state.encoded_size(*snapshot(), codec) > 0


def test_size_planning_rejects_unencodable_attention_values():
    tensors, roles = {"k": np.array([1e10], dtype=np.float32)}, {"k": "attention"}
    assert state.encoded_size(tensors, roles, {}, "raw_native") > 0
    with pytest.raises(state.StateCodecError, match="FP16 range"):
        state.encoded_size(tensors, roles, {}, "group_int4")


def test_encoding_detects_core_size_formula_drift(monkeypatch):
    original = state.encode_array
    monkeypatch.setattr(state, "encode_array", lambda *args, **kwargs: original(*args, **kwargs) + b"extra")
    with pytest.raises(state.StateCodecError, match="framing changed"):
        state.encode_state(*snapshot(), "group_int4")


@pytest.mark.parametrize("codec", state.CODECS)
def test_spec_only_size_is_exact_and_never_materializes_or_scans_arrays(monkeypatch, codec):
    tensors, roles, metadata = snapshot()
    specs = {name: (array.shape, array.dtype) for name, array in tensors.items()}
    expected = len(state.encode_state(tensors, roles, metadata, codec))
    def forbidden(*args, **kwargs):
        raise AssertionError("spec-only sizing must not touch or materialize tensor values")
    with monkeypatch.context() as patch:
        for name in ("isfinite", "array", "asarray", "zeros", "empty", "broadcast_to"):
            patch.setattr(state.np, name, forbidden)
        patch.setattr(state, "encode_array", forbidden)
        assert state.encoded_size_from_specs(specs, roles, metadata, codec) == expected


@pytest.mark.parametrize("shape", [(state.MAX_ELEMENTS, 2), (0, state.MAX_ELEMENTS, state.MAX_ELEMENTS), (-1,)])
def test_spec_only_size_rejects_oversized_or_invalid_shapes_without_allocation(shape):
    with pytest.raises(state.StateCodecError, match="shape|dimension|element|stride"):
        state.encoded_size_from_specs({"s": (shape, np.dtype("float32"))}, {"s": "recurrent"}, {}, "raw_native")


def test_spec_size_does_not_claim_value_validity_and_still_enforces_dtype_rules():
    tensors = {"k": np.array([np.inf], dtype=np.float32)}
    assert state.encoded_size_from_specs({"k": ((1,), np.dtype("float32"))}, {"k": "attention"}, {}, "group_int4") > 0
    with pytest.raises(state.StateCodecError, match="finite"):
        state.encode_state(tensors, {"k": "attention"}, {}, "group_int4")
    with pytest.raises(state.StateCodecError, match="floating"):
        state.encoded_size_from_specs({"k": ((1,), np.dtype("uint16"))}, {"k": "attention"}, {}, "group_int4")
