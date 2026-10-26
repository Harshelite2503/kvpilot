"""Codec invariants and malformed-input tests; no model-quality assertions."""

import json
import struct
import unittest
import zlib
from unittest.mock import patch

import numpy as np

from kvpilot.codecs import CODECS, MAX_ELEMENTS, CodecError, decode, encode, fp16_reference


def parts(payload):
    length = struct.unpack_from("<I", payload, 4)[0]
    return json.loads(payload[8:8 + length]), payload[8 + length:]


def envelope(metadata, body):
    header = json.dumps(metadata).encode()
    return b"KVP\x00" + struct.pack("<I", len(header)) + header + body


class CodecTests(unittest.TestCase):
    def test_lossless_round_trip_bit_exact_including_negative_zero(self):
        reference = np.array([0, -0., 1, -3.125, 65504, 6e-8], dtype=np.float16)
        for codec in CODECS[:2]:
            with self.subTest(codec=codec):
                restored = decode(encode(reference, codec))
                self.assertEqual(restored.dtype, np.dtype("float16"))
                self.assertEqual(reference.tobytes(), restored.tobytes())

    def test_all_codecs_preserve_shape_and_finiteness(self):
        rng = np.random.default_rng(8)
        for shape in [(), (1,), (5,), (3, 5), (2, 3, 17), (0,), (2, 0), (0, 5)]:
            source = np.asarray(rng.normal(size=shape), dtype=np.float16)
            for codec in CODECS:
                with self.subTest(shape=shape, codec=codec):
                    result = decode(encode(source, codec, group_size=3))
                    self.assertEqual(result.shape, source.shape)
                    self.assertTrue(np.isfinite(result).all())

    def test_int4_signs_and_odd_tail(self):
        source = np.array([-7, -3, -1, 0, 1, 3, 7], dtype=np.float16)
        payload = encode(source, "group_int4", group_size=7)
        _, body = parts(payload)
        self.assertEqual(body[2:], bytes([0xD9, 0x0F, 0x31, 0x07]))
        np.testing.assert_array_equal(decode(payload), source)

    def test_groups_do_not_cross_rows(self):
        source = np.array([[1000, 0, .125], [1, -1, .001]], dtype=np.float16)
        result = decode(encode(source, "group_int4", group_size=2))
        np.testing.assert_allclose(result[:, -1], source[:, -1], rtol=.01, atol=1e-7)
        self.assertEqual(float(result[1, 0]), 1.)
        self.assertEqual(float(result[1, 1]), -1.)

    def test_zero_and_extreme_finite_values(self):
        for source in [np.zeros((4, 7)), np.array([65504, -65504, 6e-8, -6e-8])]:
            for codec in CODECS:
                with self.subTest(codec=codec):
                    result = decode(encode(source, codec, group_size=1))
                    self.assertTrue(np.isfinite(result).all())
                    np.testing.assert_allclose(result, source, rtol=.001, atol=1e-8)

    def test_error_bound_for_random_groups(self):
        source = np.random.default_rng(7).normal(size=(5, 37)).astype(np.float16)
        for codec, limit in [("group_int8", 127), ("group_int4", 7)]:
            result = decode(encode(source, codec, group_size=8)).astype(np.float32)
            for start in range(0, 37, 8):
                block = source[:, start:start + 8].astype(np.float32)
                error = np.abs(result[:, start:start + 8] - block)
                bound = np.max(np.abs(block), axis=1) / (2 * limit) + .004
                self.assertTrue((error <= bound[:, None]).all())

    def test_sizes_include_metadata_scales_and_packing(self):
        source = np.arange(15).reshape(3, 5)
        for codec in CODECS:
            payload = encode(source, codec, group_size=3)
            metadata, body = parts(payload)
            self.assertEqual(metadata["shape"], [3, 5])
            if codec == "raw_fp16":
                self.assertEqual(len(body), 30)
            elif codec == "group_int8":
                self.assertEqual(len(body), 12 + 15)
            elif codec == "group_int4":
                self.assertEqual(len(body), 12 + 8)
            else:
                self.assertEqual(len(zlib.decompress(body)), 30)

    def test_noncontiguous_and_big_endian_input(self):
        source = np.arange(48, dtype=">f4").reshape(6, 8)[::2, ::2]
        np.testing.assert_array_equal(decode(encode(source, "raw_fp16")), source)

    def test_reject_invalid_source(self):
        for source in [[np.nan], [np.inf], [-np.inf], [65505], [1 + 2j], ["1"], [True]]:
            with self.subTest(source=source), self.assertRaises(CodecError):
                encode(source, "raw_fp16")

    def test_reject_unknown_codec_and_invalid_group_size(self):
        with self.assertRaises(CodecError):
            encode([1.], "cachegen")
        for group_size in [0, -1, True, 1.5, 65537]:
            with self.subTest(group_size=group_size), self.assertRaises(CodecError):
                encode([1.], "group_int4", group_size)

    def test_invalid_prefix_or_truncation(self):
        payload = encode(np.ones((2, 3)), "group_int4")
        for invalid in [b"", b"bad!", b"NOPE" + payload[4:], payload[:8], payload[:-1], payload + b"x"]:
            with self.subTest(payload=invalid[:8]), self.assertRaises(CodecError):
                decode(invalid)

    def test_byte_buffers_and_noncontiguous_memoryview(self):
        payload = encode(np.ones(3), "raw_fp16")
        np.testing.assert_array_equal(decode(bytearray(payload)), np.ones(3))
        np.testing.assert_array_equal(decode(memoryview(payload)), np.ones(3))
        with self.assertRaises(CodecError):
            decode(memoryview(payload)[::2])

    def test_reject_bad_metadata(self):
        original, body = parts(encode(np.ones(3), "raw_fp16"))
        changes = [
            {"version": 2}, {"version": True}, {"dtype": "object"},
            {"codec": "unknown"}, {"codec": []}, {"shape": [-1]},
            {"shape": [True]}, {"shape": [2.5]}, {"shape": "3"},
            {"shape": [MAX_ELEMENTS, 2]}, {"shape": [MAX_ELEMENTS, MAX_ELEMENTS, 0]},
            {"shape": [1] * 33}, {"extra": 1},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(CodecError):
                decode(envelope({**original, **change}, body))

    def test_duplicate_metadata_key_and_json_nonobject(self):
        for header in [b'{"version":1,"version":1}', b'[]', b'{', b'\xff']:
            with self.subTest(header=header), self.assertRaises(CodecError):
                decode(b"KVP\x00" + struct.pack("<I", len(header)) + header)

    def test_oversized_shape_rejected_before_numpy_allocation(self):
        metadata, body = parts(encode(np.ones(1), "group_int4"))
        metadata["shape"] = [MAX_ELEMENTS, MAX_ELEMENTS]
        with patch("kvpilot.codecs.np.empty", side_effect=AssertionError("must not allocate")):
            with self.assertRaises(CodecError):
                decode(envelope(metadata, body))

    def test_reject_invalid_group_scale_and_unused_codes(self):
        for codec in ["group_int8", "group_int4"]:
            metadata, body = parts(encode(np.ones(3), codec))
            for scale in [0, -1, np.inf, np.nan]:
                altered = np.array([scale], dtype="<f2").tobytes() + body[2:]
                with self.subTest(codec=codec, scale=scale), self.assertRaises(CodecError):
                    decode(envelope(metadata, altered))
            altered = bytearray(body)
            altered[2] = 128 if codec == "group_int8" else 8
            with self.assertRaises(CodecError):
                decode(envelope(metadata, altered))

    def test_reject_nonzero_int4_padding(self):
        metadata, body = parts(encode(np.ones(3), "group_int4"))
        altered = body[:-1] + bytes([body[-1] | 0x10])
        with self.assertRaises(CodecError):
            decode(envelope(metadata, altered))

    def test_reject_nonfinite_raw_payload(self):
        metadata, _ = parts(encode(np.ones(1), "raw_fp16"))
        with self.assertRaises(CodecError):
            decode(envelope(metadata, np.array([np.nan], dtype="<f2").tobytes()))

    def test_zlib_rejects_trailing_truncated_and_oversized_output(self):
        metadata, body = parts(encode(np.ones(3), "zlib_fp16"))
        for invalid in [body[:-1], body + b"trailing", zlib.compress(b"\x00" * 1_000_000), b"bad"]:
            with self.subTest(length=len(invalid)), self.assertRaises(CodecError):
                decode(envelope(metadata, invalid))

    def test_reference_conversion_is_independent(self):
        source = np.array([1], dtype=np.float16)
        converted = fp16_reference(source)
        converted[0] = 2
        self.assertEqual(source[0], 1)


if __name__ == "__main__":
    unittest.main()
