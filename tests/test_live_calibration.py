import numpy as np
import pytest

from kvpilot.live.calibration import measure_codec


def test_measure_codec_reports_real_bytes_and_nonnegative_timing():
    array = np.random.default_rng(0).normal(size=(32, 256)).astype(np.float32)
    result = measure_codec(array, "group_int4", repeats=2)
    assert result.raw_bytes == array.astype("<f2").nbytes
    assert result.stored_bytes > 0
    assert result.encode_ms >= 0
    assert result.decode_ms >= 0
    assert result.max_absolute_error > 0  # int4 is lossy


def test_measure_codec_raw_fp16_is_lossless():
    array = np.ones((4, 64), dtype=np.float32)
    result = measure_codec(array, "raw_fp16", repeats=1)
    assert result.max_absolute_error == 0.0
    profile = result.to_profile()
    assert profile.lossless is True
    assert profile.name == "raw_fp16"


def test_measure_codec_rejects_empty_array():
    with pytest.raises(ValueError, match="empty"):
        measure_codec(np.zeros((0,), dtype=np.float32), "raw_fp16")


def test_measure_codec_rejects_nonpositive_repeats():
    with pytest.raises(ValueError, match="repeats"):
        measure_codec(np.ones((4, 4), dtype=np.float32), "raw_fp16", repeats=0)


def test_to_profile_custom_name_avoids_reserved_collision():
    array = np.ones((4, 64), dtype=np.float32)
    result = measure_codec(array, "group_int8", repeats=1)
    profile = result.to_profile("measured_group_int8")
    assert profile.name == "measured_group_int8"
    assert profile.ratio == pytest.approx(result.raw_bytes / result.stored_bytes)
