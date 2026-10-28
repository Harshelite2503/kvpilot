import numpy as np
import pytest

from kvpilot.live.calibration import measure_codec
from kvpilot.policy import CostModel, Forecast, choose_action


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


@pytest.mark.parametrize("codec", ["group_int4", "group_int8"])
def test_exact_zero_sample_cannot_approve_a_lossy_codec(codec):
    result = measure_codec(np.zeros((4, 64), dtype=np.float16), codec, repeats=1)
    assert result.max_absolute_error == 0
    profile = result.to_profile()
    assert profile.lossless is False
    kwargs = dict(raw_bytes=result.raw_bytes, recompute_ms=100, profiles=[profile],
                  model=CostModel(bandwidth_bytes_per_ms=1_000_000, host_budget_bytes=1_000_000),
                  forecast=Forecast())
    assert choose_action(**kwargs).selected.action == "recompute"
    with pytest.raises(ValueError, match="no quality approval"):
        choose_action(**kwargs, force_codec=codec)
    # A different calibration name must not change the underlying guarantee.
    assert result.to_profile("measured_zero_sample").lossless is False


@pytest.mark.parametrize("codec", ["raw_fp16", "zlib_fp16"])
def test_lossless_reference_codec_reports_source_conversion_separately(codec):
    source = np.array([1.0001, -2.0001], dtype=np.float32)
    result = measure_codec(source, codec, repeats=1)
    assert result.reference_dtype == "<f2"
    assert result.source_dtype == "float32"
    assert result.raw_bytes == source.size * 2
    assert result.max_absolute_error == 0
    assert result.source_conversion_max_absolute_error == pytest.approx(
        float(np.max(np.abs(source.astype(np.float64) - source.astype(np.float16).astype(np.float64)))))
    assert result.source_conversion_max_absolute_error > 0
    assert result.to_profile().lossless is True  # Relative to the declared FP16 reference.


def test_measure_codec_rejects_empty_array():
    with pytest.raises(ValueError, match="empty"):
        measure_codec(np.zeros((0,), dtype=np.float32), "raw_fp16")


@pytest.mark.parametrize("repeats", [0, -1, 1.5, True])
def test_measure_codec_rejects_invalid_repeats(repeats):
    with pytest.raises(ValueError, match="repeats"):
        measure_codec(np.ones((4, 4), dtype=np.float32), "raw_fp16", repeats=repeats)


def test_to_profile_custom_name_avoids_reserved_collision():
    array = np.ones((4, 64), dtype=np.float32)
    result = measure_codec(array, "group_int8", repeats=1)
    profile = result.to_profile("measured_group_int8")
    assert profile.name == "measured_group_int8"
    assert profile.ratio == pytest.approx(result.raw_bytes / result.stored_bytes)
