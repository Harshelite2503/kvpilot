import pytest

from kvpilot.policy import CodecProfile, CostModel, Forecast, HistoricalForecast, MIB, choose_action


RAW = CodecProfile("raw_fp16", 1, 0, 0)


def model(**kwargs):
    return CostModel(**dict({"bandwidth_bytes_per_ms": MIB, "host_budget_bytes": 100 * MIB}, **kwargs))


def test_compression_can_cost_more_than_transfer_saved():
    slow = CodecProfile("slow", 4, 10, 10)
    decision = choose_action(MIB, 100, [RAW, slow], model(), Forecast(1, 100))
    assert decision.selected.action == "raw_fp16"


def test_never_resumed_work_does_not_deserve_encoding():
    assert choose_action(MIB, 100, [RAW], model(), Forecast(0, 100)).selected.action == "recompute"


def test_lossy_is_disallowed_without_explicit_exploration():
    lossy = CodecProfile("int4", 4, 0.01, 0.01, lossless=False)
    decision = choose_action(MIB, 100, [RAW, lossy], model(), Forecast(1, 100))
    assert all(c.action != "int4" for c in decision.alternatives)
    with pytest.raises(ValueError, match="quality"):
        choose_action(MIB, 100, [lossy], model(), Forecast(), force_codec="int4")


def test_deadline_can_exclude_cheaper_recompute():
    d = choose_action(MIB, 100, [RAW], model(resume_slo_ms=5), Forecast(0.01, 100))
    assert d.selected.action == "raw_fp16"


def test_large_fixed_cache_falls_back_instead_of_exceeding_budget():
    d = choose_action(2 * MIB, 100, [RAW], model(host_budget_bytes=MIB), Forecast(), force_codec="raw_fp16")
    assert d.selected.action == "recompute"


def test_forecast_uses_resolved_history_only():
    h = HistoricalForecast(123)
    assert h.predict("tool") == Forecast(0.5, 123)
    h.observe("tool", True, 10)
    assert h.predict("tool") == Forecast(2/3, 10)
    h.observe("tool", False, 30)
    assert h.predict("tool") == Forecast(0.5, 20)
    assert h.predict("human") == Forecast(0.5, 123)


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_reject_invalid_profiles(bad):
    with pytest.raises(ValueError):
        CodecProfile("bad", bad, 0, 0)


def test_duplicate_codec_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        choose_action(MIB, 1, [RAW, RAW], model(), Forecast())


def test_infeasible_target_fallback_includes_unfinished_encoding():
    slow = CodecProfile("slow", 1, 100, 0)
    d = choose_action(MIB, 50, [slow], model(resume_slo_ms=1), Forecast(1, 0))
    assert d.selected.action == "recompute"
    assert d.alternatives[1].predicted_resume_ms == 102


def test_shared_pages_restore_cost_is_part_of_session_target():
    d = choose_action(MIB, 5, [RAW], model(resume_slo_ms=10), Forecast(1, 100), shared_restore_ms=20)
    assert not d.alternatives[1].within_resume_slo
    assert d.selected.action == "recompute"


@pytest.mark.parametrize("bad", ["false", "true", 0, 1, None])
def test_lossy_flags_require_booleans(bad):
    with pytest.raises(ValueError, match="Boolean"):
        model(allow_unvalidated_lossy=bad)
    with pytest.raises(ValueError, match="Boolean"):
        CodecProfile("int4", 4, 0, 0, lossless=bad)


def test_boolean_is_not_a_byte_budget():
    with pytest.raises(ValueError, match="positive integer"):
        model(host_budget_bytes=True)


@pytest.mark.parametrize("name", ["adaptive", "recompute", 1, ""])
def test_codec_names_cannot_collide_with_policy_names(name):
    with pytest.raises(ValueError, match="codec name"):
        CodecProfile(name, 1, 0, 0)
