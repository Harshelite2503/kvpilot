from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path

import pytest

from kvpilot.policy import AdmissionImpact, CodecProfile, CostModel, Forecast, HistoricalForecast, MIB, choose_action
from kvpilot.replay import Entry, estimate_eviction_cost, load_config, load_trace, plan_lru_evictions, replay


RAW = CodecProfile("raw_fp16", 1, 0, 0)
INT4 = CodecProfile("group_int4", 4, 0.25, 0.15, lossless=False)


def owner(keys, cost=100, at=0):
    return {"keys": keys, "recompute_ms": cost, "at_ms": at, "kind": "tool"}


def pause(sid, at, blocks, recompute=100):
    return {"at_ms": at, "type": "pause", "session_id": sid, "pause_kind": "tool",
            "namespace": {"tenant": "t", "model_revision": "m", "adapter_revision": "none", "layout": "v1"},
            "blocks": [{"prefix_hash": k, "raw_bytes": size} for k, size in blocks], "recompute_ms": recompute}


def test_two_victims_charge_one_complete_owner_once():
    store = OrderedDict((k, Entry(100, 100, "raw_fp16", 0, 0)) for k in ["a", "b"])
    loss, detail = estimate_eviction_cost(["a", "b"], [], store, {"s": owner(["a", "b"])},
                                         HistoricalForecast(), CostModel(100, 200), 0)
    assert loss == 0.5 * (100 - 2)
    assert len(detail) == 1


def test_one_shared_victim_charges_each_owner_but_frees_one_page():
    store = OrderedDict(a=Entry(100, 100, "raw_fp16", 0, 0))
    victims = plan_lru_evictions(store, 100, 100, 100, set())
    assert victims == ("a",)
    loss, detail = estimate_eviction_cost(victims, ["new"], store,
                                         {"s1": owner(["a"], 100), "s2": owner(["a"], 200)},
                                         HistoricalForecast(), CostModel(100, 100), 0)
    assert loss == 0.5 * 99 + 0.5 * 199
    assert len(detail) == 2


def test_incomplete_owner_is_not_charged_again_or_credited_for_repair():
    store = OrderedDict(a=Entry(100, 100, "raw_fp16", 0, 0))
    owners = {"s": owner(["a", "missing"])}
    assert estimate_eviction_cost(["a"], ["missing"], store, owners, HistoricalForecast(), CostModel(100, 200), 0) == (0, [])
    assert estimate_eviction_cost([], ["missing"], store, owners, HistoricalForecast(), CostModel(100, 200), 0) == (0, [])


def test_pending_cache_that_would_lose_to_recompute_has_zero_eviction_penalty():
    store = OrderedDict(a=Entry(100, 100, "raw_fp16", 1000, 0))
    loss, detail = estimate_eviction_cost(["a"], [], store, {"s": owner(["a"], 10)},
                                         HistoricalForecast(1), CostModel(100, 100), 20)
    assert loss == 0
    assert detail[0]["predicted_remaining_pause_ms"] == 0
    assert detail[0]["restore_ms_without_eviction"] == 981


def test_planning_does_not_mutate_recency_and_protects_prefix():
    store = OrderedDict((k, Entry(100, 100, "raw_fp16", 0, 0)) for k in ["prefix", "victim", "last"])
    before = deepcopy(store)
    assert plan_lru_evictions(store, 300, 300, 100, {"prefix"}) == ("victim",)
    assert plan_lru_evictions(store, 300, 300, 201, {"prefix"}) is None
    assert store == before


def test_eviction_penalty_can_choose_smaller_codec_and_execution_matches_plan():
    # Incoming raw costs less alone, but destroys an expensive existing session.
    events = [pause("expensive", 0, [("old", MIB)]), pause("incoming", 10, [("new", MIB)])]
    model = CostModel(16 * MIB, MIB + MIB // 4, allow_unvalidated_lossy=True)
    old = replay(events, model, [RAW, INT4], policy="adaptive")
    new = replay(events, model, [RAW, INT4], policy="adaptive_eviction")
    assert old["decisions"][1]["actions"][0]["selected"]["action"] == "raw_fp16"
    assert new["decisions"][1]["actions"][0]["selected"]["action"] == "group_int4"
    impacts = new["decisions"][1]["actions"][1]["admission_impacts"]
    assert len(impacts["raw_fp16"]["victims"]) == 1
    assert impacts["raw_fp16"]["eviction_cost_ms"] > 0
    assert impacts["group_int4"]["victims"] == []
    assert new["metrics"]["evicted_blocks"] == 0
    assert old["metrics"]["evicted_blocks"] == 1


def test_expensive_eviction_can_reject_incoming_admission():
    events = [pause("old", 0, [("old", MIB)], 100), pause("new", 10, [("new", MIB)], 1)]
    result = replay(events, CostModel(16 * MIB, MIB), [RAW], policy="adaptive_eviction")
    assert result["decisions"][1]["actions"][0]["selected"]["action"] == "recompute"
    assert result["metrics"]["evicted_blocks"] == 0


def test_incoming_slo_priority_is_explicit_even_if_victim_cost_is_high():
    impacts = {"raw_fp16": AdmissionImpact(MIB, 1_000_000)}
    result = choose_action(MIB, 100, [RAW], CostModel(MIB, MIB, resume_slo_ms=5), Forecast(1, 100), admission_impacts=impacts)
    assert result.selected.action == "raw_fp16"
    assert result.selected.eviction_cost_ms == 1_000_000


def test_exact_per_page_rounding_excludes_infeasible_candidate_before_scoring():
    # Aggregate ceil(3/4)=1 but three physical one-byte pages require 3 bytes.
    events = [pause("s", 0, [("a", 1), ("b", 1), ("c", 1)])]
    result = replay(events, CostModel(MIB, 2, allow_unvalidated_lossy=True), [INT4], policy="adaptive_eviction")
    decision = result["decisions"][0]["actions"][0]
    assert decision["selected"]["action"] == "recompute"
    assert len(decision["alternatives"]) == 1
    assert result["metrics"]["peak_host_reserved_bytes"] == 0


def test_zero_free_space_does_not_offer_an_impossible_one_byte_admission():
    events = [pause("old", 0, [("shared", 1)]), pause("new", 1, [("shared", 1), ("tail", 1)])]
    result = replay(events, CostModel(100, 1), [RAW], policy="adaptive_eviction")
    decision = result["decisions"][1]["actions"][0]
    assert decision["selected"]["action"] == "recompute"
    assert len(decision["alternatives"]) == 1


def test_future_outcomes_cannot_change_eviction_penalties_or_decisions():
    prefix = [pause("old", 0, [("old", MIB)]), pause("new", 10, [("new", MIB)])]
    model = CostModel(16 * MIB, MIB, allow_unvalidated_lossy=True)
    first = replay(prefix + [{"at_ms": 100, "type": "resume", "session_id": "old"}], model, [RAW, INT4], policy="adaptive_eviction")
    second = replay(prefix + [{"at_ms": 1000, "type": "cancel", "session_id": "old"}], model, [RAW, INT4], policy="adaptive_eviction")
    assert first["decisions"][:2] == second["decisions"][:2]


@pytest.mark.parametrize("policy", ["adaptive", "adaptive_exact", "adaptive_eviction", "raw_fp16", "recompute"])
def test_eviction_changes_preserve_budget_and_lossless_defaults(policy):
    events = [pause(str(i), i, [(str(i), MIB)]) for i in range(20)]
    result = replay(events, CostModel(16 * MIB, 2 * MIB), [RAW, INT4], policy=policy)
    assert result["metrics"]["peak_host_reserved_bytes"] <= 2 * MIB
    for event in result["decisions"]:
        for action in event.get("actions", []):
            if "selected" in action:
                assert action["selected"]["action"] != "group_int4"
    if policy == "recompute":
        assert result["metrics"]["modeled_transferred_bytes"] == 0


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_bad_eviction_costs_rejected(bad):
    with pytest.raises(ValueError):
        AdmissionImpact(100, bad)


def test_all_legacy_baseline_metrics_match_the_original_checked_in_run():
    root = Path(__file__).resolve().parents[1]
    model, profiles, config = load_config(root / "examples/assumed-costs.json")
    original = json.loads((root / "results/policy-replay.json").read_text())
    for old in original["results"]:
        current = replay(load_trace(root / "examples/synthetic-sessions.jsonl"), model, profiles,
                         policy=old["policy"], default_pause_ms=config["default_pause_ms"])
        for name, expected in old["metrics"].items():
            assert current["metrics"][name] == expected, (old["policy"], name)


def test_matched_control_has_exact_sizing_but_no_eviction_charge():
    events = [pause("expensive", 0, [("old", MIB)]), pause("incoming", 10, [("new", MIB)])]
    model = CostModel(16 * MIB, MIB + MIB // 4, allow_unvalidated_lossy=True)
    result = replay(events, model, [RAW, INT4], policy="adaptive_exact")
    decision = result["decisions"][1]["actions"][0]
    assert decision["selected"]["action"] == "raw_fp16"
    assert decision["selected"]["eviction_cost_ms"] == 0
    impact = result["decisions"][1]["actions"][1]["admission_impacts"]["raw_fp16"]
    assert impact["estimated_eviction_loss_ms"] > 0
    assert impact["eviction_cost_ms"] == 0
    tiny = [pause("tiny", 0, [("a", 1), ("b", 1), ("c", 1)], 4)]
    for policy in ["adaptive_exact", "adaptive_eviction"]:
        result = replay(tiny, CostModel(1, 10, allow_unvalidated_lossy=True), [INT4], policy=policy)
        assert result["decisions"][0]["actions"][0]["selected"]["action"] == "recompute"
