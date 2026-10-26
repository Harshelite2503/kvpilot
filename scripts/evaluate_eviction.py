"""Paired whole-run evaluation of assumed-cost KV policies; never live serving.

Holdout generation is locked behind an approved, hash-verified freeze manifest.
This script does not create or approve freezes. Run development first, review the
code, then have the experiment owner create/approve build_freeze_manifest(...).
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kvpilot.policy import CodecProfile, CostModel
from kvpilot.replay import percentile, replay

EVIDENCE = "analytical_trace_replay"
TOTAL_SERVICE_METRIC = "modeled_total_service_work_ms"


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value: object) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def source_hashes(protocol_path: Path, root: Path = ROOT) -> dict[str, str]:
    """Bind all core source, evaluator, workload protocol and base cost fixture."""
    paths = sorted((root / "src").rglob("*.py"))
    paths += [root / "scripts" / "evaluate_eviction.py"]
    base_cost = root / "examples" / "assumed-costs.json"
    if base_cost.exists():
        paths.append(base_cost)
    required = [root / "src" / "kvpilot" / name for name in ("policy.py", "replay.py", "workloads.py")]
    if any(not p.is_file() for p in required):
        raise ValueError("cannot freeze: policy, replay and workload source must exist")
    result = {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in paths}
    result["experiment_protocol"] = sha256(protocol_path.read_bytes()).hexdigest()
    return result


def declared_metrics(protocol: dict) -> tuple[list[str], list[str]]:
    primary = protocol.get("primary_metrics", ["modeled_p95_resume_ms"])
    secondary = protocol.get("secondary_metrics", [
        "modeled_mean_resume_ms", TOTAL_SERVICE_METRIC, "host_reserved_byte_ms",
        "peak_host_reserved_bytes", "modeled_transferred_bytes", "full_recomputes",
    ])
    if primary != ["modeled_p95_resume_ms"]:
        raise ValueError("this experiment preregisters p95 resume delay as its sole primary metric")
    if not isinstance(secondary, list) or any(not isinstance(m, str) or not m for m in secondary):
        raise ValueError("secondary_metrics must be a list of metric names")
    if len(set(primary + secondary)) != len(primary + secondary):
        raise ValueError("declared metrics must be unique")
    return primary, secondary


def build_freeze_manifest(protocol_path: str | Path, protocol: dict, *, root: Path = ROOT) -> dict:
    """Return a draft only. The experiment owner separately approves reviewed code."""
    primary, secondary = declared_metrics(protocol)
    baseline = protocol["baseline_policy"]
    version = protocol["baseline_version"]
    candidate = protocol["candidate_policy"]
    if not isinstance(version, str) or not version:
        raise ValueError("baseline_version must be nonempty")
    if baseline == candidate or baseline not in protocol["policies"] or candidate not in protocol["policies"]:
        raise ValueError("candidate and baseline must be distinct declared policies")
    attribution = protocol.get("attribution_baseline_policy")
    if attribution is not None and (attribution == candidate or attribution not in protocol["policies"]):
        raise ValueError("attribution baseline must be a distinct declared policy")
    return {
        "schema_version": 1,
        "status": "awaiting_code_freeze_approval",
        "approval_note": "",
        "evidence_type": EVIDENCE,
        "quality_validated": False,
        "component_sha256": source_hashes(Path(protocol_path), root),
        "baseline": {"policy": baseline, "version": version},
        "candidate_policy": candidate,
        "attribution_baseline_policy": attribution,
        "policies": list(protocol["policies"]),
        "predeclared_metrics": {"primary": primary, "secondary": secondary},
        "scenario_names": list(protocol["scenarios"]),
        "bootstrap": deepcopy(protocol["bootstrap"]),
    }


def verify_freeze(manifest: dict, protocol_path: str | Path, protocol: dict, *, root: Path = ROOT) -> None:
    if manifest.get("status") != "approved_for_holdout" or not str(manifest.get("approval_note", "")).strip():
        raise ValueError("holdout locked: a reviewed freeze must explicitly be approved_for_holdout with an approval note")
    expected = build_freeze_manifest(protocol_path, protocol, root=root)
    for key, value in expected.items():
        if key in {"status", "approval_note"}:
            continue
        if manifest.get(key) != value:
            raise ValueError(f"holdout locked: freeze mismatch in {key}; do not regenerate approval automatically")


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def extract_metrics(result: dict, model: CostModel, events: list[dict]) -> dict:
    metrics = deepcopy(result["metrics"])
    transfer_ms = _number(metrics["modeled_transferred_bytes"], "transfer bytes") / model.bandwidth_bytes_per_ms
    service = sum(_number(metrics[key], key) for key in (
        "modeled_encoding_ms", "modeled_decoding_ms", "modeled_recompute_ms")) + transfer_ms
    for alias in ("modeled_total_service_ms", TOTAL_SERVICE_METRIC):
        if alias in metrics and not math.isclose(_number(metrics[alias], alias), service, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("replay total service work disagrees with encode+decode+recompute+transfer/BW")
    metrics[TOTAL_SERVICE_METRIC] = service
    metrics["modeled_transfer_service_ms"] = transfer_ms
    if "modeled_mean_resume_ms" not in metrics:
        delays = [item["modeled_resume_delay_ms"] for item in result.get("decisions", []) if item["type"] == "resume"]
        if len(delays) != metrics["resume_count"]:
            raise ValueError("mean resume delay needs all resume observations")
        metrics["modeled_mean_resume_ms"] = statistics.fmean(delays) if delays else None
    duration = events[-1]["at_ms"] - events[0]["at_ms"] if events else 0.0
    metrics["modeled_trace_duration_ms"] = duration
    metrics["time_weighted_host_reserved_bytes"] = metrics["host_reserved_byte_ms"] / duration if duration > 0 else None
    # Service work is additive accounting, not elapsed wall time; overlap is not a speedup.
    metrics["measured_gpu_itl_ms"] = None
    metrics["measured_llm_task_success"] = None
    return metrics


def paired_bootstrap(pairs: list[tuple[int, float | None, float | None]], *, resamples: int,
                     confidence: float, seed: int) -> dict:
    """Each (seed, baseline, candidate) row is one independent whole-run pair.

    The ratio is 1 - mean(candidate run statistics)/mean(baseline run statistics).
    For p95 inputs this is a ratio of mean run-p95s, not pooled request p95.
    """
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 1:
        raise ValueError("bootstrap resamples must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if len({p[0] for p in pairs}) != len(pairs):
        raise ValueError("bootstrap seeds must identify independent unique whole runs")
    missing = [run_seed for run_seed, b, c in pairs if b is None or c is None]
    complete = [(s, _number(b, "baseline metric"), _number(c, "candidate metric"))
                for s, b, c in pairs if b is not None and c is not None]
    counts = {"wins": 0, "ties": 0, "losses": 0}
    for _, b, c in complete:
        if math.isclose(b, c, rel_tol=1e-9, abs_tol=1e-9):
            counts["ties"] += 1
        else:
            counts["wins" if c < b else "losses"] += 1
    summary = {
        "run_pairs": len(pairs), "complete_pairs": len(complete), "unavailable_seeds": missing,
        "resampling_unit": "independent whole-run seed within scenario",
        "estimand": "difference and relative reduction of paired means of per-run statistics; lower is better",
        "confidence": confidence, "bootstrap_resamples": resamples, "bootstrap_seed": seed,
        **counts, "baseline_mean": None, "candidate_mean": None,
        "improvement_delta": None, "improvement_fraction": None,
        "improvement_delta_ci": None, "improvement_fraction_ci": None,
        "status": "unavailable" if not complete else "point_estimate_only",
    }
    if not complete:
        return summary
    bmean = statistics.fmean(b for _, b, _ in complete)
    cmean = statistics.fmean(c for _, _, c in complete)
    summary.update(baseline_mean=bmean, candidate_mean=cmean, improvement_delta=bmean - cmean,
                   improvement_fraction=1 - cmean / bmean if bmean > 0 else None)
    # Missing pairs invalidate a confirmatory interval; keep them explicit, not silently dropped.
    if len(complete) < 2 or missing:
        summary["status"] = "incomplete_pairs" if missing else "insufficient_independent_runs"
        return summary
    rng = random.Random(seed)
    deltas, fractions = [], []
    for _ in range(resamples):
        sample = [complete[rng.randrange(len(complete))] for _ in complete]
        b = statistics.fmean(item[1] for item in sample)
        c = statistics.fmean(item[2] for item in sample)
        deltas.append(b - c)
        fractions.append(1 - c / b if b > 0 else None)
    alpha = (1 - confidence) / 2
    summary["improvement_delta_ci"] = [percentile(deltas, alpha), percentile(deltas, 1 - alpha)]
    if all(x is not None for x in fractions):
        summary["improvement_fraction_ci"] = [percentile(fractions, alpha), percentile(fractions, 1 - alpha)]
    summary["status"] = "estimated" if summary["improvement_fraction_ci"] is not None else "ratio_undefined_for_zero_baseline"
    return summary


def summarize_runs(rows: list[dict], protocol: dict, *, baseline_policy: str | None = None,
                   compared_policies: list[str] | None = None) -> list[dict]:
    baseline = baseline_policy or protocol["baseline_policy"]
    policies = protocol["policies"] if compared_policies is None else compared_policies
    if baseline not in protocol["policies"] or any(p not in protocol["policies"] for p in policies):
        raise ValueError("comparison policies must have been declared")
    metrics = sum(declared_metrics(protocol), [])
    bootstrap = protocol["bootstrap"]
    summaries = []
    for scenario in protocol["scenarios"]:
        cell = [r for r in rows if r["scenario"] == scenario]
        indexed = {}
        for row in cell:
            identity = (row["seed"], row["policy"])
            if identity in indexed:
                raise ValueError("duplicate run/policy row")
            indexed[identity] = row
        seeds = sorted({r["seed"] for r in cell})
        for policy in policies:
            comparisons = {}
            for metric in metrics:
                pairs = []
                for run_seed in seeds:
                    if (run_seed, baseline) not in indexed or (run_seed, policy) not in indexed:
                        raise ValueError("paired comparisons require every policy on every run")
                    b, c = indexed[run_seed, baseline], indexed[run_seed, policy]
                    if b["trace_sha256"] != c["trace_sha256"]:
                        raise ValueError("paired policies used different input traces")
                    if metric not in b["metrics"] or metric not in c["metrics"]:
                        raise ValueError(f"predeclared metric missing: {metric}")
                    pairs.append((run_seed, b["metrics"][metric], c["metrics"][metric]))
                seed = int(sha256(f"{bootstrap['seed']}:{scenario}:{metric}".encode()).hexdigest()[:16], 16)
                comparisons[metric] = paired_bootstrap(pairs, resamples=bootstrap["resamples"],
                                                      confidence=bootstrap["confidence"], seed=seed)
            summaries.append({"scenario": scenario, "policy": policy, "baseline_policy": baseline,
                              "candidate_of_interest": policy == protocol["candidate_policy"],
                              "metrics": comparisons})
    return summaries


def run_experiment(protocol_path: str | Path, *, split: str, freeze_manifest: dict | None = None,
                   root: Path = ROOT, protocol: dict | None = None,
                   workload_factory: Callable | None = None, config_factory: Callable | None = None,
                   replay_fn: Callable = replay) -> tuple[dict, list[dict], dict]:
    if split not in {"development", "holdout"}:
        raise ValueError("split must be development or holdout")
    protocol_path = Path(protocol_path)
    on_disk = json.loads(protocol_path.read_text())
    if protocol is None:
        protocol = on_disk
    elif protocol != on_disk:
        raise ValueError("protocol object differs from the hashed protocol file")
    if split == "holdout":
        if freeze_manifest is None:
            raise ValueError("holdout locked: approved freeze manifest is required before trace generation")
        verify_freeze(freeze_manifest, protocol_path, protocol, root=root)
    declaration = build_freeze_manifest(protocol_path, protocol, root=root)
    if workload_factory is None or config_factory is None:
        from kvpilot.workloads import generate_workload, scenario_config
        workload_factory = workload_factory or generate_workload
        config_factory = config_factory or scenario_config
    seeds = protocol["development_seeds" if split == "development" else "evaluation_seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("split must contain unique independent seeds")
    if set(protocol["development_seeds"]) & set(protocol["evaluation_seeds"]):
        raise ValueError("development and holdout seeds overlap")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("run seeds must be integers")
    if len(set(protocol["policies"])) != len(protocol["policies"]):
        raise ValueError("policies must be unique")
    rows, traces, assumptions = [], [], {}
    for scenario in protocol["scenarios"]:
        config = config_factory(protocol, scenario)
        if config.get("evidence_type") != "analytical_assumptions":
            raise ValueError("scenario costs must explicitly be analytical assumptions")
        model = CostModel(**config["cost_model"])
        profiles = [CodecProfile(**p) for p in config["codecs"]]
        assumptions[scenario] = config
        for run_seed in seeds:
            events = workload_factory(scenario, run_seed, protocol=protocol,
                                      sessions=protocol["sessions_per_run"], allow_evaluation=split == "holdout")
            trace_hash = digest(events)
            traces.append({"scenario": scenario, "seed": run_seed, "trace_sha256": trace_hash,
                           "event_count": len(events)})
            for policy in protocol["policies"]:
                result = replay_fn(deepcopy(events), model, profiles, policy=policy,
                                   default_pause_ms=config.get("default_pause_ms", 1000))
                rows.append({"scenario": scenario, "seed": run_seed, "policy": policy,
                             "trace_sha256": trace_hash, "evidence_type": EVIDENCE,
                             "quality_validated": False, "metrics": extract_metrics(result, model, events)})
    # Detect source edits during a long run rather than publishing results as if frozen.
    if declaration["component_sha256"] != source_hashes(protocol_path, root):
        raise ValueError("source/config changed during evaluation; discard this run and review freeze")
    manifest = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split, "evidence_type": EVIDENCE, "quality_validated": False,
        "freeze_manifest": deepcopy(freeze_manifest) if split == "holdout" else None,
        "source_declaration": declaration, "protocol": protocol, "scenario_assumptions": assumptions,
        "input_traces": traces, "python": sys.version,
    }
    report = {
        "schema_version": 1, "split": split, "evidence_type": EVIDENCE, "quality_validated": False,
        "baseline": declaration["baseline"], "candidate_policy": protocol["candidate_policy"],
        "predeclared_metrics": declaration["predeclared_metrics"],
        "comparisons": summarize_runs(rows, protocol),
        "attribution_comparison": {
            "role": "secondary exploratory ablation; does not change the primary baseline or metric",
            "baseline_policy": protocol["attribution_baseline_policy"],
            "candidate_policy": protocol["candidate_policy"],
            "comparisons": summarize_runs(rows, protocol,
                                           baseline_policy=protocol["attribution_baseline_policy"],
                                           compared_policies=[protocol["candidate_policy"]]),
        } if protocol.get("attribution_baseline_policy") is not None else None,
        "overall_winner": None,
        "interpretation": "Each scenario is reported separately; no best-case or blended universal winner is selected.",
        "limitations": [
            "Synthetic traces and supplied costs; no GPU, live LLM, PCIe or serving execution.",
            "p95 comparisons use means of per-run p95s, not pooled requests or a measured serving tail.",
            "Confidence intervals describe independent synthetic-run variation conditional on assumptions, not hardware/model uncertainty.",
            "Total service work sums codec, recompute and transfer time; it is not wall time and excludes resource contention.",
            "No active-decode queues, staging-memory fidelity or task-quality validation; lossy codec outcomes remain unvalidated.",
            "Secondary metrics are diagnostic; multiple comparisons are exploratory, without family-wise correction.",
        ],
    }
    return manifest, rows, report


def archive_results(output_dir: str | Path, manifest: dict, rows: list[dict], report: dict) -> None:
    target = Path(output_dir)
    # Immutable per-run directory; do not overwrite an earlier evaluation silently.
    target.mkdir(parents=True, exist_ok=False)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (target / "runs.jsonl").write_bytes(b"".join(canonical_bytes(row) for row in rows))
    (target / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("development", "holdout"))
    parser.add_argument("--protocol", type=Path, default=ROOT / "examples" / "holdout-protocol.json")
    parser.add_argument("--freeze-manifest", type=Path, help="reviewed and approved JSON; required for holdout")
    parser.add_argument("--output-dir", type=Path, required=True, help="new archive directory; existing directories are refused")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new run archive")
    try:
        freeze = json.loads(args.freeze_manifest.read_text()) if args.freeze_manifest else None
        manifest, rows, report = run_experiment(args.protocol, split=args.split, freeze_manifest=freeze)
        archive_results(args.output_dir, manifest, rows, report)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.error(str(exc))
    print(f"MODELED ONLY: {len(rows)} policy/run rows archived to {args.output_dir}")
    print("No GPU/serving/quality result; scenario-specific whole-run intervals, no universal winner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
