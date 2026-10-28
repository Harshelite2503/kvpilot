"""Reproduce the evidence behind docs/test-plan.md's attribution audit.

capture        Axis semantics, repetition and compressibility of a captured
               vLLM KV pool slice saved as a float32 copy of the native cache.
codec-profile  Where the reference codecs spend time on a given array.
frontier       Owned-runner candidate arms versus (1) the best baseline run
               under the same configured budget, which is measured, and (2) the
               frontier of baselines at the candidate's stored bytes, which is
               interpolated when no baseline sits at that byte level. Paired
               bootstrap by unit; too few units yields `insufficient_units`.

Every subcommand prints JSON so the output can be attached to a change.
"""

from __future__ import annotations

import argparse
import cProfile
from collections import defaultdict
import io
import json
from pathlib import Path
import pstats
import random
import statistics
import sys
import zlib

import numpy as np

MIB = 1024 * 1024


def _zlib_ratio(values: np.ndarray) -> float:
    data = np.ascontiguousarray(values).astype("<f2").tobytes()
    return len(data) / len(zlib.compress(data, 6))


def _zlib_ratio_bf16(values: np.ndarray) -> float:
    """zlib on the engine's native bf16 bytes, recovered from a float32 copy."""
    data = (np.ascontiguousarray(values, dtype=np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()
    return len(data) / len(zlib.compress(data, 6))


def _repeat_profile(rows: np.ndarray, max_lag: int) -> tuple[int, float]:
    best = (0, 0.0)
    for lag in range(1, min(max_lag, len(rows) - 1) + 1):
        rate = float(np.mean(np.all(rows[:-lag] == rows[lag:], axis=1)))
        if rate > best[1]:
            best = (lag, rate)
    return best


def analyze_capture(array: np.ndarray, head_dim: int | None = None, max_lag: int = 64) -> dict:
    """Assumes [blocks, kv_heads, block_size, K||V]; the report checks that assumption."""
    if array.ndim != 4:
        raise ValueError("expected a 4-D capture [blocks, kv_heads, block_size, channels]")
    blocks, heads, block_size, channels = array.shape
    head_dim = head_dim or channels // 2
    if 2 * head_dim != channels:
        raise ValueError("channels must equal 2 * head_dim for a K||V layout")
    tokens = array.transpose(0, 2, 1, 3).reshape(blocks * block_size, heads, channels)
    filled = ~np.all(tokens == 0, axis=(1, 2))
    if not filled.any():
        raise ValueError("capture has no written token rows")
    valid = int(np.max(np.nonzero(filled)[0]) + 1)
    tokens = tokens[:valid]
    keys, values = tokens[..., :head_dim], tokens[..., head_dim:]
    flat_k, flat_v = keys.reshape(valid, -1), values.reshape(valid, -1)
    unique_v = np.unique(flat_v, axis=0)
    k_lag, k_rate = _repeat_profile(flat_k, max_lag)
    v_lag, v_rate = _repeat_profile(flat_v, max_lag)
    report = {
        "shape": list(array.shape),
        "token_slots": blocks * block_size,
        "valid_token_rows": valid,
        "trailing_unwritten_slots": blocks * block_size - valid,
        "key_half": {"unique_rows": int(np.unique(flat_k, axis=0).shape[0]), "best_repeat_lag": k_lag,
                     "best_repeat_rate": k_rate, "zlib_ratio": _zlib_ratio(keys)},
        "value_half": {"unique_rows": int(unique_v.shape[0]), "best_repeat_lag": v_lag,
                       "best_repeat_rate": v_rate, "zlib_ratio": _zlib_ratio(values),
                       "zlib_ratio_deduplicated": _zlib_ratio(unique_v)},
        "whole_capture_zlib_ratio": _zlib_ratio(array),
    }
    if v_lag:
        # Per-slice test of the layout. If axis 1 were K/V instead of heads, one
        # axis-1 slice of the upper channels would be rotated K and never repeat.
        report["axis1_slices"] = [
            {"axis1": i, "channels": name, "unique_rows": int(np.unique(x, axis=0).shape[0]),
             "repeat_rate_at_value_lag": float(np.mean(np.all(x[:-v_lag] == x[v_lag:], axis=1)))}
            for i in range(heads)
            for name, x in (("low", tokens[:, i, :head_dim]), ("high", tokens[:, i, head_dim:]))]
        hits = np.nonzero(np.all(flat_v[:-v_lag] == flat_v[v_lag:], axis=1))[0]
        report["value_periodic_run"] = {"first_row": int(hits[0]), "last_row": int(hits[-1]) + v_lag,
                                        "rows_outside_run": valid - (int(hits[-1]) + v_lag - int(hits[0]) + 1)}
    if v_lag and v_rate > 0.5 and head_dim % 2 == 0:
        # Rotate-half RoPE preserves the norm of each (i, i + head_dim/2) pair, so
        # repeated tokens keep pair norms even though rotated key values change.
        half = head_dim // 2
        norms = np.sqrt(keys[..., :half] ** 2 + keys[..., half:] ** 2)
        def rel(x):
            return float(np.median(np.abs(x[v_lag:] - x[:-v_lag]) / (np.abs(x[:-v_lag]) + 1e-6)))
        report["rope_check_at_value_lag"] = {"pair_norm_median_rel_diff": rel(norms),
                                             "raw_key_median_rel_diff": rel(keys)}
    if array.dtype == np.float32:
        bits = tokens.reshape(-1).view(np.uint32)
        bf16_origin = bool(np.all((bits & 0xFFFF) == 0))
        report["bf16_origin"] = bf16_origin
        if bf16_origin:
            src = tokens.reshape(-1)
            converted = src.astype(np.float16).astype(np.float32)
            report["bf16_to_fp16"] = {
                "values": int(src.size), "changed": int(np.sum(converted != src)),
                "max_abs_error": float(np.max(np.abs(converted - src))),
                "fp16_low3_mantissa_zero_fraction": float(np.mean((src.astype(np.float16).view(np.uint16) & 7) == 0)),
            }
            # Compression must be judged on the bytes the engine actually stores.
            report["native_bf16_zlib_ratio"] = {
                "key_half": _zlib_ratio_bf16(keys), "value_half": _zlib_ratio_bf16(values),
                "value_half_deduplicated": _zlib_ratio_bf16(unique_v), "whole_capture": _zlib_ratio_bf16(array)}
    return report


def profile_codec(array: np.ndarray, codec: str, repeats: int, top: int) -> dict:
    from kvpilot.codecs import decode, encode
    for _ in range(3):
        decode(encode(array, codec))
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(repeats):
        decode(encode(array, codec))
    profiler.disable()
    stats = pstats.Stats(profiler, stream=io.StringIO())
    total = stats.total_tt
    rows = sorted(stats.stats.items(), key=lambda item: item[1][2], reverse=True)[:top]
    return {"codec": codec, "repeats": repeats, "total_self_seconds": total,
            "top_self_time": [{"function": f"{Path(k[0]).name}:{k[1]}({k[2]})", "calls": v[1],
                               "self_seconds": v[2], "share": v[2] / total} for k, v in rows]}


def frontier_time(baselines: list[tuple[float, float]], stored_bytes: float) -> float:
    """Lower envelope of (bytes, time) baselines at ``stored_bytes``.

    A point is reached by one baseline using no more bytes, or by splitting
    sessions between two baselines that bracket the byte level. In two
    dimensions the lower convex envelope is always reached by one or two points.
    The split is a counterfactual nobody ran. Under a capacity limit its real
    time is not linear in bytes, so an interpolated point is an estimate.
    """
    best = min((t for b, t in baselines if b <= stored_bytes), default=float("inf"))
    for b1, t1 in baselines:
        for b2, t2 in baselines:
            if b1 < stored_bytes < b2:
                best = min(best, t1 + (t2 - t1) * (stored_bytes - b1) / (b2 - b1))
    if best == float("inf"):
        raise ValueError(f"no baseline stores {stored_bytes} bytes or fewer; add a zero-byte arm such as recompute")
    return best


def _service_ms(row: dict) -> float:
    return row["pause_ms"] + row["resume_wall_ms"]


def _status(lo: float, hi: float) -> str:
    return "credited" if hi < 0 else "regression" if lo > 0 else "inconclusive"


def frontier_report(rows: list[dict], baselines: list[str], candidates: list[str],
                    resamples: int, seed: int, unit_field: str = "case_id",
                    min_units: int = 30, primary: str = "same_budget") -> dict:
    """Paired bootstrap over ``unit_field``; stored bytes are held at observed peaks.

    Every arm in a scenario ran under the same configured budget, so the
    measured comparison is against the fastest baseline in that scenario. The
    frontier comparison asks whether the candidate also improves the
    bytes/time trade-off. ``status`` follows the pre-registered ``primary``
    comparison, and is ``insufficient_units`` below ``min_units``.

    Resample the unit within which sessions interact (a shared-store block, a
    session family). Resampling cases that shared a store understates
    uncertainty, and repeats that replay identical decisions are timing
    replicas, not new units.
    """
    if primary not in {"same_budget", "frontier"}:
        raise ValueError("primary must be 'same_budget' or 'frontier'")
    by = defaultdict(list)
    for row in rows:
        by[(row["scenario"], row["policy"])].append(row)
    rng = random.Random(seed)
    out = {}
    for scenario in sorted({row["scenario"] for row in rows}):
        arms = {p: by[(scenario, p)] for p in baselines + candidates if by[(scenario, p)]}
        missing = [p for p in baselines if p not in arms]
        if missing:
            raise ValueError(f"{scenario}: baseline arms missing: {missing}")
        peak = {p: max(r["stored_bytes_after_pause"] for r in rs) for p, rs in arms.items()}
        units = sorted({r[unit_field] for rs in arms.values() for r in rs})
        per_unit = {p: defaultdict(list) for p in arms}
        for p, rs in arms.items():
            for r in rs:
                per_unit[p][r[unit_field]].append(_service_ms(r))
        unpaired = [p for p in arms if set(per_unit[p]) != set(units)]
        if unpaired:
            raise ValueError(f"{scenario}: arms do not share every {unit_field}: {unpaired}")

        def means(sample):
            return {p: statistics.mean(t for u in sample for t in per_unit[p][u]) for p in arms}

        point = means(units)
        result = {"unit_field": unit_field, "units": len(units),
                  "arms": {p: {"mean_service_ms": point[p], "peak_stored_mib": peak[p] / MIB} for p in arms}}
        for cand in candidates:
            if cand not in arms:
                continue
            pts = [(peak[b], point[b]) for b in baselines]
            front = frontier_time(pts, peak[cand])
            single = min(t for b, t in pts if b <= peak[cand]) if any(b <= peak[cand] for b, _ in pts) else None
            best_base = min(baselines, key=lambda b: point[b])
            budget_draws, front_draws = [], []
            for _ in range(resamples):
                sample = [rng.choice(units) for _ in units]
                m = means(sample)
                fastest = min(m[b] for b in baselines)
                budget_draws.append((m[cand] - fastest) / fastest)
                f = frontier_time([(peak[b], m[b]) for b in baselines], peak[cand])
                front_draws.append((m[cand] - f) / f)

            def interval(draws):
                draws.sort()
                return [draws[int(0.025 * resamples)], draws[int(0.975 * resamples) - 1]]

            budget_ci, front_ci = interval(budget_draws), interval(front_draws)
            comparisons = {
                "same_budget": {"baseline": best_base, "baseline_ms": point[best_base],
                                "delta": (point[cand] - point[best_base]) / point[best_base],
                                "ci95": budget_ci, "status": _status(*budget_ci)},
                "frontier": {"frontier_ms": front,
                             "point": "measured" if single is not None and single <= front else "interpolated",
                             "delta": (point[cand] - front) / front, "ci95": front_ci,
                             "status": _status(*front_ci)},
            }
            result[cand] = {**comparisons, "primary": primary,
                            "status": "insufficient_units" if len(units) < min_units
                            else comparisons[primary]["status"]}
        out[scenario] = result
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("npy", type=Path)
    cap.add_argument("--head-dim", type=int)
    prof = sub.add_parser("codec-profile")
    prof.add_argument("npy", type=Path)
    prof.add_argument("--codec", default="group_int4")
    prof.add_argument("--repeats", type=int, default=30)
    prof.add_argument("--top", type=int, default=6)
    fr = sub.add_parser("frontier")
    fr.add_argument("rows", type=Path, help="owned-runner rows.jsonl")
    fr.add_argument("--baselines", nargs="+", default=["recompute", "raw_native", "group_int4"])
    fr.add_argument("--candidates", nargs="+", default=["adaptive_exact", "adaptive_eviction"])
    fr.add_argument("--resamples", type=int, default=2000)
    fr.add_argument("--seed", type=int, default=20261028)
    fr.add_argument("--unit-field", default="case_id",
                    help="row field to resample; use the shared-store block or session family when rows carry it")
    fr.add_argument("--min-units", type=int, default=30, help="below this, status is insufficient_units")
    fr.add_argument("--primary", choices=["same_budget", "frontier"], default="same_budget",
                    help="the pre-registered comparison that sets status")
    args = parser.parse_args(argv)
    if args.command == "capture":
        report = analyze_capture(np.load(args.npy), args.head_dim)
    elif args.command == "codec-profile":
        report = profile_codec(np.load(args.npy), args.codec, args.repeats, args.top)
    else:
        rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
        report = frontier_report(rows, args.baselines, args.candidates, args.resamples, args.seed,
                                 args.unit_field, args.min_units, args.primary)
    json.dump(report, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
