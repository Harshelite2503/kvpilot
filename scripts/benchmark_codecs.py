#!/usr/bin/env python3
"""Measure CPU reference codecs; synthetic data is not an LLM KV benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

# Permit a direct checkout invocation without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kvpilot.codecs import CODECS, MAX_ELEMENTS, CodecError, decode, encode, fp16_reference


def shape_argument(text: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(piece) for piece in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must contain comma-separated integers") from exc
    count = 1
    for dimension in shape:
        if dimension < 1:
            raise argparse.ArgumentTypeError("benchmark dimensions must be positive")
        count *= dimension
    if len(shape) > 32 or count > MAX_ELEMENTS:
        raise argparse.ArgumentTypeError("shape exceeds reference codec limits")
    return shape


def fixture(shape: tuple[int, ...], kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        return rng.normal(size=shape).astype(np.float16)
    # Shared feature vectors plus small independent noise. This deliberately
    # constructed correlation is an algorithm test, not a measured KV profile.
    base_shape = (1,) * (len(shape) - 1) + (shape[-1],)
    base = rng.normal(size=base_shape)
    return (base + .02 * rng.normal(size=shape)).astype(np.float16)


def timing_summary(samples: list[float]) -> dict:
    return {"mean_ms": float(np.mean(samples) * 1000), "median_ms": float(np.median(samples) * 1000),
            "p95_ms": float(np.percentile(samples, 95) * 1000)}


def attention_proxy(reference: np.ndarray, restored: np.ndarray, seed: int) -> dict | None:
    """Isolate key error in a toy single-head attention operation.

    Up to the first 256 rows form keys; queries/values are seeded synthetic
    vectors. Values are identical in both runs. This is not model accuracy,
    and does not measure jointly quantized K/V or a complete attention layer.
    """
    if reference.ndim < 2 or reference.shape[-1] == 0 or reference.size == 0:
        return None
    dimension = reference.shape[-1]
    if dimension > 4096:
        return {"label": "skipped: last-axis dimension exceeds the toy proxy limit of 4096"}
    keys = reference.reshape(-1, dimension)[:256].astype(np.float64)
    approximated = restored.reshape(-1, dimension)[:256].astype(np.float64)
    rng = np.random.default_rng(seed)
    queries = rng.normal(size=(8, dimension))
    values = rng.normal(size=(keys.shape[0], 16))

    def weights(matrix):
        logits = queries @ matrix.T / np.sqrt(dimension)
        logits -= logits.max(axis=-1, keepdims=True)
        exponentials = np.exp(logits)
        return exponentials / exponentials.sum(axis=-1, keepdims=True)

    truth, candidate = weights(keys), weights(approximated)
    delta = candidate @ values - truth @ values
    return {
        "label": "synthetic single-head key-only attention proxy; not LLM quality",
        "key_rows": int(keys.shape[0]), "synthetic_queries": 8,
        "mean_attention_total_variation": float(.5 * np.abs(truth - candidate).sum(axis=-1).mean()),
        "output_mse": float(np.mean(delta ** 2)),
    }


def benchmark(reference: np.ndarray, codec: str, group_size: int, repeats: int, warmup: int, seed: int) -> dict:
    for _ in range(warmup):
        decode(encode(reference, codec, group_size))
    encode_times, decode_times = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        payload = encode(reference, codec, group_size)
        middle = time.perf_counter()
        restored = decode(payload)
        end = time.perf_counter()
        encode_times.append(middle - start)
        decode_times.append(end - middle)
    error = restored.astype(np.float64) - reference.astype(np.float64)
    return {
        "codec": codec,
        "serialized_bytes": len(payload),
        "payload_includes": "format header, shape metadata, scales where applicable, packed data",
        "fp16_reference_bytes": int(reference.nbytes),
        "fp16_bytes_divided_by_serialized_bytes": float(reference.nbytes / len(payload)),
        "size_reduction_fraction_vs_unframed_fp16": float(1 - len(payload) / reference.nbytes) if reference.nbytes else None,
        "encode": timing_summary(encode_times), "decode": timing_summary(decode_times),
        "mse_vs_fp16_reference": float(np.mean(error ** 2)) if error.size else 0.,
        "max_absolute_error_vs_fp16_reference": float(np.max(np.abs(error))) if error.size else 0.,
        "bit_exact_fp16_roundtrip": reference.tobytes() == restored.tobytes(),
        "attention_proxy": attention_proxy(reference, restored, seed + 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="single numeric .npy array, loaded with allow_pickle=False")
    parser.add_argument("--fixture", choices=("gaussian", "correlated"), default="correlated")
    parser.add_argument("--shape", type=shape_argument, default=(2, 8, 256, 64), help="synthetic shape, default 2,8,256,64")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--codecs", nargs="+", choices=CODECS, default=list(CODECS))
    parser.add_argument("--output", type=Path, help="optional JSON path; results are also printed")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10_000 or not 0 <= args.warmup <= 100:
        parser.error("repeats must be 1..10000 and warmup 0..100")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    try:
        if args.input:
            if args.input.suffix.lower() != ".npy":
                parser.error("--input must be a .npy file")
            # Memory mapping avoids allocating the full input before validation.
            source = np.load(args.input, allow_pickle=False, mmap_mode="r")
        else:
            source = fixture(args.shape, args.fixture, args.seed)
        reference = fp16_reference(source)
        rows = [benchmark(reference, codec, args.group_size, args.repeats, args.warmup, args.seed) for codec in args.codecs]
    except (CodecError, ValueError, OSError) as exc:
        parser.error(str(exc))
    report = {
        "schema_version": 1,
        "evidence_type": "cpu_codec_microbenchmark",
        "benchmark": "kvpilot_numpy_cpu_reference_codecs",
        "device": "CPU", "timings": "measured wall-clock; includes Python serialization and validation",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "fixture_type": "external_npy_unverified_origin" if args.input else f"synthetic_{args.fixture}",
        "input_path": str(args.input.resolve()) if args.input else None,
        "seed": args.seed, "shape": list(reference.shape), "source_dtype": str(source.dtype),
        "reference_dtype": "float16", "group_size": args.group_size,
        "repeats": args.repeats, "warmup": args.warmup,
        "limitations": [
            "Original illustrative codecs; not CacheGen, TurboQuant, or a production KV-cache implementation.",
            "All error metrics compare against FP16-converted input, not original higher-precision data.",
            "No real-model accuracy, perplexity, GPU timing, transfer overlap, or end-to-end serving claims.",
            "Attention proxy uses synthetic queries/values and only the first 256 flattened key rows.",
            "CPU timing depends on this host and includes format overhead; compression ratios include all serialized bytes.",
        ],
        "results": rows,
    }
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
