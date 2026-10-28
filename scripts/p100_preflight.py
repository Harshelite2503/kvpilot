#!/usr/bin/env python3
"""Check a local tensor runtime; no downloads, model load or remote execution."""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


class PreflightError(RuntimeError):
    """A named, actionable environment check failed."""


def arithmetic_smoke(torch: Any, device: str) -> dict[str, Any]:
    """Exercise FP16 matmul and explicit math SDPA against FP32 CPU references."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import torch.nn.functional as functional

    source = torch.arange(256, dtype=torch.float32).reshape(16, 16).remainder(17) / 17
    matrix = source.to(device=device, dtype=torch.float16)
    product = matrix @ matrix.T
    expected = source @ source.T
    if not torch.isfinite(product).all().item() or not torch.allclose(
        product.float().cpu(), expected, atol=0.02, rtol=0.01
    ):
        raise PreflightError("fp16_matmul_reference_mismatch")
    q_cpu = source.reshape(1, 2, 8, 16)
    k_cpu = q_cpu.flip(-1).contiguous()
    v_cpu = q_cpu / 2
    q, k, v = [item.to(device=device, dtype=torch.float16) for item in (q_cpu, k_cpu, v_cpu)]
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        actual = functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        reference = functional.scaled_dot_product_attention(
            q_cpu, k_cpu, v_cpu, dropout_p=0.0, is_causal=True
        )
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    if not torch.isfinite(actual).all().item() or not torch.allclose(
        actual.float().cpu(), reference, atol=0.003, rtol=0.01
    ):
        raise PreflightError("fp16_math_sdpa_reference_mismatch")
    return {"fp16_matmul": "passed", "fp16_causal_math_sdpa": "passed",
            "attention_backend": "MATH", "model_loaded": False}


def run_preflight(device: str = "cuda", *, allow_other_cuda: bool = False,
                  torch_module: Any = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "evidence_type": "environment_preflight", "device": device,
        "python": platform.python_version(), "platform": platform.platform(),
        "ok": False, "p100_ready": False, "stage": "import_torch",
    }
    try:
        torch = torch_module if torch_module is not None else importlib.import_module("torch")
        report.update(torch_version=str(torch.__version__), cuda_runtime=torch.version.cuda)
        report["stage"] = "device_support"
        if device == "cuda":
            if not torch.cuda.is_available():
                raise PreflightError("cuda_unavailable: check driver, GPU assignment and CUDA wheel")
            name = torch.cuda.get_device_name(0)
            capability = tuple(torch.cuda.get_device_capability(0))
            arches = list(torch.cuda.get_arch_list())
            report.update(device_name=name, compute_capability=list(capability), compiled_arches=arches,
                          total_device_bytes=int(torch.cuda.get_device_properties(0).total_memory))
            is_p100 = capability == (6, 0) and "P100" in name
            if not is_p100 and not allow_other_cuda:
                raise PreflightError("not_a_p100: use --allow-other-cuda for an explicitly labeled diagnostic")
            if capability == (6, 0) and "sm_60" not in arches:
                raise PreflightError("missing_sm_60: install torch==2.7.1+cu126 from the official cu126 index")
            if capability < (6, 0):
                raise PreflightError("unsupported_compute_capability: this runner targets Pascal or newer")
        elif device == "mps":
            if not torch.backends.mps.is_available():
                raise PreflightError("mps_unavailable: use --device cpu for an offline diagnostic")
            is_p100 = False
        elif device == "cpu":
            is_p100 = False
        else:
            raise PreflightError("invalid_device")
        report["stage"] = "arithmetic"
        report.update(arithmetic_smoke(torch, device))
        report.update(ok=True, p100_ready=is_p100, stage="complete")
    except Exception as error:
        # Do not dump environment variables or third-party exception messages.
        report["error"] = str(error) if isinstance(error, PreflightError) else type(error).__name__
        report["next_step"] = "See docs/p100-runner.md; a successful preflight is not a model benchmark."
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--allow-other-cuda", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional new JSON file (never overwritten).")
    args = parser.parse_args(argv)
    report = run_preflight(args.device, allow_other_cuda=args.allow_other_cuda)
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as target:
            target.write(content)
    print(content, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
