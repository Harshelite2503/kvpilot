#!/usr/bin/env python3
"""Read-only runtime comparison guard and separate evidence packaging helper."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys

EXPECTED_PYTHON = "3.11.18"
EXPECTED = {"torch": "2.7.1+cu126", "numpy": "2.2.6", "tokenizers": "0.21.2",
            "safetensors": "0.5.3", "psutil": "7.0.0", "pytest": "8.4.2"}
ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG",
    "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "TOKENIZERS_PARALLELISM",
)


def attempt(function):
    try:
        return function()
    except Exception as error:
        return {"unavailable": type(error).__name__}


def read(path):
    path = Path(path)
    return path.read_text().strip() if path.is_file() else None


def write(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def cgroups():
    paths = ["/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpuset.cpus.effective",
             "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
             "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us",
             "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us"]
    membership = attempt(lambda: read("/proc/self/cgroup"))
    if isinstance(membership, str):
        for line in membership.splitlines():
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            hierarchy, controllers, relative = parts
            relative = PurePosixPath(relative)
            if ".." in relative.parts:
                continue
            relative = str(relative).lstrip("/")
            if hierarchy == "0" and not controllers:
                paths += [str(Path("/sys/fs/cgroup") / relative / name)
                          for name in ("cpu.max", "cpuset.cpus.effective")]
            if "cpu" in controllers.split(","):
                for directory in ("cpu", "cpu,cpuacct"):
                    paths += [str(Path("/sys/fs/cgroup") / directory / relative / name)
                              for name in ("cpu.cfs_quota_us", "cpu.cfs_period_us")]
    return {"process_membership": membership,
            "read_only_controls": {path: attempt(lambda path=path: read(path)) for path in dict.fromkeys(paths)},
            "limits": "Absent/inaccessible controls are recorded; nested mount namespaces can obscure the host quota."}


def package_validation(python_version, packages):
    errors = []
    if python_version != EXPECTED_PYTHON:
        errors.append(f"Python: expected {EXPECTED_PYTHON}; found {python_version}")
    for name, expected in EXPECTED.items():
        if packages.get(name) != expected:
            errors.append(f"{name}: expected {expected}; found {packages.get(name)}")
    return errors


def requirement_pins(repo):
    paths = ("experiments/torch/requirements.txt", "experiments/torch/requirements-p100.txt")
    pins = {}
    for name in paths:
        for line in (repo / name).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                package, version = line.split("==", 1)
                pins[package] = version
    return pins


def capture(output, repo):
    packages = {name: attempt(lambda name=name: importlib.metadata.version(name)) for name in EXPECTED}
    python_version = platform.python_version()
    errors = package_validation(python_version, packages)
    pins = attempt(lambda: requirement_pins(repo))
    intended = {name: version for name, version in EXPECTED.items() if name != "pytest"}
    if pins != intended:
        errors.append("Checked-out runner requirement pins differ from the intended comparison pins")
    report = {
        "evidence_type": "p100_comparison_runtime_sidecar",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "settings_changed_by_capture": False,
        "python": python_version, "python_executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": attempt(lambda: sorted(os.sched_getaffinity(0))),
        "environment_allowlist": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
        "packages": packages, "checked_out_requirement_pins": pins, "cgroups": cgroups(),
        "comparability": "P100 CUDA 12.6 versus RTX CUDA 12.8 is a cross-system comparison. CPU threads, affinity, quota and TF32 flags are observed, not changed or forced equal.",
    }
    # Fail version mismatches before importing the tensor engine or loading any model.
    if not errors:
        try:
            import torch
            report["torch"] = {
                "version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                "num_threads": torch.get_num_threads(),
                "num_interop_threads": torch.get_num_interop_threads(),
                "default_dtype": str(torch.get_default_dtype()),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "matmul_allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cuda_available": torch.cuda.is_available(),
                "compiled_architectures": attempt(torch.cuda.get_arch_list),
            }
            if str(torch.__version__) != EXPECTED["torch"] or torch.version.cuda != "12.6":
                errors.append("Imported torch version/CUDA runtime differs from the pinned P100 build")
            if not torch.cuda.is_available():
                errors.append("CUDA device unavailable; benchmark not started")
            else:
                report["torch"]["devices"] = [
                    {"index": index, "name": torch.cuda.get_device_name(index),
                     "compute_capability": list(torch.cuda.get_device_capability(index)),
                     "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory)}
                    for index in range(torch.cuda.device_count())]
        except Exception as error:
            errors.append("Runtime capture failed: " + type(error).__name__)
    write(output / "runtime.json", report)
    validation = {"ok": not errors, "expected_python": EXPECTED_PYTHON,
                  "expected_packages": EXPECTED, "errors": errors,
                  "scope": "Exact Python and direct runner pins plus pytest match the RTX setup, with the required P100 CUDA wheel exception. Transitive packages and host hardware are not guaranteed identical."}
    write(output / "runtime-validation.json", validation)
    print(json.dumps(validation, indent=2))
    return 0 if not errors else 1


def finalize(output, code, phase, suite):
    write(output / "companion-status.json", {
        "finished_utc": datetime.now(timezone.utc).isoformat(), "exit_code": code,
        "last_phase": phase, "suite_completed": code == 0 and phase == "complete",
        "suite_directory": str(suite), "suite_directory_exists": suite.is_dir(),
        "suite_archive_exists": Path(str(suite) + ".tar.gz").is_file(),
        "runtime_validation_passed": (json.loads((output / "runtime-validation.json").read_text()).get("ok")
                                      if (output / "runtime-validation.json").is_file() else False),
        "scope": "Companion evidence is separate from the unchanged benchmark suite; a failure preserves partial evidence."})
    hashes = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[path.relative_to(output).as_posix()] = digest.hexdigest()
    write(output / "companion-artifact-sha256.json", hashes)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--phase")
    parser.add_argument("--suite-directory", type=Path)
    args = parser.parse_args()
    if not args.output.is_dir():
        parser.error("--output must be an existing separate companion directory")
    if args.finalize:
        if args.exit_code is None or args.phase is None or args.suite_directory is None:
            parser.error("--finalize requires --exit-code, --phase and --suite-directory")
        return finalize(args.output, args.exit_code, args.phase, args.suite_directory)
    if args.repo is None:
        parser.error("--repo is required for runtime capture")
    return capture(args.output, args.repo)


if __name__ == "__main__":
    raise SystemExit(main())
