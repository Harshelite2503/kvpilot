"""Offline, standard-library verification of the recorded P100/RTX policy experiment.

Hashes establish internal consistency, not an independent signature or proof of
physical execution. No tensor engine, inference, network or historical Git clone
is needed. Summary numbers are recomputed from artifacts audited by the existing
independent result validator; they are descriptive, not performance gates.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD = ROOT / "results/policy-replication-2026-10-29"
SOURCE_COMMIT = "ca6f6feaaf7faef4b942bd37d78ee29ded99a27a"
# SHA256 of the 25-file source map at SOURCE_COMMIT, encoded as UTF-8 JSON with
# sort_keys=True and separators=(",", ":"). This anchor is code-reviewed here,
# not accepted from the record being verified, and works in shallow CI clones.
SOURCE_MAP_SHA256 = "a93bd43c0973caa3638d8813cdd3c00a4db78a46839b4d90ea76e89e489733f5"
MODEL_DIGEST = "c3b57e4a3bcb257287b2b3d5b368115727de1174bcdd4314c548104815769ff0"
# Same canonical encoding for the five loaded file {bytes, sha256} entries in
# the pinned model-download manifest; LICENSE is not loaded by inference.
MODEL_FILES_SHA256 = "a9ae50f24dc546e929b6811e5138e5265ae643f0796a60a649adec2b229437c3"
COMMON_VERSIONS = {"numpy": "2.2.6", "tokenizers": "0.21.2", "safetensors": "0.5.3", "psutil": "7.0.0"}
ARCHIVES = {"p100": "p100/run.tar.gz", "rtx5060ti": "rtx5060ti/run.tar.gz"}
RUN_NAMES = tuple(f"{kind}-{mode}" for kind in ("smoke", "full")
                  for mode in ("equal_work", "free_generation"))
RUN_FILES = frozenset({"artifact-sha256.json", "protocol.json", "fixture.json", "model-provenance.json",
                       "calibration.json", "rows.jsonl", "events.jsonl", "blocks.json", "summary.json",
                       "status.json", "report.md"})
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_MEMBERS = 512
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_relative(name):
    require(isinstance(name, str) and bool(name), "empty or non-string relative path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == name
            and all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) and part not in (".", "..") for part in path.parts),
            f"unsafe or noncanonical relative path: {name}")
    return path


def regular_file(path, limit=MAX_FILE_BYTES):
    path = Path(path)
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode) and metadata.st_size <= limit,
            f"input is not a bounded regular file: {path.name}")
    return path


def file_hash(path):
    regular_file(path)
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(regular_file(path).read_text(), object_pairs_hook=unique_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {value}")))


def verify_manifest(folder):
    folder = Path(folder)
    require(folder.is_dir() and not folder.is_symlink(), "record root must be a real directory")
    manifest = read_json(folder / "manifest.json")
    require(set(manifest) == {"schema_version", "record_scope", "files"}, "unexpected manifest fields")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1, "unsupported record schema")
    require(isinstance(manifest["record_scope"], str) and manifest["record_scope"].strip(), "missing record scope")
    hashes = manifest["files"]
    require(isinstance(hashes, dict) and bool(hashes), "empty or invalid record file manifest")
    for name, digest in hashes.items():
        safe_relative(name)
        require(name != "manifest.json" and isinstance(digest, str) and SHA256.fullmatch(digest), "invalid file hash entry")
    actual = set()
    for path in folder.rglob("*"):
        require(not path.is_symlink(), "symlink in record directory")
        require(path.is_dir() or stat.S_ISREG(path.lstat().st_mode), "non-regular record entry")
        if path.is_file() and path != folder / "manifest.json":
            actual.add(path.relative_to(folder).as_posix())
    require(actual == set(hashes), "record file membership differs from manifest")
    require(set(ARCHIVES.values()) <= actual, "missing required hardware archive")
    for name, digest in hashes.items():
        require(file_hash(folder / name) == digest, f"record hash mismatch: {name}")
    return manifest


def extract_suite(archive_path, destination):
    """Extract one bounded root of regular files/directories, never links/extractall.

    Decompress to a bounded temporary stream before parsing tar metadata. This
    also bounds malicious PAX headers/padding that member-size checks miss.
    """
    archive_path = regular_file(archive_path, MAX_ARCHIVE_BYTES)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    seen, roots, total = set(), set(), 0
    with tempfile.TemporaryFile() as expanded:
        with gzip.open(archive_path, "rb") as source:
            remaining = MAX_EXPANDED_BYTES
            while True:
                block = source.read(min(1024 * 1024, remaining + 1))
                if not block:
                    break
                remaining -= len(block)
                require(remaining >= 0, "archive exceeds expanded stream size limit")
                expanded.write(block)
        expanded.seek(0)
        with tarfile.open(fileobj=expanded, mode="r:") as archive:
            for index, member in enumerate(archive, 1):
                require(index <= MAX_MEMBERS, "archive exceeds member count limit")
                name = member.name.rstrip("/") if member.isdir() else member.name
                path = safe_relative(name)
                require(name not in seen, f"duplicate archive member: {name}")
                require((member.isfile() or member.isdir()) and not member.issparse(), "non-regular archive member")
                require(0 <= member.size <= MAX_FILE_BYTES, "archive member exceeds size limit")
                total += member.size
                require(total <= MAX_EXPANDED_BYTES, "archive exceeds expanded content size limit")
                seen.add(name)
                roots.add(path.parts[0])
                require(len(roots) == 1, "archive must have exactly one root directory")
                target = destination.joinpath(*path.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                require(len(path.parts) > 1, "archive root must be a directory")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                require(source is not None, "unreadable archive member")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                require(target.stat().st_size == member.size, "archive member length mismatch")
    require(len(roots) == 1, "empty archive")
    return destination / next(iter(roots))


def source_anchor(source):
    require(isinstance(source, dict) and len(source) == 25, "unexpected recorded source inventory")
    for name, digest in source.items():
        safe_relative(name)
        require(isinstance(digest, str) and SHA256.fullmatch(digest), "invalid recorded source hash")
    digest = sha256(json.dumps(source, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(digest == SOURCE_MAP_SHA256, "recorded source hashes differ from pinned source anchor")
    return digest


def load_validator():
    path = ROOT / "scripts/validate_p100_results.py"
    spec = importlib.util.spec_from_file_location("policy_replication_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_run(folder, name, validator):
    run = folder / name
    require(run.is_dir() and not run.is_symlink(), f"missing run: {name}")
    require({path.name for path in run.iterdir()} == RUN_FILES, f"unexpected run file membership: {name}")
    for path in run.iterdir():
        regular_file(path)
    protocol = read_json(run / "protocol.json")
    source_anchor(protocol["source_sha256"])
    kind, mode = name.split("-", 1)
    args = protocol["args"]
    expected = {"contexts": [256] if kind == "smoke" else [512, 2048],
                "cases": 4 if kind == "smoke" else 24, "repeats": 1 if kind == "smoke" else 3,
                "group_size": 4 if kind == "smoke" else 8,
                "scenarios": ["roomy", "pressure"] if kind == "smoke" else ["roomy", "pressure", "cancellation"],
                "max_tokens": 24, "chunk_size": 128, "seed": 20261102,
                "smoke": kind == "smoke", "allow_unvalidated_lossy": True, "device": "cuda", "timing_mode": mode}
    require({key: value for key, value in args.items() if key not in ("output", "model_dir")} == expected,
            f"unexpected recorded protocol arguments: {name}")
    require(protocol["schema_version"] == 2 and protocol["timing_mode"] == mode, "unexpected run schema/mode")
    require(protocol["model_identity"]["model_digest"] == MODEL_DIGEST, "unexpected model digest")
    provenance = read_json(run / "model-provenance.json")
    files = {path: {"bytes": provenance["source_file_bytes"][path], "sha256": digest}
             for path, digest in provenance["file_sha256"].items()}
    require(set(provenance["source_file_bytes"]) == set(files), "model file inventory differs")
    require(sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            == MODEL_FILES_SHA256, "loaded model files differ from pinned model anchor")
    require(provenance["model_digest"] == MODEL_DIGEST and provenance["activation_dtype"] == "float16"
            and protocol["model_identity"]["activation_dtype"] == "float16", "model execution identity differs")
    result = validator.audit(run)
    require(result["valid"] is True, f"independent audit failed: {name}")
    expected_sessions, expected_blocks = (48, 12) if kind == "smoke" else (2592, 108)
    require(result["sessions_checked"] == expected_sessions and result["blocks_checked"] == expected_blocks,
            "unexpected session/block counts")
    return result, protocol, read_json(run / "summary.json")


def policy_totals(summary):
    totals = {}
    for cell in summary["summaries"]:
        value = totals.setdefault(cell["policy"], {"sessions": 0, "responses": 0, "total_session_ms": 0.0,
                   "peak_serialized_payload_bytes": 0, "peak_cuda_allocated_bytes": 0,
                   "correct": 0, "token_identical_to_recompute": 0, "quality_regressions_vs_recompute": 0,
                   "quality_improvements_vs_recompute": 0, "truncated": 0})
        value["sessions"] += cell["sessions"]
        value["responses"] += cell["responses"]
        value["total_session_ms"] += cell["total_ms_including_initial_prefill_and_cancellations"]
        value["peak_serialized_payload_bytes"] = max(value["peak_serialized_payload_bytes"], cell["peak_paused_payload_bytes"])
        value["peak_cuda_allocated_bytes"] = max(value["peak_cuda_allocated_bytes"], cell["peak_cuda_allocated_bytes"] or 0)
        for key in ("correct", "token_identical_to_recompute", "quality_regressions_vs_recompute",
                    "quality_improvements_vs_recompute", "truncated"):
            if cell[key] is None:
                value[key] = None
            elif value[key] is not None:
                value[key] += cell[key]
    for value in totals.values():
        value["mean_total_ms_per_started_session"] = value["total_session_ms"] / value["sessions"]
    return totals


def verify_record(folder=DEFAULT_RECORD):
    folder = Path(folder)
    manifest = verify_manifest(folder)
    validator = load_validator()
    hardware, prompts, all_protocols = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="kvpilot-policy-replication-") as temporary:
        for side, archive in ARCHIVES.items():
            suite = extract_suite(folder / archive, Path(temporary) / side)
            runs, protocols, totals = {}, {}, {}
            for name in RUN_NAMES:
                validation, protocol, summary = audit_run(suite, name, validator)
                runs[name] = validation
                protocols[name] = protocol
                all_protocols[side, name] = protocol
                cuda_build = "2.7.1+cu126" if side == "p100" else "2.7.1+cu128"
                require(protocol["versions"] == dict(COMMON_VERSIONS, torch=cuda_build), "recorded runtime package versions differ")
                require(protocol["model_identity"]["torch_version"] == cuda_build, "model Torch build differs")
                expected_device = "P100" if side == "p100" else "RTX 5060 Ti"
                require(expected_device in protocol["device_name"], "hardware archive/device label mismatch")
                if name.startswith("full-"):
                    totals[name.removeprefix("full-")] = policy_totals(summary)
                fixture = read_json(suite / name / "fixture.json")
                prompts[side, name] = fixture["cases_by_context"]
            timing = totals["equal_work"]
            adaptive = timing["adaptive_eviction"]["total_session_ms"]
            hardware[side] = {"recorded_source_commit": (suite / "source-commit.txt").read_text().strip(),
                              "device_name": protocols["full-equal_work"]["device_name"],
                              "versions": protocols["full-equal_work"]["versions"], "runs": runs,
                              "full_equal_work": timing, "full_free_generation": totals["free_generation"],
                              "adaptive_eviction_speed_ratio": {
                                  "vs_raw_admit_if_fits": timing["raw_admit_if_fits"]["total_session_ms"] / adaptive,
                                  "vs_group_int4": timing["group_int4"]["total_session_ms"] / adaptive},
                              "speed_ratio_definition": "baseline full-session time divided by adaptive full-session time; above one means adaptive used less time"}
        continuation_differences = []
        for name in RUN_NAMES:
            left_protocol, right_protocol = all_protocols["p100", name], all_protocols["rtx5060ti", name]
            for key in ("jobs", "schedules", "budgets_bytes", "policies", "cases", "cache_ownership"):
                require(left_protocol[key] == right_protocol[key], f"cross-system protocol differs: {key}")
            left, right = prompts["p100", name], prompts["rtx5060ti", name]
            require(set(left) == set(right), "cross-system fixture contexts differ")
            for context in left:
                require(len(left[context]) == len(right[context]), "cross-system case counts differ")
                for a, b in zip(left[context], right[context]):
                    omitted = {"fixed_continuation", "continuation_preparation_ms"}
                    require({key: value for key, value in a.items() if key not in omitted}
                            == {key: value for key, value in b.items() if key not in omitted},
                            "cross-system prompt inputs or case ordering differ")
                    if a.get("fixed_continuation") != b.get("fixed_continuation"):
                        continuation_differences.append({"run": name, "context_target": int(context), "case_id": a["id"]})
    return {"valid": True, "record_scope": manifest["record_scope"], "files_checked": len(manifest["files"]),
            "runs_checked": 8, "source_reference_commit": SOURCE_COMMIT, "source_map_sha256": SOURCE_MAP_SHA256,
            "current_validator_sha256": file_hash(ROOT / "scripts/validate_p100_results.py"),
            "hardware": hardware, "cross_system_prompt_inputs_identical": True,
            "fixed_continuation_differences": continuation_differences,
            "scope": "Offline artifact/accounting/source-anchor audit. No inference or independent execution attestation; repeated responses are not independent quality samples. GPU/runtime/host differences and continuation differences prohibit a pure GPU-only causal claim. No serving-throughput, dollar-savings or general quality claim is certified."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", nargs="?", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--output", type=Path, help="Optional JSON output outside the immutable record directory")
    args = parser.parse_args(argv)
    if args.output is not None:
        require(not args.output.resolve().is_relative_to(args.record.resolve()), "output must be outside the immutable record")
    try:
        report = verify_record(args.record)
    except (ValueError, OSError, KeyError, TypeError, tarfile.TarError, EOFError) as error:
        report = {"valid": False, "error": str(error)}
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(content)
    print(content, end="")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
