"""Offline regression checks for historical P100 and measured-cost records.

This checks backward-compatible audit/analysis behavior and record consistency,
not hardware execution, present-day source identity, or performance thresholds.
No model, tensor engine, network, or historical Git checkout is needed. The
measured-cost summary check imports the core evaluator (and thus NumPy).
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import gzip
import importlib.util
import io
import json
import math
from pathlib import Path
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUN_FILES = frozenset({
    "artifact-sha256.json", "protocol.json", "fixture.json", "model-provenance.json",
    "calibration.json", "rows.jsonl", "events.jsonl", "blocks.json", "summary.json",
    "status.json", "report.md",
})
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_FILE_BYTES,
            f"JSON input is not a bounded regular file: {path}")
    return json.loads(path.read_text())


def file_hash(path):
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"regression_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_run(archive_path, destination):
    """Copy only bounded, flat, regular allowlisted members; never extractall."""
    archive_path, destination = Path(archive_path), Path(destination)
    require(not archive_path.is_symlink() and archive_path.stat().st_size <= MAX_ARCHIVE_BYTES,
            "archive exceeds compressed size limit or is a symlink")
    destination.mkdir(parents=True, exist_ok=False)
    seen, total = set(), 0
    # Bound the entire expanded stream, including metadata/padding, before the
    # tar parser can read oversized PAX headers or sparse-file metadata.
    tar_limit = MAX_TOTAL_BYTES + 1024 * 1024
    with gzip.open(archive_path, "rb") as compressed:
        archive_bytes = compressed.read(tar_limit + 1)
    require(len(archive_bytes) <= tar_limit, "archive exceeds expanded stream size limit")
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        for member in archive:
            require(member.name in {f"run/{name}" for name in RUN_FILES},
                    f"unsafe or unexpected archive member: {member.name}")
            require(member.name not in seen, f"duplicate archive member: {member.name}")
            require(member.isfile() and not member.issparse(), f"non-regular archive member: {member.name}")
            require(0 <= member.size <= MAX_FILE_BYTES, f"archive member exceeds size limit: {member.name}")
            total += member.size
            require(total <= MAX_TOTAL_BYTES, "archive exceeds expanded size limit")
            seen.add(member.name)
            stream = archive.extractfile(member)
            require(stream is not None, f"unreadable archive member: {member.name}")
            with stream:
                payload = stream.read(MAX_FILE_BYTES + 1)
            require(len(payload) == member.size, f"archive member length mismatch: {member.name}")
            (destination / member.name.removeprefix("run/")).write_bytes(payload)
    require(seen == {f"run/{name}" for name in RUN_FILES}, "missing archive members")


def audit_p100_record(folder):
    folder = Path(folder)
    hashes = read_json(folder / "record-sha256.json")
    require(set(hashes) == {"record.json", "report.md", "run.tar.gz", "validation.json"},
            "unexpected outer record hash entries")
    for name, digest in hashes.items():
        require((folder / name).stat().st_size <= (MAX_ARCHIVE_BYTES if name == "run.tar.gz" else MAX_FILE_BYTES),
                f"outer record exceeds size limit: {name}")
        require(file_hash(folder / name) == digest, f"outer record hash mismatch: {name}")
    record = read_json(folder / "record.json")
    require(record["schema_version"] == 1 and record["evidence_type"] == "measured_owned_cache_torch",
            "invalid P100 record declaration")
    require(record["record_id"] == folder.name, "record ID does not match directory")
    require(record["archive_sha256"] == hashes["run.tar.gz"], "record archive digest mismatch")
    require(len(record["source_commit"]) == 40 and all(c in "0123456789abcdef" for c in record["source_commit"]),
            "invalid recorded source commit")
    validator = load_script("validate_p100_results")
    with tempfile.TemporaryDirectory(prefix="kvpilot-evidence-") as temporary:
        run = Path(temporary) / "run"
        extract_run(folder / "run.tar.gz", run)
        manifest = read_json(run / "artifact-sha256.json")
        require(set(manifest) == RUN_FILES - {"artifact-sha256.json"}, "unexpected inner artifact hash entries")
        require(file_hash(run / "artifact-sha256.json") == record["original_artifact_manifest_sha256"],
                "original artifact manifest digest mismatch")
        require(file_hash(run / "protocol.json") == record["protocol_sha256"], "protocol digest mismatch")
        require((run / "report.md").read_bytes() == (folder / "report.md").read_bytes(), "outer report mismatch")
        protocol = read_json(run / "protocol.json")
        require(record["device_name"] == protocol["device_name"], "recorded device name disagrees with protocol")
        require(record["run_created_utc"] == protocol["created_utc"], "recorded run timestamp disagrees with protocol")
        require(protocol["source_sha256"] == record["source_sha256"], "recorded source hashes disagree")
        require(record["validator_sha256"] == record["source_sha256"]["scripts/validate_p100_results.py"],
                "recorded validator hash disagrees with source freeze")
        result = validator.audit(run)
        validator.equal(read_json(folder / "validation.json"), result, "recorded validation")
    return {"record": folder.name, **result}


def check_rows(rows, manifest, protocol):
    """Verify complete pairing and accounting before any summary is recomputed."""
    traces = {(trace["scenario"], trace["seed"]): trace for trace in manifest["input_traces"]}
    expected_traces = {(scenario, seed) for scenario in protocol["scenarios"]
                       for seed in protocol["evaluation_seeds"]}
    require(len(traces) == len(manifest["input_traces"]) and set(traces) == expected_traces,
            "missing or duplicate trace declarations")
    expected = {(scenario, seed, policy) for scenario, seed in expected_traces for policy in protocol["policies"]}
    identities = [(row["scenario"], row["seed"], row["policy"]) for row in rows]
    require(len(identities) == len(set(identities)) and set(identities) == expected,
            "missing, duplicate, or unexpected paired rows")
    require(len(expected) == 840, "historical sensitivity record must contain 840 paired rows")
    for row in rows:
        require(row["evidence_type"] == "analytical_trace_replay" and row["quality_validated"] is False,
                "analytical rows cannot claim live measurements or validated quality")
        trace = traces[row["scenario"], row["seed"]]
        require(row["trace_sha256"] == trace["trace_sha256"], "paired input trace mismatch")
        require(trace["event_count"] == 2 * protocol["sessions_per_run"], "unexpected historical event count")
        metrics = row["metrics"]
        for name, value in metrics.items():
            if name.startswith("measured_"):
                require(value is None, f"analytical replay fabricated {name}")
            elif value is not None:
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                        f"invalid analytical metric: {name}")
        require(metrics["measured_gpu_itl_ms"] is None and metrics["measured_llm_task_success"] is None,
                "analytical replay cannot certify GPU timing or task quality")
        bandwidth = manifest["scenario_assumptions"][row["scenario"]]["cost_model"]["bandwidth_bytes_per_ms"]
        transfer = metrics["modeled_transferred_bytes"] / bandwidth
        service = transfer + sum(metrics[key] for key in ("modeled_encoding_ms", "modeled_decoding_ms", "modeled_recompute_ms"))
        for key, expected_value in (("modeled_transfer_service_ms", transfer),
                                    ("modeled_total_service_ms", service), ("modeled_total_service_work_ms", service)):
            require(math.isclose(metrics[key], expected_value, rel_tol=1e-9, abs_tol=1e-7), f"incorrect accounting: {key}")
        require(metrics["resume_count"] + metrics["cancel_count"] == protocol["sessions_per_run"]
                and metrics["unresolved_count"] == 0, "incomplete historical sessions")


def audit_measured_costs(root=ROOT):
    root = Path(root)
    protocol_path = root / "examples/holdout-protocol-measured-costs.json"
    original_path = root / "examples/holdout-protocol.json"
    protocol, original = read_json(protocol_path), read_json(original_path)
    measured = read_json(root / "results/measured-costs-v1.json")
    require(measured["evidence_type"] == "live_measured_calibration", "unexpected codec calibration evidence")
    freeze = read_json(root / "results/eviction-freeze-measured-costs-v1.json")
    original_freeze = read_json(root / "results/eviction-freeze-v1.json")
    folder = root / "results/eviction-holdout-measured-costs-v1"
    manifest, summary = read_json(folder / "manifest.json"), read_json(folder / "summary.json")
    rows = [json.loads(line) for line in (folder / "runs.jsonl").read_text().splitlines()]
    require(protocol["status"] == "sensitivity_rerun_on_revealed_evaluation_seeds", "sensitivity rerun mislabeled as new holdout")
    normalized = []
    for source in (protocol, original):
        value = deepcopy(source)
        for key in ("status", "description", "protocol_version"):
            value.pop(key, None)
        for key in ("description", "source", "codecs"):
            value["base_cost_config"].pop(key, None)
        normalized.append(value)
    require(normalized[0] == normalized[1], "sensitivity protocol changed more than codecs and metadata")
    codecs = protocol["base_cost_config"]["codecs"]
    require(len(codecs) == len(measured["codecs"]) and {c["name"] for c in codecs} == set(measured["codecs"]),
            "calibration codec set mismatch")
    for codec in codecs:
        require(codec == {"name": codec["name"], **{key: measured["codecs"][codec["name"]][key]
                for key in ("ratio", "encode_ms_per_mib", "decode_ms_per_mib", "lossless")}},
                f"codec differs from measured calibration: {codec['name']}")
    for archived_freeze, path in ((freeze, protocol_path), (original_freeze, original_path)):
        require(archived_freeze["component_sha256"]["experiment_protocol"] == file_hash(path), "frozen protocol hash mismatch")
    require({k: v for k, v in freeze["component_sha256"].items() if k != "experiment_protocol"}
            == {k: v for k, v in original_freeze["component_sha256"].items() if k != "experiment_protocol"},
            "sensitivity rerun changed frozen core sources")
    require(manifest["freeze_manifest"] == freeze and manifest["protocol"] == protocol, "manifest freeze/protocol mismatch")
    for key, value in freeze.items():
        if key not in ("approval_note", "status"):
            require(manifest["source_declaration"][key] == value, f"source declaration mismatch: {key}")
    original_manifest = read_json(root / "results/eviction-holdout-v1/manifest.json")
    require(manifest["input_traces"] == original_manifest["input_traces"], "sensitivity rerun changed original input traces")
    require(set(manifest["scenario_assumptions"]) == set(protocol["scenarios"]), "missing scenario assumptions")
    for scenario, definition in protocol["scenarios"].items():
        config = deepcopy(protocol["base_cost_config"])
        config["cost_model"].update(definition.get("cost_model_overrides", {}))
        require(manifest["scenario_assumptions"][scenario] == config, "scenario cost assumptions mismatch")
    for document in (freeze, manifest, summary):
        require(document["evidence_type"] == "analytical_trace_replay" and document["quality_validated"] is False,
                "sensitivity evidence scope mismatch")
    require(manifest["split"] == summary["split"] == "holdout" and summary["overall_winner"] is None,
            "historical split or overall winner changed")
    require(summary["baseline"] == freeze["baseline"] and summary["candidate_policy"] == protocol["candidate_policy"]
            and summary["predeclared_metrics"] == freeze["predeclared_metrics"], "summary declaration mismatch")
    check_rows(rows, manifest, protocol)
    evaluator, validator = load_script("evaluate_eviction"), load_script("validate_p100_results")
    validator.equal(summary["comparisons"], evaluator.summarize_runs(rows, protocol), "primary comparisons")
    ablation = summary["attribution_comparison"]
    require(ablation["baseline_policy"] == protocol["attribution_baseline_policy"]
            and ablation["candidate_policy"] == protocol["candidate_policy"], "attribution declaration mismatch")
    validator.equal(ablation["comparisons"], evaluator.summarize_runs(
        rows, protocol, baseline_policy=protocol["attribution_baseline_policy"],
        compared_policies=[protocol["candidate_policy"]]), "attribution comparisons")
    return {"valid": True, "rows_checked": len(rows), "trace_pairs_checked": len(manifest["input_traces"]),
            "comparisons_recomputed": len(summary["comparisons"]) + len(ablation["comparisons"])}


def audit(root=ROOT):
    root = Path(root)
    return {"valid": True, "p100_records": [audit_p100_record(root / f"results/p100-2026-10-28-{name}")
            for name in ("full", "smoke")], "measured_costs": audit_measured_costs(root),
            "scope": "Historical consistency and regression audit; no inference rerun, hardware proof, or performance threshold."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the audit report as JSON")
    args = parser.parse_args(argv)
    try:
        result = audit()
    except (KeyError, TypeError, ValueError, OSError, tarfile.TarError, StopIteration) as exc:
        result = {"valid": False, "error": str(exc)}
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
