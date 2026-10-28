"""Historical evidence remains auditable; tampering fails closed, offline."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import io
import gzip
import json
from pathlib import Path
import shutil
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_regression_evidence", ROOT / "scripts/check_regression_evidence.py")
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


@pytest.fixture
def sensitivity():
    folder = ROOT / "results/eviction-holdout-measured-costs-v1"
    return (evidence.read_json(ROOT / "examples/holdout-protocol-measured-costs.json"),
            evidence.read_json(folder / "manifest.json"),
            [json.loads(line) for line in (folder / "runs.jsonl").read_text().splitlines()])


def test_checked_in_records_pass_current_validators_without_model_or_historical_git():
    result = evidence.audit()
    assert result["valid"]
    assert [r["sessions_checked"] for r in result["p100_records"]] == [2160, 40]
    assert result["measured_costs"] == {"valid": True, "rows_checked": 840,
                                        "trace_pairs_checked": 140, "comparisons_recomputed": 49}


def test_historical_p95_attribution_tie_does_not_erase_secondary_regressions(sensitivity):
    _, _, rows = sensitivity
    pair = {row["policy"]: row["metrics"] for row in rows
            if row["scenario"] == "heterogeneous" and row["seed"] == 1105
            and row["policy"] in ("adaptive_exact", "adaptive_eviction")}
    baseline, candidate = pair["adaptive_exact"], pair["adaptive_eviction"]
    assert candidate["modeled_p95_resume_ms"] == baseline["modeled_p95_resume_ms"]
    assert candidate["modeled_total_service_work_ms"] > baseline["modeled_total_service_work_ms"]
    assert candidate["peak_host_reserved_bytes"] > baseline["peak_host_reserved_bytes"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "different_trace", "false_quality", "fabricated_gpu",
                                    "fabricated_success", "nan", "work_accounting", "incomplete_sessions"])
def test_invalid_paired_records_are_rejected(sensitivity, mutation):
    protocol, manifest, rows = sensitivity
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif mutation == "different_trace":
        rows[0]["trace_sha256"] = "0" * 64
    elif mutation == "false_quality":
        rows[0]["quality_validated"] = True
    elif mutation == "fabricated_gpu":
        rows[0]["metrics"]["measured_gpu_itl_ms"] = 1.0
    elif mutation == "fabricated_success":
        rows[0]["metrics"]["measured_llm_task_success"] = 0.99
    elif mutation == "nan":
        rows[0]["metrics"]["modeled_p95_resume_ms"] = float("nan")
    elif mutation == "work_accounting":
        rows[0]["metrics"]["modeled_total_service_work_ms"] += 1
    else:
        rows[0]["metrics"]["resume_count"] -= 1
    with pytest.raises(ValueError):
        evidence.check_rows(rows, manifest, protocol)


@pytest.fixture
def copied_record(tmp_path):
    source = ROOT / "results/p100-2026-10-28-smoke"
    target = tmp_path / source.name
    shutil.copytree(source, target)
    return target


def test_outer_record_corruption_is_rejected(copied_record):
    with (copied_record / "report.md").open("a") as stream:
        stream.write("changed")
    with pytest.raises(ValueError, match="outer record hash mismatch: report.md"):
        evidence.audit_p100_record(copied_record)


@pytest.mark.parametrize("field,error", [
    ("source_sha256", "recorded source hashes disagree"),
    ("device_name", "recorded device name disagrees with protocol"),
    ("run_created_utc", "recorded run timestamp disagrees with protocol"),
])
def test_record_metadata_is_bound_to_the_archived_protocol(copied_record, field, error):
    path = copied_record / "record.json"
    record = evidence.read_json(path)
    if field == "source_sha256":
        record[field]["experiments/torch/benchmark.py"] = "0" * 64
    elif field == "device_name":
        record[field] = "unrecorded device"
    else:
        record[field] = "2000-01-01T00:00:00+00:00"
    path.write_text(json.dumps(record))
    hashes = evidence.read_json(copied_record / "record-sha256.json")
    hashes["record.json"] = sha256(path.read_bytes()).hexdigest()
    (copied_record / "record-sha256.json").write_text(json.dumps(hashes))
    with pytest.raises(ValueError, match=error):
        evidence.audit_p100_record(copied_record)


@pytest.mark.parametrize("name,kind", [
    ("../outside", tarfile.REGTYPE), ("/tmp/outside", tarfile.REGTYPE),
    ("run/../outside", tarfile.REGTYPE), ("run/.env", tarfile.REGTYPE),
    ("run/protocol.json", tarfile.SYMTYPE), ("run/protocol.json", tarfile.LNKTYPE),
    ("run/protocol.json", tarfile.DIRTYPE), ("run/protocol.json", tarfile.FIFOTYPE),
])
def test_archive_rejects_unsafe_paths_and_nonregular_members(tmp_path, name, kind):
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.linkname = "../../outside"
        archive.addfile(member, io.BytesIO(b""))
    with pytest.raises(ValueError, match="unsafe or unexpected|non-regular"):
        evidence.extract_run(archive_path, tmp_path / "extracted")
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "file_size", "total_size", "archive_size"])
def test_archive_member_completeness_and_limits(tmp_path, monkeypatch, mutation):
    archive_path = tmp_path / "bounded.tar.gz"
    names = sorted(evidence.RUN_FILES)
    if mutation == "duplicate":
        names.append(names[0])
    if mutation == "missing":
        names.pop()
    with tarfile.open(archive_path, "w:gz") as archive:
        for name in names:
            member = tarfile.TarInfo(f"run/{name}")
            member.size = 2
            archive.addfile(member, io.BytesIO(b"{}"))
    if mutation == "file_size":
        monkeypatch.setattr(evidence, "MAX_FILE_BYTES", 1)
    if mutation == "total_size":
        monkeypatch.setattr(evidence, "MAX_TOTAL_BYTES", 3)
    if mutation == "archive_size":
        monkeypatch.setattr(evidence, "MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(ValueError, match="duplicate|missing|size limit"):
        evidence.extract_run(archive_path, tmp_path / "extracted")


def test_archive_bounds_expansion_before_parsing_tar_headers(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "MAX_TOTAL_BYTES", 1)
    archive_path = tmp_path / "oversized-padding.tar.gz"
    with gzip.open(archive_path, "wb") as stream:
        stream.write(bytes(1024 * 1024 + 2))
    with pytest.raises(ValueError, match="expanded stream size limit"):
        evidence.extract_run(archive_path, tmp_path / "extracted")


@pytest.fixture
def copied_sensitivity(tmp_path):
    names = ["examples/holdout-protocol-measured-costs.json", "examples/holdout-protocol.json",
             "results/measured-costs-v1.json", "results/eviction-freeze-measured-costs-v1.json",
             "results/eviction-freeze-v1.json", "results/eviction-holdout-v1/manifest.json"]
    names += [f"results/eviction-holdout-measured-costs-v1/{name}"
              for name in ("manifest.json", "runs.jsonl", "summary.json")]
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


@pytest.mark.parametrize("mutation,error", [
    ("workload", "changed more than codecs"), ("codec", "differs from measured calibration"),
    ("freeze", "frozen protocol hash mismatch"), ("scope", "evidence scope mismatch"),
    ("summary", "summary declaration mismatch"),
])
def test_sensitivity_records_keep_their_original_scope_and_provenance(copied_sensitivity, mutation, error):
    if mutation in ("workload", "codec"):
        target = copied_sensitivity / "examples/holdout-protocol-measured-costs.json"
    elif mutation == "freeze":
        target = copied_sensitivity / "results/eviction-freeze-measured-costs-v1.json"
    else:
        target = copied_sensitivity / "results/eviction-holdout-measured-costs-v1/summary.json"
    data = evidence.read_json(target)
    if mutation == "workload":
        data["sessions_per_run"] += 1
    elif mutation == "codec":
        data["base_cost_config"]["codecs"][0]["encode_ms_per_mib"] = 0.0
    elif mutation == "freeze":
        data["component_sha256"]["experiment_protocol"] = "0" * 64
    elif mutation == "scope":
        data["quality_validated"] = True
    else:
        data["baseline"]["policy"] = "recompute"
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=error):
        evidence.audit_measured_costs(copied_sensitivity)


@pytest.mark.parametrize("valid", [False, True])
def test_cli_writes_machine_readable_report_and_returns_failure(tmp_path, monkeypatch, capsys, valid):
    def fake_audit():
        if not valid:
            raise ValueError("tampered evidence")
        return {"valid": True}
    monkeypatch.setattr(evidence, "audit", fake_audit)
    output = tmp_path / "nested/report.json"
    assert evidence.main(["--output", str(output)]) == (0 if valid else 1)
    assert evidence.read_json(output)["valid"] is valid
    assert json.loads(capsys.readouterr().out) == evidence.read_json(output)
