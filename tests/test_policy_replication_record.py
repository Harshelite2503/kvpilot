"""Portable record checks reject corruption even when outer hashes are refreshed."""
from hashlib import sha256
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_policy_replication_record", ROOT / "scripts/verify_policy_replication_record.py")
record = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(record)


def manifest_for(folder):
    value = {"schema_version": 1, "record_scope": "Synthetic packaging fixture; not performance evidence",
             "files": {path.relative_to(folder).as_posix(): sha256(path.read_bytes()).hexdigest()
                       for path in folder.rglob("*") if path.is_file() and path.name != "manifest.json"}}
    (folder / "manifest.json").write_text(json.dumps(value))


@pytest.fixture
def outer_record(tmp_path):
    for name in record.ARCHIVES.values():
        path = tmp_path / name
        path.parent.mkdir(parents=True)
        path.write_bytes(b"outer-hash-test-only")
    (tmp_path / "notes.txt").write_text("Archival scope remains limited.")
    manifest_for(tmp_path)
    return tmp_path


@pytest.fixture
def small_suite(tmp_path):
    """Reuse only the 48-session captured smoke case; no model or Torch needed."""
    target = tmp_path / "small-root/smoke-equal_work"
    target.mkdir(parents=True)
    with tarfile.open(record.DEFAULT_RECORD / "rtx5060ti/run.tar.gz", "r:gz") as source:
        for member in source:
            parts = Path(member.name).parts
            if len(parts) == 3 and parts[1] == "smoke-equal_work" and member.isfile():
                assert parts[2] in record.RUN_FILES
                with source.extractfile(member) as stream:
                    (target / parts[2]).write_bytes(stream.read())
    assert {path.name for path in target.iterdir()} == record.RUN_FILES
    return target.parent


def archive_members(path, entries):
    with tarfile.open(path, "w:gz") as target:
        for name, kind, content in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.linkname = "../../outside"
            info.size = len(content) if kind == tarfile.REGTYPE else 0
            target.addfile(info, io.BytesIO(content))


def test_small_valid_capture_survives_safe_extraction_and_real_independent_audit(small_suite, tmp_path):
    archive = tmp_path / "small.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        target.add(small_suite, arcname="original-root-name")
    extracted = record.extract_suite(archive, tmp_path / "unpacked")
    result, protocol, summary = record.audit_run(extracted, "smoke-equal_work", record.load_validator())
    assert result["valid"] and result["sessions_checked"] == 48 and result["events_checked"] == 96
    assert result["blocks_checked"] == 12
    assert record.source_anchor(protocol["source_sha256"]) == record.SOURCE_MAP_SHA256
    totals = record.policy_totals(summary)
    assert len(totals) == 6 and all(value["sessions"] == 8 for value in totals.values())
    assert all(value["correct"] is None for value in totals.values())


def test_refreshed_inner_hash_cannot_redefine_trusted_source_anchor(small_suite):
    run = small_suite / "smoke-equal_work"
    protocol = json.loads((run / "protocol.json").read_text())
    protocol["source_sha256"]["experiments/torch/benchmark.py"] = "0" * 64
    (run / "protocol.json").write_text(json.dumps(protocol))
    hashes = json.loads((run / "artifact-sha256.json").read_text())
    hashes["protocol.json"] = sha256((run / "protocol.json").read_bytes()).hexdigest()
    (run / "artifact-sha256.json").write_text(json.dumps(hashes))
    with pytest.raises(ValueError, match="pinned source anchor"):
        record.audit_run(small_suite, "smoke-equal_work", record.load_validator())


def test_loaded_model_provenance_is_bound_even_if_artifact_hash_is_refreshed(small_suite):
    run = small_suite / "smoke-equal_work"
    provenance = json.loads((run / "model-provenance.json").read_text())
    provenance["file_sha256"]["model.safetensors"] = "0" * 64
    (run / "model-provenance.json").write_text(json.dumps(provenance))
    hashes = json.loads((run / "artifact-sha256.json").read_text())
    hashes["model-provenance.json"] = sha256((run / "model-provenance.json").read_bytes()).hexdigest()
    (run / "artifact-sha256.json").write_text(json.dumps(hashes))
    with pytest.raises(ValueError, match="pinned model anchor"):
        record.audit_run(small_suite, "smoke-equal_work", record.load_validator())


def test_valid_outer_manifest_and_tampered_hash(outer_record):
    assert len(record.verify_manifest(outer_record)["files"]) == 3
    (outer_record / "notes.txt").write_text("changed scope")
    with pytest.raises(ValueError, match="record hash mismatch: notes.txt"):
        record.verify_manifest(outer_record)


@pytest.mark.parametrize("change", ["extra", "missing", "symlink", "directory_symlink"])
def test_record_membership_and_symlinks_fail_closed(outer_record, tmp_path, change):
    if change == "extra":
        (outer_record / "extra.txt").write_text("unlisted")
    elif change == "missing":
        (outer_record / "notes.txt").unlink()
    elif change == "symlink":
        (outer_record / "notes.txt").unlink()
        (outer_record / "notes.txt").symlink_to(outer_record / "manifest.json")
    else:
        (outer_record / "linked-directory").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="membership|symlink"):
        record.verify_manifest(outer_record)


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "a/../outside", "a//file", "./file", "a\\file"])
def test_unsafe_manifest_paths_rejected_before_file_access(outer_record, name):
    data = json.loads((outer_record / "manifest.json").read_text())
    data["files"][name] = "0" * 64
    (outer_record / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="relative path"):
        record.verify_manifest(outer_record)


def test_duplicate_json_keys_are_not_silently_overwritten(outer_record):
    (outer_record / "manifest.json").write_text('{"schema_version":1,"schema_version":2}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        record.verify_manifest(outer_record)


@pytest.mark.parametrize("name,kind", [
    ("../outside", tarfile.REGTYPE), ("/tmp/outside", tarfile.REGTYPE),
    ("root/../../outside", tarfile.REGTYPE), ("root/./file", tarfile.REGTYPE),
    ("root/file", tarfile.SYMTYPE), ("root/file", tarfile.LNKTYPE),
    ("root/file", tarfile.FIFOTYPE),
])
def test_unsafe_archive_paths_and_links_never_escape(tmp_path, name, kind):
    archive = tmp_path / "bad.tar.gz"
    archive_members(archive, [(name, kind, b"x")])
    with pytest.raises(ValueError, match="relative path|non-regular"):
        record.extract_suite(archive, tmp_path / "out")
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("entries,error", [
    ([("root/file", tarfile.REGTYPE, b"x"), ("root/file", tarfile.REGTYPE, b"y")], "duplicate"),
    ([("root/file", tarfile.REGTYPE, b"x"), ("other/file", tarfile.REGTYPE, b"y")], "exactly one root"),
    ([("root", tarfile.REGTYPE, b"x")], "root must be a directory"),
])
def test_archive_duplicate_roots_and_root_file_rejected(tmp_path, entries, error):
    archive = tmp_path / "bad.tar.gz"
    archive_members(archive, entries)
    with pytest.raises(ValueError, match=error):
        record.extract_suite(archive, tmp_path / "out")


def test_expanded_tar_metadata_is_bounded_before_parsing(tmp_path, monkeypatch):
    monkeypatch.setattr(record, "MAX_EXPANDED_BYTES", 1024)
    archive = tmp_path / "expanded.tar.gz"
    with gzip.open(archive, "wb") as target:
        target.write(bytes(1025))
    with pytest.raises(ValueError, match="expanded stream size limit"):
        record.extract_suite(archive, tmp_path / "out")


@pytest.mark.parametrize("limit", ["MAX_ARCHIVE_BYTES", "MAX_FILE_BYTES", "MAX_MEMBERS"])
def test_other_archive_resource_limits_are_enforced(tmp_path, monkeypatch, limit):
    archive = tmp_path / "large.tar.gz"
    archive_members(archive, [("root/file", tarfile.REGTYPE, b"xx"), ("root/next", tarfile.REGTYPE, b"xx")])
    monkeypatch.setattr(record, limit, 1)
    with pytest.raises(ValueError, match="bounded regular|size limit|member count"):
        record.extract_suite(archive, tmp_path / "out")


def test_archived_full_replication_remains_auditable_without_torch_or_git():
    result = record.verify_record()
    assert result["valid"] and result["runs_checked"] == 8
    assert result["cross_system_prompt_inputs_identical"]
    assert result["fixed_continuation_differences"] == [
        {"run": "full-equal_work", "context_target": 512, "case_id": "case-0002"}]
    for hardware in result["hardware"].values():
        assert hardware["runs"]["full-equal_work"]["sessions_checked"] == 2592
        assert hardware["runs"]["full-free_generation"]["sessions_checked"] == 2592
        assert hardware["full_equal_work"]["adaptive_eviction"]["sessions"] == 432
        assert hardware["full_equal_work"]["adaptive_eviction"]["responses"] == 396
        assert hardware["full_equal_work"]["adaptive_eviction"]["correct"] is None
        assert hardware["full_free_generation"]["adaptive_eviction"]["correct"] == 243
        assert hardware["full_free_generation"]["adaptive_eviction"]["token_identical_to_recompute"] == 396
