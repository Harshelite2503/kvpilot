"""Offline recording/publication checks using an audited fixture and local Git only."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


@pytest.fixture
def recording(tmp_path):
    repo = tmp_path / "working repo"
    repo.mkdir()
    git(repo, "init", "-b", "experiment")
    git(repo, "config", "user.name", "Recording Test")
    git(repo, "config", "user.email", "test@example.invalid")
    (repo / "scripts").mkdir()
    for name in ("record_p100_results.sh", "validate_p100_results.py"):
        shutil.copyfile(ROOT / "scripts" / name, repo / "scripts" / name)
    source = b"# Synthetic source for packaging tests, not benchmark evidence.\n"
    (repo / "runner.py").write_bytes(source)
    git(repo, "add", "scripts", "runner.py")
    git(repo, "commit", "-m", "Fixture source")
    commit = git(repo, "rev-parse", "HEAD").decode().strip()
    run = tmp_path / "original run"
    run.mkdir()
    fixture = ROOT / "results" / "p100-readiness-mac-smoke-v1"
    manifest = json.loads((fixture / "artifact-sha256.json").read_text())
    for name in manifest:
        shutil.copyfile(fixture / name, run / name)
    protocol = json.loads((run / "protocol.json").read_text())
    protocol["source_sha256"] = {"runner.py": sha256(source).hexdigest()}
    (run / "protocol.json").write_text(json.dumps(protocol))
    manifest["protocol.json"] = sha256((run / "protocol.json").read_bytes()).hexdigest()
    (run / "artifact-sha256.json").write_text(json.dumps(manifest))
    return repo, run, commit


def record(recording, *args, record_id="test-record"):
    repo, run, commit = recording
    return subprocess.run([
        "bash", str(repo / "scripts" / "record_p100_results.sh"), str(run),
        "--record-id", record_id, "--source-commit", commit,
        "--python", sys.executable, *args], cwd=repo, capture_output=True, text=True)


def test_deterministic_archive_excludes_unlisted_files_and_is_idempotent(recording):
    repo, run, commit = recording
    (run / ".env").write_text("not published")
    (run / "model.safetensors").write_text("not published")
    first = record(recording)
    assert first.returncode == 0, first.stderr
    folder = repo / "results/test-record"
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    second = record(recording)
    assert second.returncode == 0, second.stderr
    assert "equivalent" in second.stdout
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
    assert record(recording, record_id="test-record-2").returncode == 0
    assert (folder / "run.tar.gz").read_bytes() == (repo / "results/test-record-2/run.tar.gz").read_bytes()
    with tarfile.open(folder / "run.tar.gz") as archive:
        assert set(archive.getnames()) == {"run/" + name for name in json.loads((run / "artifact-sha256.json").read_text())} | {"run/artifact-sha256.json"}
        assert archive.extractfile("run/protocol.json").read() == (run / "protocol.json").read_bytes()
        assert all(member.mtime == 0 and member.uid == 0 for member in archive.getmembers())
    assert json.loads((folder / "record.json").read_text())["source_commit"] == commit


@pytest.mark.parametrize("mutation", ["hash", "symlink", "extra_manifest_entry", "wrong_source"])
def test_invalid_inputs_do_not_create_record(recording, mutation):
    repo, run, commit = recording
    if mutation == "hash":
        (run / "rows.jsonl").write_text("tampered")
    elif mutation == "symlink":
        report = run / "report.md"
        saved = run / "other.md"
        report.rename(saved)
        report.symlink_to(saved)
    elif mutation == "extra_manifest_entry":
        manifest = json.loads((run / "artifact-sha256.json").read_text())
        manifest[".env"] = sha256(b"secret").hexdigest()
        (run / "artifact-sha256.json").write_text(json.dumps(manifest))
        (run / ".env").write_bytes(b"secret")
    else:
        (repo / "runner.py").write_text("changed")
        git(repo, "add", "runner.py")
        git(repo, "commit", "-m", "Different source")
        recording = (repo, run, git(repo, "rev-parse", "HEAD").decode().strip())
    outcome = record(recording)
    assert outcome.returncode != 0
    assert not (repo / "results/test-record").exists()


@pytest.mark.parametrize("record_id", ["../outside", "local-run", "-bad", "with/slash", ""])
def test_unsafe_ids_are_rejected(recording, record_id):
    assert record(recording, record_id=record_id).returncode != 0


def test_existing_changed_record_is_never_overwritten(recording):
    repo, _, _ = recording
    assert record(recording).returncode == 0
    report = repo / "results/test-record/report.md"
    report.write_text("user edit")
    outcome = record(recording)
    assert outcome.returncode != 0 and "refusing to overwrite" in outcome.stderr
    assert report.read_text() == "user edit"


def test_push_preserves_dirty_checkout_and_uses_original_source_commit(recording, tmp_path):
    repo, _, original = recording
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    git(repo, "remote", "add", "origin", str(remote))
    (repo / "runner.py").write_text("new benchmark implementation")
    git(repo, "add", "runner.py")
    git(repo, "commit", "-m", "Source has advanced after the run")
    head = git(repo, "rev-parse", "HEAD")
    (repo / "runner.py").write_text("staged user work")
    git(repo, "add", "runner.py")
    (repo / "runner.py").write_text("unstaged user work")
    (repo / "private.txt").write_text("unrelated untracked work")
    staged, unstaged = git(repo, "diff", "--cached"), git(repo, "diff")
    outcome = record(recording, "--push")
    assert outcome.returncode == 0, outcome.stderr
    assert git(repo, "branch", "--show-current").decode().strip() == "experiment"
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "diff", "--cached") == staged
    assert git(repo, "diff") == unstaged
    assert (repo / "private.txt").read_text() == "unrelated untracked work"
    branch = "refs/heads/results/test-record"
    changed = git(remote, "diff-tree", "--no-commit-id", "--name-only", "-r", branch).decode().splitlines()
    assert len(changed) == 5 and all(name.startswith("results/test-record/") for name in changed)
    provenance = json.loads(git(remote, "show", branch + ":results/test-record/record.json"))
    assert provenance["source_commit"] == original
