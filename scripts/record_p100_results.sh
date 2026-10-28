#!/usr/bin/env bash
# Freeze one audited run; optionally publish only that record on a new branch.
set -euo pipefail

KVPILOT_SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
KVPILOT_PYTHON="$KVPILOT_SCRIPT_ROOT/.venv-p100/bin/python"
KVPILOT_ARGS=()
while (($#)); do
  if [[ "$1" == "--python" ]]; then
    if (($# < 2)); then echo "--python requires an executable" >&2; exit 2; fi
    KVPILOT_PYTHON="$2"
    shift 2
  else
    KVPILOT_ARGS+=("$1")
    shift
  fi
done
if ! command -v "$KVPILOT_PYTHON" >/dev/null 2>&1; then
  echo "Python not found: $KVPILOT_PYTHON; pass --python /path/to/python3.11" >&2
  exit 2
fi

"$KVPILOT_PYTHON" - "$KVPILOT_SCRIPT_ROOT" "${KVPILOT_ARGS[@]}" <<'PY'
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def git(root, *args, capture=True):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular_bytes(path):
    require(not path.is_symlink() and path.is_file(), f"Not a regular file: {path}")
    return path.read_bytes()


def main():
    root = Path(sys.argv[1]).resolve()
    parser = argparse.ArgumentParser(
        description="Audit and preserve one completed owned-cache run without model files or VM logs.",
        epilog="Pass --python EXECUTABLE to the shell script to override .venv-p100/bin/python.")
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--source-commit", required=True,
                        help="Original benchmark commit; every frozen source hash must match it")
    parser.add_argument("--push", action="store_true",
                        help="Commit only this record on results/<id> in an isolated worktree and push")
    parser.add_argument("--remote", default="origin")
    args = parser.parse_args(sys.argv[2:])
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", args.record_id) is not None
            and not args.record_id.startswith("local-"),
            "record-id must be 1–80 lowercase letters/digits/hyphens and must not start with local-")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.remote) is not None,
            "remote must be a configured remote name")
    require(Path(os.fsdecode(git(root, "rev-parse", "--show-toplevel")).strip()).resolve() == root,
            "Script must be inside the repository root's scripts directory")
    run = args.run_directory.resolve()
    manifest_bytes = regular_bytes(run / "artifact-sha256.json")
    manifest = json.loads(manifest_bytes)
    allowed = {"protocol.json", "fixture.json", "model-provenance.json", "calibration.json",
               "rows.jsonl", "events.jsonl", "blocks.json", "summary.json", "status.json", "report.md"}
    require(isinstance(manifest, dict) and set(manifest) == allowed,
            "Unexpected artifact allowlist; update the recorder deliberately for a new run schema")
    raw = {name: regular_bytes(run / name) for name in sorted(manifest)}
    for name, content in raw.items():
        require(digest(content) == manifest[name], f"Artifact hash mismatch: {name}")
    raw["artifact-sha256.json"] = manifest_bytes
    protocol = json.loads(raw["protocol.json"])
    source = protocol["source_sha256"]
    require(isinstance(source, dict) and source, "Missing frozen benchmark source hashes")
    # Resolve only an object ID; later Git commands never reinterpret user text.
    commit = os.fsdecode(git(root, "rev-parse", "--verify", "--end-of-options",
                            args.source_commit + "^{commit}")).strip()
    for name, expected in source.items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts and "\\" not in name,
                "Unsafe frozen source path")
        require(digest(git(root, "show", f"{commit}:{name}")) == expected,
                f"Source commit does not match frozen hash: {name}; use the original benchmark commit")
    validator = root / "scripts" / "validate_p100_results.py"
    # Audit the exact byte snapshot that will be archived, not mutable input paths.
    with tempfile.TemporaryDirectory(prefix="kvpilot-record-audit-") as temporary:
        snapshot = Path(temporary)
        for name, content in raw.items():
            (snapshot / name).write_bytes(content)
        checked = subprocess.run([sys.executable, str(validator), str(snapshot)],
                                 check=True, stdout=subprocess.PIPE)
        validation = json.loads(checked.stdout)
        require(validation.get("valid") is True, "Independent archive validation failed")

    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, content in sorted(raw.items()):
                member = tarfile.TarInfo("run/" + name)
                member.size, member.mode, member.mtime = len(content), 0o644, 0
                archive.addfile(member, io.BytesIO(content))
    archive_bytes = buffer.getvalue()
    record = {
        "schema_version": 1, "record_id": args.record_id,
        "evidence_type": protocol["evidence_type"], "run_created_utc": protocol["created_utc"],
        "device_name": protocol["device_name"], "source_commit": commit,
        "source_commit_basis": "Supplied by recorder; all frozen source hashes verified against Git commit. The benchmark protocol did not capture a Git commit automatically.",
        "source_sha256": source,
        "protocol_sha256": digest(raw["protocol.json"]),
        "original_artifact_manifest_sha256": digest(manifest_bytes),
        "archive_sha256": digest(archive_bytes),
        "validator_sha256": digest(regular_bytes(validator)),
        "recorder_sha256": digest(regular_bytes(root / "scripts" / "record_p100_results.sh")),
        "scope": "Audited sequential reference-runner evidence; not serving throughput, independent quality certification, or proof of physical execution.",
    }
    files = {"run.tar.gz": archive_bytes, "record.json": json_bytes(record),
             "validation.json": json_bytes(validation), "report.md": raw["report.md"]}
    files["record-sha256.json"] = json_bytes({name: digest(content) for name, content in sorted(files.items())})
    results = root / "results"
    require(not results.is_symlink(), "results must not be a symlink")
    results.mkdir(exist_ok=True)
    destination = results / args.record_id
    if destination.exists() or destination.is_symlink():
        require(destination.is_dir() and not destination.is_symlink(), "Record destination is not a directory")
        require({p.name for p in destination.iterdir()} == set(files),
                "Existing record has different files; choose a fresh record-id")
        for name, content in files.items():
            require(regular_bytes(destination / name) == content,
                    f"Existing record differs: {name}; refusing to overwrite")
        print(f"Verified equivalent existing record: {destination}")
    else:
        destination.mkdir()  # No exist_ok: do not overwrite an existing record.
        for name, content in files.items():
            with (destination / name).open("xb") as output:
                output.write(content)
        print(f"Recorded: {destination}")
    print(json.dumps(validation, indent=2))
    if not args.push:
        print("No commit or push performed. Add --push to publish only this record on a new branch.")
        return

    relative = destination.relative_to(root).as_posix()
    already = subprocess.run(["git", "-C", str(root), "show", f"HEAD:{relative}/record-sha256.json"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if already.returncode == 0 and already.stdout == files["record-sha256.json"]:
        print("This exact record is already present at HEAD; no new branch or push performed.")
        return
    branch = "results/" + args.record_id
    holder = Path(tempfile.mkdtemp(prefix="kvpilot-record-push-"))
    checkout = holder / "checkout"
    try:
        git(root, "worktree", "add", "-b", branch, str(checkout), "HEAD", capture=False)
        target = checkout / relative
        target.mkdir(parents=True, exist_ok=False)
        for name, content in files.items():
            with (target / name).open("xb") as output:
                output.write(content)
        git(checkout, "add", "--", relative, capture=False)
        git(checkout, "commit", "-m", f"Record audited benchmark {args.record_id}", capture=False)
        git(checkout, "push", "--set-upstream", args.remote, f"HEAD:refs/heads/{branch}", capture=False)
        git(root, "worktree", "remove", str(checkout), capture=False)
        holder.rmdir()
        print(f"Pushed {branch}; original checkout, branch and index were not changed.")
    except Exception:
        print(f"Publication did not complete. Any temporary checkout is retained at {checkout}; original checkout was not switched.", file=sys.stderr)
        raise


try:
    main()
except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
    print(f"record_p100_results: {error}", file=sys.stderr)
    sys.exit(1)
PY
