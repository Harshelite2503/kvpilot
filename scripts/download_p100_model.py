#!/usr/bin/env python3
"""Explicit public model download or offline SHA256 handoff verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
FILES = ("config.json", "generation_config.json", "tokenizer.json",
         "tokenizer_config.json", "model.safetensors", "LICENSE")
MANIFEST = "model-manifest.json"
MAX_FILE_BYTES = 2 * 1024**3
CHUNK_BYTES = 1024**2


class ModelFilesError(ValueError):
    pass


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Allow HTTPS redirects only to Hugging Face's download domains."""
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> Any:
        parsed = urllib.parse.urlsplit(newurl)
        host = parsed.hostname or ""
        allowed = host == "huggingface.co" or host.endswith((".huggingface.co", ".hf.co"))
        if parsed.scheme != "https" or parsed.username or parsed.password or not allowed:
            raise ModelFilesError("download_redirect_not_allowed")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fingerprint(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ModelFilesError(f"missing_or_symlink_file: {path.name}")
    size = path.stat().st_size
    if not 0 < size <= MAX_FILE_BYTES:
        raise ModelFilesError(f"invalid_file_size: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_BYTES):
            digest.update(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def check_model_dir(model_dir: Path) -> dict[str, Any]:
    """Verify an offline transfer against its pinned-download manifest, without network."""
    if not model_dir.is_dir() or {path.name for path in model_dir.iterdir()} != set(FILES) | {MANIFEST}:
        # In particular, an extra shard index can redirect the runner to
        # different weights despite an unchanged, correctly hashed main file.
        raise ModelFilesError("unexpected_directory_contents: require exactly the six model files and model-manifest.json")
    manifest_path = model_dir / MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size > 65536:
        raise ModelFilesError("missing_or_invalid_manifest: transfer model-manifest.json with all model files")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise ModelFilesError("invalid_manifest_json") from error
    if not isinstance(manifest, dict) or manifest.get("version") != 1 or manifest.get("repo") != REPO or manifest.get("revision") != REVISION:
        raise ModelFilesError("manifest_repository_or_revision_mismatch")
    records = manifest.get("files")
    if not isinstance(records, dict) or set(records) != set(FILES):
        raise ModelFilesError("manifest_file_allowlist_mismatch")
    for name in FILES:
        expected = records[name]
        if not isinstance(expected, dict) or type(expected.get("bytes")) is not int or not isinstance(expected.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) is None:
            raise ModelFilesError(f"invalid_manifest_record: {name}")
        if fingerprint(model_dir / name) != expected:
            raise ModelFilesError(f"file_hash_or_size_mismatch: {name}")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("model_type") != "qwen2" or config.get("hidden_size") != 896 or config.get("num_hidden_layers") != 24:
        raise ModelFilesError("unexpected_model_config")
    return {"ok": True, "mode": "offline_check", "repo": REPO, "revision": REVISION,
            "files_checked": len(FILES), "total_bytes": sum(record["bytes"] for record in records.values()),
            "verification": "SHA256 matches supplied download manifest; this is not an independent signature verification",
            "network_used": False}


def download_model(model_dir: Path, *, opener: Any = None) -> dict[str, Any]:
    """Download only six data files at one immutable revision. Never reads auth tokens."""
    model_dir.mkdir(parents=True, exist_ok=False)
    if opener is None:
        # No cookies, auth headers, HF SDK, remote Python files or pickle loader.
        opener = urllib.request.build_opener(SafeRedirect())
    records: dict[str, Any] = {}
    for name in FILES:
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
        request = urllib.request.Request(url, headers={"User-Agent": "kvpilot-public-model-download/1"})
        temporary = model_dir / (name + ".partial")
        size = 0
        with opener.open(request, timeout=120) as response, temporary.open("xb") as output:
            while chunk := response.read(CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    raise ModelFilesError(f"download_size_limit: {name}")
                output.write(chunk)
        temporary.rename(model_dir / name)
        records[name] = fingerprint(model_dir / name)
    manifest = {"version": 1, "repo": REPO, "revision": REVISION, "files": records,
                "source": "public HTTPS resolve URLs pinned to revision; hashes calculated after download"}
    (model_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = check_model_dir(model_dir)
    report.update(mode="download_and_check", network_used=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download", action="store_true", help="Explicitly fetch ~1 GB from official public Hugging Face URLs into a NEW directory.")
    mode.add_argument("--check-only", action="store_true", help="Verify an existing manifest and files without any network call.")
    args = parser.parse_args(argv)
    try:
        report = download_model(args.model_dir) if args.download else check_model_dir(args.model_dir)
    except Exception as error:
        # Third-party/network messages can contain signed URLs. Print only our
        # bounded diagnostics or the exception type, never a URL/credential dump.
        detail = str(error) if isinstance(error, ModelFilesError) else type(error).__name__
        print(json.dumps({"ok": False, "error": detail, "mode": "download" if args.download else "offline_check",
                          "hint": "Use a new destination for downloads; transfer all six files plus model-manifest.json for offline checks."}))
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
