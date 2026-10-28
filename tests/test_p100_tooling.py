"""Offline tests: no model weights, CUDA runtime or network required."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


download = load_script("download_p100_model")
preflight = load_script("p100_preflight")


class OfflineOpener:
    def __init__(self):
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        name = request.full_url.rsplit("/", 1)[1]
        content = b"fixture, not actual model weights"
        if name == "config.json":
            content = json.dumps({"model_type": "qwen2", "hidden_size": 896, "num_hidden_layers": 24}).encode()
        return io.BytesIO(content)


def make_model(tmp_path):
    target = tmp_path / "model"
    opener = OfflineOpener()
    result = download.download_model(target, opener=opener)
    return target, opener, result


def test_fixed_revision_exact_allowlist_and_offline_hash_check(tmp_path, monkeypatch):
    target, opener, result = make_model(tmp_path)
    assert result["ok"] and result["network_used"]
    assert set(path.name for path in target.iterdir()) == set(download.FILES) | {download.MANIFEST}
    assert len(opener.requests) == len(download.FILES)
    for request in opener.requests:
        assert request.full_url.startswith(f"https://huggingface.co/{download.REPO}/resolve/{download.REVISION}/")
        assert "Authorization" not in request.headers
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: pytest.fail("offline check used network"))
    report = download.check_model_dir(target)
    assert report["ok"] and not report["network_used"]
    assert report["files_checked"] == 6
    assert report["total_bytes"] == sum((target / name).stat().st_size for name in download.FILES)


def test_download_does_not_overwrite_existing_directory(tmp_path):
    target, _, _ = make_model(tmp_path)
    with pytest.raises(FileExistsError):
        download.download_model(target, opener=OfflineOpener())


@pytest.mark.parametrize("mutation", ["bytes", "revision", "allowlist", "symlink", "missing"])
def test_offline_handoff_rejects_tampering(tmp_path, mutation):
    target, _, _ = make_model(tmp_path)
    manifest_path = target / download.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    if mutation == "bytes":
        (target / "model.safetensors").write_bytes(b"changed")
    elif mutation == "revision":
        manifest["revision"] = "main"
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "allowlist":
        manifest["files"]["../outside"] = manifest["files"].pop("LICENSE")
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "symlink":
        weights = target / "model.safetensors"
        weights.unlink()
        weights.symlink_to(target / "LICENSE")
    else:
        manifest_path.unlink()
    with pytest.raises(download.ModelFilesError):
        download.check_model_dir(target)


def test_download_requires_explicit_mode(tmp_path):
    with pytest.raises(SystemExit) as error:
        download.main(["--model-dir", str(tmp_path / "model")])
    assert error.value.code == 2


@pytest.mark.parametrize("name", ["../outside", "/absolute/file", "..\\outside", "%2e%2e%2foutside", "model.safetensors\u0000", "\u202emodel.safetensors"])
def test_manifest_names_cannot_redirect_file_reads(tmp_path, name, monkeypatch):
    target, _, _ = make_model(tmp_path)
    path = target / download.MANIFEST
    manifest = json.loads(path.read_text())
    manifest["files"][name] = manifest["files"].pop("model.safetensors")
    # JSON escapes are decoded before checking the exact filename allowlist.
    path.write_text(json.dumps(manifest, ensure_ascii=True))
    monkeypatch.setattr(download, "fingerprint", lambda *_: pytest.fail("must reject names before reading any model file"))
    with pytest.raises(download.ModelFilesError, match="allowlist"):
        download.check_model_dir(target)


def test_check_only_cli_reports_offline_and_does_not_create_files(tmp_path, monkeypatch, capsys):
    target, _, _ = make_model(tmp_path)
    before = {path.name: path.stat().st_mtime_ns for path in target.iterdir()}
    monkeypatch.setattr(download, "download_model", lambda *_: pytest.fail("check-only attempted download"))
    assert download.main(["--check-only", "--model-dir", str(target)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "offline_check" and report["network_used"] is False
    assert before == {path.name: path.stat().st_mtime_ns for path in target.iterdir()}


@pytest.mark.parametrize("extra", ["model.safetensors.index.json", "other.safetensors", "remote_code.py", "unexpected-directory"])
def test_manifest_cannot_pass_with_untracked_loader_inputs(tmp_path, extra, monkeypatch):
    target, _, _ = make_model(tmp_path)
    if extra == "unexpected-directory":
        (target / extra).mkdir()
    else:
        (target / extra).write_text("{}")
    monkeypatch.setattr(download, "fingerprint", lambda *_: pytest.fail("must reject extra entries before reading model files"))
    with pytest.raises(download.ModelFilesError, match="unexpected_directory_contents"):
        download.check_model_dir(target)


@pytest.mark.parametrize("url", ["http://huggingface.co/file", "https://example.com/file", "https://huggingface.co.evil.example/file", "https://user:password@huggingface.co/file"])
def test_redirect_restriction(url):
    with pytest.raises(download.ModelFilesError):
        download.SafeRedirect().redirect_request(None, None, 302, "", {}, url)


def fake_torch(*, available=True, capability=(6, 0), arches=("sm_60",), name="Tesla P100-PCIE-16GB"):
    cuda = SimpleNamespace(is_available=lambda: available, get_device_name=lambda _: name,
                           get_device_capability=lambda _: capability, get_arch_list=lambda: arches,
                           get_device_properties=lambda _: SimpleNamespace(total_memory=16 * 1024**3))
    return SimpleNamespace(__version__="2.7.1+cu126", version=SimpleNamespace(cuda="12.6"), cuda=cuda)


def test_p100_preflight_requires_runtime_sm60_and_actual_arithmetic(monkeypatch):
    calls = []
    monkeypatch.setattr(preflight, "arithmetic_smoke", lambda torch, device: calls.append(device) or {"fp16_matmul": "passed"})
    report = preflight.run_preflight(torch_module=fake_torch())
    assert report["ok"] and report["p100_ready"] and calls == ["cuda"]
    bad = preflight.run_preflight(torch_module=fake_torch(arches=("sm_75",)))
    assert not bad["ok"] and "missing_sm_60" in bad["error"]
    assert calls == ["cuda"]


def test_preflight_detects_missing_cuda_and_labels_other_devices(monkeypatch):
    monkeypatch.setattr(preflight, "arithmetic_smoke", lambda *_: {})
    assert not preflight.run_preflight(torch_module=fake_torch(available=False))["ok"]
    other = fake_torch(capability=(8, 6), arches=("sm_86",), name="RTX 3090")
    assert not preflight.run_preflight(torch_module=other)["ok"]
    report = preflight.run_preflight(allow_other_cuda=True, torch_module=other)
    assert report["ok"] and not report["p100_ready"]
    cpu = preflight.run_preflight("cpu", torch_module=fake_torch())
    assert cpu["ok"] and not cpu["p100_ready"]


def test_arithmetic_failure_cannot_report_ready_or_expose_exception_data(monkeypatch):
    def fail(*_):
        raise RuntimeError("signed URL or secret that must not enter reports")
    monkeypatch.setattr(preflight, "arithmetic_smoke", fail)
    report = preflight.run_preflight(torch_module=fake_torch())
    assert not report["ok"] and not report["p100_ready"]
    assert report["stage"] == "arithmetic" and report["error"] == "RuntimeError"
