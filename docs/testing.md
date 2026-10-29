# Regression testing

The evidence job now also runs `python scripts/verify_policy_replication_record.py` against the [paired P100/RTX record](../results/policy-replication-2026-10-29/README.md). It validates bounded archives, source anchors, eight completed runs and recomputed descriptive results without a GPU. It does not enforce historical speed ratios as noisy performance-test thresholds. Corruption, path traversal, unsafe archive entries and refreshed-hash source forgery are covered by `tests/test_policy_replication_record.py`.

Every pull request and push runs `.github/workflows/ci.yml` on GitHub-hosted
Ubuntu CPUs. No model download, GPU rental, CUDA installation or repository secret
is required. Dependency installation does access the package indexes.

## What the checks cover

| Check | Environment | Purpose |
| --- | --- | --- |
| Core | Python 3.11 and 3.13, NumPy 2.2.6, pytest 8.4.2 | Run the full available test suite, including policy, accounting, codecs, validators and offline tooling. Optional tensor-engine tests may skip here. |
| Owned runner / CPU Torch | Python 3.11, CPU PyTorch 2.7.1 and the pinned runner dependencies | Run the suite with actual tiny tensor models, including native cache restore, comparison with independent math, both benchmark timing modes and audit rejection tests. |
| Recorded evidence audit | Python 3.11 and the core dependencies | Check explicitly registered committed benchmark fixtures and records with independent validators, and recompute measured-cost comparison summaries. It does not rerun model inference. |
| Regression gate | Aggregate check | Pass only when all of the above jobs succeed. A failed, cancelled or skipped dependency does not produce a passing gate. |

The CPU runner explicitly imports PyTorch, psutil, safetensors and tokenizers
before pytest. A missing or broken dependency therefore fails that job rather
than silently hiding the tensor tests behind `pytest.importorskip`. Apple
MLX/Metal tests intentionally skip on these Linux jobs.

JUnit results, the installed dependency lists and the evidence-audit JSON are
uploaded as workflow artifacts for 14 days, including after test failures when
the files exist. A job that fails before tests start may have no JUnit file;
inspect its log. Superseded runs are cancelled to limit wasted work. There are no
path filters, so documentation-only pull requests still receive the gate.

## Run locally

From the repository root, create an isolated core environment:

```bash
python3.11 -m venv .venv-ci
.venv-ci/bin/python -m pip install -r requirements-ci.txt
.venv-ci/bin/python -m pip install --no-deps -e .
.venv-ci/bin/python -m pip check
.venv-ci/bin/python -m pytest -q -ra --junitxml=artifacts/local-core-tests.xml
.venv-ci/bin/python scripts/check_regression_evidence.py --output artifacts/local-evidence-audit.json
```

To exercise the CPU tensor runner on Linux, add its dependencies to that
environment and rerun the suite:

```bash
.venv-ci/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.venv-ci/bin/python -m pip install -r experiments/torch/requirements.txt
.venv-ci/bin/python -m pip check
.venv-ci/bin/python -c 'import torch, psutil, safetensors.torch, tokenizers; from experiments.torch.qwen_runner import QwenRunner; print(torch.__version__)'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-ci/bin/python -m pytest -q -ra --junitxml=artifacts/local-torch-tests.xml
```

On macOS, install `torch==2.7.1` from the default PyPI index instead of the
Linux CPU wheel index. These tiny tests still use CPU tensors; they do not need
the local Qwen weights or Ollama. To check the other core version, use
`python3.13 -m venv .venv-ci-313` and substitute `.venv-ci-313` in the core commands.

`requirements-ci.txt` pins direct dependencies for a portable test environment.
It is not a complete, hash-verified lockfile. The workflow saves `pip freeze` so
changes in transitive dependencies can be diagnosed. The older
`requirements-dev.lock.txt` is a snapshot of a particular macOS environment and
is not the Linux CI install source.

## Maintaining the gate

A repository maintainer should require the **Regression gate** check in the
target branch's GitHub ruleset or branch protection after the first successful
workflow run. Merely adding this workflow does not prevent merging a red pull
request; the required-check setting is separate and is not changed by this PR.
Use the stable aggregate name instead of enumerating the matrix jobs. If a
merge queue is introduced, add its `merge_group` trigger before requiring this
workflow for queued merges.

Actions are pinned to full commits verified against their official release
pages: [checkout v7.0.1](https://github.com/actions/checkout/commit/3d3c42e5aac5ba805825da76410c181273ba90b1),
[setup-python v7.1.0](https://github.com/actions/setup-python/commit/6d4fd22b20465ccc0f7354476588e8df7c9ca0e0),
and [upload-artifact v7.0.2](https://github.com/actions/upload-artifact/commit/cf430e030ddbb5b0abf93d22962f4752f3646cd9).
Update pins deliberately and run the same suite. The workflow uses read-only
repository permissions and does not retain checkout credentials or use
`pull_request_target`.

When adding a bug fix, prefer a test that reproduces the faulty behavior and
passes only with the fix. Keep fixtures small and deterministic. Add new
benchmark evidence to the explicit audit registry; do not turn validation
failures into skips or regenerate historical measurements to make tests pass.

## What passing CI does not establish

These checks guard functionality and evidence accounting. They do not prove
P100 or RTX-3090 compatibility, real-model answer quality, speedups, GPU-memory
savings, serving throughput or customer cost reduction. Hosted CPU timings are
not stable performance baselines, so CI does not assert wall-clock speed ratios.
The P100 experiment and its independent result validator remain necessary for
hardware and model claims. MLX/Metal arithmetic still requires a Mac check.
