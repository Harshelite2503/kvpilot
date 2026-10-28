# Mac readiness evidence for the admission experiment

Executed the offline suite helper on MPS using pinned real Qwen2.5-0.5B weights and source commit `af48abc9dec6c3e5f13a37d9f1799f3431f22a55`:

```bash
bash scripts/run_p100_admission_suite.sh --python .venv-torch/bin/python --model-dir data/qwen2.5-0.5b-instruct --device mps --smoke-only --output results/local-admission-mac-v2
```

The first sandboxed attempt reported MPS unavailable and stopped before model execution. Running with Mac GPU access passed preflight. There was no model download or paid GPU use.

- `tests.xml`: 512 tests and 89 subtests passed, 11 skips. The lightweight environment separately passed 431 tests and 89 subtests, with 4 skips.
- `preflight.json`: actual local device/runtime; explicitly `p100_ready: false`.
- `model-check.log`: offline supplied-manifest verification; not an independent signature.
- `sweep.tar.gz`: deterministic archive of the 96-block/eight-scale **counterfactual** sweep on the equal-work smoke, including its input/source provenance and inner hashes. This did not execute any cheaper codec.
- `artifact-sha256.json`: hashes for the preserved evidence files.
- Actual benchmark records: [equal-work](../admission-readiness-mac-equal-work/) and [free-generation](../admission-readiness-mac-free-generation/), each 48 sessions, 96 events and 12 blocks.

The helper completed both independent validations and packaged the full local suite successfully. The new P100 suite has not yet run. Four distinct questions and one repeat only establish readiness. Both wins and losses against simple admission remain in the raw reports. No fleet cost, general quality, or serving-throughput benefit is claimed.
