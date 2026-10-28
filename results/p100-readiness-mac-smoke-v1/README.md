# P100 code readiness: real Mac GPU smoke, not P100 results

This archive ran the new owned PyTorch runner on the local Apple M1 Pro GPU
(`mps`), using the pinned Qwen2.5-0.5B-Instruct standard weights converted to
FP16. It did not use the remote P100, a 3090, Ollama or an external cache manager.

The [frozen protocol](protocol.json), [calibration](calibration.json),
[raw sessions](rows.jsonl), [event accounting](events.jsonl) and
[summary](summary.json) preserve 40 sessions: five policies, two budgets,
four distinct generated questions and one repetition. See [the readable table](report.md).

Native restoration produced byte-identical continuation logits, maximum
absolute difference zero, after the original live cache was released.
The [independent audit](validation.json) checked all 40 rows, 80 events and
10 blocks. Hashes in `artifact-sha256.json` bind the artifacts created by the
benchmark. This README, audit output, preflight and JUnit files were added
after completion and are supplementary; they are not part of that original
artifact hash manifest.

Recompute, native and adaptive scored 2/4 in each scenario; fixed INT4 scored
1/4. The arithmetic task expected `11`; the exact path returned `11`, while
INT4 returned `13`. The other two baseline failures are retained. Adaptive
was faster than INT4 with roomy storage but slower under pressure, and used
more retained payload bytes in both scenarios. No policy is declared a winner.
Single-repeat timings can include warmup/order effects. Four correlated tasks
cannot establish broad answer quality, and these results cannot be substituted
for the planned P100 experiment.

Commands run from the repository root:

```bash
.venv-torch/bin/python scripts/benchmark_p100_kvpilot.py --model-dir data/qwen2.5-0.5b-instruct --device mps --output results/p100-readiness-mac-smoke-v1 --smoke --allow-unvalidated-lossy
.venv-torch/bin/python scripts/validate_p100_results.py results/p100-readiness-mac-smoke-v1 --output results/p100-readiness-mac-smoke-v1/validation.json
```

The CPU suite passed 420 tests plus 89 subtests, with four accelerator/optional
module skips. The separate Mac GPU runner test result is saved in
`tests-torch-mps.xml`; `preflight-mps.json` records successful FP16 arithmetic.
Neither a CPU test nor a Mac GPU check validates P100 compatibility in advance.
Use the [P100 handoff guide](../../docs/p100-runner.md) on the protected VM.
