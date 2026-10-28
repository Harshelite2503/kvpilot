# Admission, equal-work timing and break-even experiment

This follows the [first P100 result](p100-results-2026-10-28.md). It adds the missing simple baseline and removes answer-length differences from a separate timing control. It does not claim a speedup before the new P100 run. All live K/V tensors, codecs, retention, eviction and restoration remain ours; Torch is the numerical engine.

## What changes

Six policies run against the same capacity, case order and resolution schedule:

- `recompute`: save nothing; rebuild the prefix when the session returns.
- `raw_native`: save native state and evict oldest snapshots when necessary.
- **`raw_admit_if_fits`**: save native state only when the entire serialized snapshot fits unused capacity; otherwise recompute. Never evict, compress or consult predicted future reuse.
- `group_int4`: the existing fixed owned INT4 reference, including its known limitations.
- `adaptive_exact`: the existing calibrated selector without eviction-loss pricing.
- `adaptive_eviction`: the existing selector including predicted eviction loss.

Every saved snapshot includes framing in the budget. A refused admission performs no cache export or encoding. Admission planning remains timed. Canceled and evicted admissions remain charged. This tests whether adaptive decisions add value beyond a cheap conservative policy.

Two independent runs answer different questions:

| Run mode | Inputs after resume | What to read |
|---|---|---|
| `equal_work` | Same frozen continuation token IDs and number of prediction steps for every policy | Full-session time, restore/recompute time, bytes, allocator peaks and ratio versus admit-if-fits |
| `free_generation` | Normal greedy generation, including normal stopping | Actual answers, strict scores, token agreement, lengths, truncations and observed session time |

For equal work, a separate native reference prepares `--max-tokens` continuation IDs before calibration and measured blocks, deliberately continuing beyond EOS if needed. The fixture records the IDs and preparation time. Each measured response processes the suffix and the first N−1 fixed targets to obtain N predictions. Predictions never choose the next input or shorten the run. The final target is not appended to state. This is a computational control, not an answer; quality fields and output-token counts are null and `forced_token_steps` is separate. Reference preparation is excluded from timed blocks and disclosed in the fixture. Do not compare absolute times across the two modes as though they execute the same work.

Jobs are shuffled within each run. All repetitions and policies share the frozen fixture. The protocol uses schema 2; the independent validator continues to support the historical five-policy schema 1 records. It audits fixed-work inputs and timing aliases, rejects manufactured quality scores, and checks the exact admission rule against reconstructed residency.

These are generated development tasks with a new seed, not a secret holdout or customer traces. Repeating 24 questions does not create thousands of independent quality samples. TTL, idleness ranking and concurrent serving are not implemented by this change.

## Exact commands on the existing P100 VM

The VM already has the pinned CUDA 12.6 Torch environment and the downloaded model from the earlier run. Keep those. The following Git commands fetch the new branch without resetting local files. If Git reports conflicting edits, preserve them before switching; do not force checkout.

```bash
cd /home/nutanix/kvpilot-p100
git fetch origin refs/heads/feat/p100-admission-equal-work:refs/remotes/origin/feat/p100-admission-equal-work
if git show-ref --verify --quiet refs/heads/feat/p100-admission-equal-work; then git switch feat/p100-admission-equal-work; else git switch --no-track -c feat/p100-admission-equal-work refs/remotes/origin/feat/p100-admission-equal-work; fi
git merge --ff-only refs/remotes/origin/feat/p100-admission-equal-work
git status --short
```

The explicit fetch destination also works with the VM's original `--single-branch` clone. Each displayed line is a complete command. The branch-exists check makes the checkout safe to repeat without resetting an existing branch.

Run the complete suite. This is one command, on one line, to avoid the earlier pasted-newline problem:

```bash
bash scripts/run_p100_admission_suite.sh --model-dir data/qwen25-05b-pinned
```

The script uses `.venv-p100/bin/python`, performs no downloads or package installs, and stops at the first failed command, even through `tee`. It checks the P100 environment and offline model hashes, runs tests, runs and audits both smoke modes, runs and audits both full modes, performs the analytical sweep on the full equal-work run, and creates a compressed result bundle. It refuses to overwrite results and requires clean tracked files.

The full configuration is 512/2048 context targets, 24 distinct questions, three repeats, groups of eight, all three scenarios, 24 continuation steps and seed `20261102`. Each full mode contains 2,592 started sessions and 108 blocks. There are two full runs, so allow more time than the old single-mode suite. No completion-time estimate or speedup is promised.

If you only want readiness checks first, use a fresh output directory:

```bash
bash scripts/run_p100_admission_suite.sh --model-dir data/qwen25-05b-pinned --smoke-only
```

The smoke sweep is labeled by its actual source and does not replace the full experiment. A complete suite already includes smoke, so this separate command is optional.

If the interpreter or weights are missing, follow [the pinned setup and model handoff guide](p100-runner.md). Do not substitute an unpinned CUDA wheel on the P100. The suite does not access the network, and no remote access to the protected VM is needed.

## Result files to return

At completion, the script prints `Complete. Send back: /absolute/path/to/results/local-p100-admission-TIMESTAMP.tar.gz`. Send that entire archive. To find the most recent suite's path:

```bash
cd /home/nutanix/kvpilot-p100
cat results/local-last-admission-directory.txt
```

The bundle contains:

- `source-commit.txt`, `git-status.txt`, preflight, offline model check and test logs/XML.
- `smoke-equal_work/` and `smoke-free_generation/`, plus independent validation JSON and logs.
- `full-equal_work/` and `full-free_generation/`, plus independent validation JSON and logs (except in smoke-only mode).
- Each benchmark directory's protocol, fixture, calibration, rows, events, block measurements, summary, report, status and artifact hashes.
- `break-even/` and its log: modeled sensitivity results and provenance.

If a command fails, keep the partial directory and send its log/status files. Do not present an incomplete directory as a completed measurement. Raw artifacts contain the synthetic prompts/answers and local environment paths; they do not include model weights or credential files.

## Individual commands, if diagnosing a failure

Use unused output paths. These commands are alternatives to the suite, not required additional runs:

```bash
.venv-p100/bin/python -u scripts/benchmark_p100_kvpilot.py --model-dir data/qwen25-05b-pinned --device cuda --output results/local-admission-equal --timing-mode equal_work --contexts 512 2048 --cases 24 --repeats 3 --group-size 8 --scenarios roomy pressure cancellation --max-tokens 24 --chunk-size 128 --seed 20261102 --allow-unvalidated-lossy
.venv-p100/bin/python scripts/validate_p100_results.py results/local-admission-equal --output results/local-admission-equal-validation.json
.venv-p100/bin/python -u scripts/benchmark_p100_kvpilot.py --model-dir data/qwen25-05b-pinned --device cuda --output results/local-admission-quality --timing-mode free_generation --contexts 512 2048 --cases 24 --repeats 3 --group-size 8 --scenarios roomy pressure cancellation --max-tokens 24 --chunk-size 128 --seed 20261102 --allow-unvalidated-lossy
.venv-p100/bin/python scripts/validate_p100_results.py results/local-admission-quality --output results/local-admission-quality-validation.json
.venv-p100/bin/python scripts/sweep_p100_break_even.py --run-dir results/local-admission-equal --output results/local-admission-sweep
```

## Reading the break-even sweep

The sweep audits its input archive, then reruns the decision logic under hypothetical codec costs. Measured export/import and native serialization costs remain charged. Only lossy codec encode/decode costs are scaled from zero to their measured values; zero codec work does **not** mean free transfers or storage. A modeled sequential clock accounts for execution, not real agent tool/human waiting times.

Use the sweep to ask: *Would caching be worthwhile even if codec compute were much cheaper, given this workload and the measured native path?* Every admission is paid for, including those later canceled or evicted. Recompute and simple admission remain comparison arms. The report gives tested winning regions and adjacent sampled transition brackets rather than pretending to identify an exact continuous threshold.

All sweep durations and savings are **modeled**, even when an input cost came from a GPU experiment. This does not execute a faster codec, change measured results, prove lossy answer preservation, model concurrent transfer contention, or estimate a customer bill. A candidate region needs another live test. A result on this eager 0.5B runner does not establish modern serving-engine economics.

Read equal-work speed versus **admit-if-fits and native**, then check memory and the free-generation answers. If the simple rule matches or beats adaptive, retain that finding. Our existing weak fixed INT4 baseline is not an adequate sole competitor. The business outcome still requires a later concurrent customer deployment that does the same useful work with fewer billed resources or greater capacity at the same cost.
