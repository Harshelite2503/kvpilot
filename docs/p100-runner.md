# Owned PyTorch experiment on a P100

This is a reproducible setup and execution protocol, **not a P100 result**.
No P100 timing, memory saving or answer-quality improvement has been measured
by preparing these files. The earlier RTX 3090 first/repeated-request ratio
measured native vLLM behavior; it was not a KVPilot speedup.

The exact new code completed a [real Qwen Mac GPU smoke run](../results/p100-readiness-mac-smoke-v1/report.md):
40 sessions, five policies, two budgets, four distinct questions and one repeat.
Native restoration reproduced continuation logits bit-for-bit, and the independent
archive audit passed. Fixed INT4 scored 1/4 versus recompute/native/adaptive at 2/4
in each scenario; it changed the correct arithmetic answer `11` to `13`.
Adaptive was faster with roomy storage and slower under pressure, using more
stored bytes than fixed INT4 in both. These small, single-repeat MPS timings
are readiness evidence with possible warmup/order effects, not a P100 result
or a statistically established performance win. All answers and raw measurements
are retained; the two baseline failures also remain in the denominator.

The runner uses PyTorch for tensor operations, `safetensors` to read weights,
and `tokenizers` to tokenize text. KVPilot owns the model's live K/V tensors,
snapshot framing, codec selection, paused payload store, eviction and restore.
It does not use Ollama, vLLM, Transformers cache classes or an external cache
manager. Model weights remain external and are never committed.

## Hardware and environment

The [Tesla P100 has compute capability 6.0](https://developer.nvidia.com/cuda/gpus/legacy).
[vLLM 0.32.0 requires capability 7.5 or newer](https://docs.vllm.ai/en/v0.32.0/getting_started/installation/gpu/),
so installing that vLLM release is not the P100 route. A 3090 can be used for
a later, separate serving-engine integration test.

Use Linux x86_64 and Python 3.11 for this pinned environment. PyTorch 2.7.1's
[CUDA 12.6 build configuration includes `6.0`](https://github.com/pytorch/pytorch/blob/v2.7.1/.ci/manywheel/build_cuda.sh).
The CUDA 12.8 build differs; do not substitute it or install an unpinned latest
CUDA wheel. The host must already provide a working NVIDIA driver compatible
with CUDA 12.6. `nvidia-smi` reports the driver; its CUDA version field alone
does not show which wheel or architectures PyTorch contains.

From the repository root, on a machine where package installation is allowed:

```bash
python3.11 -m venv .venv-p100
source .venv-p100/bin/activate
python -m pip install -r experiments/torch/requirements-p100.txt --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r experiments/torch/requirements.txt
python -m pip install --no-deps -e .
nvidia-smi
python scripts/p100_preflight.py --output p100-preflight.json
```

The PyTorch install uses the [official CUDA 12.6 index and release pin](https://pytorch.org/get-started/previous-versions/#v271).
The second requirements file pins NumPy 2.2.6, tokenizers 0.21.2,
safetensors 0.5.3 and psutil 7.0.0. These are top-level pins, not a complete
transitive lock. Tokenizers may install Hugging Face Hub as a dependency;
the runner does not invoke its downloader. For an installation without network access, prepare the
same Python/Linux/CUDA wheel set on an allowed machine and install from a
local wheel directory with `--no-index --find-links`; include transitive
dependencies and the build dependencies needed for the editable package.
Do not copy a macOS virtual environment onto Linux.

The preflight reports the actual device, capability, compiled architectures,
PyTorch version and CUDA runtime. On P100 it requires `sm_60`, then executes
FP16 matrix multiplication and explicitly selected **math SDPA** against
small FP32 CPU references. Failure exits nonzero with a named stage. A pass
establishes these operations work, not model correctness or performance.
`--device cpu` and `--device mps` are local diagnostics and always report
`p100_ready: false`. `--allow-other-cuda` explicitly permits a different CUDA
GPU without labeling it a P100.

## Obtain and hand off weights

The allowed model is [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
pinned to [revision `7ae557604adf67be50417f59c2c2f167def9a775`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/commit/7ae557604adf67be50417f59c2c2f167def9a775).
Use its standard safetensors weights; the Mac's unrelated Qwen3.5 MLX/NVFP4
export is not an input to this runner. The published source weights are
BF16; this P100 experiment loads them for FP16 computation and stores native
attention state in FP16. That conversion is part of the tested model
configuration, not evidence of equivalence to a BF16 reference implementation.

On a machine allowed to fetch public weights, run the **explicit download**
command. It transfers approximately 1 GB and requires a new destination:

```bash
python scripts/download_p100_model.py --download --model-dir data/qwen25-05b-pinned
python scripts/download_p100_model.py --check-only --model-dir data/qwen25-05b-pinned
```

Only `config.json`, `generation_config.json`, `tokenizer.json`,
`tokenizer_config.json`, `model.safetensors` and `LICENSE` are fetched from
fixed public HTTPS revision URLs. No login token, remote Python module,
pickle file or automatic model discovery is used. The helper writes
`model-manifest.json` containing revision, file sizes and SHA256 hashes.
It does not overwrite an existing directory. A failed download may leave a
partial destination; inspect it before removing it or choosing another path.
The offline check requires exactly those six files and the manifest, and
rejects added shard indexes or files that could change the runner's inputs.

For a protected VM, transfer that entire directory, including the manifest,
through the permitted file-transfer channel. Then use the offline check and
pass the local absolute path to the benchmark:

```bash
python scripts/download_p100_model.py --check-only --model-dir /approved/models/qwen25-05b-pinned
```

`--check-only` makes no network calls. It detects files changed during handoff
by comparing their hashes with the supplied manifest. The manifest is not an
independently signed authenticity certificate: keep its provenance and the
pinned download tool with the handoff. The benchmark itself accepts local
files and does not fetch missing weights.

## Run a small check, then the experiment

First run the smoke workload into a new output directory:

```bash
python scripts/benchmark_p100_kvpilot.py --model-dir /approved/models/qwen25-05b-pinned --device cuda --output results/local-p100-smoke --smoke --allow-unvalidated-lossy
python scripts/validate_p100_results.py results/local-p100-smoke
```

The intended full comparison is:

```bash
python scripts/benchmark_p100_kvpilot.py --model-dir /approved/models/qwen25-05b-pinned --device cuda --output results/local-p100-v1 --contexts 512 2048 --cases 24 --repeats 3 --group-size 8 --scenarios roomy pressure cancellation --max-tokens 24 --chunk-size 128 --seed 20261029 --allow-unvalidated-lossy
python scripts/validate_p100_results.py results/local-p100-v1
```

Keep the preflight JSON, protocol, calibration, raw records, `summary.json`,
the generated comparative `report.md` and validator output together. The full
command uses groups of eight paused sessions; smoke uses four. A local model download or CPU/Mac smoke run does
not count as executing this P100 experiment. Check the benchmark's `--help`
for the final CLI. The owned runner caps total context at 8192 tokens;
prompts plus reserved generation must fit. For a later longer-context test,
use `--contexts 8000` rather than 8192 to leave room for generation and
prompt framing. The prepared token count is checked before evaluation.

The five arms are recompute, exact native state, fixed INT4, adaptive selection
without eviction pricing, and adaptive selection with eviction pricing.
Adaptive choices use measured calibration costs and resolved history. INT4
stores approximate attention tensors; decode restores full active FP16
tensors. Exact native state must round-trip bit-for-bit. `--allow-unvalidated-lossy`
permits testing, not quality approval; keep incorrect and changed answers in
the reported denominators.

## What this experiment can establish

It can measure the owned snapshot path on the actual named device: pause and
resume work, exact serialized payload bytes under a capacity limit, eviction
behavior and answers on the declared synthetic tasks. It can expose when a
simple fixed policy beats adaptive selection. Preserve calibration costs,
setup work and all negative results; do not count unmeasured transfer or
codec estimates as observed timings.

The store retains whole paused sessions; it does not implement shared-prefix
deduplication, continuous batching, concurrent serving or a production vLLM
connector. Smaller paused payloads do not imply equally smaller total GPU
allocation, and restored FP16 attention still needs its full active memory.
A result on a P100, this small model and this workload cannot establish
3090 serving throughput, general coding quality or customer savings.
