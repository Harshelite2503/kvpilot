# Preserved P100 and RTX policy replication

This record preserves the completed P100 admission suite and the third RTX rental's completed suite. The first RTX attempt's timing-only evidence is retained separately. No observations from that incomplete attempt are pooled with the successful run.

Read the [findings and limitations](../../docs/policy-replication-2026-10-29.md), the [HTML comparison](report.html) or the [next product experiment](../../docs/product-plan-2026-10-29.md).

## Offline verification

From the repository root, with Python 3.11 or newer:

```bash
mkdir -p artifacts
python3 scripts/verify_policy_replication_record.py \
  --output artifacts/policy-replication-audit.json
```

No GPU, model weights, tensor engine, network or historical Git checkout is needed. The verifier checks this directory's exact file membership and SHA256 values, bounded archive extraction, all eight smoke/full runs, the frozen benchmark source-map anchor, prompt equivalence and independently recomputed accounting/scores. It prints descriptive timing/memory/quality totals and records the known cross-system continuation difference. The required **Regression gate** includes this check through the evidence CI job.

The source anchor matches the 25 benchmark files at `ca6f6feaaf7faef4b942bd37d78ee29ded99a27a`. The P100 labels `b91ebb37ead55d21dabe23ef0311ae2a9d2dc697`; those benchmark files are identical there. The audit is a consistency check, not a signature, physical-execution attestation or performance acceptance threshold.

## Contents

| Path | Meaning |
|---|---|
| `p100/run.tar.gz` | Exact preserved P100 archive: two smoke and two full modes, sweep, tests and provenance. Its single root is `local-p100-admission-20261028T180805Z`. |
| `rtx5060ti/run.tar.gz` | Exact downloaded third-attempt RTX archive, rooted at `local-rtx5060ti-replication`, including runtime metadata and the enclosing suite manifest. |
| `analysis/` | Descriptive summaries and balanced-mixture comparison. Absolute analysis-workspace paths were normalized for publication; underlying measurements were not changed. |
| `reviews/` | Independent timing, quality and source/control reviews; analysis-workspace paths normalized. These are recorded reviews, while the verifier above recomputes the core facts. |
| `p100/import-receipt.json` | Hashes of the 64 files received from the user's P100 folder; not an original VM-side signature or enclosing manifest. |
| `rtx5060ti/download-verification.json` | Original download checksum verification with analysis-workspace paths normalized. |
| `history/` | First-attempt timing archive, later offline modeled sweep, and complete attempt-status/billing context. Not part of the paired result. |
| `analysis-source/` | Byte-preserved historical calculation scripts. Their old workspace defaults are historical provenance, not supported CLI entry points. |
| `manifest.json` | SHA256 and exact membership for every record file except the manifest itself. |

The raw archives are unchanged, including original VM paths and the RTX suite's nested checkpoint archives. Model weights, account API keys, SSH private keys, private rental orchestration state, virtual environments and Python caches are not published. Separate local checkpoint copies are not duplicated. Portable analysis entry points are under `scripts/`; the historical snapshot scripts retain the original calculation bytes referenced by review digests.

Full raw archive SHA256 values:

- P100: `d7c8f421bd69497ac349447aa82308c9c43988a18b63557c88d1311837218b74`
- RTX: `225d559971a790b01f9dea6f85c1162dc7a50bf2066d95ea6afbd23b88673c53`

## Recompute a detailed summary

Run the verifier first. Then unpack the now-verified archives into an empty local directory and analyze them:

```bash
mkdir -p artifacts/policy-inputs
tar -xzf results/policy-replication-2026-10-29/p100/run.tar.gz -C artifacts/policy-inputs
tar -xzf results/policy-replication-2026-10-29/rtx5060ti/run.tar.gz -C artifacts/policy-inputs
python3 scripts/analyze_policy_replication.py \
  artifacts/policy-inputs/local-p100-admission-20261028T180805Z \
  --output artifacts/p100-analysis.json
python3 scripts/analyze_policy_replication.py \
  artifacts/policy-inputs/local-rtx5060ti-replication \
  --output artifacts/rtx-analysis.json
```

The analyzer refuses to overwrite an existing output. Generated timestamps and paths will differ from the recorded reviews; measurements should not. `scripts/review_policy_replication_timing.py --help` describes the deeper timing/decision review; it additionally requires the recorded Git objects locally to load their exact validators.

For a future VM experiment, `scripts/p100-runtime-sidecar.py --help` describes runtime capture and `scripts/run-matching-p100.sh --help` describes the pinned reproduction handoff. The latter deliberately checks out the historical comparison commit and runs its unchanged suite. **It is not required to analyze these results or to begin the next experiment.** It installs no packages or model weights. Preserve the helper beside the handoff script when copying them to a VM.

## Interpretation

Adaptive loses to simple admission in the balanced workload on both systems, selects no lossy codec, and reduces no measured GPU peak. Quality observations repeat 24 questions. CPU/driver/runtime settings are not held equal across machines, and one forced continuation differs after EOS. This record supports within-machine policy comparisons; it certifies neither generic answer preservation nor customer savings.
