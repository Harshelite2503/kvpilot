# Derived P100 audit — 28 October 2026

Start with the [human-readable results](../../docs/p100-results-2026-10-28.md).

- `analysis.json`: all policy/context/scenario statistics, paired repeat reductions, stage timings, deduplicated scores and an explicitly unmeasured no-eviction residency calculation.
- `analyze.py`: standard-library reproduction, bound to the full run's exact manifest and source hashes. It reads results; it never runs inference.
- `report.html`: standalone explanatory report. Open locally in a browser.
- `original-report.txt`: original benchmark-generated full table, unchanged.
- `full-validation.json`, `smoke-validation.json`: independent validation performed on the Mac.
- `vm-preflight.json`: the supplied P100 arithmetic/device preflight.
- `vm-tests.xml`: the supplied VM pytest report (421 passed tests, 89 successful subtests and 3 skips; the XML counts subtests in its total).
- `model-download-manifest.json`: supplied pinned model download manifest; not an independent signature.

Original observations live in sibling [full](../p100-2026-10-28-full/) and [smoke](../p100-2026-10-28-smoke/) records. These derived explanations must not replace the frozen raw evidence. The source revision measured by the GPU is `fac2ed29beb602b67725417b6b2f384a14ea5aa7`.
