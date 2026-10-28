# Owned-cache Torch experiment

Device: `mps`. This is a sequential reference-runner experiment, not a serving-throughput benchmark.

Mode: **equal_work**. Every policy consumes the same frozen continuation IDs for the same number of prediction steps, including EOS. These are teacher-forced computational controls, not generated answers; no quality score is assigned. Compare timings within this mode only.

Full-session time includes original prefill, pause/compression, restore or recomputation, generation and canceled-session work. A speed ratio above 1 means less full-session time than fixed INT4 in this run. Memory is peak retained serialized cache, not total GPU or process memory.

| Context target | Scenario | Policy | Mean full session (ms) | Speed ratio vs INT4 | Ratio vs admit-if-fits | Peak stored MiB | Correct / responses | Answer tokens identical to recompute |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | pressure | adaptive_eviction | 3126.96 | 1.071× | 1.191× | 2.526 | n/a | n/a |
| 256 | pressure | adaptive_exact | 3866.95 | 0.866× | 0.963× | 2.526 | n/a | n/a |
| 256 | pressure | group_int4 | 3349.84 | 1.000× | 1.112× | 1.992 | n/a | n/a |
| 256 | pressure | raw_admit_if_fits | 3723.57 | 0.900× | 1.000× | 2.526 | n/a | n/a |
| 256 | pressure | raw_native | 3751.80 | 0.893× | 0.992× | 2.526 | n/a | n/a |
| 256 | pressure | recompute | 3875.12 | 0.864× | 0.961× | 0.000 | n/a | n/a |
| 256 | roomy | adaptive_eviction | 3554.68 | 1.008× | 0.977× | 9.926 | n/a | n/a |
| 256 | roomy | adaptive_exact | 3748.27 | 0.956× | 0.927× | 9.926 | n/a | n/a |
| 256 | roomy | group_int4 | 3581.75 | 1.000× | 0.970× | 2.672 | n/a | n/a |
| 256 | roomy | raw_admit_if_fits | 3473.38 | 1.031× | 1.000× | 9.926 | n/a | n/a |
| 256 | roomy | raw_native | 3337.79 | 1.073× | 1.041× | 9.926 | n/a | n/a |
| 256 | roomy | recompute | 3822.05 | 0.937× | 0.909× | 0.000 | n/a | n/a |

These are descriptive results on generated lookup, two-hop and arithmetic tasks. Repeated questions are not independent quality observations. Unchanged accuracy does not imply unchanged answers. No quality-preservation or general performance claim is certified.

`summary.json` also reports p95 resume time, truncations, baseline-relative quality regressions, sampled process RSS and CUDA allocator peaks. RSS sampling can miss short peaks. CPU compression follows native export, so compressed stored bytes are not reduced GPU-to-host transfer bytes.

Run `python scripts/validate_p100_results.py RUN_DIRECTORY` to independently audit archive hashes, accounting, completeness and scores.
