# Owned-cache Torch experiment

Device: `cuda`. This is a sequential reference-runner experiment, not a serving-throughput benchmark.

Full-session time includes original prefill, pause/compression, restore or recomputation, generation and canceled-session work. A speed ratio above 1 means less full-session time than fixed INT4 in this run. Memory is peak retained serialized cache, not total GPU or process memory.

| Context target | Scenario | Policy | Mean full session (ms) | Speed ratio vs INT4 | Peak stored MiB | Correct / responses | Tokens identical to recompute |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 256 | pressure | adaptive_eviction | 237.95 | 1.186× | 2.526 | 2/4 | 4/4 |
| 256 | pressure | adaptive_exact | 241.66 | 1.168× | 2.619 | 2/4 | 4/4 |
| 256 | pressure | group_int4 | 282.28 | 1.000× | 2.042 | 1/4 | 2/4 |
| 256 | pressure | raw_native | 242.78 | 1.163× | 2.619 | 2/4 | 4/4 |
| 256 | pressure | recompute | 247.85 | 1.139× | 0.000 | 2/4 | 4/4 |
| 256 | roomy | adaptive_eviction | 202.90 | 1.390× | 10.114 | 2/4 | 4/4 |
| 256 | roomy | adaptive_exact | 202.62 | 1.392× | 10.114 | 2/4 | 4/4 |
| 256 | roomy | group_int4 | 282.04 | 1.000× | 2.722 | 1/4 | 1/4 |
| 256 | roomy | raw_native | 202.54 | 1.393× | 10.114 | 2/4 | 4/4 |
| 256 | roomy | recompute | 248.43 | 1.135× | 0.000 | 2/4 | 4/4 |

These are descriptive results on generated lookup, two-hop and arithmetic tasks. Repeated questions are not independent quality observations. Unchanged accuracy does not imply unchanged answers. No quality-preservation or general performance claim is certified.

`summary.json` also reports p95 resume time, truncations, baseline-relative quality regressions, sampled process RSS and CUDA allocator peaks. RSS sampling can miss short peaks. CPU compression follows native export, so compressed stored bytes are not reduced GPU-to-host transfer bytes.

Run `python scripts/validate_p100_results.py RUN_DIRECTORY` to independently audit archive hashes, accounting, completeness and scores.
