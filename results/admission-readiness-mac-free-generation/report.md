# Owned-cache Torch experiment

Device: `mps`. This is a sequential reference-runner experiment, not a serving-throughput benchmark.

Mode: **free_generation**. Answers and their lengths may differ; this run tests quality and observed session behavior, not equal computational work.

Full-session time includes original prefill, pause/compression, restore or recomputation, generation and canceled-session work. A speed ratio above 1 means less full-session time than fixed INT4 in this run. Memory is peak retained serialized cache, not total GPU or process memory.

| Context target | Scenario | Policy | Mean full session (ms) | Speed ratio vs INT4 | Ratio vs admit-if-fits | Peak stored MiB | Correct / responses | Answer tokens identical to recompute |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | pressure | adaptive_eviction | 1190.26 | 1.065× | 0.856× | 2.526 | 1/4 | 4/4 |
| 256 | pressure | adaptive_exact | 1163.03 | 1.090× | 0.876× | 2.526 | 1/4 | 4/4 |
| 256 | pressure | group_int4 | 1267.33 | 1.000× | 0.804× | 1.992 | 0/4 | 1/4 |
| 256 | pressure | raw_admit_if_fits | 1018.56 | 1.244× | 1.000× | 2.526 | 1/4 | 4/4 |
| 256 | pressure | raw_native | 1242.28 | 1.020× | 0.820× | 2.526 | 1/4 | 4/4 |
| 256 | pressure | recompute | 1147.01 | 1.105× | 0.888× | 0.000 | 1/4 | 4/4 |
| 256 | roomy | adaptive_eviction | 836.69 | 1.719× | 1.170× | 9.926 | 1/4 | 4/4 |
| 256 | roomy | adaptive_exact | 840.34 | 1.712× | 1.165× | 9.926 | 1/4 | 4/4 |
| 256 | roomy | group_int4 | 1438.59 | 1.000× | 0.681× | 2.672 | 0/4 | 0/4 |
| 256 | roomy | raw_admit_if_fits | 979.27 | 1.469× | 1.000× | 9.926 | 1/4 | 4/4 |
| 256 | roomy | raw_native | 1073.91 | 1.340× | 0.912× | 9.926 | 1/4 | 4/4 |
| 256 | roomy | recompute | 1101.52 | 1.306× | 0.889× | 0.000 | 1/4 | 4/4 |

These are descriptive results on generated lookup, two-hop and arithmetic tasks. Repeated questions are not independent quality observations. Unchanged accuracy does not imply unchanged answers. No quality-preservation or general performance claim is certified.

`summary.json` also reports p95 resume time, truncations, baseline-relative quality regressions, sampled process RSS and CUDA allocator peaks. RSS sampling can miss short peaks. CPU compression follows native export, so compressed stored bytes are not reduced GPU-to-host transfer bytes.

Run `python scripts/validate_p100_results.py RUN_DIRECTORY` to independently audit archive hashes, accounting, completeness and scores.
