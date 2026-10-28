# Owned-cache Torch experiment

Device: `cuda`. This is a sequential reference-runner experiment, not a serving-throughput benchmark.

Full-session time includes original prefill, pause/compression, restore or recomputation, generation and canceled-session work. A speed ratio above 1 means less full-session time than fixed INT4 in this run. Memory is peak retained serialized cache, not total GPU or process memory.

| Context target | Scenario | Policy | Mean full session (ms) | Speed ratio vs INT4 | Peak stored MiB | Correct / responses | Tokens identical to recompute |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 512 | cancellation | adaptive_eviction | 268.09 | 1.513× | 11.086 | 30/54 | 54/54 |
| 512 | cancellation | adaptive_exact | 285.76 | 1.419× | 11.098 | 30/54 | 54/54 |
| 512 | cancellation | group_int4 | 405.49 | 1.000× | 10.345 | 27/54 | 33/54 |
| 512 | cancellation | raw_native | 286.44 | 1.416× | 11.098 | 30/54 | 54/54 |
| 512 | cancellation | recompute | 315.60 | 1.285× | 0.000 | 30/54 | 54/54 |
| 512 | pressure | adaptive_eviction | 323.01 | 1.341× | 11.086 | 39/72 | 72/72 |
| 512 | pressure | adaptive_exact | 336.62 | 1.286× | 11.098 | 39/72 | 72/72 |
| 512 | pressure | group_int4 | 433.04 | 1.000× | 10.345 | 33/72 | 39/72 |
| 512 | pressure | raw_native | 335.84 | 1.289× | 11.098 | 39/72 | 72/72 |
| 512 | pressure | recompute | 348.83 | 1.241× | 0.000 | 39/72 | 72/72 |
| 512 | roomy | adaptive_eviction | 260.30 | 1.638× | 44.193 | 39/72 | 72/72 |
| 512 | roomy | adaptive_exact | 260.43 | 1.637× | 44.193 | 39/72 | 72/72 |
| 512 | roomy | group_int4 | 426.42 | 1.000× | 11.810 | 30/72 | 36/72 |
| 512 | roomy | raw_native | 259.58 | 1.643× | 44.193 | 39/72 | 72/72 |
| 512 | roomy | recompute | 344.85 | 1.237× | 0.000 | 39/72 | 72/72 |
| 2048 | cancellation | adaptive_eviction | 941.90 | 1.559× | 47.110 | 36/54 | 54/54 |
| 2048 | cancellation | adaptive_exact | 1019.31 | 1.441× | 47.122 | 36/54 | 54/54 |
| 2048 | cancellation | group_int4 | 1468.89 | 1.000× | 43.809 | 15/54 | 18/54 |
| 2048 | cancellation | raw_native | 1022.86 | 1.436× | 47.122 | 36/54 | 54/54 |
| 2048 | cancellation | recompute | 1025.26 | 1.433× | 0.000 | 36/54 | 54/54 |
| 2048 | pressure | adaptive_eviction | 1110.28 | 1.451× | 47.110 | 51/72 | 72/72 |
| 2048 | pressure | adaptive_exact | 1127.81 | 1.428× | 47.122 | 51/72 | 72/72 |
| 2048 | pressure | group_int4 | 1610.94 | 1.000× | 43.809 | 21/72 | 27/72 |
| 2048 | pressure | raw_native | 1121.72 | 1.436× | 47.122 | 51/72 | 72/72 |
| 2048 | pressure | recompute | 1189.31 | 1.355× | 0.000 | 51/72 | 72/72 |
| 2048 | roomy | adaptive_eviction | 772.47 | 2.100× | 188.241 | 51/72 | 72/72 |
| 2048 | roomy | adaptive_exact | 768.69 | 2.111× | 188.241 | 51/72 | 72/72 |
| 2048 | roomy | group_int4 | 1622.44 | 1.000× | 50.073 | 15/72 | 18/72 |
| 2048 | roomy | raw_native | 773.56 | 2.097× | 188.241 | 51/72 | 72/72 |
| 2048 | roomy | recompute | 1189.50 | 1.364× | 0.000 | 51/72 | 72/72 |

These are descriptive results on generated lookup, two-hop and arithmetic tasks. Repeated questions are not independent quality observations. Unchanged accuracy does not imply unchanged answers. No quality-preservation or general performance claim is certified.

`summary.json` also reports p95 resume time, truncations, baseline-relative quality regressions, sampled process RSS and CUDA allocator peaks. RSS sampling can miss short peaks. CPU compression follows native export, so compressed stored bytes are not reduced GPU-to-host transfer bytes.

Run `python scripts/validate_p100_results.py RUN_DIRECTORY` to independently audit archive hashes, accounting, completeness and scores.
