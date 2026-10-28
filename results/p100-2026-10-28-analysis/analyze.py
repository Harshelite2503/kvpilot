#!/usr/bin/env python3
"""Reproduce descriptive analysis of a validated KVPilot P100 archive.

Only reads JSON and source files; does not import or run the benchmark/model.
The narrative is specific to the received 2026-10-28 run, enforced by a manifest
digest check. It must not be reused as a report template for different results.
Run the repository's independent validator before using this analysis.
Usage: python3 analyze.py RUN_DIRECTORY REPO_DIRECTORY OUTPUT_DIRECTORY
"""
import collections
import hashlib
import html
import json
from pathlib import Path
import statistics
import sys


MIB = 1024 ** 2


def read(path):
    return json.loads(path.read_text())


def total(row):
    return (row['initial_prefill_ms'] + row['pause_ms']
            + row.get('resume_wall_ms', row.get('cancel_ms', 0)))


def analyze(run, repo):
    manifest_path = run / 'artifact-sha256.json'
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_digest != 'b09b0ebc70e7ab73fd5b1bca2b0600de6edab2399b0ea5de1dabc17fb3ba76ca':
        raise ValueError('This narrative report is specific to the received full-run manifest')
    for name, digest in read(manifest_path).items():
        if Path(name).name != name or hashlib.sha256((run / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Artifact differs from the audited run: ' + name)
    protocol = read(run / 'protocol.json')
    rows = [json.loads(line) for line in (run / 'rows.jsonl').read_text().splitlines()]
    blocks = read(run / 'blocks.json')
    summaries = read(run / 'summary.json')['summaries']
    source_matches = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest
                      for name, digest in protocol['source_sha256'].items()}
    if not all(source_matches.values()):
        raise ValueError('Local source differs from the recorded benchmark source')
    index = {(s['context_target'], s['scenario'], s['policy']): s for s in summaries}
    comparisons, diagnostics, modeled = [], [], []
    for context in protocol['args']['contexts']:
        for scenario in protocol['args']['scenarios']:
            arms = {p: index[context, scenario, p] for p in protocol['policies']}
            adaptive = arms['adaptive_eviction']
            entry = {'context_target': context, 'scenario': scenario,
                     'mean_full_session_ms': adaptive['mean_total_ms_per_started_session'],
                     'comparisons': {}}
            for baseline in ('group_int4', 'raw_native', 'adaptive_exact', 'recompute'):
                control = arms[baseline]
                repeat_changes = []
                for repeat in range(protocol['args']['repeats']):
                    means = {}
                    for policy in ('adaptive_eviction', baseline):
                        subset = [r for r in rows if (r['context_target'], r['scenario'], r['repeat'], r['policy'])
                                  == (context, scenario, repeat, policy)]
                        means[policy] = statistics.mean(map(total, subset))
                    repeat_changes.append(100 * (1 - means['adaptive_eviction'] / means[baseline]))
                entry['comparisons'][baseline] = {
                    'latency_reduction_percent': 100 * (1 - adaptive['mean_total_ms_per_started_session']
                                                         / control['mean_total_ms_per_started_session']),
                    'peak_payload_ratio': (adaptive['peak_paused_payload_bytes'] / control['peak_paused_payload_bytes']
                                           if control['peak_paused_payload_bytes'] else None),
                    'paired_repeat_latency_reductions_percent': repeat_changes,
                }
            comparisons.append(entry)
            for policy in protocol['policies']:
                subset = [r for r in rows if (r['context_target'], r['scenario'], r['policy'])
                          == (context, scenario, policy)]
                answered = [r for r in subset if r['resolution'] == 'resume']
                unique = {}
                for r in answered:
                    key = r['case_id']
                    result = (tuple(r['generation']['tokens']), r['correct'], r['generation']['truncated'])
                    if key in unique and unique[key] != result:
                        raise ValueError('Output varied across repeats; deduplication needs revision')
                    unique[key] = result
                group_blocks = [b for b in blocks if (b['context_target'], b['scenario'], b['policy'])
                                == (context, scenario, policy)]
                s = arms[policy]
                diagnostics.append({
                    **s,
                    'deduplicated_correct': sum(result[1] for result in unique.values()),
                    'deduplicated_responses': len(unique),
                    'evicted_states': sum(b['evicted_states'] for b in group_blocks),
                    'peak_cuda_reserved_bytes': max(b['memory']['after']['device_peak_reserved_bytes'] for b in group_blocks),
                    'mean_initial_prefill_ms_per_started_session': statistics.mean(r['initial_prefill_ms'] for r in subset),
                    'mean_pause_ms_per_started_session': statistics.mean(r['pause_ms'] for r in subset),
                    'mean_restore_ms_per_started_session': sum(r.get('restore_or_recompute_ms', 0) for r in subset) / len(subset),
                    'mean_generation_ms_per_started_session': sum(r.get('generation', {}).get('wall_ms', 0) for r in subset) / len(subset),
                    'mean_generation_ms_per_response': statistics.mean(r['generation']['wall_ms'] for r in answered),
                    'mean_tokens_per_response': statistics.mean(len(r['generation']['tokens']) for r in answered),
                    'full_session_means_by_repeat_ms': [statistics.mean(total(r) for r in subset if r['repeat'] == rep)
                                                        for rep in range(protocol['args']['repeats'])],
                })
            # Counterfactual residency accounting ONLY: no timing or inference claim.
            native_rows = {r['case_id']: r for r in rows if (r['context_target'], r['scenario'], r['policy'], r['repeat'])
                           == (context, scenario, 'raw_native', 0)}
            resident, admitted, hit_cases = {}, [], []
            budget = protocol['budgets_bytes'][str(context)][scenario]
            peak = 0
            for event, name in protocol['schedules'][scenario]:
                if event == 'pause':
                    size = native_rows[name]['exact_candidate_bytes']['raw_native']
                    if sum(resident.values()) + size <= budget:
                        resident[name] = size
                        admitted.append(name)
                        peak = max(peak, sum(resident.values()))
                else:
                    if event == 'resume' and name in resident:
                        hit_cases.append(name)
                    resident.pop(name, None)
            modeled.append({'context_target': context, 'scenario': scenario,
                            'evidence_type': 'counterfactual_residency_only_not_measured_runtime',
                            'policy': 'native_if_free_space_else_recompute_no_eviction',
                            'admitted_per_repeat': len(admitted), 'hits_per_repeat': len(hit_cases),
                            'evictions_per_repeat': 0, 'peak_payload_bytes': peak,
                            'admitted_case_ids': admitted, 'hit_case_ids': hit_cases})
    return {'run_directory_name': run.name,
            'artifact_manifest_sha256': manifest_digest,
            'original_archive_sha256': 'a2a7269e70fda815b3d258abd9f28b672be67bb276c708d12d4f429d3be3a128',
            'source_files_matched': source_matches,
            'model_identity': protocol['model_identity'], 'arguments': protocol['args'],
            'started_sessions': len(rows), 'responses': sum(r['resolution'] == 'resume' for r in rows),
            'blocks': len(blocks), 'distinct_base_questions': len(protocol['cases']),
            'truncated_responses': sum(r.get('generation', {}).get('truncated', False) for r in rows),
            'comparisons': comparisons, 'diagnostics': diagnostics,
            'unmeasured_no_eviction_baseline': modeled}


def table(headers, rows):
    esc = lambda value: html.escape(str(value))
    return ('<div class="scroll"><table><thead><tr>' + ''.join('<th>' + esc(h) + '</th>' for h in headers)
            + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + esc(c) + '</td>' for c in row)
                                             + '</tr>' for row in rows) + '</tbody></table></div>')


def render(data):
    timing = []
    for row in data['comparisons']:
        i, n = (row['comparisons'][p] for p in ('group_int4', 'raw_native'))
        timing.append([row['context_target'], row['scenario'], f"{row['mean_full_session_ms']:.2f}",
                       f"{i['latency_reduction_percent']:.1f}%", f"{n['latency_reduction_percent']:.1f}%",
                       f"{i['peak_payload_ratio']:.2f}×",
                       ', '.join(f'{x:.2f}%' for x in n['paired_repeat_latency_reductions_percent'])])
    quality, actions, tails = [], [], []
    index = {(r['context_target'], r['scenario'], r['policy']): r for r in data['diagnostics']}
    for row in data['comparisons']:
        key = row['context_target'], row['scenario']
        a, n, i = (index[*key, p] for p in ('adaptive_eviction', 'raw_native', 'group_int4'))
        quality.append([*key, f"{a['deduplicated_correct']}/{a['deduplicated_responses']}",
                        f"{i['deduplicated_correct']}/{i['deduplicated_responses']}",
                        f"{a['output_tokens']}", f"{i['output_tokens']}"])
        actions.append([*key, str(a['actions']), f"{a['hits']} / {n['hits']}",
                        f"{a['evicted_states']} / {n['evicted_states']}"])
        tails.append([*key, f"{a['p95_resume_ttft_ms_descriptive']:.1f}",
                      f"{n['p95_resume_ttft_ms_descriptive']:.1f}",
                      f"{a['peak_cuda_allocated_bytes']/MIB:.3f}",
                      f"{i['peak_cuda_allocated_bytes']/MIB:.3f}"])
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>KVPilot — P100 result audit, 28 October 2026</title><style>
body{font:16px/1.6 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#172b3a;background:#f6f8fa}
h1,h2{line-height:1.25}h1{font-size:32px}h2{margin-top:36px;font-size:23px}.lead{font-size:20px}
.card{background:white;border:1px solid #d7e0e7;border-radius:10px;padding:20px;margin:20px 0}.note{border-left:4px solid #c67e16;padding-left:16px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{text-align:left;padding:10px;border-bottom:1px solid #d7e0e7;vertical-align:top}th{background:#e8eff4}code{font-size:13px;overflow-wrap:anywhere}li{margin:10px 0}a{color:#075b94}.muted{color:#4b6271}
</style><body><p class="muted">28 October 2026 · Tesla P100-PCIE-16GB · Qwen2.5-0.5B-Instruct</p>
<h1>A useful admission-control signal; no demonstrated GPU-memory saving</h1>
<p class="lead">Adaptive eviction took 25–52% less full-session time than this implementation of fixed INT4. The more informative native-cache comparison shows 6.4–7.9% less time in cancellation scenarios, with the same observed answers and effectively the same retained-payload peak.</p>
<div class="card"><strong>Audit passed.</strong> The independent validator checked 2,160 sessions, 4,320 events and 90 blocks. All 24 recorded source hashes match commit <code>fac2ed29beb602b67725417b6b2f384a14ea5aa7</code>. The successful smoke archive also validates. This is an audit of records, not an independent GPU rerun.</div>
<h2>Timing and retained memory</h2><p>All percentages below mean reduction in mean full-session wall time per started session; positive is faster. Full time includes initial prefill, pause, restoration or recomputation, generation, and cancelled-session work. Compare policies within each scenario. The three repeat values are descriptive; they are not confidence intervals.</p>
''' + table(['Context target','Scenario','Adaptive ms','Less time vs INT4','Less time vs native','Payload vs INT4','Vs native by repeat 0, 1, 2'], timing) + '''
<p class="note">Adaptive uses 7.2–7.5% more retained cache than INT4 under pressure/cancellation and 3.74–3.76× in roomy runs. It does not achieve a simultaneous speed-and-storage win over INT4. Native caching essentially ties adaptive when there is plenty of capacity.</p>
<h2>What the policy actually did</h2>
<p>Across its 432 started sessions, adaptive eviction chose 222 native snapshots and 210 recomputations. It never chose INT8 or INT4. Adaptive exact chose native snapshots for all 432 sessions. Therefore these results evaluate admission and eviction decisions, not successful adaptive lossy compression.</p>
''' + table(['Context target','Scenario','Adaptive actions, all repeats','Resume hits: adaptive / native','Evictions: adaptive / native'], actions) + '''
<p>Under pressure, adaptive avoids serializing states that would displace useful existing states. Under cancellation it also retains more states that later resume. At 2,048 tokens with cancellation, average pause cost falls from 57.28 ms to 17.54 ms per started session; restore/recompute cost falls from 336.10 ms to 296.06 ms. Generation is essentially unchanged.</p>
<p class="note"><strong>A crucial baseline is missing:</strong> store a native snapshot only when free capacity exists; otherwise recompute, without evicting anyone. Replaying only the recorded sizes and event order predicts the same 6 hits per pressure repeat and 5 hits per cancellation repeat as adaptive. In cancellation this simple rule would create 6 snapshots instead of adaptive’s 7 per repeat. This is an accounting counterfactual, not a measured speed result. The present experiment does not establish that adaptive forecasting beats this simple rule.</p>
<h2>Answer quality and generation length</h2>
<p>There are 24 distinct base questions (8 lookup, 8 two-hop, 8 addition), rendered at two context lengths. Outputs were identical across timing repeats. Scores below remove those repeats. Questions still overlap across contexts and scenarios, so do not add these denominators as independent samples.</p>
''' + table(['Context target','Scenario','Adaptive/native/recompute correct','INT4 correct','Adaptive output tokens, 3 repeats','INT4 output tokens, 3 repeats'], quality) + '''
<p>Both adaptive policies and native caching match recomputation’s exact generated tokens on all 396 answered observations each. INT4 matches on 171/396. None of the 1,980 total responses was truncated. These are observed baseline agreements, not a general quality guarantee or an external correctness audit of the model runner.</p>
<p><strong>Do not turn format failures into factual-error claims.</strong> At 2,048 tokens in the roomy scenario, 11 of the 12 distinct baseline-correct → INT4-failed cases retain the correct number but add prose, violating the explicit number-only instruction. One changes port 6627 to service identifier 79947. At 512 tokens, genuine errors include sum 10 → 12 and pager code 7285 → team_23209. INT4 also fixes some baseline errors.</p>
<p><strong>Generation length contributes to the speed ratio:</strong> 2,048-token roomy INT4 produces four times as many output tokens (1,092 vs 273); mean generation takes 444.63 ms vs 133.51 ms. Full-session timing correctly includes this difference, but the 2.10× ratio cannot be attributed entirely to cache mechanics. Native comparisons have identical outputs and avoid this confound.</p>
<h2>Tail latency and GPU memory</h2>
''' + table(['Context target','Scenario','Adaptive p95 resume TTFT, ms','Native p95 resume TTFT, ms','Adaptive CUDA peak allocated, MiB','INT4 CUDA peak allocated, MiB'], tails) + '''
<p>CUDA peak allocated memory is identical across all five policies: 977.932 MiB at 512 tokens and 1,044.450 MiB at 2,048. Maximum reserved memory across repeats is 1,112 MiB for every policy and condition; individual early blocks report 1,102 MiB. CPU payload compression therefore demonstrates no GPU-memory saving here. Active states are restored to native precision; compression runs after full native export and does not reduce that initial device-to-host copy.</p>
<p class="note">At 2,048 tokens under pressure, adaptive’s mean gain over native is only 1.0%, one of three repeats is slower, and descriptive p95 resume time is worse (637.6 vs 555.5 ms). Report this alongside the favorable cancellation result. Process RSS is sampled and affected by allocator history; it is not a clean per-policy memory comparison.</p>
<h2>What to claim, and what to test next</h2>
<p><strong>Defensible wording:</strong> “In a controlled P100 experiment with our owned cache and a 0.5B model, eviction-aware admission reduced mean full-session time by 6.4–7.9% against always-store-native LRU under cancellation, with identical observed outputs and approximately equal retained-payload memory. This needs confirmation against simpler admission rules and unseen workloads.”</p>
<ol><li>Add the native-only admission rule above as a measured baseline; keep the current results and policy frozen.</li>
<li>Freeze fresh workload seeds and new schedule families before execution: variable session lengths, reuse probabilities, cancellation rates and capacity levels. The current full run repeats the four smoke questions and is development evidence.</li>
<li>Use a fixed-generation-work timing companion to isolate cache overhead, alongside the existing free-generation answer-quality test. Retain the strict rubric and separately report reviewed semantic/value errors.</li>
<li>Run on the free P100 first. Require a repeatable gain over the best simple quality-preserving baseline under the same capacity limit, with acceptable tail latency, before paying for the 3090 or making a broad performance claim.</li></ol>
<h2>Reproducibility</h2><p>Original archive: <code>kvpilot-p100-results-20261028T141453Z.tar.gz</code><br>SHA256: <code>a2a7269e70fda815b3d258abd9f28b672be67bb276c708d12d4f429d3be3a128</code><br>Model revision: <code>7ae557604adf67be50417f59c2c2f167def9a775</code><br>Seed: <code>20261029</code>; three repeats; group size 8; 24-token generation cap. Sequential eager reference runner, not concurrent serving throughput. All cache lifecycle decisions and codecs are KVPilot-owned.</p>
<p><a href="analysis.json">Machine-readable analysis</a> · <a href="analyze.py">Reproduction script</a> · <a href="full-validation.json">Independent validation</a> · <a href="original-report.txt">Original generated table</a></p>
</body></html>'''


if __name__ == '__main__':
    run, repo, output = map(Path, sys.argv[1:])
    data = analyze(run, repo)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'analysis.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    (output / 'report.html').write_text(render(data))
    print(json.dumps({key: data[key] for key in ('started_sessions', 'responses', 'blocks', 'distinct_base_questions', 'truncated_responses')}, indent=2))
