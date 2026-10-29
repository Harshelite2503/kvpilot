"""Build a descriptive, within-machine comparison from audited result files."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/p100-vs-rtx-20261029"
INPUTS = {
    "P100": OUT / "p100-analysis.json",
    "RTX 5060 Ti": ROOT / "outputs/rtx5060ti-20261029/analysis.json",
}
RAW = {
    "P100": OUT / "local-p100-admission-20261028T180805Z",
    "RTX 5060 Ti": ROOT / "outputs/rtx5060ti-20261029/extracted/local-rtx5060ti-replication",
}
POLICIES = {
    "adaptive_eviction": "Adaptive",
    "raw_admit_if_fits": "Simple admission",
    "group_int4": "Fixed CPU INT4",
    "raw_native": "Native with eviction",
    "recompute": "Recompute",
    "adaptive_exact": "Adaptive exact",
}


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(
        '<th>' + escape(str(x)) + '</th>' for x in headers
    ) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(
        '<td>' + escape(str(x)) + '</td>' for x in row
    ) + '</tr>' for row in rows) + '</tbody></table></div>'


def reduction(a, b):
    return 100 * (1 - a / b)


def sweep_screen(folder):
    sweep = read(folder / 'break-even/sweep.json')
    assert sweep['measured_performance'] is False
    assert sweep['quality_certified'] is False
    assert sweep['dollar_savings_estimated'] is False
    assert sweep['input_timing_mode'] == 'equal_work'
    for name, digest in read(folder / 'break-even/artifact-sha256.json').items():
        assert Path(name).name == name
        assert sha256((folder / 'break-even' / name).read_bytes()).hexdigest() == digest
    for name, digest in sweep['provenance']['input_artifact_sha256'].items():
        assert Path(name).name == name
        assert sha256((folder / 'full-equal_work' / name).read_bytes()).hexdigest() == digest
    groups = {}
    for row in sweep['summaries']:
        key = row['context_target'], row['scenario'], row['codec_scale']
        group = groups.setdefault(key, {})
        assert row['policy'] not in group
        group[row['policy']] = row
    cells = {}
    for (context, scenario, scale), policies in sorted(groups.items()):
        assert len({p['sessions'] for p in policies.values()}) == 1
        costs = {k: p['total_modeled_ms'] for k, p in policies.items()}
        simple = min(costs[k] for k in ('raw_admit_if_fits', 'raw_native', 'recompute'))
        record = cells.setdefault((context, scenario), {
            'context': context, 'scenario': scenario, 'wins_native_recompute': [], 'wins_including_int4': []})
        if scale < 1 and costs['adaptive_eviction'] < simple - 1e-7:
            record['wins_native_recompute'].append(scale)
        if scale < 1 and costs['adaptive_eviction'] < min(simple, costs['group_int4']) - 1e-7:
            record['wins_including_int4'].append(scale)
    return list(cells.values())


def main():
    analyses = {name: read(path) for name, path in INPUTS.items()}
    comparison = {
        'evidence_type': 'descriptive_within_system_policy_replication',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': {name: {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()}
                   for name, path in INPUTS.items()},
        'scope': 'Within-system policy effects. Not a pure GPU-only comparison, production throughput claim, or significance test.',
        'systems': {},
    }
    overview, timing_rows, quality_rows, memory_rows, actions_rows, sweep_rows = [], [], [], [], [], []
    detail = []
    for name, data in analyses.items():
        equal, free = data['modes']['equal_work'], data['modes']['free_generation']
        assert equal['validation']['valid'] and free['validation']['valid']
        means = {
            policy: sum(c['policies'][policy]['mean_total_ms_per_started_session'] * c['policies'][policy]['sessions']
                        for c in equal['cells']) / sum(c['policies'][policy]['sessions'] for c in equal['cells'])
            for policy in POLICIES
        }
        reductions = {policy: reduction(means['adaptive_eviction'], mean) for policy, mean in means.items()}
        quality = {}
        for policy in POLICIES:
            totals = Counter()
            for cell in free['cells']:
                values = cell['policies'][policy]['answer_quality']
                for key in ('correct', 'responses', 'token_identical_to_recompute',
                            'quality_regressions_vs_recompute', 'quality_improvements_vs_recompute', 'truncated'):
                    totals[key] += values[key]
            quality[policy] = dict(totals)
            quality_rows.append([name, POLICIES[policy], f"{totals['correct']}/{totals['responses']}",
                                 f"{totals['token_identical_to_recompute']}/{totals['responses']}",
                                 totals['quality_regressions_vs_recompute'], totals['truncated']])
        action_totals = {}
        for mode in ('equal_work', 'free_generation'):
            totals = Counter()
            for cell in data['modes'][mode]['cells']:
                totals.update(cell['policies']['adaptive_eviction']['actions'])
            action_totals[mode] = dict(totals)
            actions_rows.append([name, mode, totals['raw_native'], totals['recompute'],
                                 totals['group_int4'], totals['group_int8']])
        screen = sweep_screen(RAW[name])
        for cell in screen:
            sweep_rows.append([name, cell['context'], cell['scenario'],
                               ', '.join(map(str, cell['wins_native_recompute'])) or 'None',
                               ', '.join(map(str, cell['wins_including_int4'])) or 'None'])
        comparison['systems'][name] = {'source_commit': data['source_commit'], 'mean_session_ms_balanced_mix': means,
            'adaptive_time_reduction_percent': reductions, 'free_generation_totals': quality,
            'adaptive_actions': action_totals, 'modeled_sweep': screen}
        overview.append([name, f"{means['adaptive_eviction']:.2f}", f"{means['raw_admit_if_fits']:.2f}",
                         f"{reductions['raw_admit_if_fits']:+.2f}%", f"{reductions['group_int4']:+.2f}%"])
        details_rows = []
        for cell in equal['cells']:
            context, scenario = cell['context_target'], cell['scenario']
            for base in ('raw_admit_if_fits', 'group_int4', 'raw_native', 'recompute'):
                values = cell['adaptive_eviction_vs'][base]
                timing_rows.append([name, context, scenario, POLICIES[base],
                    f"{values['adaptive_full_session_latency_reduction_percent']:+.2f}%",
                    ', '.join(f"{100 * (1 - 1 / r['speed_ratio_baseline_over_adaptive']):+.2f}%"
                              for r in values['per_repeat_speed_ratios'])])
            for policy, entry in cell['policies'].items():
                details_rows.append([context, scenario, POLICIES[policy],
                                    f"{entry['mean_total_ms_per_started_session']:.2f}",
                                    f"{entry['p95_resume_ttft_ms_descriptive']:.2f}", entry['hits'],
                                    f"{entry['peak_paused_payload_bytes'] / 2**20:.3f}",
                                    f"{entry['peak_cuda_allocated_bytes'] / 2**20:.2f}"])
            if scenario == 'roomy':
                a, fixed = cell['policies']['adaptive_eviction'], cell['policies']['group_int4']
                peaks = {v['peak_cuda_allocated_bytes'] for v in cell['policies'].values()}
                assert len(peaks) == 1
                memory_rows.append([name, context, f"{a['peak_cuda_allocated_bytes'] / 2**20:.2f}",
                                    f"{a['peak_paused_payload_bytes'] / 2**20:.2f}",
                                    f"{fixed['peak_paused_payload_bytes'] / 2**20:.2f}",
                                    f"{a['peak_paused_payload_bytes'] / fixed['peak_paused_payload_bytes']:.2f}×"])
        assert all(len({v['peak_cuda_allocated_bytes'] for v in c['policies'].values()}) == 1 for c in equal['cells'])
        detail.append('<details><summary>' + escape(name) + ': every measured timing cell</summary>' +
                      table(['Context', 'Scenario', 'Policy', 'Session ms', 'p95 resume TTFT ms', 'Hits',
                             'Peak host MiB', 'Peak CUDA allocated MiB'], details_rows) + '</details>')
    (OUT / 'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False) + '\n')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KVPilot — P100 and RTX policy results</title><style>
:root{font:16px/1.6 system-ui,sans-serif;color:#202a35;background:#f3f5f7}body{margin:0}main{max-width:1160px;margin:auto;padding:36px 24px 80px}h1{font-size:36px;line-height:1.15;max-width:820px}h2{font-size:23px;margin-top:0}section,header{background:white;border:1px solid #d6dde4;border-radius:14px;padding:26px;margin-bottom:22px}.lead{border-left:6px solid #906025}.eyebrow{font-weight:700;color:#506276;font-size:13px;letter-spacing:.08em}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 12px;border-bottom:1px solid #dce2e7;text-align:left;white-space:nowrap}th{background:#eaf0f4}tbody tr:nth-child(even){background:#f8fafb}.scroll{overflow:auto;margin:18px 0}a{color:#175aa0}code{overflow-wrap:anywhere}p{max-width:980px}.note{color:#526170;font-size:14px}summary{cursor:pointer;font-weight:650;margin:14px 0}li{margin:7px 0}@media(max-width:650px){main{padding:18px 12px}section,header{padding:18px}h1{font-size:28px}}</style><main>
<header class="lead"><div class="eyebrow">KVPILOT · MEASURED REFERENCE RUNNER · 29 OCTOBER 2026</div>
<h1>Both systems favor avoiding expensive compression. Neither proves an adaptive-policy advantage.</h1>
<p>The supplied P100 run is the full admission experiment. Its benchmark implementation matches the RTX run, despite different Git commit labels. On the balanced workload, adaptive took <strong>3.66% more time than simple admission on P100</strong> and <strong>0.29% more on RTX</strong>. These are descriptive measurements, not significance estimates.</p>
<p>Simple admission means: retain a native snapshot if it fits; otherwise recompute. It is the immediate baseline to beat.</p></header>
<section><h2>Time: compare policies within each machine</h2><p>Positive percentages mean less full-session time for adaptive. Negative means more time. Each mean weights the same number of started sessions across six cells; this artificial balanced mix is not a customer workload or a dollar-saving estimate. Full-session time includes original prefill, pause/export/codec work, resume or recomputation, generation, and canceled work.</p>'''
    document += table(['System', 'Adaptive ms/session', 'Simple admission ms/session', 'Adaptive vs admission', 'Adaptive vs CPU INT4'], overview)
    document += '''<p>Adaptive beats this owned CPU INT4 reference in all six cells on both systems. P100 gains range from 8.57% to 29.42%; RTX gains range from 6.91% to 28.98%. This does not compare against optimized GPU INT4 or a production serving engine.</p>
<details><summary>Per-cell reductions and each of the three repeats</summary>'''
    document += table(['System', 'Context', 'Scenario', 'Baseline', 'Adaptive time reduction', 'Repeats 0, 1, 2'], timing_rows) + '</details>'
    document += ''.join(detail) + '</section><section><h2>Memory: no demonstrated GPU reduction</h2>'
    document += '<p>All six policies have identical peak CUDA allocation within each context and system. Host snapshots differ: in roomy cases, adaptive retains native data and uses about 3.75× the serialized host storage of INT4.</p>'
    document += table(['System', 'Context', 'All policies: CUDA MiB', 'Roomy adaptive host MiB', 'Roomy INT4 host MiB', 'Host ratio'], memory_rows)
    document += '''<p class="note">Serialized host payload is distinct from GPU allocation, GPU reservation, total process RSS, and capacity. CPU compression follows native GPU export; smaller stored payload does not imply fewer GPU-to-host bytes. The small model and sequential runner use roughly 1 GiB of peak CUDA allocation, so these runs do not test a full 16 GiB serving workload.</p></section>
<section><h2>Answers: preserved by the native/recompute path</h2><p>Adaptive and simple admission each reproduce recomputation token-for-token for all 396 response observations on each machine. There are only 24 underlying questions, repeated across contexts, scenarios and three runs. Recompute's free-generation tokens also match across devices for all 396 paired responses.</p>'''
    document += table(['System', 'Policy', 'Strict exact matches', 'Token-identical to recompute', 'Exact-match regressions', 'Truncations'], quality_rows)
    document += '''<p>These generated lookup, two-hop and arithmetic questions are a narrow test. The strict score can reject an answer containing the correct number plus extra words; score regressions are not automatically factual errors. Exact-match preservation does not establish strong baseline accuracy or broad task quality. Equal-work timing forces continuation tokens and is not a quality evaluation.</p>
<p><strong>No adaptive lossy compression was exercised on either system.</strong> Timing and quality calibrate costs separately; their selected actions differ on the P100.</p>'''
    document += table(['System', 'Mode', 'Native admissions', 'Recompute decisions', 'INT4 decisions', 'INT8 decisions'], actions_rows)
    document += '''</section><section><h2>What to improve next</h2>
<p>The P100 equal-work records show inefficient admissions: in pressure cells, adaptive makes 20 and 19 native admissions versus simple admission's 18, but still gets the same 18 hits. In cancellation cells it admits 24 and 26 snapshots versus 18, yet gets only 12 and 13 hits versus 15. More cache work does not necessarily buy more reuse.</p>
<p>Timing variability also matters. P100 generation and prefill components vary across blocks despite fixed token counts. The entire elapsed-time gap cannot be attributed to the admission algorithm. The observed gap is enough to reject a proven adaptive win; it is not a precise causal estimate of policy overhead.</p>
<ol><li>Keep simple admission as the default benchmark to beat. Separate cache-control and restoration time from prefill/generation, and record CPU scheduling, threads, GPU clocks and utilization on future runs.</li>
<li>Investigate replacements that cause extra exports without a net hit gain. Develop a conservative admission rule using only signals available at decision time; cancellation labels known after the fact are not valid prediction inputs.</li>
<li>Freeze a new realistic trace workload and a meaningful win threshold before evaluating changes. Compare simple admission, TTL, idleness and recomputation; include end-to-end latency, constrained resource memory and answer preservation.</li></ol>
<p>Do not rent another GPU merely to repeat this fixture. These P100 results already answer the main baseline question. A new paired run is useful after instrumentation or a candidate policy change, followed by an untouched evaluation workload.</p></section>
<section><h2>Break-even sweep: modeled sensitivity, not measured performance</h2>
<p>The supplied sweeps are bound to their respective equal-work artifacts. They reduce only lossy encode/decode cost, holding the rest of the model fixed. Faster codecs also help fixed INT4, so it belongs in the comparator set. Entries below are tested scales below 1 at which adaptive has strictly lower modeled time; scale 0 removes codec compute cost. No interpolation or monotonicity is assumed.</p>'''
    document += table(['System', 'Context', 'Scenario', 'Beats native/recompute at scales', 'Also beats fixed INT4 at same scales'], sweep_rows)
    document += '''<p class="note">No quality, GPU-memory, serving-throughput or dollar claim is inferred from the sweep. A modeled opportunity is not evidence that a codec with those costs exists or that adaptive would make those choices in a live run.</p></section>
<section><h2>Controls, audits and limits</h2><ul>
<li>Four P100 runs pass independent artifact, score and accounting audits: two smoke runs of 48 sessions each and two full runs of 2,592 sessions each. Each full mode has 108 blocks and 5,184 events. The test log reports 512 passed, 11 skipped and 89 subtests passed. The XML includes subtests, so its total must not be presented as 601 ordinary tests.</li>
<li>P100 source label: <code>b91ebb37ead55d21dabe23ef0311ae2a9d2dc697</code>; RTX: <code>ca6f6feaaf7faef4b942bd37d78ee29ded99a27a</code>. All 25 recorded benchmark source hashes match. Intervening repository changes concern CI, tests, documentation and saved evidence.</li>
<li>Same pinned Qwen2.5-0.5B-Instruct model digest, prompts, input tokens, expected answers, schedules, budgets, policy order and benchmark arguments apart from paths. Contexts 512/2048; 24 cases; 3 repeats; 24 forced steps in equal_work and at most 24 output tokens in free_generation.</li>
<li>Frozen equal-work continuations match for 47/48 context-case pairs. At 512/case-0002, the 24-step sequence differs from the eighth token onward, after EOS at the third token. Within each system all policies use identical forced inputs; cross-system token identity is incomplete. The full free-generation recompute answers match.</li>
<li>Torch 2.7.1 uses CUDA 12.6 on P100 and CUDA 12.8 on RTX; operating systems differ. CPU, PCIe and driver settings are not matched or fully documented. The P100 folder lacks the later runtime sidecar, so CPU affinity, cgroup quota and thread settings cannot be reconstructed. This is not a GPU-only causal comparison.</li>
<li>The supplied P100 folder has per-run manifests but no outer suite manifest. The preserved copy has a new import receipt, which records the files as received and does not retroactively attest their origin. Artifact audits establish internal consistency, not an independent signature or physical-execution proof.</li>
</ul></section><section><h2>Evidence and reproducibility</h2><p>
<a href="p100-analysis.json">P100 analysis</a> · <a href="comparison.json">Derived comparison</a> ·
<a href="comparability-review.json">Independent controls review</a> · <a href="independent-timing-review.json">Independent timing review</a> ·
<a href="independent-quality-review.json">Independent quality review</a> · <a href="import-receipt.json">Original-file import hashes</a> ·
<a href="p100-original-results.tar.gz">Preserved P100 results</a> · <a href="p100-original-results.tar.gz.sha256">Archive checksum</a> ·
<a href="../rtx5060ti-20261029/report.html">Prior RTX report</a></p>
<p class="note">No model inference, GPU rental, repository edit or result publication was required to analyze this folder. All comparisons here are local and descriptive.</p></section></main></html>'''
    (OUT / 'report.html').write_text(document)
    print(json.dumps({'report': str(OUT / 'report.html'), 'comparison': str(OUT / 'comparison.json')}))


if __name__ == '__main__':
    main()
