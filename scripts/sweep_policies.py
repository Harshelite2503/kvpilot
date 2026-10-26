"""Exploratory sensitivity cells, not held-out serving performance evidence."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kvpilot.replay import load_config, replay
from make_demo_trace import make_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/assumed-costs.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 99])
    parser.add_argument("--budgets-mib", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--bandwidths-gib-s", type=float, nargs="+", default=[1, 16, 32])
    parser.add_argument("--output", default="results/sensitivity.json")
    args = parser.parse_args()
    model, profiles, config = load_config(args.config)
    policies = ["recompute"] + [p.name for p in profiles if p.lossless or model.allow_unvalidated_lossy] + ["adaptive"]
    rows = []
    for seed in args.seeds:
        events = make_trace(seed=seed)
        for budget in args.budgets_mib:
            for bandwidth in args.bandwidths_gib_s:
                cell_model = replace(model, host_budget_bytes=budget * 1024 ** 2,
                                     bandwidth_bytes_per_ms=bandwidth * 1024 ** 3 / 1000)
                for policy in policies:
                    result = replay(iter(events), cell_model, profiles, policy=policy,
                                    default_pause_ms=config.get("default_pause_ms", 1000))
                    rows.append({"seed": seed, "budget_mib": budget, "bandwidth_gib_s": bandwidth,
                                 "policy": policy, "metrics": result["metrics"]})
    report = {"evidence_type": "analytical_trace_replay", "quality_validated": False,
              "purpose": "Exploratory sensitivity; generated development fixtures, not a frozen final holdout",
              "assumptions": config,
              "limitations": ["No LLM or GPU execution; no resource queues or active-decode contention",
                              "All codec and transfer costs are supplied assumptions",
                              "Do not select a best baseline on these cells and call it a held-out win"],
              "results": rows}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(rows)} modeled policy/cell results to {path}; no serving claims")


if __name__ == "__main__":
    main()
