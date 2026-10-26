"""Compare reference policies under identical assumed costs and finite host budget."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kvpilot.replay import load_config, load_trace, replay


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", default="examples/synthetic-sessions.jsonl")
    p.add_argument("--config", default="examples/assumed-costs.json")
    p.add_argument("--output", default="results/policy-replay.json")
    args = p.parse_args()
    model, profiles, config = load_config(args.config)
    policies = ["recompute"] + [c.name for c in profiles if c.lossless or model.allow_unvalidated_lossy] + ["adaptive"]
    records = [replay(load_trace(args.trace), model, profiles, policy=policy,
                      default_pause_ms=config.get("default_pause_ms", 1000)) for policy in policies]
    output = {"evidence_type": "analytical_trace_replay", "quality_validated": False,
              "trace": str(args.trace), "costs": config, "results": records}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print("MODELED ONLY — no GPU, serving benchmark, or task-quality validation")
    for r in records:
        m = r["metrics"]
        print(f"{r['policy']:14s} p95={m['modeled_p95_resume_ms']}ms transferred={m['modeled_transferred_bytes']}B peak_host={m['peak_host_reserved_bytes']}B")
    print(f"Saved detailed decisions and assumptions to {target}")
