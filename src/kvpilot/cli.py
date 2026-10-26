import argparse
import json
from pathlib import Path

from .replay import load_config, load_trace, replay


def main():
    parser = argparse.ArgumentParser(description="KV cache research harness; modeled results are not GPU measurements")
    sub = parser.add_subparsers(dest="command", required=True)
    rp = sub.add_parser("replay", help="causal analytical paused-store replay")
    rp.add_argument("--trace", required=True)
    rp.add_argument("--config", required=True)
    rp.add_argument("--policy", default="adaptive")
    rp.add_argument("--output")
    args = parser.parse_args()
    model, profiles, config = load_config(args.config)
    result = replay(load_trace(args.trace), model, profiles, policy=args.policy,
                    default_pause_ms=config.get("default_pause_ms", 1000.0))
    result["scenario_description"] = config.get("description", "")
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        print(json.dumps({"output": str(target), "evidence_type": result["evidence_type"], "metrics": result["metrics"]}, indent=2))
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
