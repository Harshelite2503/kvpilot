"""Create deterministic synthetic pause/resume/cancel fixtures (not production traces)."""
import argparse
import json
from pathlib import Path
import random


def make_trace(seed=42, count=60):
    rng = random.Random(seed)
    events = []
    for i in range(count):
        at = i * 50
        kind = "human" if i % 3 == 0 else "tool"
        namespace = {"tenant": f"demo-tenant-{i % 2}", "model_revision": "synthetic-model@v1",
                     "adapter_revision": "none", "layout": "synthetic-fp16-kv-v1"}
        events.append({"at_ms": at, "type": "pause", "session_id": f"session-{i}", "pause_kind": kind,
                       "namespace": namespace, "blocks": [
                           {"prefix_hash": "synthetic-shared-prefix", "raw_bytes": 4 * 1024 * 1024},
                           {"prefix_hash": f"synthetic-branch-{i}", "raw_bytes": 4 * 1024 * 1024}], "recompute_ms": 35})
        # Fixture generator sees outcomes; the controller receives events online and
        # cannot inspect them until the corresponding timestamp.
        duration = rng.choice([2000, 5000, 10000]) if kind == "human" else rng.choice([1, 25, 100, 1000])
        canceled = rng.random() < (0.65 if kind == "human" else 0.1)
        events.append({"at_ms": at + duration, "type": "cancel" if canceled else "resume", "session_id": f"session-{i}"})
    return sorted(events, key=lambda e: e["at_ms"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="examples/synthetic-sessions.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--count", type=int, default=60)
    args = p.parse_args()
    if args.count <= 0:
        p.error("--count must be positive")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in make_trace(args.seed, args.count)))
    print(f"Wrote {2 * args.count} SYNTHETIC events to {out}")
