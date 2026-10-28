"""Screen codec-cost break-even using audited measurements; no GPU is required."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.torch.break_even import main

if __name__ == "__main__":
    main()
