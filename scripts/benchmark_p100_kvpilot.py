"""Run the owned PyTorch cache benchmark; use --help for the frozen protocol options."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.torch.benchmark import main

if __name__ == "__main__":
    main()
