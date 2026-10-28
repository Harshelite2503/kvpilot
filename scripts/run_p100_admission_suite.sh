#!/usr/bin/env bash
# Offline execution on an already configured VM; does not install or download.
set -euo pipefail

KVPILOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$KVPILOT_ROOT"
KVPILOT_PYTHON=".venv-p100/bin/python"
KVPILOT_MODEL="data/qwen25-05b-pinned"
KVPILOT_OUTPUT="results/local-p100-admission-$(date -u +%Y%m%dT%H%M%SZ)"
KVPILOT_DEVICE="cuda"
KVPILOT_SMOKE_ONLY=0
usage() {
  cat <<'EOF'
Usage: bash scripts/run_p100_admission_suite.sh [options]
  --model-dir DIR  Existing pinned Qwen2.5-0.5B model + manifest
  --output DIR     New results directory (never overwritten)
  --python PATH    Existing interpreter (default .venv-p100/bin/python)
  --device DEVICE  cuda (P100 default), mps or cpu (local diagnostics)
  --smoke-only     Run/audit both smoke modes and sweep, without full runs

Full suite: preflight, offline model check, tests, both smoke modes, both full
modes, analytical break-even sweep and a .tar.gz containing all result files.
No package install, weight download, Git commit or network call is performed.
EOF
}
while (($#)); do
  case "$1" in
    --model-dir|--output|--python|--device)
      if (($# < 2)); then usage >&2; exit 2; fi
      case "$1" in
        --model-dir) KVPILOT_MODEL="$2" ;;
        --output) KVPILOT_OUTPUT="$2" ;;
        --python) KVPILOT_PYTHON="$2" ;;
        --device) KVPILOT_DEVICE="$2" ;;
      esac
      shift 2 ;;
    --smoke-only) KVPILOT_SMOKE_ONLY=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
case "$KVPILOT_DEVICE" in cuda|mps|cpu) ;; *) usage >&2; exit 2 ;; esac
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Commit or preserve tracked edits before running; use a stable checkout." >&2
  exit 1
fi
if [[ ! -x "$KVPILOT_PYTHON" || ! -f "$KVPILOT_MODEL/config.json" ]]; then
  echo "Missing interpreter or model config. See docs/p100-admission-experiment.md." >&2
  exit 1
fi
if [[ -e "$KVPILOT_OUTPUT" || -e "$KVPILOT_OUTPUT.tar.gz" ]]; then
  echo "Output already exists; choose a fresh --output directory." >&2
  exit 1
fi
mkdir -p "$(dirname "$KVPILOT_OUTPUT")"
mkdir "$KVPILOT_OUTPUT"
KVPILOT_OUTPUT="$(cd "$KVPILOT_OUTPUT" && pwd)"
printf '%s\n' "$KVPILOT_OUTPUT" > results/local-last-admission-directory.txt
git rev-parse HEAD > "$KVPILOT_OUTPUT/source-commit.txt"
git status --short > "$KVPILOT_OUTPUT/git-status.txt"
echo "Results directory: $KVPILOT_OUTPUT"
trap 'echo "Suite stopped. Keep partial evidence in: $KVPILOT_OUTPUT" >&2' ERR

"$KVPILOT_PYTHON" scripts/p100_preflight.py --device "$KVPILOT_DEVICE" --output "$KVPILOT_OUTPUT/preflight.json" 2>&1 | tee "$KVPILOT_OUTPUT/preflight.log"
"$KVPILOT_PYTHON" scripts/download_p100_model.py --check-only --model-dir "$KVPILOT_MODEL" 2>&1 | tee "$KVPILOT_OUTPUT/model-check.log"
"$KVPILOT_PYTHON" -m pytest -q --junitxml="$KVPILOT_OUTPUT/tests.xml" 2>&1 | tee "$KVPILOT_OUTPUT/tests.log"

for KVPILOT_MODE in equal_work free_generation; do
  "$KVPILOT_PYTHON" -u scripts/benchmark_p100_kvpilot.py --model-dir "$KVPILOT_MODEL" --device "$KVPILOT_DEVICE" --output "$KVPILOT_OUTPUT/smoke-$KVPILOT_MODE" --smoke --timing-mode "$KVPILOT_MODE" --seed 20261102 --allow-unvalidated-lossy 2>&1 | tee "$KVPILOT_OUTPUT/smoke-$KVPILOT_MODE.log"
  "$KVPILOT_PYTHON" scripts/validate_p100_results.py "$KVPILOT_OUTPUT/smoke-$KVPILOT_MODE" --output "$KVPILOT_OUTPUT/smoke-$KVPILOT_MODE-validation.json"
done
KVPILOT_SWEEP_INPUT="$KVPILOT_OUTPUT/smoke-equal_work"
if ((KVPILOT_SMOKE_ONLY == 0)); then
  for KVPILOT_MODE in equal_work free_generation; do
    "$KVPILOT_PYTHON" -u scripts/benchmark_p100_kvpilot.py --model-dir "$KVPILOT_MODEL" --device "$KVPILOT_DEVICE" --output "$KVPILOT_OUTPUT/full-$KVPILOT_MODE" --contexts 512 2048 --cases 24 --repeats 3 --group-size 8 --scenarios roomy pressure cancellation --max-tokens 24 --chunk-size 128 --timing-mode "$KVPILOT_MODE" --seed 20261102 --allow-unvalidated-lossy 2>&1 | tee "$KVPILOT_OUTPUT/full-$KVPILOT_MODE.log"
    "$KVPILOT_PYTHON" scripts/validate_p100_results.py "$KVPILOT_OUTPUT/full-$KVPILOT_MODE" --output "$KVPILOT_OUTPUT/full-$KVPILOT_MODE-validation.json"
  done
  KVPILOT_SWEEP_INPUT="$KVPILOT_OUTPUT/full-equal_work"
fi
"$KVPILOT_PYTHON" scripts/sweep_p100_break_even.py --run-dir "$KVPILOT_SWEEP_INPUT" --output "$KVPILOT_OUTPUT/break-even" 2>&1 | tee "$KVPILOT_OUTPUT/break-even.log"
tar -czf "$KVPILOT_OUTPUT.tar.gz" -C "$(dirname "$KVPILOT_OUTPUT")" "$(basename "$KVPILOT_OUTPUT")"
trap - ERR
echo "Complete. Send back: $KVPILOT_OUTPUT.tar.gz"
