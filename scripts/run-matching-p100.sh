#!/usr/bin/env bash
# Run the same KVPilot commit and full protocol as the RTX 5060 Ti experiment.
# Git fetch needs network access; the benchmark itself is offline.
# Keep the working P100 PyTorch 2.7.1 + CUDA 12.6 environment. The RTX run uses
# CUDA 12.8, so this compares complete hardware/software setups, not GPU alone.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash run-matching-p100.sh [MODEL_DIRECTORY]

Run on the existing P100 VM. The repository is /home/nutanix/kvpilot-p100.
MODEL_DIRECTORY defaults to data/qwen25-05b-20261028T134203Z; relative paths
are resolved from that repository. Reuse the existing .venv-p100 environment.
Keep p100-runtime-sidecar.py in the same directory as this script.

This fetches main, checks out the exact comparison commit in detached-HEAD
mode, and runs the unchanged full admission suite. It installs no packages,
downloads no weights, resets no files, and publishes nothing. Existing tracked
edits cause an immediate stop. Git also refuses to overwrite conflicting
untracked files. No branch is moved by the checkout.

Before the suite it requires Python 3.11.18, torch 2.7.1+cu126, numpy 2.2.6,
tokenizers 0.21.2, safetensors 0.5.3, psutil 7.0.0 and pytest 8.4.2.
CPU threads, affinity, cgroup quota and TF32 settings are captured without
changing them. Return both the suite archive and its -runtime-companion archive.
EOF
}

if (($# == 1)) && [[ "$1" == "--help" || "$1" == "-h" ]]; then
  usage
  exit 0
fi
if (($# > 1)); then
  usage >&2
  exit 2
fi

KVPILOT_REPOSITORY="/home/nutanix/kvpilot-p100"
KVPILOT_COMMIT="ca6f6feaaf7faef4b942bd37d78ee29ded99a27a"
KVPILOT_MODEL="${1:-data/qwen25-05b-20261028T134203Z}"
KVPILOT_HANDOFF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KVPILOT_HANDOFF_SCRIPT="$KVPILOT_HANDOFF_DIR/$(basename "${BASH_SOURCE[0]}")"
KVPILOT_HELPER="$KVPILOT_HANDOFF_DIR/p100-runtime-sidecar.py"

if [[ ! -f "$KVPILOT_HELPER" ]]; then
  echo "Keep p100-runtime-sidecar.py next to this handoff script." >&2
  exit 1
fi

cd "$KVPILOT_REPOSITORY"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Tracked edits exist. Preserve or commit them before running this script." >&2
  git status --short
  exit 1
fi
if [[ ! -x .venv-p100/bin/python ]]; then
  echo "Missing executable: $KVPILOT_REPOSITORY/.venv-p100/bin/python" >&2
  exit 1
fi
if [[ ! -f "$KVPILOT_MODEL/config.json" || ! -f "$KVPILOT_MODEL/model-manifest.json" ]]; then
  echo "Missing model config or manifest in: $KVPILOT_MODEL" >&2
  echo "Pass the existing model directory as the script's first argument." >&2
  exit 1
fi

KVPILOT_PYTHON="$KVPILOT_REPOSITORY/.venv-p100/bin/python"
KVPILOT_OUTPUT="$KVPILOT_REPOSITORY/results/local-p100-match-ca6f6fea-$(date -u +%Y%m%dT%H%M%SZ)"
KVPILOT_COMPANION="$KVPILOT_OUTPUT-runtime-companion"
KVPILOT_PHASE="handoff-start"
if [[ -e "$KVPILOT_OUTPUT" || -e "$KVPILOT_OUTPUT.tar.gz" || -e "$KVPILOT_COMPANION" || -e "$KVPILOT_COMPANION.tar.gz" ]]; then
  echo "Output already exists; retry with a fresh timestamp." >&2
  exit 1
fi
mkdir -p "$(dirname "$KVPILOT_OUTPUT")"
mkdir "$KVPILOT_COMPANION"

finish() {
  local result=$?
  local packaging=0
  trap - EXIT
  set +e
  # The unchanged suite may stop before its own final tar command.
  if [[ -d "$KVPILOT_OUTPUT" && ! -f "$KVPILOT_OUTPUT.tar.gz" ]]; then
    tar -czf "$KVPILOT_OUTPUT.tar.gz" -C "$(dirname "$KVPILOT_OUTPUT")" "$(basename "$KVPILOT_OUTPUT")" || packaging=1
  fi
  "$KVPILOT_PYTHON" "$KVPILOT_HELPER" --output "$KVPILOT_COMPANION" --finalize \
    --exit-code "$result" --phase "$KVPILOT_PHASE" --suite-directory "$KVPILOT_OUTPUT" || packaging=1
  tar -czf "$KVPILOT_COMPANION.tar.gz" -C "$(dirname "$KVPILOT_COMPANION")" "$(basename "$KVPILOT_COMPANION")" || packaging=1
  if [[ -f "$KVPILOT_OUTPUT.tar.gz" ]]; then
    sha256sum "$KVPILOT_OUTPUT.tar.gz" | tee "$KVPILOT_OUTPUT.tar.gz.sha256" || packaging=1
  fi
  if [[ -f "$KVPILOT_COMPANION.tar.gz" ]]; then
    sha256sum "$KVPILOT_COMPANION.tar.gz" | tee "$KVPILOT_COMPANION.tar.gz.sha256" || packaging=1
  fi
  printf '\nKeep and return these available archives and their SHA256 files:\n'
  if [[ -f "$KVPILOT_OUTPUT.tar.gz" ]]; then printf '%s.tar.gz\n' "$KVPILOT_OUTPUT"; fi
  if [[ -f "$KVPILOT_COMPANION.tar.gz" ]]; then printf '%s.tar.gz\n' "$KVPILOT_COMPANION"; fi
  if ((result != 0 || packaging != 0)); then
    echo "Run or packaging failed. Preserve both directories, including partial evidence." >&2
    echo "Suite directory: $KVPILOT_OUTPUT" >&2
    echo "Companion directory: $KVPILOT_COMPANION" >&2
  fi
  if ((result == 0 && packaging != 0)); then result=1; fi
  exit "$result"
}
trap finish EXIT
cp "$KVPILOT_HELPER" "$KVPILOT_COMPANION/p100-runtime-sidecar.py"
cp "$KVPILOT_HANDOFF_SCRIPT" "$KVPILOT_COMPANION/run-matching-p100.sh"
KVPILOT_HELPER="$KVPILOT_COMPANION/p100-runtime-sidecar.py"
git rev-parse HEAD > "$KVPILOT_COMPANION/source-before-checkout.txt"

# The explicit destination works with the VM's original --single-branch clone.
KVPILOT_PHASE="git-fetch"
git fetch origin refs/heads/main:refs/remotes/origin/main 2>&1 | tee "$KVPILOT_COMPANION/git-fetch.log"
if ! git cat-file -e "$KVPILOT_COMMIT^{commit}" || ! git merge-base --is-ancestor "$KVPILOT_COMMIT" refs/remotes/origin/main; then
  echo "The comparison commit is unavailable or is not an ancestor of fetched main." >&2
  echo "Stop here rather than substituting a different benchmark revision." >&2
  exit 1
fi
KVPILOT_PHASE="checkout"
git switch --detach "$KVPILOT_COMMIT" 2>&1 | tee "$KVPILOT_COMPANION/git-checkout.log"
if [[ "$(git rev-parse HEAD)" != "$KVPILOT_COMMIT" ]]; then
  echo "Unexpected checkout revision; refusing to run." >&2
  exit 1
fi

git rev-parse HEAD > "$KVPILOT_COMPANION/source-commit.txt"
printf 'Comparison commit: %s\nModel directory: %s\nResults: %s\n' "$KVPILOT_COMMIT" "$KVPILOT_MODEL" "$KVPILOT_OUTPUT"
echo "Keeping the existing P100 CUDA 12.6 environment; the RTX comparison uses CUDA 12.8."
KVPILOT_PHASE="runtime-validation"
"$KVPILOT_PYTHON" "$KVPILOT_HELPER" --repo "$KVPILOT_REPOSITORY" --output "$KVPILOT_COMPANION" \
  2>&1 | tee "$KVPILOT_COMPANION/runtime-validation.log"

# No additional benchmark options: use the unchanged full suite and its audits.
KVPILOT_PHASE="full-suite"
bash scripts/run_p100_admission_suite.sh --model-dir "$KVPILOT_MODEL" --output "$KVPILOT_OUTPUT"

KVPILOT_PHASE="complete"
echo "The repository remains on the comparison commit in detached-HEAD mode."
