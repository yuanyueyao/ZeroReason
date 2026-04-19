#!/usr/bin/env bash
# Run run.sh to completion, then run_2.sh (same directory). Extra args are passed to both.
# All stdout/stderr from this script and both children go to one log file and the terminal.
# Override path: LOG_FILE=/path/to/run.log bash run_sequential.sh ...
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${LOG_FILE:-$DIR/logs/run_sequential_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "Sequential run — logging to: $LOG"

echo "=== [1/2] Starting run.sh ==="
bash "$DIR/run.sh" "$@"
echo "=== run.sh finished ==="
echo "=== [2/2] Starting run_2.sh ==="
bash "$DIR/run_2.sh" "$@"
echo "=== run_2.sh finished ==="
