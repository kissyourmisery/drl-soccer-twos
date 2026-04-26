#!/usr/bin/env bash
set -euo pipefail

JOB_ID="${1:-5053883}"
INTERVAL_SEC="${2:-900}"
REPO_DIR="${3:-/home/hice1/cxu371/scratch/drl-soccer-twos}"
OUT_FILE="$REPO_DIR/soccer_opt2_sp-${JOB_ID}.out"
RESULT_ROOT="$REPO_DIR/ray_results/PPO_option2_selfplay_dense_cpu_slurm"
LOG_FILE="$REPO_DIR/monitor_opt2_${JOB_ID}.log"

check_once() {
  local ts latest progress iter timesteps ckpt_count fatal_count socket_count line_count
  ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[$ts] health snapshot job=$JOB_ID"

  if [ -f "$OUT_FILE" ]; then
    line_count="$(wc -l < "$OUT_FILE" || echo 0)"
    socket_count="$(rg -c 'socket\.gaierror' "$OUT_FILE" 2>/dev/null || echo 0)"
    fatal_count="$(rg -c 'Connection closed by server|Unexpected error starting runner|TuneError|Aborted|UnityWorkerInUseException|Trial .*ERROR' "$OUT_FILE" 2>/dev/null || echo 0)"
    echo "  log_lines=$line_count socket_gaierrors=$socket_count fatal_patterns=$fatal_count"
  else
    echo "  log_missing=$OUT_FILE"
  fi

  latest="$(ls -td "$RESULT_ROOT"/PPO_Soccer_* 2>/dev/null | head -n 1 || true)"
  if [ -n "$latest" ] && [ -f "$latest/progress.csv" ]; then
    progress="$latest/progress.csv"
    iter="$(tail -n 1 "$progress" | awk -F',' '{print $11}')"
    timesteps="$(tail -n 1 "$progress" | awk -F',' '{print $7}')"
    ckpt_count="$(find "$latest" -maxdepth 3 -type f | rg -c 'checkpoint-[0-9]+$' || true)"
    echo "  trial=$(basename "$latest") iter=$iter timesteps=$timesteps checkpoints=$ckpt_count"
  else
    echo "  trial_progress_not_found"
  fi

  echo
}

mkdir -p "$REPO_DIR"

while true; do
  check_once >> "$LOG_FILE"
  sleep "$INTERVAL_SEC"
done
