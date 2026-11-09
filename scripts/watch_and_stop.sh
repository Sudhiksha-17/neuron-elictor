#!/bin/bash
# Waits for the phase 2 full run to exit, records its exit code, commits its config/summary locally,
# then ALWAYS stops this RunPod pod (also on failure, via the EXIT trap).
# If the run ended without summary.json (e.g. it was killed together with the Claude Code session),
# it is resumed once from its last checkpoint before stopping.
# DRY_RUN=1: print the stop command instead of running it, and skip resume/commit.
set -u
RUN_PID=${RUN_PID:?}
RUN_DIR=${RUN_DIR:?}                 # e.g. /workspace/results/phase2/20261107_233933_full1M
RUN_LOG=${RUN_LOG:?}                 # the run's log file
TASK_OUT=${TASK_OUT:-}               # file where the launching shell writes "exit N"
POD_ID=${RUNPOD_POD_ID:?RUNPOD_POD_ID not set}   # no hardcoded fallback: the old pod ID was wrong after moving pods
DRY_RUN=${DRY_RUN:-0}
cd /workspace
log() { echo "[watcher $(date '+%F %T')] $*"; }

stop_pod() {
  trap - EXIT
  log "stopping pod $POD_ID"
  if [ "$DRY_RUN" = 1 ]; then log "DRY_RUN: would run: runpodctl stop pod $POD_ID"; exit 0; fi
  for i in 1 2 3 4 5; do
    runpodctl stop pod "$POD_ID" && { log "stop command sent"; exit 0; }
    log "runpodctl stop failed (attempt $i), retrying in 60s"; sleep 60
  done
  log "ERROR: could not stop pod"; exit 1
}
trap stop_pod EXIT            # whatever happens below, the pod gets stopped

log "watching PID $RUN_PID (run dir $RUN_DIR)"
while kill -0 "$RUN_PID" 2>/dev/null; do sleep ${POLL:-30}; done

CODE=$(grep -oE '^exit [0-9]+' "$TASK_OUT" 2>/dev/null | tail -1 | awk '{print $2}')
CODE=${CODE:-unknown}
log "run process $RUN_PID exited; exit code: $CODE"
echo "[watcher $(date '+%F %T')] run process exited with exit code: $CODE" >> "$RUN_LOG"

if [ ! -f "$RUN_DIR/summary.json" ] && [ "$DRY_RUN" != 1 ]; then
  log "no summary.json -> run incomplete; resuming once from last checkpoint"
  echo "[watcher $(date '+%F %T')] run incomplete; resuming once" >> "$RUN_LOG"
  python scripts/phase2_baseline.py --resume "$RUN_DIR" >> "$RUN_LOG" 2>&1
  CODE=$?
  log "resumed run exited with exit code: $CODE"
  echo "[watcher $(date '+%F %T')] resumed run exited with exit code: $CODE" >> "$RUN_LOG"
fi

if [ "$DRY_RUN" != 1 ]; then
  git add -- .gitignore "$RUN_DIR/config.json" 2>&1
  [ -f "$RUN_DIR/summary.json" ] && git add -- "$RUN_DIR/summary.json"
  if git diff --cached --quiet; then
    log "nothing to commit"
  else
    STATUS=$([ -f "$RUN_DIR/summary.json" ] && echo "complete" || echo "INCOMPLETE")
    git commit -q -m "Phase 2 full 1M baseline run: config and summary ($STATUS, exit code $CODE)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "committed locally: $(git log --oneline -1)"
  fi
fi
# EXIT trap stops the pod
