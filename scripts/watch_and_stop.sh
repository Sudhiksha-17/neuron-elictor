#!/bin/bash
# Waits for the phase 2 full run to exit, then (success or failure):
#   1. writes the run's exit code to the run log
#   2. if there is no summary.json (run incomplete), resumes once from the last checkpoint
#   3. writes a final summary report to $REPO/reports/ (scripts/final_report.py)
#   4. creates $REPO/code_backup.tar.gz (scripts, hermes, CLAUDE.md, README.md, reports, .gitignore)
#   5. commits config/summary/reports locally (no push)
#   6. ALWAYS stops this RunPod pod with "runpodctl stop pod" (also on any failure, via the EXIT trap).
#      Only "stop" is ever used, never "remove"/"terminate".
# KIND=grpo (default phase2): for scripts/real_run.sh. Skips steps 2, 3 and 5 (phase 2 specific; the GRPO pipeline and
#            real_run.sh write their own reports and commit). Backup + pod stop as above.
# OUTER_PID: after RUN_PID exits, also wait (at most OUTER_TIMEOUT s, default 3600) while OUTER_PID is alive, so the
#            caller (real_run.sh) can write its report, commit and push before the pod stops.
# DRY_RUN=1: print the stop command instead of running it, and skip resume/commit (for testing;
#            REPORT_DIR / BACKUP_PATH / PILOT_TAG may then point to a scratch location).
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_PID=${RUN_PID:?}
RUN_DIR=${RUN_DIR:?}                 # e.g. $REPO/results/phase2/20261109_204500_full1M
RUN_LOG=${RUN_LOG:?}                 # the run's log file
TASK_OUT=${TASK_OUT:-}               # file where the launching shell writes "exit N"
POD_ID=${RUNPOD_POD_ID:?RUNPOD_POD_ID not set}   # no hardcoded fallback: the old pod ID was wrong after moving pods
DRY_RUN=${DRY_RUN:-0}
REPORT_DIR=${REPORT_DIR:-$REPO/reports}
BACKUP_PATH=${BACKUP_PATH:-$REPO/code_backup.tar.gz}
PILOT_TAG=${PILOT_TAG:-full1M_pilots}
KIND=${KIND:-phase2}
OUTER_PID=${OUTER_PID:-}
OUTER_TIMEOUT=${OUTER_TIMEOUT:-3600}
cd "$REPO"
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

log "watching PID $RUN_PID (run dir $RUN_DIR, log $RUN_LOG): $(ps -o args= -p "$RUN_PID" 2>/dev/null || echo 'NOT RUNNING')"
while kill -0 "$RUN_PID" 2>/dev/null; do sleep ${POLL:-30}; done

CODE=$(grep -oE '^exit [0-9]+' "$TASK_OUT" 2>/dev/null | tail -1 | awk '{print $2}')
CODE=${CODE:-unknown}
log "run process $RUN_PID exited; exit code: $CODE"
echo "[watcher $(date '+%F %T')] run process exited with exit code: $CODE" >> "$RUN_LOG"

if [ -n "$OUTER_PID" ]; then
  log "waiting up to ${OUTER_TIMEOUT}s for outer process $OUTER_PID to finish"
  T0=$(date +%s)
  while kill -0 "$OUTER_PID" 2>/dev/null && [ $(( $(date +%s) - T0 )) -lt "$OUTER_TIMEOUT" ]; do sleep ${POLL:-30}; done
  kill -0 "$OUTER_PID" 2>/dev/null && log "outer process still alive after ${OUTER_TIMEOUT}s; stopping anyway" \
    || log "outer process $OUTER_PID exited"
fi

if [ "$KIND" = phase2 ] && [ ! -f "$RUN_DIR/summary.json" ] && [ "$DRY_RUN" != 1 ] && [ -f "$RUN_DIR/state.pt" ]; then
  log "no summary.json -> run incomplete; resuming once from last checkpoint"
  echo "[watcher $(date '+%F %T')] run incomplete; resuming once" >> "$RUN_LOG"
  python scripts/phase2_baseline.py --resume "$RUN_DIR" >> "$RUN_LOG" 2>&1
  CODE=$?
  log "resumed run exited with exit code: $CODE"
  echo "[watcher $(date '+%F %T')] resumed run exited with exit code: $CODE" >> "$RUN_LOG"
fi

if [ "$KIND" = phase2 ]; then
  log "writing final report"
  timeout 1800 python scripts/final_report.py "$RUN_DIR" "$CODE" "$RUN_LOG" "$REPORT_DIR" "$PILOT_TAG" 2>&1 \
    || log "final report step failed"
fi

log "creating code backup $BACKUP_PATH"
mkdir -p reports
tar czf "$BACKUP_PATH" --exclude=__pycache__ --exclude=.ipynb_checkpoints \
    scripts hermes CLAUDE.md README.md reports .gitignore 2>&1 \
  && log "backup ok: $(tar tzf "$BACKUP_PATH" | wc -l) entries, $(du -h "$BACKUP_PATH" | cut -f1)" \
  || log "backup FAILED"

if [ "$KIND" = phase2 ] && [ "$DRY_RUN" != 1 ]; then
  git add -- .gitignore "$RUN_DIR/config.json" reports 2>&1
  [ -f "$RUN_DIR/summary.json" ] && git add -- "$RUN_DIR/summary.json"
  git add -- results/pilots 2>&1
  if git diff --cached --quiet; then
    log "nothing to commit"
  else
    STATUS=$([ -f "$RUN_DIR/summary.json" ] && echo "complete" || echo "INCOMPLETE")
    git commit -q -m "Phase 2 full 1M baseline run: config, summary, final report ($STATUS, exit code $CODE)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "committed locally: $(git log --oneline -1)"
  fi
fi
# EXIT trap stops the pod
