#!/bin/bash
# Full GRPO pipeline, launched detached so it does not depend on the Claude Code session:
#   setsid nohup scripts/grpo_pipeline.sh > logs/grpo_pipeline_<ts>.out 2>&1 < /dev/null &
# Steps: phase 1 setup check -> FineWeb max on this GPU (5 neurons) -> Phase A (train + evals) -> gate
#        (Phase A (d) > FineWeb max this GPU, strict) -> Phase B if the gate passes -> final report ->
#        git commit (+ push if authenticated) -> ALWAYS "runpodctl stop pod" (EXIT trap; never remove/terminate).
# Flags: --dry          -> tiny phases (grpo_phase.py --dry), no commit (lists what it would commit), no push,
#                           no pod stop.
#        --force_phase_b -> run Phase B even if the gate fails. ONLY allowed together with --dry; a real run can
#                           never bypass the gate (grpo_phase.py also refuses a non-dry Phase B without a passed
#                           gate.json).
# Env:   PHASE_A_ARGS / PHASE_B_ARGS -> extra args for scripts/grpo_phase.py (e.g. "--steps 200 --merged_gen").
#        The old DRY_RUN / FORCE_PHASE_B env vars are rejected (so a stale env var cannot silently change the mode).
# Changes 2026-11-16 (flags, gate hardening): UNTESTED, run on GPU.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export HF_HOME=/workspace/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
if [ -n "${DRY_RUN:-}" ] || [ -n "${FORCE_PHASE_B:-}" ]; then
  echo "ERROR: DRY_RUN / FORCE_PHASE_B env vars are no longer used; pass --dry [--force_phase_b]" >&2; exit 2
fi
DRY_RUN=0; FORCE_B=0
for arg in "$@"; do
  case "$arg" in
    --dry) DRY_RUN=1 ;;
    --force_phase_b) FORCE_B=1 ;;
    *) echo "ERROR: unknown argument $arg" >&2; exit 2 ;;
  esac
done
if [ "$FORCE_B" = 1 ] && [ "$DRY_RUN" != 1 ]; then
  echo "ERROR: --force_phase_b is only allowed with --dry (a real run cannot bypass the gate)" >&2; exit 2
fi
POD_ID=${RUNPOD_POD_ID:?RUNPOD_POD_ID not set}
TS=$(date +%Y%m%d_%H%M%S)
RUN=${RUN_DIR:-$REPO/results/grpo/${TS}$([ "$DRY_RUN" = 1 ] && echo _dry)}
LOGP=$REPO/logs/grpo_${TS}
STATUS=$RUN/pipeline_status.txt
mkdir -p "$RUN" "$REPO/logs"
echo $$ > "$REPO/logs/grpo_pipeline.pid"
log() { echo "[pipeline $(date '+%F %T')] $*" | tee -a "$STATUS"; }
DRY=$([ "$DRY_RUN" = 1 ] && echo --dry)

finish() {
  local code=$?
  trap - EXIT
  log "pipeline ending (last exit code $code); writing final report"
  timeout 1800 python scripts/grpo_report.py final "$RUN" "$STATUS" 2>&1 | tail -3 || log "final report step failed"
  log "committing"
  git add -- scripts hermes CLAUDE.md README.md .gitignore reports logs 2>&1 | tail -2
  git add -- "$RUN" 2>&1 | tail -2           # .gitignore keeps only small text files (json/jsonl/md/csv) under results/
  # never commit files > 5 MB, .pt/.npz/.safetensors, or anything token-like
  git diff --cached --name-only | while read -r f; do
    if [ -f "$f" ] && { [ "$(stat -c %s "$f")" -gt 5000000 ] || [[ "$f" =~ \.(pt|npz|npy|safetensors|bin)$ ]]; }; then
      git reset -q -- "$f"; echo "unstaged $f (size/type)" >> "$STATUS"; fi
  done
  if [ "$DRY_RUN" = 1 ]; then
    log "DRY_RUN: would commit $(git diff --cached --name-only | wc -l) files:"; git diff --cached --name-only >> "$STATUS"
    git reset -q
  elif git diff --cached --quiet; then log "nothing to commit"; else
    git commit -q -m "GRPO elicitor run $(basename "$RUN"): code, configs, metrics, final report$([ "$DRY_RUN" = 1 ] && echo ' (DRY RUN)')

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "committed: $(git log --oneline -1)"
  fi
  if [ "$DRY_RUN" != 1 ]; then
    if timeout 300 git push origin HEAD:main >> "$STATUS" 2>&1; then log "pushed to origin main"; else
      log "push FAILED (not authenticated or offline); commit is local only"; fi
    # amend nothing after push; the report already lists pipeline steps up to the commit
    log "stopping pod $POD_ID (runpodctl stop pod; never remove/terminate)"
    touch "$REPO/logs/watchdog.disable"
    for i in 1 2 3 4 5; do
      runpodctl stop pod "$POD_ID" >> "$STATUS" 2>&1 && { log "stop command sent"; exit 0; }
      log "runpodctl stop failed (attempt $i); retrying in 60s"; sleep 60
    done
    log "ERROR: could not stop pod"
  else
    log "DRY_RUN: not pushing, not stopping the pod"
  fi
}
trap finish EXIT

log "pipeline start: run dir $RUN, git $(git rev-parse --short HEAD), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader), dry=$DRY_RUN force_phase_b=$FORCE_B"

log "step 1: phase 1 setup check"
python scripts/phase1_setup_check.py > "${LOGP}_phase1.log" 2>&1
P1=$?; log "phase 1 setup check exit $P1 ($(grep -cE '^\s+PASS' "${LOGP}_phase1.log") PASS, $(grep -cE '^\s+FAIL' "${LOGP}_phase1.log") FAIL)"
grep -E '^\s+(PASS|FAIL)' "${LOGP}_phase1.log" >> "$STATUS"
[ $P1 -eq 0 ] || { log "ABORT: phase 1 setup check failed"; exit 1; }

log "step 2: FineWeb max on this GPU (top-10 contexts, 5 neurons)"
python scripts/grpo_fwmax.py "$RUN" > "${LOGP}_fwmax.log" 2>&1 || { log "ABORT: grpo_fwmax.py failed"; exit 1; }
grep -E "^L[0-9]+N|confirmed" "${LOGP}_fwmax.log" >> "$STATUS"

log "step 3: Phase A (args: ${PHASE_A_ARGS:-none} $DRY)"
python scripts/grpo_phase.py --phase A --run_dir "$RUN" $DRY ${PHASE_A_ARGS:-} > "${LOGP}_phaseA.log" 2>&1
PA=$?; log "Phase A exit $PA"
grep -E "PROJECTED" "${LOGP}_phaseA.log" | tail -1 >> "$STATUS"
grep -E "check \(|INTEGRITY FAIL|REFUSING" "${LOGP}_phaseA.log" >> "$STATUS"
[ $PA -eq 0 ] || { log "ABORT: Phase A failed"; tail -20 "${LOGP}_phaseA.log" >> "$STATUS"; exit 1; }

log "step 4: gate"
python scripts/grpo_report.py gate "$RUN" 2>&1 | tee -a "$STATUS"
GATE=${PIPESTATUS[0]}
if [ "$GATE" -ne 0 ]; then
  if [ "$DRY_RUN" = 1 ] && [ "$FORCE_B" = 1 ]; then log "gate FAILED: running Phase B anyway (--dry --force_phase_b, testing only)"
  else log "gate FAILED: Phase B skipped"; exit 0; fi
fi

log "step 5: Phase B (args: ${PHASE_B_ARGS:-none} $DRY)"
python scripts/grpo_phase.py --phase B --run_dir "$RUN" $DRY ${PHASE_B_ARGS:-} > "${LOGP}_phaseB.log" 2>&1
PB=$?; log "Phase B exit $PB"
grep -E "PROJECTED" "${LOGP}_phaseB.log" | tail -1 >> "$STATUS"
grep -E "check \(|INTEGRITY FAIL|REFUSING" "${LOGP}_phaseB.log" >> "$STATUS"
[ $PB -eq 0 ] || { log "Phase B failed"; tail -20 "${LOGP}_phaseB.log" >> "$STATUS"; exit 1; }
log "all phases done"
