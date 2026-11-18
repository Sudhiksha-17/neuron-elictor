#!/bin/bash
# Collapse-fix experiments (2026-11-18), unattended, as ONE pipeline. Launch from the repo root:
#   setsid nohup scripts/collapse_run.sh > logs/collapse_run_$(date +%Y%m%d_%H%M%S).out 2>&1 < /dev/null &
# Runs, in order (all: 300 steps, unmerged generation, --gen_batch 256 --micro_bs 32, all integrity checks,
# --stop_flat_frac 0.9 --stop_patience 25):
#   R1 "B-KL"         Phase B, nK_text, --kl_beta 0.04, seed 0, (a)(b)(c) reused from run 1 (results/grpo/20261118_013749)
#   R2 "L10-solo"     --phase S --neuron L10N4358, prompt "Text:\n", --kl_beta 0, own (a)(b)(c)
#   R3 "L10-solo-KL"  R2 with --kl_beta 0.04, (a)(b)(c) reused from R2
#   R4 "B-KL seed 1"  R1 with --seed 1
# Reuse is only done if the GPU name (and prompt, sizes, FineWeb max, model revision) match; otherwise grpo_phase.py
# recomputes and logs why. Phase B uses run 1's passed gate (--gate_from). An integrity failure (grpo_phase.py exit 3)
# skips the remaining runs; any other failure is logged and the next run starts.
# Steps before the runs: re-arm the watchdog, phase 1 setup check (abort on failure), data/ sha256 check (abort),
# FineWeb max on this GPU (abort). Then (EXIT trap, also on failure): reports/<ts>_collapse_fix.md
# (scripts/collapse_report.py), commit (small files only), push, "runpodctl stop pod" (never remove/terminate).
# The run part executes in a child process watched by scripts/watch_and_stop.sh (KIND=grpo, OUTER_PID=this script),
# so the pod is stopped even if this script dies without its EXIT trap.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
POD_ID=${RUNPOD_POD_ID:?RUNPOD_POD_ID not set}
unset DRY_RUN FORCE_PHASE_B TRANSFORMERS_OFFLINE
export HF_HOME=/workspace/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
TS=$(date +%Y%m%d_%H%M%S)
SMOKE=${SMOKE:-0}
TOP=$REPO/results/grpo_collapse/$TS
[ "$SMOKE" = 1 ] && TOP=$REPO/results/smoke/${TS}_collapse_pipeline
LOGD=$REPO/logs/collapse_run_$TS
STATUS=$TOP/pipeline_status.txt
RUN1=results/grpo/20261118_013749
COMMON="--gen_batch 256 --micro_bs 32 --steps 300 --stop_flat_frac 0.9 --stop_patience 25"
# Testing only: SMOKE=1 -> tiny runs (2 steps, 2 fresh samples, (a)/(b) sizes 4/8: reuse from run 1 is refused for size),
# results/smoke/<ts>_collapse_pipeline, report moved there, no commit, no push, no pod stop (watcher in DRY_RUN mode).
if [ "$SMOKE" = 1 ]; then
  COMMON="--gen_batch 256 --micro_bs 32 --steps 2 --n_fresh 2 --n_untrained_best 4 --n_matched 8 --stop_flat_frac 0.9 --stop_patience 25"
fi
SHA_BASELINE=70cde47f08fbcd786f1d0dfacd3e858526fee54b6845e41cebe70c978d649817
SHA_CONTEXTS=d519c23a590961df190291e916464bcb186f1942500fc465dca2187f7143ff2f
RUN_TIMEOUT=${RUN_TIMEOUT:-18000}     # 5 h per run
[ -e "$TOP" ] && { echo "ERROR: $TOP exists (never overwrite results)"; exit 1; }
mkdir -p "$TOP" "$LOGD" "$REPO/reports"
log() { echo "[collapse $(date '+%F %T')] $*" | tee -a "$STATUS"; }

finish() {
  local code=$?
  trap - EXIT
  log "pipeline ending (last exit $code); writing comparison report"
  timeout 1800 python scripts/collapse_report.py "$TOP" "$STATUS" 2>&1 | tail -3 | tee -a "$STATUS" || log "report step failed"
  if [ "$SMOKE" = 1 ]; then
    mv "$(ls -t reports/*_collapse_fix.md | head -1)" "$TOP/" && log "SMOKE: report moved to $TOP; no commit, push or pod stop"
    exit 0
  fi
  log "committing"
  git add -- scripts hermes CLAUDE.md README.md .gitignore reports logs 2>&1 | tail -2
  git add -- "$TOP" results/grpo_tests results/smoke 2>&1 | tail -2
  for f in "$TOP"/R*/phase*/b_untrained_scores_*.npy; do [ -f "$f" ] && git add -f -- "$f"; done   # tiny score arrays
  git diff --cached --name-only | while read -r f; do
    if [ -f "$f" ] && { [ "$(stat -c %s "$f")" -gt 5000000 ] || [[ "$f" =~ \.(pt|npz|safetensors|bin)$ ]] || [[ "$f" =~ /adapter/ ]]; }; then
      git reset -q -- "$f"; echo "unstaged $f (size/type)" >> "$STATUS"; fi
  done
  if git diff --cached --quiet; then log "nothing to commit"; else
    git commit -q -m "Collapse-fix experiments $TS: KL penalty, L10 solo, seed check; comparison report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "committed: $(git log --oneline -1)"
  fi
  gh auth status > /dev/null 2>&1 && gh auth setup-git > /dev/null 2>&1
  if GIT_TERMINAL_PROMPT=0 timeout 300 git push origin HEAD:main >> "$STATUS" 2>&1; then log "pushed to origin main"
  else log "push FAILED (not authenticated or offline); commit is local only"; fi
  if [ "${NO_STOP:-0}" = 1 ]; then log "NO_STOP=1: not stopping the pod"; exit 0; fi
  log "stopping pod $POD_ID (runpodctl stop pod; never remove/terminate)"
  for i in 1 2 3 4 5; do
    runpodctl stop pod "$POD_ID" >> "$STATUS" 2>&1 && { log "stop command sent"; exit 0; }
    log "runpodctl stop failed (attempt $i); retrying in 60s"; sleep 60
  done
  log "ERROR: could not stop pod"
}
trap finish EXIT

runs_part() {     # the GPU work; runs in a child process (watched)
  set -u
  log "watchdog: $(bash scripts/ensure_watchdog.sh 2>&1)"
  log "start: dir $TOP, git $(git rev-parse --short HEAD), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader), pod $POD_ID"
  printf '%s  %s\n' "$SHA_BASELINE" data/baseline.pt "$SHA_CONTEXTS" data/contexts.npz | sha256sum -c - >> "$STATUS" 2>&1 \
    || { log "ABORT: data/ sha256 mismatch"; return 1; }
  log "data/ sha256 verified"
  python scripts/phase1_setup_check.py > "$LOGD/phase1.log" 2>&1
  local p1=$?
  log "phase 1 setup check exit $p1 ($(grep -cE '^\s+PASS' "$LOGD/phase1.log") PASS, $(grep -cE '^\s+FAIL' "$LOGD/phase1.log") FAIL)"
  [ $p1 -eq 0 ] || { log "ABORT: phase 1 setup check failed"; return 1; }
  python scripts/grpo_fwmax.py "$TOP" > "$LOGD/fwmax.log" 2>&1 || { log "ABORT: grpo_fwmax.py failed"; return 1; }
  grep -E "^L[0-9]+N" "$LOGD/fwmax.log" >> "$STATUS"
  local spec label name dir args code
  for spec in \
    "R1|B-KL|--phase B --phase_b_prompt nK_text --kl_beta 0.04 --seed 0 --reuse_untrained_from $RUN1 --gate_from $RUN1" \
    "R2|L10-solo|--phase S --neuron L10N4358 --kl_beta 0 --seed 0" \
    "R3|L10-solo-KL|--phase S --neuron L10N4358 --kl_beta 0.04 --seed 0 --reuse_untrained_from ${TOP#$REPO/}/R2" \
    "R4|B-KL seed 1|--phase B --phase_b_prompt nK_text --kl_beta 0.04 --seed 1 --reuse_untrained_from $RUN1 --gate_from $RUN1"; do
    IFS='|' read -r label name args <<< "$spec"
    dir=$TOP/$label; mkdir -p "$dir"; cp "$TOP/fineweb_max.json" "$TOP/fwmax_config.json" "$dir/"
    log "START $label \"$name\": grpo_phase.py $args $COMMON"
    timeout "$RUN_TIMEOUT" python scripts/grpo_phase.py --run_dir "$dir" $args $COMMON > "$LOGD/$label.log" 2>&1
    code=$?
    log "END $label exit $code"
    grep -E "PROJECTED|EARLY STOP|REUSED|NOT reusing|INTEGRITY FAIL|REFUSING" "$LOGD/$label.log" | cut -c1-600 >> "$STATUS"
    grep -E "check \(" "$LOGD/$label.log" | grep -v PASS >> "$STATUS"
    if [ $code -eq 3 ]; then log "INTEGRITY FAILURE in $label: skipping the remaining runs, going to the report"; return 3; fi
    [ $code -eq 0 ] || { log "$label failed (exit $code); continuing with the next run"; tail -20 "$LOGD/$label.log" >> "$STATUS"; }
  done
  log "all runs done"
}

python - "$TOP" "$RUN1" <<'EOF'
import json, sys, os
top, run1 = sys.argv[1], sys.argv[2]
rel = os.path.relpath(top, os.getcwd())
json.dump(dict(run1=run1, runs=[dict(label="R1", name="B-KL", dir=f"{rel}/R1"), dict(label="R2", name="L10-solo", dir=f"{rel}/R2"),
                                dict(label="R3", name="L10-solo-KL", dir=f"{rel}/R3"), dict(label="R4", name="B-KL seed 1", dir=f"{rel}/R4")]),
          open(f"{top}/runs.json", "w"), indent=1)
EOF
log "collapse_run start (outer PID $$)"
runs_part > "$LOGD/runs_part.log" 2>&1 &
PIPE_PID=$!
RUN_PID=$PIPE_PID RUN_DIR=$TOP RUN_LOG=$LOGD/runs_part.log TASK_OUT=$LOGD/runs_exit.txt KIND=grpo OUTER_PID=$$ \
  OUTER_TIMEOUT=3600 DRY_RUN=$SMOKE BACKUP_PATH=$REPO/code_backup_$TS.tar.gz \
  setsid nohup bash scripts/watch_and_stop.sh > "$LOGD/watcher.log" 2>&1 < /dev/null &
log "runs PID $PIPE_PID, watcher PID $! (log $LOGD/watcher.log)"
echo "$PIPE_PID" > "$LOGD/runs.pid"
wait "$PIPE_PID"; CODE=$?
echo "exit $CODE" > "$LOGD/runs_exit.txt"
log "runs part exit $CODE"
# EXIT trap: report, commit, push, stop pod
