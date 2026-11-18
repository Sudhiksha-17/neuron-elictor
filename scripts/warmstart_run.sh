#!/bin/bash
# Warm-start experiments (2026-11-18, RTX 4090), unattended, as ONE pipeline. Launch from the repo root:
#   setsid nohup scripts/warmstart_run.sh > logs/warmstart_run_$(date +%Y%m%d_%H%M%S).out 2>&1 < /dev/null &
# Design (user, 2026-11-18; following MAEM arXiv 2611.08465, NLA as comparison): cross-entropy warm start on the
# neuron's top-10 FineWeb contexts, then GRPO. Main recipe no KL; comparison KL toward the warm-started policy (beta
# 0.01). Never KL toward the untrained base in the warm-start runs.
# Runs, in order (all: 300 steps, no early stop, unmerged generation, --gen_batch 192 --micro_bs 32, all integrity checks):
#   Rep       run 1 exactly (beta 0, no warm start), seed 1: Phase A (L14N8949) -> gate (Phase A (d) > FineWeb max,
#             strict, as in run 1) -> Phase B (4 neurons, prompt nK_text = run 1's pre-registered choice)
#   W0-L9-s0  --phase S --neuron L9N5711 --prompt "Text:\n" --warm_start --kl_beta 0, seed 0
#   W0-L9-s1  same, seed 1                    ((a)(b)(c) reused from W0-L9-s0: same GPU, prompt, sizes)
#   W0-L10-s0 --phase S --neuron L10N4358, same, seed 0
#   W0-L10-s1 same, seed 1                    (reuse from W0-L10-s0)
#   WKL-L10   L10N4358 --warm_start --kl_ref warm --kl_beta 0.01, seed 0   (reuse from W0-L10-s0)
# Untrained baselines and FineWeb maxes are computed on this GPU (reuse only within this pipeline; grpo_phase.py
# checks the GPU name). PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (also set by hermes/__init__.py; logged in
# every run config via hermes/runinfo.py).
# Steps before the runs: re-arm the watchdog, data/ sha256 check (abort), phase 1 setup check (abort), FineWeb max on
# this GPU (abort). An integrity failure (grpo_phase.py exit 3) skips the remaining runs; any other failure is logged
# and the next run starts. Then (EXIT trap, also on failure): reports/<ts>_warmstart.md (scripts/warmstart_report.py),
# commit (small files only), push, "runpodctl stop pod" (never remove/terminate). The run part executes in a child
# process watched by scripts/watch_and_stop.sh (KIND=grpo, OUTER_PID=this script), so the pod is stopped even if this
# script dies without its EXIT trap.
# SMOKE=1 (testing only): tiny --dry runs (2 steps, small evals; Phase B runs regardless of the gate), results under
# results/smoke/<ts>_warmstart_pipeline, report moved there, no commit, no push, no pod stop.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
POD_ID=${RUNPOD_POD_ID:?RUNPOD_POD_ID not set}
unset DRY_RUN FORCE_PHASE_B TRANSFORMERS_OFFLINE
export HF_HOME=/workspace/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
TS=$(date +%Y%m%d_%H%M%S)
SMOKE=${SMOKE:-0}
TOP=$REPO/results/grpo_warmstart/$TS
[ "$SMOKE" = 1 ] && TOP=$REPO/results/smoke/${TS}_warmstart_pipeline
LOGD=$REPO/logs/warmstart_run_$TS
STATUS=$TOP/pipeline_status.txt
COMMON="--gen_batch 192 --micro_bs 32 --steps 300"
[ "$SMOKE" = 1 ] && COMMON="--gen_batch 192 --micro_bs 32 --dry --steps 2 --n_fresh 2 --n_untrained_best 4 --n_matched 8 --n_diversity 8"
SHA_BASELINE=70cde47f08fbcd786f1d0dfacd3e858526fee54b6845e41cebe70c978d649817
SHA_CONTEXTS=d519c23a590961df190291e916464bcb186f1942500fc465dca2187f7143ff2f
RUN_TIMEOUT=${RUN_TIMEOUT:-18000}     # 5 h per grpo_phase.py call
[ -e "$TOP" ] && { echo "ERROR: $TOP exists (never overwrite results)"; exit 1; }
mkdir -p "$TOP" "$LOGD" "$REPO/reports"
log() { echo "[warmstart $(date '+%F %T')] $*" | tee -a "$STATUS"; }

finish() {
  local code=$?
  trap - EXIT
  log "pipeline ending (last exit $code); writing report"
  timeout 1800 python scripts/warmstart_report.py "$TOP" "$STATUS" 2>&1 | tail -3 | tee -a "$STATUS" || log "report step failed"
  if [ "$SMOKE" = 1 ]; then
    mv "$(ls -t reports/*_warmstart.md | head -1)" "$TOP/" && log "SMOKE: report moved to $TOP; no commit, push or pod stop"
    exit 0
  fi
  log "committing"
  git add -- scripts hermes CLAUDE.md README.md .gitignore reports 2>&1 | tail -2
  git add -f -- "$LOGD" logs/watchdog.log 2>&1 | tail -2
  git add -- "$TOP" 2>&1 | tail -2
  for f in "$TOP"/*/phase*/b_untrained_scores_*.npy; do [ -f "$f" ] && git add -f -- "$f"; done   # tiny score arrays
  git diff --cached --name-only | while read -r f; do
    if [ -f "$f" ] && { [ "$(stat -c %s "$f")" -gt 5000000 ] || [[ "$f" =~ \.(pt|npz|safetensors|bin)$ ]] || [[ "$f" =~ /adapter/ ]]; }; then
      git reset -q -- "$f"; echo "unstaged $f (size/type)" >> "$STATUS"; fi
  done
  if git diff --cached --quiet; then log "nothing to commit"; else
    git commit -q -m "Warm-start experiments $TS: Rep, W0-L9 x2, W0-L10 x2, WKL-L10; report

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

run_phase() {   # run_phase <label> <dir> <args...>; returns grpo_phase.py's exit code
  local label=$1 dir=$2; shift 2
  log "START $label: grpo_phase.py --run_dir ${dir#$REPO/} $* $COMMON"
  timeout "$RUN_TIMEOUT" python scripts/grpo_phase.py --run_dir "$dir" "$@" $COMMON > "$LOGD/$label.log" 2>&1
  local code=$?
  log "END $label exit $code"
  grep -E "PROJECTED|EARLY STOP|REUSED|NOT reusing|INTEGRITY FAIL|REFUSING|diversity pre-check|redoing the warm start|warm start epoch|\(w\)|\(d\)" \
    "$LOGD/$label.log" | cut -c1-600 >> "$STATUS"
  grep -E "check \(" "$LOGD/$label.log" | grep -v PASS >> "$STATUS"
  [ $code -eq 0 ] || [ $code -eq 3 ] || { log "$label failed (exit $code)"; tail -20 "$LOGD/$label.log" >> "$STATUS"; }
  return $code
}

runs_part() {     # the GPU work; runs in a child process (watched)
  set -u
  log "watchdog: $(bash scripts/ensure_watchdog.sh 2>&1)"
  log "start: dir $TOP, git $(git rev-parse --short HEAD), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader), pod $POD_ID, PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
  printf '%s  %s\n' "$SHA_BASELINE" data/baseline.pt "$SHA_CONTEXTS" data/contexts.npz | sha256sum -c - >> "$STATUS" 2>&1 \
    || { log "ABORT: data/ sha256 mismatch"; return 1; }
  log "data/ sha256 verified"
  python scripts/phase1_setup_check.py > "$LOGD/phase1.log" 2>&1
  local p1=$?
  log "phase 1 setup check exit $p1 ($(grep -cE '^\s+PASS' "$LOGD/phase1.log") PASS, $(grep -cE '^\s+FAIL' "$LOGD/phase1.log") FAIL)"
  [ $p1 -eq 0 ] || { log "ABORT: phase 1 setup check failed"; return 1; }
  python scripts/grpo_fwmax.py "$TOP" > "$LOGD/fwmax.log" 2>&1 || { log "ABORT: grpo_fwmax.py failed"; return 1; }
  grep -E "^L[0-9]+N" "$LOGD/fwmax.log" >> "$STATUS"
  local d code rel=${TOP#$REPO/}
  for d in Rep W0-L9-s0 W0-L9-s1 W0-L10-s0 W0-L10-s1 WKL-L10; do
    mkdir -p "$TOP/$d"; cp "$TOP/fineweb_max.json" "$TOP/fwmax_config.json" "$TOP/$d/"
  done

  # Rep: run 1's Phase A -> gate -> Phase B, seed 1, beta 0, no warm start
  run_phase Rep-A "$TOP/Rep" --phase A --kl_beta 0 --seed 1; code=$?
  [ $code -eq 3 ] && { log "INTEGRITY FAILURE in Rep-A: skipping the remaining runs"; return 3; }
  python scripts/grpo_report.py gate "$TOP/Rep" 2>&1 | tee -a "$STATUS"; local gate=${PIPESTATUS[0]}
  if [ "$gate" -eq 0 ] || [ "$SMOKE" = 1 ]; then
    [ "$gate" -ne 0 ] && log "SMOKE: gate failed, running Phase B anyway (--dry)"
    run_phase Rep-B "$TOP/Rep" --phase B --phase_b_prompt nK_text --kl_beta 0 --seed 1; code=$?
    [ $code -eq 3 ] && { log "INTEGRITY FAILURE in Rep-B: skipping the remaining runs"; return 3; }
  else log "gate FAILED: Rep Phase B skipped (as in run 1's pipeline)"; fi

  local spec label args
  for spec in \
    "W0-L9-s0|--phase S --neuron L9N5711 --warm_start --kl_beta 0 --seed 0" \
    "W0-L9-s1|--phase S --neuron L9N5711 --warm_start --kl_beta 0 --seed 1 --reuse_untrained_from $rel/W0-L9-s0" \
    "W0-L10-s0|--phase S --neuron L10N4358 --warm_start --kl_beta 0 --seed 0" \
    "W0-L10-s1|--phase S --neuron L10N4358 --warm_start --kl_beta 0 --seed 1 --reuse_untrained_from $rel/W0-L10-s0" \
    "WKL-L10|--phase S --neuron L10N4358 --warm_start --kl_ref warm --kl_beta 0.01 --seed 0 --reuse_untrained_from $rel/W0-L10-s0"; do
    IFS='|' read -r label args <<< "$spec"
    # shellcheck disable=SC2086
    run_phase "$label" "$TOP/$label" $args; code=$?      # prompt: grpo_phase.py's --phase S default "Text:\n"
    [ $code -eq 3 ] && { log "INTEGRITY FAILURE in $label: skipping the remaining runs"; return 3; }
  done
  log "all runs done"
}

python - "$TOP" <<'EOF'
import json, sys, os
top = sys.argv[1]; rel = os.path.relpath(top, os.getcwd())
runs = [dict(label="Rep", name="Rep (run 1 exactly, seed 1)", dir=f"{rel}/Rep", warm=False, kl_ref="base", kl_beta=0.0, seed=1),
        dict(label="W0-L9-s0", name="W0-L9 seed 0", dir=f"{rel}/W0-L9-s0", warm=True, kl_ref="warm", kl_beta=0.0, seed=0, neuron="L9N5711"),
        dict(label="W0-L9-s1", name="W0-L9 seed 1", dir=f"{rel}/W0-L9-s1", warm=True, kl_ref="warm", kl_beta=0.0, seed=1, neuron="L9N5711"),
        dict(label="W0-L10-s0", name="W0-L10 seed 0", dir=f"{rel}/W0-L10-s0", warm=True, kl_ref="warm", kl_beta=0.0, seed=0, neuron="L10N4358"),
        dict(label="W0-L10-s1", name="W0-L10 seed 1", dir=f"{rel}/W0-L10-s1", warm=True, kl_ref="warm", kl_beta=0.0, seed=1, neuron="L10N4358"),
        dict(label="WKL-L10", name="WKL-L10 (KL to warm, beta 0.01) seed 0", dir=f"{rel}/WKL-L10", warm=True, kl_ref="warm", kl_beta=0.01, seed=0, neuron="L10N4358")]
json.dump(dict(runs=runs), open(f"{top}/runs.json", "w"), indent=1)
EOF
log "warmstart_run start (outer PID $$)"
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
