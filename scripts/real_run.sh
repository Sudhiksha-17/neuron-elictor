#!/bin/bash
# The REAL GRPO run, unattended, on a fresh GPU pod (written 2026-11-17 on a CPU-only pod: syntax-checked only,
# UNTESTED end to end). Launch from the repo root:
#   setsid nohup scripts/real_run.sh > logs/real_run_$(date +%Y%m%d_%H%M%S).out 2>&1 < /dev/null &
# Steps:
#   w. re-arm the idle watchdog (scripts/ensure_watchdog.sh: deletes logs/watchdog.disable, starts idle_watchdog.sh)
#   0. install pinned packages + pinned model (scripts/install_env.sh, network)            abort on failure
#   d. data/: download baseline.pt / contexts.npz from HF gray171/hermes-baseline if missing, copy pilots.json from
#      results/pilots/20261109_213918_full1M_pilots/ if missing; ALWAYS verify the sha256 hashes (CLAUDE.md, README)
#                                                                                           abort on failure
#   1. environment check (scripts/gpu_bench.py env, HF_HUB_OFFLINE=1 from here on)         abort on failure
#   s. SMOKE TEST (scripts/smoke_test.sh, added 2026-11-18): 1 step Phase A + 1 step Phase B (both prompts nK and
#      nK_text) + tiny prompt check, tiny eval sizes, throwaway dir results/smoke/<ts>     abort on failure (report,
#      commit, pod stop; the real run is NOT started)
#   2. Phase B prompt check, pre-registered rule (scripts/phase_b_prompt_check.py): FineWeb max on this GPU, then
#      best of 64 untrained samples per Phase B neuron for (a) "<|nK|>\n" and (b) "<|nK|>\nText:\n"; (b) iff its mean
#      best-of-64 ratio to FineWeb max is strictly higher, else (a). If the check fails, (a) is kept (logged).
#   3. REAL pipeline: scripts/grpo_pipeline.sh --no_stop (no --dry, no --force_phase_b; the gate decides Phase B),
#      unmerged LoRA generation (no --merged_gen), --gen_batch 192 --micro_bs 32, Phase B prompt from step 2.
#      Runs under scripts/watch_and_stop.sh (KIND=grpo, OUTER_PID=this script): if this script dies without its
#      EXIT trap, the watcher still stops the pod. Pipeline timeout 12 h.
# Then (EXIT trap, also on failure): reports/<ts>_real_run.md, commit (small files only), push if git is
# authenticated, "runpodctl stop pod $RUNPOD_POD_ID" (never remove/terminate).
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
POD_ID=${RUNPOD_POD_ID:?RUNPOD_POD_ID not set}
unset DRY_RUN FORCE_PHASE_B HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_HOME=/workspace/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
TS=$(date +%Y%m%d_%H%M%S)
LOGD=$REPO/logs/real_run_$TS
REPORT=$REPO/reports/${TS}_real_run.md
CHK=$REPO/results/phase_b_prompt_check/$TS
RUN=$REPO/results/grpo/$TS
SMOKE=$REPO/results/smoke/$TS
HF_DATA=https://huggingface.co/datasets/gray171/hermes-baseline/resolve/main
SHA_BASELINE=70cde47f08fbcd786f1d0dfacd3e858526fee54b6845e41cebe70c978d649817
SHA_CONTEXTS=d519c23a590961df190291e916464bcb186f1942500fc465dca2187f7143ff2f
PILOTS_SRC=$REPO/results/pilots/20261109_213918_full1M_pilots/pilots.json
RUN_ARGS="--gen_batch 192 --micro_bs 32"      # 192 decided 2026-11-18 (RTX 4090: 256 OOMs); no --merged_gen
B_PROMPT=nK; B_PROMPT_NOTE="not run"
PIPE_CODE=not_run
SMOKE_RESULT=not_run
mkdir -p "$LOGD" "$REPO/reports"
START_EPOCH=$(date +%s)
STEPS=()          # "name|exit|seconds|log"
log() { echo "[real_run $(date '+%F %T')] $*" | tee -a "$LOGD/session.log"; }

run_step() {      # run_step <name> <timeout_s> <cmd...>
  local name=$1 to=$2; shift 2
  local lf="$LOGD/$(printf '%s' "$name" | tr -c 'A-Za-z0-9._-' '_').log" t0=$(date +%s)
  log "START $name: $*"
  timeout "$to" "$@" > "$lf" 2>&1
  local code=$?
  [ $code -eq 124 ] && echo "TIMEOUT after ${to}s" >> "$lf"
  STEPS+=("$name|$code|$(( $(date +%s) - t0 ))|$lf")
  log "END $name: exit $code ($(( $(date +%s) - t0 ))s)"
  return $code
}

ensure_data() {   # download missing inputs, then verify both hashes (always)
  mkdir -p data
  local f
  for f in baseline.pt contexts.npz; do
    if [ ! -s "data/$f" ]; then
      echo "downloading $f from $HF_DATA/$f"
      curl -fL --retry 5 --retry-delay 20 -o "data/$f.part" "$HF_DATA/$f" && mv "data/$f.part" "data/$f" || return 1
    else echo "data/$f present"; fi
  done
  if [ ! -s data/pilots.json ]; then cp "$PILOTS_SRC" data/pilots.json && echo "copied pilots.json from $PILOTS_SRC" || return 1; fi
  cmp -s data/pilots.json "$PILOTS_SRC" || { echo "FAIL data/pilots.json differs from $PILOTS_SRC"; return 1; }
  printf '%s  %s\n' "$SHA_BASELINE" data/baseline.pt "$SHA_CONTEXTS" data/contexts.npz | sha256sum -c - || return 1
  echo "PASS data/ sha256 verified"
}

write_report() {
  {
    echo "# Real GRPO run $TS"
    echo
    echo "Unattended run of scripts/real_run.sh. Commit $(git rev-parse --short HEAD) at start of report,"
    echo "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1), pod $POD_ID,"
    echo "total $(( ($(date +%s) - START_EPOCH) / 60 )) min. Logs: \`logs/real_run_$TS/\`."
    echo
    echo "Settings: real pipeline (no --dry, no --force_phase_b), unmerged LoRA generation, \`$RUN_ARGS\`,"
    echo "Phase B prompt \`$B_PROMPT\` ($B_PROMPT_NOTE). Pipeline exit: $PIPE_CODE."
    echo
    echo "## Steps"
    echo
    echo "| step | exit | minutes | log |"
    echo "|---|---|---|---|"
    for s in "${STEPS[@]}"; do
      IFS='|' read -r n c sec lf <<< "$s"
      echo "| $n | $([ "$c" = 0 ] && echo 0 || echo "**$c**") | $(( sec / 60 )).$(( (sec % 60) / 6 )) | \`${lf#$REPO/}\` |"
    done
    echo
    echo "## Smoke test: $SMOKE_RESULT"
    echo
    if [ -f "$LOGD/s_smoke_test.log" ]; then
      echo "Dir \`${SMOKE#$REPO/}\`, log \`logs/real_run_$TS/s_smoke_test.log\` (progress lines + last 30 lines):"; echo; echo '```'
      grep -E '^\[smoke' "$LOGD/s_smoke_test.log"; echo '...'; tail -30 "$LOGD/s_smoke_test.log"; echo '```'
    fi
    echo
    echo "## Phase B prompt check (pre-registered rule)"
    echo
    if [ -f "$CHK/prompt_check.json" ]; then
      echo "Results: \`${CHK#$REPO/}/prompt_check.json\`"; echo; echo '```'
      grep -E '^\((a|b)\)|^choice' "$LOGD"/2b_*.log 2>/dev/null; echo '```'
    else echo "No prompt_check.json; Phase B prompt kept at (a) \`<|nK|>\\n\` ($B_PROMPT_NOTE)."; fi
    echo
    echo "## Pipeline (\`${RUN#$REPO/}\`)"
    echo
    echo '```'; cat "$RUN/pipeline_status.txt" 2>/dev/null || echo "(no pipeline_status.txt)"; echo '```'
    local fr; fr=$(ls -t "$REPO"/reports/*_grpo_final.md 2>/dev/null | head -1)
    if [ -n "$fr" ] && [ "$(stat -c %Y "$fr")" -ge "$START_EPOCH" ]; then
      echo; echo "Full GRPO report: \`${fr#$REPO/}\`"; echo; cat "$fr"
    fi
  } > "$REPORT" 2>&1
}

finish() {
  local code=$?
  trap - EXIT
  log "real_run ending (last exit $code); writing report $REPORT"
  write_report || log "report writing failed"
  log "committing"
  git add -- scripts hermes CLAUDE.md README.md .gitignore reports logs 2>&1 | tail -2
  git add -- results/gpu_bench results/phase_b_prompt_check "$RUN" "$SMOKE" 2>&1 | tail -2
  git diff --cached --name-only | while read -r f; do
    if [ -f "$f" ] && { [ "$(stat -c %s "$f")" -gt 5000000 ] || [[ "$f" =~ \.(pt|npz|npy|safetensors|bin)$ ]]; }; then
      git reset -q -- "$f"; log "unstaged $f (size/type)"; fi
  done
  if git diff --cached --quiet; then log "nothing to commit"; else
    git commit -q -m "Real GRPO run $TS: prompt check, pipeline results, report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "committed: $(git log --oneline -1)"
  fi
  gh auth status > /dev/null 2>&1 && gh auth setup-git > /dev/null 2>&1
  if GIT_TERMINAL_PROMPT=0 timeout 300 git push origin HEAD:main >> "$LOGD/session.log" 2>&1; then log "pushed to origin main"
  else log "push FAILED (not authenticated or offline); commit is local only"; fi
  log "stopping pod $POD_ID (runpodctl stop pod; never remove/terminate)"
  for i in 1 2 3 4 5; do
    runpodctl stop pod "$POD_ID" >> "$LOGD/session.log" 2>&1 && { log "stop command sent"; exit 0; }
    log "runpodctl stop failed (attempt $i); retrying in 60s"; sleep 60
  done
  log "ERROR: could not stop pod"
}
trap finish EXIT

log "real run start: commit $(git rev-parse --short HEAD), pod $POD_ID"
log "watchdog: $(bash "$REPO/scripts/ensure_watchdog.sh" 2>&1)"
if pgrep -f grpo_pipeline.sh > /dev/null; then log "ABORT: a grpo_pipeline.sh is already running"; exit 1; fi

run_step "0 install packages + model (network)" 3600 bash scripts/install_env.sh || { log "ABORT: install failed"; exit 1; }
export HF_DATA SHA_BASELINE SHA_CONTEXTS PILOTS_SRC
run_step "d data download + sha256" 3600 bash -c "$(declare -f ensure_data); ensure_data" \
  || { log "ABORT: data/ missing or hash mismatch"; exit 1; }
export HF_HUB_OFFLINE=1
run_step "1 environment check" 1200 python scripts/gpu_bench.py env || { log "ABORT: environment check failed"; exit 1; }

if run_step "s smoke test" 3600 bash scripts/smoke_test.sh "$SMOKE"; then SMOKE_RESULT=PASS
else SMOKE_RESULT=FAIL; log "ABORT: smoke test failed; real run NOT started"; exit 1; fi

# Phase B prompt check (pre-registered): FineWeb max on this GPU into its own dir, then (a) vs (b)
mkdir -p "$CHK"
if run_step "2a FineWeb max this GPU (prompt check)" 1800 python scripts/grpo_fwmax.py "$CHK" \
   && run_step "2b Phase B prompt check" 3600 python scripts/phase_b_prompt_check.py "$CHK" --n 64; then
  B_PROMPT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["choice"])' "$CHK/prompt_check.json")
  B_PROMPT_NOTE="rule: $(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["choice_reason"])' "$CHK/prompt_check.json")"
  case "$B_PROMPT" in nK|nK_text) ;; *) log "unexpected choice '$B_PROMPT'; keeping nK"; B_PROMPT=nK; B_PROMPT_NOTE="bad choice value, kept (a)";; esac
else
  B_PROMPT=nK; B_PROMPT_NOTE="prompt check FAILED, kept (a)"
fi
log "Phase B prompt: $B_PROMPT ($B_PROMPT_NOTE)"

# REAL pipeline under watch_and_stop.sh
export PHASE_A_ARGS="$RUN_ARGS" PHASE_B_ARGS="$RUN_ARGS --phase_b_prompt $B_PROMPT"
log "START 3 real pipeline: RUN_DIR=$RUN PHASE_A_ARGS='$PHASE_A_ARGS' PHASE_B_ARGS='$PHASE_B_ARGS'"
T3=$(date +%s); PIPE_LOG=$LOGD/3_pipeline.log
RUN_DIR=$RUN timeout 43200 bash scripts/grpo_pipeline.sh --no_stop > "$PIPE_LOG" 2>&1 &
PIPE_PID=$!
RUN_PID=$PIPE_PID RUN_DIR=$RUN RUN_LOG=$PIPE_LOG TASK_OUT=$LOGD/pipeline_exit.txt KIND=grpo OUTER_PID=$$ \
  OUTER_TIMEOUT=3600 DRY_RUN=0 BACKUP_PATH=$REPO/code_backup_$TS.tar.gz \
  setsid nohup bash scripts/watch_and_stop.sh > "$LOGD/watcher.log" 2>&1 < /dev/null &
log "pipeline PID $PIPE_PID, watcher PID $! (log $LOGD/watcher.log)"
wait "$PIPE_PID"; PIPE_CODE=$?
echo "exit $PIPE_CODE" > "$LOGD/pipeline_exit.txt"
[ $PIPE_CODE -eq 124 ] && echo "TIMEOUT after 43200s" >> "$PIPE_LOG"
STEPS+=("3 real pipeline|$PIPE_CODE|$(( $(date +%s) - T3 ))|$PIPE_LOG")
log "END 3 real pipeline: exit $PIPE_CODE"
log "all steps done"
# EXIT trap: report, commit, push, stop pod
