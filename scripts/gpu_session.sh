#!/bin/bash
# One unattended GPU session (written 2026-11-16 on a CPU-only pod: UNTESTED, run on GPU).
#   setsid nohup scripts/gpu_session.sh > logs/gpu_session_$(date +%Y%m%d_%H%M%S).out 2>&1 < /dev/null &
# Steps (each with a timeout; a failing step is recorded and the session continues, except step 1):
#   1. environment check: GPU, versions, pinned model loads OFFLINE from /workspace/hf_cache, fast kernels, data/
#      (if this fails, steps 2-5 are skipped: they would all fail the same way)
#   2. setup and tests: scripts/phase1_setup_check.py, scripts/test_scoring.py, scripts/test_grpo.py --micro_bs 32
#      (incl. the merge-path tests)
#   3. generation benchmark, KV-cache check FIRST: scripts/gpu_bench.py gen
#   4. micro-batch check: scripts/gpu_bench.py pg
#   5. short dry pipeline: scripts/grpo_pipeline.sh --dry --force_phase_b (3 steps per phase, merged generation)
# Then (EXIT trap, also on failure): writes reports/<ts>_gpu_session.md, commits locally (small files only, no
# push), and stops the pod with "runpodctl stop pod $RUNPOD_POD_ID" (never remove/terminate).
# It NEVER starts the real run: the only pipeline call is --dry.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
POD_ID=${RUNPOD_POD_ID:?RUNPOD_POD_ID not set}
export HF_HOME=/workspace/hf_cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
TS=$(date +%Y%m%d_%H%M%S)
LOGD=$REPO/logs/gpu_session_$TS
REPORT=$REPO/reports/${TS}_gpu_session.md
mkdir -p "$LOGD" "$REPO/reports"
START_EPOCH=$(date +%s)
STEPS=()          # "name|exit|seconds|log"
log() { echo "[gpu_session $(date '+%F %T')] $*" | tee -a "$LOGD/session.log"; }

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

write_report() {
  {
    echo "# GPU session $TS"
    echo
    echo "Unattended run of scripts/gpu_session.sh (steps 1-5 of the GPU plan; the real run was NOT started)."
    echo "Commit $(git rev-parse --short HEAD), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1),"
    echo "total $(( ($(date +%s) - START_EPOCH) / 60 )) min. Logs: \`logs/gpu_session_$TS/\`."
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
    echo "## Checks (PASS/FAIL lines from the step logs)"
    echo
    echo '```'
    for s in "${STEPS[@]}"; do
      IFS='|' read -r n c sec lf <<< "$s"
      echo "--- $n"
      grep -E '^\s*(PASS|FAIL)\b|WARN|TIMEOUT|Error|error:' "$lf" 2>/dev/null | grep -v FutureWarning | head -60
    done
    echo '```'
    echo
    for d in $(ls -d "$REPO"/results/gpu_bench/*_env "$REPO"/results/gpu_bench/*_gen "$REPO"/results/gpu_bench/*_pg 2>/dev/null); do
      if [ "$(stat -c %Y "$d")" -ge "$START_EPOCH" ] && [ -f "$d/summary.md" ]; then
        cat "$d/summary.md"; echo "(results: \`${d#$REPO/}/results.json\`)"; echo
      fi
    done
    local run; run=$(ls -td "$REPO"/results/grpo/*_dry 2>/dev/null | head -1)
    if [ -n "$run" ] && [ "$(stat -c %Y "$run")" -ge "$START_EPOCH" ]; then
      echo "## Dry pipeline (\`${run#$REPO/}\`)"
      echo
      echo '```'; cat "$run/pipeline_status.txt" 2>/dev/null; echo '```'
      local fr; fr=$(ls -t "$REPO"/reports/*_grpo_final.md 2>/dev/null | head -1)
      [ -n "$fr" ] && [ "$(stat -c %Y "$fr")" -ge "$START_EPOCH" ] && echo "Full dry-run report: \`${fr#$REPO/}\`"
      echo
    fi
    echo "## Not done"
    echo
    echo "- The real GRPO run was not started (needs the user's go-ahead)."
  } > "$REPORT" 2>&1
}

finish() {
  local code=$?
  trap - EXIT
  log "session ending (last exit $code); writing report $REPORT"
  write_report || log "report writing failed"
  log "committing (local only, no push)"
  git add -- scripts hermes CLAUDE.md README.md .gitignore reports logs 2>&1 | tail -2
  git add -- results/gpu_bench results/grpo results/grpo_tests results/phase1 results/scoring_tests 2>&1 | tail -2
  git diff --cached --name-only | while read -r f; do
    if [ -f "$f" ] && { [ "$(stat -c %s "$f")" -gt 5000000 ] || [[ "$f" =~ \.(pt|npz|safetensors|bin)$ ]]; }; then
      git reset -q -- "$f"; log "unstaged $f (size/type)"; fi
  done
  if git diff --cached --quiet; then log "nothing to commit"; else
    git commit -q -m "GPU session $TS: env check, tests, generation + micro-batch benchmarks, dry pipeline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "committed: $(git log --oneline -1)"
  fi
  log "stopping pod $POD_ID (runpodctl stop pod; never remove/terminate)"
  for i in 1 2 3 4 5; do
    runpodctl stop pod "$POD_ID" >> "$LOGD/session.log" 2>&1 && { log "stop command sent"; exit 0; }
    log "runpodctl stop failed (attempt $i); retrying in 60s"; sleep 60
  done
  log "ERROR: could not stop pod"
}
trap finish EXIT

log "session start: commit $(git rev-parse --short HEAD), pod $POD_ID"
if ! run_step "1 environment check" 1200 python scripts/gpu_bench.py env; then
  log "ABORT: environment check failed; skipping steps 2-5"; exit 1
fi
run_step "2a phase1 setup check" 1800 python scripts/phase1_setup_check.py
run_step "2b test_scoring L15" 1800 python scripts/test_scoring.py --layer 15
run_step "2c test_grpo (incl. merge path)" 3600 python scripts/test_grpo.py --micro_bs 32
run_step "3 generation benchmark (KV cache first)" 5400 python scripts/gpu_bench.py gen
run_step "4 micro-batch check" 3600 python scripts/gpu_bench.py pg
export PHASE_A_ARGS="--merged_gen" PHASE_B_ARGS="--merged_gen"
run_step "5 dry pipeline" 7200 bash scripts/grpo_pipeline.sh --dry --force_phase_b
log "all steps done"
# EXIT trap: report, commit, stop pod
