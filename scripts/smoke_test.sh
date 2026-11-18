#!/bin/bash
# Smoke test of the real GRPO code path (added 2026-11-18): 1 step of Phase A and 1 step of Phase B with tiny eval
# sizes, Phase B once per prompt variant (nK and nK_text), plus a tiny Phase B prompt check (--n 4) and the gate.
# Same generation / micro-batch settings as the real run (--gen_batch 192 --micro_bs 32, unmerged LoRA generation).
# Uses --dry (so Phase B runs without a passed gate) with explicit tiny sizes. Throwaway run dir (must not exist):
#   bash scripts/smoke_test.sh results/smoke/<ts>
# Called by scripts/real_run.sh after the environment check; exits non-zero on the first failure.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
S=${1:?usage: smoke_test.sh <throwaway run dir>}
[ -e "$S" ] && { echo "ERROR: $S exists (never overwrite results)"; exit 1; }
export HF_HOME=/workspace/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
ARGS="--gen_batch 192 --micro_bs 32"
TINY="--dry --steps 1 --n_untrained_best 4 --n_matched 4 --n_fresh 2 --sample_every 1"
say() { echo "[smoke $(date '+%F %T')] $*"; }
mkdir -p "$S"
T0=$(date +%s)

say "1/6 FineWeb max on this GPU"
python scripts/grpo_fwmax.py "$S"
say "2/6 Phase A, 1 step"
python scripts/grpo_phase.py --phase A --run_dir "$S" $TINY $ARGS
say "3/6 gate (exit 0 or 1 both fine; anything else is a crash)"
set +e; python scripts/grpo_report.py gate "$S"; g=$?; set -e
[ $g -le 1 ] || { say "FAIL gate crashed (exit $g)"; exit 1; }
say "4/6 Phase B prompt check, n=4"
python scripts/phase_b_prompt_check.py "$S" --n 4
grep -q '"choice"' "$S/prompt_check.json"
say "5/6 Phase B, 1 step, prompt nK"
python scripts/grpo_phase.py --phase B --run_dir "$S" $TINY $ARGS --phase_b_prompt nK
say "6/6 Phase B, 1 step, prompt nK_text"
mkdir -p "$S/nK_text"; cp "$S/fineweb_max.json" "$S/nK_text/"
python scripts/grpo_phase.py --phase B --run_dir "$S/nK_text" $TINY $ARGS --phase_b_prompt nK_text
say "SMOKE PASS in $(( $(date +%s) - T0 ))s ($S)"
