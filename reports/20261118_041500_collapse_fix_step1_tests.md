# Collapse-fix experiments: Step 1 test results (pipeline NOT launched)

2026-11-18, RTX 5090 pod 054chxspcxgs42. Log: `logs/test_grpo_20261118_kl.log`, results `results/grpo_tests/20261118_040622/`.

## Code added (uncommitted)

- `hermes/grpo.py`: `pg_backward(..., kl_beta=0.0, log_kl=True)` with the per-token k3 KL penalty against the LoRA-off reference.
  When kl_beta is 0 it takes the run 1 code path (adv == 0 completions skipped, same micro-batches). When kl_beta > 0 every
  completion is trained. Mean per-token KL over all completions is logged in both cases (extra no-grad forwards).
  `kl_stats()` is a diagnostic helper.
- `scripts/grpo_phase.py`: new flags `--kl_beta`, `--stop_flat_frac/--stop_patience` (early stop, `early_stop.json`),
  `--reuse_untrained_from` (reuses (a), (b), (c) only if the GPU name, model revision, new tokens, prompt, eval sizes,
  FineWeb max and the b npy file all match; otherwise it logs why and recomputes), `--phase S --neuron --prompt`
  (single-neuron run), and `--gate_from` (lets a Phase B run without its own Phase A use run 1's passed gate).
  Per-step metrics now include `mean_kl` and `loss`.
- `scripts/gpu_bench.py`: callers updated to the new return value.

## test_grpo.py: 47 PASS, 2 FAIL

New KL checks, all PASS:

| check | result |
|---|---|
| KL at init (LoRA off = reference) | exactly 0.0 (kl_stats and pg_backward with beta 0.04) |
| KL after one optimizer step | 0.00776 > 0 |
| kl_beta=0: same trained set and mean logp as the old code | identical |
| kl_beta=0: loss equals the old code | bitwise equal (0.03629125840961933) |
| kl_beta=0: grads equal the old code | max diff 0.00146; two runs of the OLD code already differ by 0.00098 (the backward is nondeterministic run to run), so the check is "within 2x that noise", not bitwise |
| kl_beta 0.04 changes the gradient after a step | max diff 0.244 |
| pg_backward with kl_beta 0.04, micro_bs 32 | peak 21.1 GB, 6.3 s for 256 completions |

**FAIL** (existing checks, not changed by this work):

- `peft_scorer_matches_quickcheck_rescore_3.046875`
- `reference_scores_helper_exact`

Both compare L14N8949's top FineWeb context with the hardcoded value 3.046875. That value was measured on an earlier RTX 5090
pod (2026-11-16, both checks passed there). On this pod it scores 3.03125, one bf16 step lower and exactly the value run 1
recorded as FineWeb max (this GPU) for L14N8949. The integrity checks in `grpo_phase.py` use the per-run
`fineweb_max.json` value, not this constant, and they passed throughout run 1.

## Step 2 (safety part only)

- Environment installed (`logs/install_env_20261118_040219.out`).
- data/ sha256 OK for both files; pilots.json matches.
- Watchdog started (PID 3475); `logs/watchdog.disable` does not exist.
- `runpodctl get pod` reports POD TYPE **RESERVED**, which I read as On-Demand (not INTERRUPTABLE/spot).

## Not done (waiting for the user)

Pipeline script, comparison report script, launch, commit and push.

## Update 04:50 UTC: resolved, pipeline launched (user chose "fix constant, rerun, launch")

- The two checks now compare against L14N8949's top-context score from a plain (non-PEFT) Scorer run on this GPU
  (`grpo_fwmax.py`, 3.03125 here) instead of the constant 3.046875.
- On the rerun, `kl_beta0_grads_equal_old` failed (max abs diff 0.0078 vs a single old-vs-old pair at 0.00098). A diagnostic
  (3 old runs, 2 new without KL logging, 2 new with it) showed the backward pass is not bitwise reproducible on this stack:
  old vs old differs by 0.08-0.13% relative grad norm, and new vs old by 0.08-0.17%. No systematic difference. The check now
  uses relative grad norm over 3 old and 2 new runs (new-vs-old <= 2x the old-vs-old spread). Loss, trained set and mean logp
  must still match exactly, and they do.
- Final test run: **49/49 PASS** (`logs/test_grpo_20261118_kl_rerun2.log`, `results/grpo_tests/20261118_042155/`).
- Smoke test of the new flags (`results/smoke/20261118_042557_collapse`) and a tiny end-to-end pipeline
  (`results/smoke/20261118_043738_collapse_pipeline`, SMOKE=1): all PASS, integrity checks PASS, report generated.
- Real pipeline `scripts/collapse_run.sh` launched 04:48:38 UTC -> `results/grpo_collapse/20261118_044838/`.
