# GRPO fixes before the GPU session (CPU-only pod)

Written 2026-11-16 22:04 UTC. **All code changes below are UNTESTED. They must be run on a GPU**, via
`scripts/gpu_session.sh`. On this pod I only syntax-checked them (`py_compile`, `bash -n`) and ran CPU-only checks:
the pipeline flag guards, the report rendered from the old dry run with torch blocked, and the session script's
control flow with every command stubbed out.

## Decisions applied
- CLAUDE.md now matches the code: LoRA r=32/alpha=64, groups of 16, prompts `Text:\n` (Phase A) and `<|nK|>\n`
  (Phase B, one trained special token per neuron). The tracked-results rule now allows small force-added files.
- The force-added small files stay in git.

## Changes
1. **Integrity checks** (`scripts/grpo_phase.py`, `hermes/grpo.py`):
   - Bitwise base-weight check (`Elicitor.check_base_bitwise`) at the start, every `--check_every` steps
     (default 50), at the last step, and at the end.
   - Reference-text check (`Elicitor.reference_scores`): each phase neuron's best FineWeb context is scored with
     LoRA off and must exactly equal `fineweb_max.json`'s this-GPU value. It runs at the start, after training and
     at the end.
   - Both are logged and written to `phaseX/integrity.json` and results.json. On failure the phase logs
     `INTEGRITY FAIL` and exits with code 3, and the pipeline aborts.
   - `scripts/grpo_fwmax.py` now stores the reference tokens (`ref_tokens`). For older run dirs the tokens are read
     from `data/` instead.
2. **Exact backup covers the neuron-token embedding rows.** The backup is taken once at init, before any merge or
   optimizer step, and restored together with the LoRA base weights. The rows are read from the original
   `nn.Embedding` captured before `get_peft_model`.
3. **`merge_adapter()` runs inside try/finally.** A failure in the merge, in generation or in the unmerge always
   ends with an exact restore.
4. **The out-of-memory fallback remembers the smaller batch** (`Elicitor.gen_cap`, per mode: untrained / merged /
   lora). It is logged, and recorded per step in metrics.jsonl.
5. **Merge timing**: merge, unmerge and the exact restore copy are timed for every call. They are logged on each
   step line, recorded in metrics.jsonl, summarized in results.json `merge_timing`, and shown in the report.
6. **The gate cannot be bypassed in a real run.**
   - `grpo_pipeline.sh` takes `--dry` and `--force_phase_b`. `--force_phase_b` without `--dry` exits with
     code 2.
   - The old `DRY_RUN` / `FORCE_PHASE_B` environment variables are rejected.
   - A non-dry `grpo_phase.py --phase B` refuses to start unless `gate.json` says passed and Phase A was not a
     dry run.
7. **Report fixes** (`scripts/grpo_report.py`):
   - Best-of-N headers show the actual N in each cell.
   - The "Why the gate failed" text says Phase B ran anyway when it did (dry run, forced).
   - The "(d) N fresh samples" and "every N steps" texts are no longer hard-coded.
   - New integrity and merge-timing sections.
   - Dry-mode (b) is now compute-matched: steps x groups per neuron x group size, which is 192 per neuron in a dry
     run, instead of 64.
8. **Scope flags** (defaults unchanged): `--steps`, `--groups_per_step` (a multiple of the number of neurons),
   `--n_matched`, `--n_untrained_best`, `--n_fresh`. Explicit flags now also override the `--dry` defaults.
9. **`scripts/test_grpo.py` section 6 (merge path)**, run after a real optimizer step:
   - Checks: base weights bitwise equal to the backup, LoRA-on logits bitwise unchanged, LoRA-off score
     unchanged, and neuron-token rows restored. These run after a merged forward, after an exception inside
     `merged()`, after an injected failure in `merge_adapter`, and after merged generation.
   - Merged vs unmerged LoRA: top-1 agreement must be at least 95%. The logit-delta cosine is reported.
   - Merge timing, a simulated out-of-memory check that the smaller batch is remembered, and an exact check of
     `reference_scores()`.

## New scripts
- `scripts/gpu_bench.py env|gen|pg`:
  - `env`: environment check.
  - `gen`: generation benchmark, with the **KV-cache check first**. It checks whether the cache is actually used,
    times use_cache on vs off, and times gradient checkpointing on vs off. It then runs one real training step and
    sweeps 3 modes x batches 64-256 with 64 tokens forced. It also reports merge timing and a bitwise check, and
    projects the real-run time.
  - `pg`: micro-batch check (16/32/48/64, warm-up run then measured run).
- `scripts/gpu_session.sh`: runs GPU steps 1-5 unattended with timeouts, in this order:
  1. environment check, offline (`HF_HUB_OFFLINE=1`)
  2. phase-1 setup check, `test_scoring`, `test_grpo`
  3. `gpu_bench gen`
  4. `gpu_bench pg`
  5. `grpo_pipeline.sh --dry --force_phase_b` with merged generation

  It then writes `reports/<ts>_gpu_session.md`, commits locally (no push), and runs `runpodctl stop pod
  $RUNPOD_POD_ID`. It never starts the real run. If the environment check fails, it skips steps 2-5 and still
  reports and stops.

  Launch: `setsid nohup scripts/gpu_session.sh > logs/gpu_session_$(date +%Y%m%d_%H%M%S).out 2>&1 < /dev/null &`

## Things to watch in the GPU-session report
- **Merged-policy lag.** Merged weights are rounded to bf16. Early in training the LoRA update can be smaller than
  that rounding, so generation may sample from a policy that lags behind the trained one. test_grpo reports the
  logit-delta cosine; if it is low, the merge is a speed/accuracy trade-off to decide on.
- **Host memory.** The exact backup now always allocates about 7 GB of pinned host RAM, also without
  `--merged_gen`, because the bitwise check needs it.
- **Watchdog side effect.** The existing pipeline's real-run stop path runs `touch logs/watchdog.disable` before
  stopping. That file persists on /workspace and leaves the watchdog off after the next boot. I left it unchanged;
  `gpu_session.sh` does not touch it.
