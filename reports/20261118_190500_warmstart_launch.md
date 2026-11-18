# Warm-start pipeline launch (2026-11-18, RTX 4090)

Pipeline `scripts/warmstart_run.sh`, launched 2026-11-18 18:59:40 UTC (setsid nohup; run part under
`scripts/watch_and_stop.sh`), dir `results/grpo_warmstart/20261118_185940`, code commit `b8ac81e`.

## Tests (scripts/test_grpo.py, RTX 4090): 78/78 PASS
Log `logs/test_grpo_warm_20261118_183505.log`, results `results/grpo_tests/20261118_183505`. New warm-start checks:
- defaults (no warm start, kl_ref base, kl_beta 0): single "default" adapter; old run-1 loss reproduced exactly (loss,
  trained set, mean logp equal; grads equal) -- PASS
- warm-start CE decreases over epochs: eval CE on the 10 contexts 3.327 -> 2.208 -> 1.170 -- PASS
- scorer (all adapters off) bitwise identical before/after warm start and after 3 RL steps -- PASS
- kl_ref=warm: KL exactly 0 at RL step 0 (kl_stats and first pg_backward), 0.0024 after one optimizer step -- PASS
- warm adapter bitwise unchanged after 3 RL steps; only policy params trainable; use_adapter restores flags -- PASS
- base weights bitwise, reference-text score exact after warm start and RL -- PASS
- 3-step grpo_phase.py --dry warm-start solo run (WKL-L10 config) end to end: (w), (d), diversity pre-check
  (64/64 distinct, 0 near-copies), copy check on every text, all integrity checks -- PASS
- 1-epoch redo path (--test_force_redo): two attempts (2, 1 epochs) reported, 1 used -- PASS
- all earlier checks (merge path, OOM fallback, KL, trainable params, ...) -- PASS
Smoke pipeline (SMOKE=1, all 7 runs, 2 steps): all exit 0, reuse within the pipeline works, report renders
(`results/smoke/20261118_184438_warmstart_pipeline`).

## Settings
300 steps, no early stop, unmerged generation, gen batch 192, micro-batch 32, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
Warm start: lr 1e-4, 2 epochs, batch 1 (10 optimizer steps per epoch; batch size not in the spec, chosen here),
AdamW constant lr, grad clip 1.0. Rep Phase B prompt: nK_text (run 1's pre-registered choice), gated on Rep's Phase A as in run 1.

## FineWeb max on this GPU (recomputed)
L14N8949 3.046875, L8N2027 5.0, L9N5711 1.3984375, L10N4358 2.90625, L11N3081 3.375.

## Projection (measured 9.2 s/step on Rep-A)
| part | estimate |
|---|---|
| Rep-A | ~0.95 h (training 0.76 h + untrained evals ~0.15 h) |
| Rep-B | ~2.5 h (~23 s/step x 300 + 4 x 19,200 untrained samples ~0.6 h); skipped if the gate fails |
| 5 warm-start runs | ~4.4 h (0.8 h each + (a)(b) in W0-L9-s0 and W0-L10-s0) |
| report, commit, push, stop | ~0.1 h |
| **total** | **~8 h -> ends ~03:00 UTC 2026-11-19** (~00:30 if Rep's gate fails) |

Idle watchdog: back to 60 min (PID 15238) after the pipeline started.
