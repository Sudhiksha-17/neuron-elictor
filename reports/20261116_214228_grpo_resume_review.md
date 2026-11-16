# GRPO resume review (CPU-only pod, no model loaded)

Written 2026-11-16 21:42 UTC after the pod restart. Nothing was run on a GPU. Sources:
logs/grpo_pipeline_dry.out, logs/grpo_20261116_175439_*.log, reports/20261116_181729_grpo_final.md,
hermes/grpo.py, scripts/grpo_phase.py, scripts/grpo_pipeline.sh, scripts/test_grpo.py.

## 1. What the dry run (results/grpo/20261116_175439_dry, commit 6c7a96c, RTX 4090) did

The run was a pipeline smoke test (`DRY_RUN=1`, `--dry`): 3 steps per phase, (a) best of 8, (b) capped at 64.
Phase B ran even though the gate failed, so it was presumably started with FORCE_PHASE_B=1.

| step | result |
|---|---|
| Phase 1 setup check | 9/9 PASS |
| FineWeb max rescored on the 4090 (5 neurons) | Matches the A100 values to within bf16 noise (e.g. L14N8949: 3.0625 on A100, 3.046875 here). The rescored max is rank 0 of the top-10 for all 5. |
| Phase A (L14N8949, prompt `Text:\n`, 64 completions/step, `--merged_gen`, gen_batch 128) | Step 1 took 148 s (gen 136 s, warm-up). Steps 2-3 took 33 s (gen 24-25 s, score 1 s, train 7 s). Peak memory 18.6 GB. |
| Gate | FAIL: (d) 0.147 vs FineWeb 3.047. This was expected after 3 steps. |
| Phase B (4 neurons, `<\|nK\|>` tokens, 256/step, `--merged_gen --gen_batch 256`) | gen_batch 256 **ran out of memory on every step** and fell back to 128. Steps 2-3 took 81-83 s (gen 49-51 s = 2 x 128, train 31-32 s). Peak memory 24.5 GB. |
| Commit step | Dry-run mode only listed 29 files. Nothing was committed, pushed, or stopped. |

What it showed:
- The pipeline works end to end: setup check, fwmax, A, gate, B, report, commit listing.
- **Merged generation took about 25 s per 128-sequence batch**, which matches the earlier 24 s LoRA-off figure and beats 41 s with LoRA unmerged. So the merge removes the LoRA overhead.
- Generation time is nearly flat in batch size: 64 sequences took 24 s and 128 took 25 s. That is about 380 ms per decode step for a 4B model on a 4090, which suggests decoding is latency-bound. This is a hypothesis to benchmark (section 4).
- The scores carry no signal yet (3 steps). (d) is below (b) for all 5 neurons. There are no empty outputs (≤0.4%) and no flat groups.
- Phase B outputs collapsed into wiktionary-style `===...===` CJK text, both untrained and trained.

Problems found in the dry run and its report:
- The OOM fallback resets on every `generate` call (`per` is local), so Phase B pays one OOM and retry per step, plus one in eval (b).
- The report's summary-table header hard-codes "(a) untrained best of 64" even though the dry run used 8.
- The "Why the gate failed" text says "Phase B was not run", but it was run (forced).
- Under `--dry`, (b) uses N=64, but training generated 192 per neuron. In the dry run, (b) is not compute-matched.
- The run settings differ from CLAUDE.md's "Fixed design decisions": LoRA r=32/alpha=64 (CLAUDE.md says r=16/32), group size 16 (CLAUDE.md says 8), prompts `Text:\n` / `<|nK|>\n` (CLAUDE.md says "Write text that maximally activates neuron L{l}N{n}."), and reward normalized by FineWeb max. I could not find these approvals recorded in CLAUDE.md. **Either CLAUDE.md needs updating or the code does.**

## 2. hermes/grpo.py: merge-and-restore status

Done:
- `Elicitor.merged()` context manager (grpo.py:90-107). On first use it saves a pinned CPU copy of every LoRA `base_layer.weight`. It then runs `merge_adapter()`, yields, runs `unmerge_adapter()`, and copies the saved base weights back (`copy_` + `cuda.synchronize`).
- `generate(..., merged=True)` uses it. `grpo_phase.py --merged_gen` turns it on for training generation and for (d). The untrained evaluations (a)/(b) use `adapter_off`, never the merge.
- `merged_generation` is recorded in config.json.

Missing:
1. **No bitwise-identity check every 50 steps.** Nothing compares the GPU base weights with `_orig` (or a hash of them) during training.
2. **No fixed reference-text score at the start and end of each phase.** Something like L14N8949's top FineWeb context, which must score exactly 3.046875 LoRA-off before and after training. scripts/test_grpo.py checks this once, but without the merge path.
3. **No test of the merge path.** test_grpo.py (run 16:51, before `merged()` existed) never calls `generate(merged=True)`. It needs to check: base weights bitwise equal after a merged generate; LoRA-off score unchanged; trainable-token embedding rows restored; and merged vs unmerged logits close.
4. The trainable-token embedding rows (Phase B) depend on PEFT's own unmerge and are not covered by the CPU copy. Either add them to `_orig` or cover them with the check in item 1.
5. `merge_adapter()` runs outside the `try`. If it fails partway, the weights are left partly merged and never restored. Moving it inside the `try` makes the `finally` always restore.
6. The pinned CPU copy holds every LoRA-targeted base weight, roughly 7 GB of pinned host RAM. Each merged call does a 7 GB host-to-GPU copy, which has not been timed.
7. The OOM fallback should persist the reduced batch size across calls (see section 1).

## 3. Artifacts

- results/grpo is 508 MB. Almost all of it is two `adapter_model.safetensors` files of 260 MB each. These stay out of git (`*.safetensors` is ignored).
- results/grpo_tests is about 7 KB of real data (one results.json). `du` shows 2.0 MB because of filesystem block overhead.
- Committed: every file under 5 MB from the dry run, grpo_tests and phase1/20261116_175458. That is json/jsonl, plus `train.log`, `pipeline_status.txt` and the tiny `.npy` score arrays (640 B each), which were force-added past `.gitignore`. The commit also includes the GRPO logs and watchdog.log.

## 4. What needs a GPU (one session)

Before the session, CPU-side and no GPU needed: implement items 1-5 and 7 from section 2, fix the report labels, and settle the design-doc mismatch.

On the GPU, in order:
1. **Cache and environment check (~3 min).** Load from the HF cache at the pinned revision with `HF_HUB_OFFLINE=1`. The cache is on disk (snapshot 851bf6e, 8.8 GB of blobs), but it has not been loaded since the restart. Confirm `assert_fast_kernels()` passes (FLA + causal-conv1d) and `nvidia-smi` shows the expected GPU.
2. **Setup and safety tests (~10 min).** Run `scripts/phase1_setup_check.py` and the extended `scripts/test_grpo.py` with the merge-path checks from section 2.
3. **Generation benchmark (~10 min).** For 64 new tokens, at batch 64/128/192, time three modes: LoRA on, merged, and LoRA off. Also time merge plus restore on its own, and the KV-cache settings: check that `use_cache=True` takes effect when `model.config.use_cache=False`, and whether gradient checkpointing affects eval-mode generation. Find the largest batch that fits next to training memory; 256 OOMs. Report ms per decode step to test the latency-bound hypothesis.
4. **pg_backward micro-batch check (~5 min).** micro_bs 16 took 344 s vs 32 at 51 s for 256 completions (16:51 test). Confirm 32 is right after the changes.
5. **Short dry pipeline (~25 min).** Rerun with the fixes and confirm that the bitwise check and the reference-text check are logged and pass.
6. **Time estimate for the real run.** From the dry run with defaults of 300 steps and n_matched = steps x 64 = 19,200 per neuron:
   - Phase A: about 300 x 33 s = 2.75 h of training, plus (b) 19,200 untrained samples at about 0.2-0.4 s each = 1-2 h. **About 4-5 h.**
   - Phase B: about 300 x 82 s = 6.8 h of training, plus (b) 4 x 19,200 = 76,800 samples, about 4-8 h. **About 11-15 h.**
   - These are rough. The (b) sample cost at large batch was never measured; the dry-run projection used 367-966 ms per sample from tiny runs. Benchmark 3 replaces these numbers.
7. Launch the real run only after your go-ahead, with `scripts/watch_and_stop.sh`.
