# Hermes neuron elicitor pilot

## Goal
Train an elicitor (Qwen3.5-4B + LoRA) via GRPO to write text that maximally
activates specific MLP neurons in a frozen target (Qwen3.5-4B). Small proof
of concept only. Keep everything simple.

## Fixed design decisions (do not change without asking)
- Model: Qwen/Qwen3.5-4B, bf16. One copy of weights: LoRA on = elicitor,
  LoRA off = reference policy and target.
- Training method: GRPO (switched from DPO on 2026-11-07, approved by user).
  Reward = target score of the cleaned elicitor text. LoRA r=16, alpha=32.
- Neurons: any (layer L, neuron N), L in 0..31, N in 0..9215 (294,912 in all);
  ID format L{layer}N{neuron}, e.g. L15N2341 (only an example).
- Neuron activation: channel N of the input to layer L's mlp.down_proj,
  i.e. silu(gate_proj(x)) * up_proj(x).
- Score of a text = max activation over token positions.
- Context length: 64 tokens for both baseline and elicitor outputs,
  measured after re-tokenizing with the target tokenizer.
- Scoring function takes (layer, neuron); the forward pass stops after
  the deepest layer needed (early exit). One forward can read several
  layers (phase 2 reads all 32; a GRPO batch reads the layers of its
  neurons). Fixed forward shape [256, 64] (right-padded, pads excluded
  from the max) so a text's score does not depend on its batch.
  Code: hermes/scoring.py.
- Elicitor input: ID tag only, prompt template
  "Write text that maximally activates neuron L{layer}N{neuron}."
  Thinking mode off.
- Pilot neurons: 1-2 per layer (n_per_layer, default 1 -> 32 neurons),
  sampled uniformly at random with a fixed seed among the layer's
  non-dead neurons (scripts/select_pilots.py; seed, rule and baseline run
  recorded in its config.json). Dead-neuron rule: FineWeb max <= 0 or
  < 0.1 x the layer's median neuron max. Decided 2026-11-09: 1 per layer
  (32), seed 0. Extension-safe: more neurons are drawn with the same
  seed without changing the first 32 (see README).

## Phases (finish and verify each before starting the next)
1. Setup check: load model, find the 32 language-model MLP layers, read
   a neuron (--layer/--neuron, default L15N2341) per token on a test
   sentence, and verify the hook against a manual
   silu(gate_proj(x)) * up_proj(x) calculation. Plus scoring safeguard
   tests (scripts/test_scoring.py: fast-kernel assertion, padded-batch
   vs one-at-a-time bitwise match, multi-layer reads).
2. Baseline: all 32 layers (294,912 neurons), 1M FineWeb (sample-10BT,
   streamed) 64-token contexts, at most 4 chunks per document (first 4;
   cap recorded in config.json). Per neuron save: max, context ID and
   token position of the max, mean and std of the per-context score, and
   top-10 context IDs (+ values, positions). Context tokens are saved, not
   text (decode with hermes/baseline.py). Per layer save: argmax-position
   histogram, fraction of neurons with max at position 0, number of
   neurons with >= 5 of their top-10 from the same document (0 by
   construction with the 4-chunk cap; the full distribution is saved).
3. GRPO training: one group = 8 completions for one neuron's prompt,
   reward = target score (hermes/reward.py; one batch may mix neurons
   from different layers). A text that is empty after cleaning scores
   NaN (not 0.0: activations can be negative). RL code MUST penalize NaN
   scores explicitly (see README). Remaining hyperparameters (lr, KL beta, steps,
   neurons per step, reward normalization, etc.) TO BE PROPOSED AND
   APPROVED by the user before phase 3 starts.
   - Step 0: before training, sample 8 completions per pilot neuron from
     the untrained elicitor, score them, and record mean/max reward and the
     flat-group rate (groups whose 8 rewards are all (near-)equal).
   - Compute-matched best-of-N baseline: sample the untrained elicitor as
     many times per neuron as GRPO generates for it during training; record
     the max per neuron.
   - Trivial check: per neuron, score its FineWeb argmax token repeated to
     64 tokens.
   - Score only the elicitor's generated text: strip <think> tags and any
     chat-template tokens. Log raw vs cleaned examples.
   - Dry-run GRPO on 2-3 neurons for a few steps first; report memory use,
     time per step, and projected total time.
4. Report per round: mean activation, fraction of neurons beating their
   FineWeb max, score/baseline ratio per neuron, sample outputs.
   - Final evaluation uses 8 fresh samples per neuron from the trained
     elicitor, not the best sample seen during training.
   - Final report table per neuron: FineWeb max, step-0 score, best-of-N
     score, trained-elicitor score, repeated-token score.

## Rules
- All code, data, checkpoints and results live under /workspace.
- HF_HOME is /workspace/hf_cache.
- Log every run's config and seed. Never overwrite previous results.
- Every run's config records pip freeze, GPU name, and the FineWeb dataset
  revision hash (and model revision), via hermes/runinfo.py. Data and model
  are loaded at the recorded revisions.
- After each phase, commit and push scripts and configs to the private
  GitHub repo. Never push large result files (only results/**/config.json
  and summary.json are tracked; see .gitignore).
- Ask before any change to the design above.
- Safety: scripts/idle_watchdog.sh (nohup) stops the pod via
  "runpodctl stop pod" after 60 min of GPU util < 5%; disable with
  `touch /workspace/logs/watchdog.disable`. Long runs are launched with
  scripts/watch_and_stop.sh so the pod stops when the run ends. Only
  ever "stop", never "remove"/"terminate".
- Model and FineWeb revisions are pinned in hermes/revisions.json.
- Do not launch long runs without the user's go-ahead.
- Save every report given to the user as timestamped markdown in
  /workspace/reports/.
- No HF token (decided 2026-11-09); rely on the datasets/hub retry logic.
  GitHub push postponed; deploy key + git config stay as set up.
