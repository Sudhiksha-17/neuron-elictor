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
- Neuron activation: channel N of the input to layer L's mlp.down_proj,
  i.e. silu(gate_proj(x)) * up_proj(x).
- Score of a text = max activation over token positions.
- Context length: 64 tokens for both baseline and elicitor outputs,
  measured after re-tokenizing with the target tokenizer.
- Forward pass stops after the target layer (early exit).
- Elicitor input: ID tag only, e.g. "Write text that maximally activates
  neuron L15N2341." Thinking mode off.
- Pilot neurons: 20-50 from layer 15, skipping dead neurons.

## Phases (finish and verify each before starting the next)
1. Setup check: load model, find the 32 language-model MLP layers, read
   neuron L15N2341 per token on a test sentence, and verify the hook
   against a manual silu(gate_proj(x)) * up_proj(x) calculation.
2. Baseline: 1M FineWeb (sample-10BT, streamed) 64-token contexts; save
   per-neuron max, argmax context, top-10 contexts.
3. GRPO training: one group = 8 completions for one neuron's prompt,
   reward = target score. Remaining hyperparameters (lr, KL beta, steps,
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
