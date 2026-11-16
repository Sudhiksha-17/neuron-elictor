# Hermes neuron elicitor pilot

See CLAUDE.md for the design and phases. Code layout:

- `hermes/scoring.py` – target scorer: any (layer, neuron), multi-layer reads, early exit, fixed [256, 64] shape
- `hermes/reward.py` – GRPO reward (prompts may mix layers), completion cleaning
- `hermes/baseline.py` – load a phase 2 run, decode contexts from saved tokens
- `hermes/runinfo.py` – run metadata; model + FineWeb revisions pinned in `hermes/revisions.json`
- `scripts/phase1_setup_check.py`, `scripts/test_scoring.py`, `scripts/test_reward.py` – checks
- `scripts/phase2_baseline.py` – FineWeb baseline, all 32 layers, 4 chunks/doc cap
- `scripts/select_pilots.py` – pilot neurons
- `scripts/idle_watchdog.sh`, `scripts/watch_and_stop.sh`, `scripts/launch_full_baseline.sh` – pod safety / long runs
- `scripts/quick_checks.py` – new-pod sanity checks (text rescore, repeated token, cross-GPU top-10 rescore)
- `hermes/paths.py` – repo-relative paths (`data/`, `results/`, `reports/`, `logs/`)
- `reports/` – every report, timestamped markdown

## Data (`data/`, gitignored)

The repo lives at `/workspace/neuron-elictor` (HF cache stays at `/workspace/hf_cache`). Inputs are read from `data/`:

- `data/baseline.pt`, `data/contexts.npz` – the full 1M phase 2 baseline (formerly `results/phase2/20261109_204110_full1M`),
  from https://huggingface.co/datasets/gray171/hermes-baseline. sha256:
  `70cde47f08fbcd786f1d0dfacd3e858526fee54b6845e41cebe70c978d649817` (baseline.pt),
  `d519c23a590961df190291e916464bcb186f1942500fc465dca2187f7143ff2f` (contexts.npz).
  Load with `hermes.baseline.Baseline(hermes.paths.BASELINE_DIR)`.
- `data/pilots.json` – the 32 pilot neurons, a copy of `results/pilots/20261109_213918_full1M_pilots/pilots.json`.
  Load with `hermes.paths.load_pilots()`. `select_pilots.py` (default `--baseline data`) reproduces it byte-for-byte.

## Pilot neurons (1 per layer = 32, seed 0)

```
python scripts/select_pilots.py --seed 0 --n_per_layer 1
```

Rule: per layer L, low-activation = FineWeb max <= 0 or max < 0.1 x median neuron max of layer L
(`--low_act_rel`, default 0.1). The other neurons of layer L (ascending index) are shuffled once with `numpy.random.default_rng([seed, L]).permutation`;
the layer's pilots are the first `n_per_layer` entries ("ranks") of that permutation.

### Drawing more neurons without changing the first 32

Use the **same baseline run, same `--seed`, same `--low_act_rel`** and raise `--n_per_layer`. Ranks only get
appended, so the first 32 (rank 0) never change:

```
# all 64 (ranks 0 and 1): the first 32 are exactly the original list
python scripts/select_pilots.py --seed 0 --n_per_layer 2
# only the 32 new ones (rank 1)
python scripts/select_pilots.py --seed 0 --n_per_layer 2 --start_rank 1
```

Changing the baseline run or `--low_act_rel` changes the eligible (not low-activation) set and therefore the permutation, so
keep those fixed when extending (both are recorded in each selection's config.json). Selections made
before 2026-11-10 record the same threshold as `dead_rel` / `n_dead_per_layer`; `--dead_rel` is still
accepted as an alias, and the rule and the selected neurons are unchanged.

## Empty completions score NaN – RL code must penalize NaN

`Scorer.score_pairs` / `score_texts` and the GRPO reward return **NaN** for a text that is empty after
cleaning (no tokens = no activation). 0.0 is deliberately not used: activations can be negative, so 0.0
could beat real attempts. **Any RL / evaluation code must handle NaN explicitly**: replace it with a
penalty (e.g. below the group's lowest real reward) before computing advantages, count it as a failure
in evaluation, and never let NaN propagate into losses or silently become 0.

## Using the phase 2 baseline (`data/`, the full 1M run formerly `results/phase2/20261109_204110_full1M`)

### Scores are bf16: ties are common, use strict `>`

Activations are **computed** in bf16 (the model runs in bf16), not just stored in bf16; the scorer
returns them as float32 but every value is a bf16 number. bf16 has 8 significant bits, so the
spacing between representable values is coarse (0.125 at 16-32, 1.0 at 128-256). An elicitor text
therefore often scores **exactly** the FineWeb max. **A tie does not count as beating the baseline:
"beats" means `score > fineweb_max` (strict greater-than).**

### Report ratios, not just beat / not-beat

Report `score / fineweb_max` per neuron (and its distribution over neurons), not only the count of
neurons beaten. Given the bf16 resolution, a ratio just above 1.0 can be a single quantization step.
**Proposed** threshold for a clear win: **ratio >= 1.05**. This is a proposal only; the team should
agree on it before any results are in.

### "Low-activation", not "dead"

The rule (max <= 0, or max < 0.1 x the layer's median neuron max) is **relative**. It excludes 81
neurons in total, 59 of them in layers 30 (22) and 31 (37). In those layers they are low-activation
neurons, not dead ones: every excluded neuron in layers 30 and 31 has a positive FineWeb max (layer 30's
lowest max is 1.01, and the cutoffs are about 1.36 and 1.32). Only 2 of all 294,912 neurons never
go positive. Scripts and reports from now on use "low-activation". Older outputs say "dead" for the
same set.

### Outliers and argmax positions

- The top of each layer's max distribution has a heavy tail. The single largest neuron maxima are far
  above the layer's 99th percentile: **layer 31: 255** (99%: 72), **layer 27: 140** (99%: 45),
  layer 18: 122 (99%: 14), layer 30: 119 (99%: 52), layer 28: 96, layer 22: 89. Means across neurons,
  or plots on a linear scale, are dominated by these. Prefer per-neuron ratios and medians.
- **Late positions dominate the argmax in most layers.** From layer 5 on, more than half of a layer's
  neurons reach their max at positions 32-63 (about 60-70% in layers 11-31), and almost none at
  position 0 (layer 0: 15.7%, layer 1: 3%, below 0.5% elsewhere). So in a 64-token context, the max
  comes mostly from tokens with long left context. Expect elicitor texts shorter than 64 tokens to be
  at a disadvantage. Full histograms: summary.json `per_layer.argmax_pos_hist`.

## Open items

- (Resolved 2026-11-16) `scripts/select_pilots.py` reproduces the saved 32 pilots byte-for-byte from `data/baseline.pt`.
