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
- `reports/` – every report, timestamped markdown

## Pilot neurons (1 per layer = 32, seed 0)

```
python scripts/select_pilots.py --baseline results/phase2/<full run> --seed 0 --n_per_layer 1
```

Rule: per layer L, dead = FineWeb max <= 0 or max < 0.1 x median neuron max of layer L. The non-dead
neurons of layer L (ascending index) are shuffled once with `numpy.random.default_rng([seed, L]).permutation`;
the layer's pilots are the first `n_per_layer` entries ("ranks") of that permutation.

### Drawing more neurons without changing the first 32

Use the **same baseline run, same `--seed`, same `--dead_rel`** and raise `--n_per_layer`. Ranks only get
appended, so the first 32 (rank 0) never change:

```
# all 64 (ranks 0 and 1): the first 32 are exactly the original list
python scripts/select_pilots.py --baseline results/phase2/<full run> --seed 0 --n_per_layer 2
# only the 32 new ones (rank 1)
python scripts/select_pilots.py --baseline results/phase2/<full run> --seed 0 --n_per_layer 2 --start_rank 1
```

Changing the baseline run or `--dead_rel` changes the non-dead set and therefore the permutation, so
keep those fixed when extending (both are recorded in each selection's config.json).

## Empty completions score NaN – RL code must penalize NaN

`Scorer.score_pairs` / `score_texts` and the GRPO reward return **NaN** for a text that is empty after
cleaning (no tokens = no activation). 0.0 is deliberately not used: activations can be negative, so 0.0
could beat real attempts. **Any RL / evaluation code must handle NaN explicitly**: replace it with a
penalty (e.g. below the group's lowest real reward) before computing advantages, count it as a failure
in evaluation, and never let NaN propagate into losses or silently become 0.
