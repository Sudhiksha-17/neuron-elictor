# README update: notes for baseline users

2026-11-10. Documentation and naming only. No design change, no new runs, no earlier results modified.

## README.md: new section "Using the phase 2 baseline"
- **bf16 scores:** activations are computed in bf16 (not only stored in bf16). Ties with the FineWeb max are common. "Beats baseline" means strict `score > fineweb_max`.
- **Ratios:** report `score / fineweb_max` per neuron, not only beat/not-beat. A clear win at **ratio >= 1.05** is *proposed* and needs team agreement before any results are in.
- **"Low-activation", not "dead":** the rule is relative. 81 neurons are excluded, 59 of them in layers 30 (22) and 31 (37). All excluded neurons in layers 30-31 have a positive max (cutoffs about 1.36 and 1.32). Only 2 of 294,912 neurons never go positive.
- **Outliers:** the largest maxima are L31 255 (99th pct 72), L27 140 (45), L18 122 (14), L30 119 (52). **Late positions dominate the argmax:** from layer 5 on, more than half of the neurons peak at positions 32-63 (about 60-70% in layers 11-31). Position 0 is rare except in layers 0 and 1. Figures are from `reports/20261109_213910_phase2_final.md`.

## Rename in outputs (future runs only)
- `scripts/select_pilots.py`: `--low_act_rel` (with `--dead_rel` kept as an alias). The config/summary keys are now `low_act_rel`, `n_low_activation_per_layer` and `n_low_activation_total`, and the rule text says "low-activation". The selection logic is the same, so the same 32 pilots are selected.
- `scripts/final_report.py`: the column and section are now "low-activation".
- Earlier results and reports keep "dead" (never overwritten). The README explains how the old names map to the new ones.
- Not changed: CLAUDE.md still says "Dead-neuron rule". It is the design file, so changing it is left to the user.

## Verification
- Both scripts pass `py_compile`. They could **not** be run in this session: the container now has a 512 MB memory limit and no GPU (`import torch` is OOM-killed). A rerun of `select_pilots.py` to confirm identical pilots is still pending.

## Backup
- `/workspace/code_backup.tar.gz` refreshed: scripts, hermes, CLAUDE.md, README.md, reports.

## Follow-up (same day)
- CLAUDE.md: "Dead-neuron rule" is now "Low-activation rule", with a note that older results use the old name. Approved by the user.
- README: new "Open items" section, which says to re-run select_pilots.py on a GPU pod and confirm it reproduces the saved 32 pilots.
- Committed locally (not pushed). Backup refreshed after the commit.
