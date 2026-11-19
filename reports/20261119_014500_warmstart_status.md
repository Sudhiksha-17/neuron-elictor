# Warm-start pipeline status, 2026-11-19 01:42 UTC

Pipeline results/grpo_warmstart/20261118_185940 (launched 18:59:40). 5 of 7 runs done, all exit 0; every bitwise, reference and warm-frozen check PASS. Gate PASS (Rep-A (d) 9.375 vs FineWeb max 3.047).

| run | neuron | FineWeb max | (b) | (w) | (d) | ratio | max Jaccard (d) |
|---|---|---|---|---|---|---|---|
| Rep-A | L14N8949 | 3.047 | 2.328 | – | 9.375 | 3.08 | 0.14 |
| Rep-B | L8N2027 | 5.000 | 2.609 | – | 6.781 | 1.36 | 0.09 |
| Rep-B | L9N5711 | 1.398 | 0.719 | – | 0.637 | 0.46 | 0.02 |
| Rep-B | L10N4358 | 2.906 | 1.117 | – | 2.078 | 0.72 | 0.02 |
| Rep-B | L11N3081 | 3.375 | 2.078 | – | 4.625 | 1.37 | 0.03 |
| W0-L9-s0 | L9N5711 | 1.398 | 0.672 | 0.297 | 1.531 | 1.09 | 0.25 |
| W0-L9-s1 | L9N5711 | 1.398 | 0.672 | 0.451 | 2.047 | 1.46 | 0.23 |
| W0-L10-s0 | L10N4358 | 2.906 | 1.742 | 1.031 | 7.313 | 2.52 | 0.16 |

Diversity pre-checks: 0/64 near-copies in all 4 warm starts (2 epochs kept, no redo).
Running: W0-L10-s1 step 242/300 (~10 s/step). Remaining: WKL-L10. Projected end ~02:50 UTC, then report, commit, push, pod stop.
