## gpu_bench gen

### KV cache (checked first)

- PASS KV cache used during generate (layer-0 input lengths for 8 new tokens: [2, 1, 1, 1, 1, 1, 1, 1]; expected [2, 1, 1, ...])
- config use_cache: model False, text True, generation True; gradient checkpointing enabled: True
- LoRA, batch 64, 32 tokens: use_cache=True 114 ms/step, use_cache=False 243 ms/step; checkpointing off 109 ms/step, on 112 ms/step

### Generation sweep (64 new tokens forced; second run of each config; optimizer state resident)

| mode | batch | gen s | ms / decode step | seq / s | peak GB | first-run gen s |
|---|---|---|---|---|---|---|
| untrained | 64 | 4.4 | 69 | 14.46 | 14.0 | 4.3 |
| untrained | 128 | 4.3 | 67 | 29.80 | 17.9 | 4.9 |
| untrained | 192 | 4.8 | 75 | 40.07 | 21.8 | 5.9 |
| untrained | 256 | OOM | | | | |
| merged | 64 | 4.5 | 70 | 14.34 | 14.0 | 4.0 |
| merged | 128 | 4.7 | 73 | 27.39 | 17.9 | 4.3 |
| merged | 192 | 4.9 | 76 | 39.25 | 21.8 | 4.8 |
| merged | 256 | OOM | | | | |
| lora | 64 | 7.2 | 113 | 8.87 | 15.1 | 7.5 |
| lora | 128 | 7.3 | 115 | 17.43 | 18.8 | 7.5 |
| lora | 192 | 8.1 | 127 | 23.65 | 22.4 | 7.7 |
| lora | 256 | OOM | | | | |

- merged(): 7 calls, mean merge 0.08s, unmerge 0.09s, exact restore copy 0.59s; backup 7.14 GB pinned
- PASS bitwise base-weight check after the sweep (249 tensors, 0.6s)
- train step (256 completions, micro_bs 32): pg_backward 7.4s, peak 19.4 GB; scoring 256 texts 0.6s

### Projection for the real run (300 steps, 64 completions per neuron per step, (b) = 19,200 per neuron)

- Phase A: step 7s merged (9s unmerged LoRA); untrained evals 0.15 h; total ~0.8 h
- Phase B: step 18s merged (24s unmerged LoRA); untrained evals 0.58 h; total ~2.2 h
- Assumes every completion runs the full 64 tokens (upper bound for generation) and the step's other costs scale linearly with completions.

