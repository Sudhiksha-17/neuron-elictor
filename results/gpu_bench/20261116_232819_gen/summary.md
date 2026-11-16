## gpu_bench gen

### KV cache (checked first)

- PASS KV cache used during generate (layer-0 input lengths for 8 new tokens: [2, 1, 1, 1, 1, 1, 1, 1]; expected [2, 1, 1, ...])
- config use_cache: model False, text True, generation True; gradient checkpointing enabled: True
- LoRA, batch 64, 32 tokens: use_cache=True 100 ms/step, use_cache=False 170 ms/step; checkpointing off 102 ms/step, on 95 ms/step

### Generation sweep (64 new tokens forced; second run of each config; optimizer state resident)

| mode | batch | gen s | ms / decode step | seq / s | peak GB | first-run gen s |
|---|---|---|---|---|---|---|
| untrained | 64 | 3.6 | 56 | 17.78 | 14.0 | 3.3 |
| untrained | 128 | 3.9 | 61 | 32.81 | 17.9 | 3.5 |
| untrained | 192 | 3.8 | 60 | 49.88 | 21.8 | 4.9 |
| untrained | 256 | 3.8 | 60 | 66.87 | 25.6 | 4.2 |
| merged | 64 | 3.2 | 50 | 20.06 | 14.0 | 3.3 |
| merged | 128 | 3.4 | 54 | 37.26 | 17.9 | 3.4 |
| merged | 192 | 3.5 | 55 | 54.89 | 21.8 | 3.6 |
| merged | 256 | 3.8 | 60 | 67.12 | 25.6 | 3.9 |
| lora | 64 | 5.5 | 87 | 11.55 | 15.1 | 5.7 |
| lora | 128 | 5.8 | 91 | 21.99 | 18.8 | 5.9 |
| lora | 192 | 6.1 | 95 | 31.52 | 22.4 | 6.4 |
| lora | 256 | 6.3 | 98 | 40.77 | 26.1 | 6.4 |

- merged(): 8 calls, mean merge 0.04s, unmerge 0.04s, exact restore copy 0.26s; backup 7.14 GB pinned
- PASS bitwise base-weight check after the sweep (249 tensors, 0.3s)
- train step (256 completions, micro_bs 32): pg_backward 5.4s, peak 21.4 GB; scoring 256 texts 0.4s

### Projection for the real run (300 steps, 64 completions per neuron per step, (b) = 19,200 per neuron)

- Phase A: step 5s merged (7s unmerged LoRA); untrained evals 0.09 h; total ~0.6 h
- Phase B: step 10s merged (12s unmerged LoRA); untrained evals 0.35 h; total ~1.3 h
- Assumes every completion runs the full 64 tokens (upper bound for generation) and the step's other costs scale linearly with completions.

