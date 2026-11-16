"""Pre-run for the GRPO pipeline: FineWeb max on THIS GPU for every target neuron.

Rescores each neuron's stored top-10 FineWeb contexts (tokens from data/contexts.npz) with the base model
(hermes.scoring.Scorer, fixed [256, 64] shape) and takes the max as "FineWeb max (this GPU)". The stored A100
max from data/baseline.pt is kept alongside. Also checks that the Phase B neurons are the first four pilots
from layers 8-24 in data/pilots.json.

Usage: python scripts/grpo_fwmax.py <run_dir>   -> <run_dir>/fineweb_max.json (+ config.json)
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from hermes.paths import BASELINE_DIR, load_pilots
import torch
from hermes.runinfo import run_info
from hermes.scoring import Scorer, parse_neuron_id, CONTEXT_LEN
from hermes.baseline import Baseline

PHASE_A = ["L14N8949"]
PHASE_B = ["L8N2027", "L9N5711", "L10N4358", "L11N3081"]

run_dir = sys.argv[1]
os.makedirs(run_dir, exist_ok=True)
pil = [p for p in load_pilots() if 8 <= p["layer"] <= 24 and p["rank"] == 0]
first4 = [p["id"] for p in sorted(pil, key=lambda p: p["layer"])[:4]]
assert first4 == PHASE_B, f"Phase B neurons {PHASE_B} != first four pilots from layers 8-24 {first4}"
print("Phase B neurons confirmed = first four pilots from layers 8-24:", first4)

sc = Scorer()
base = Baseline(BASELINE_DIR, sc.tok)
out = {}
for nid in PHASE_A + PHASE_B:
    L, N = parse_neuron_id(nid); nb = base.neuron(L, N)
    ids = torch.tensor([base.context_tokens(c) for c in nb["top_ctx"]], dtype=torch.long)
    s, p = sc.score_pairs(ids, torch.full((len(ids),), CONTEXT_LEN), [(L, N)] * len(ids))
    s, p = s.tolist(), p.tolist()
    k = max(range(len(s)), key=lambda i: s[i])
    out[nid] = dict(layer=L, neuron=N, fw_max_a100=nb["max"], fw_max_this_gpu=s[k], this_gpu_best_rank=k,
                    argmax_token_id=base.argmax_token(L, N), argmax_token=sc.tok.decode([base.argmax_token(L, N)]),
                    top10=[dict(rank=i, ctx=c, stored=v, stored_pos=vp, this_gpu=s[i], this_gpu_pos=p[i],
                                text=base.context_text(c))
                           for i, (c, v, vp) in enumerate(zip(nb["top_ctx"], nb["top_val"], nb["top_pos"]))])
    print(f"{nid}: FineWeb max A100 {nb['max']}, this GPU {s[k]} (rank {k}); argmax token {out[nid]['argmax_token']!r}")
json.dump(dict(phase_a=PHASE_A, phase_b=PHASE_B, gpu=torch.cuda.get_device_name(0), env=run_info(),
               kernels=sc.kernels), open(f"{run_dir}/fwmax_config.json", "w"), indent=1)
json.dump(out, open(f"{run_dir}/fineweb_max.json", "w"), indent=1)
print("saved:", f"{run_dir}/fineweb_max.json")
