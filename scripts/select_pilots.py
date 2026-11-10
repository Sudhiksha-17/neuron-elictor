"""Select pilot neurons: n_per_layer per layer, sampled uniformly at random (fixed seed) among the
layer's neurons that are not low-activation. Low-activation = FineWeb max <= 0, or FineWeb max <
low_act_rel * (median FineWeb max of that layer). This is a relative rule (formerly called "dead"; in late
layers these neurons do activate, just weakly). The baseline run, seed and rule are recorded in the output config.json.

Extension-safe: for each layer L, the eligible (not low-activation) neurons (ascending index) are shuffled once with
numpy default_rng([seed, L]).permutation; the layer's pilots are the first n_per_layer entries of that
permutation ("rank" 0, 1, ...). Asking for more per layer (same baseline, seed, low_act_rel) only appends
neurons; the earlier ones never change. --start_rank k outputs only ranks >= k (the new neurons).

Usage: python scripts/select_pilots.py --baseline results/phase2/<run> [--n_per_layer 1] [--seed 0]
       python scripts/select_pilots.py --baseline results/phase2/<run> --n_per_layer 2 --start_rank 1
"""
import os, sys, json, time, argparse
sys.path.insert(0, "/workspace")
import numpy as np, torch
from hermes.runinfo import run_info
from hermes.scoring import neuron_id

ap = argparse.ArgumentParser()
ap.add_argument("--baseline", required=True)
ap.add_argument("--n_per_layer", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--low_act_rel", "--dead_rel", dest="low_act_rel", type=float, default=0.1,
                help="low-activation threshold relative to the layer median (--dead_rel: old name)")
ap.add_argument("--start_rank", type=int, default=0, help="output only ranks >= this (to extend a list)")
ap.add_argument("--tag", default="")
a = ap.parse_args()

b = torch.load(f"{a.baseline}/baseline.pt")
mx = b["max"].float()                                     # [32, 9216]
med = mx.median(dim=1, keepdim=True).values
low = (mx <= 0) | (mx < a.low_act_rel * med)
pilots = []
for rank in range(a.start_rank, a.n_per_layer):            # rank-major: all rank-0 pilots first
    for L in range(mx.shape[0]):
        alive = np.flatnonzero(~low[L].numpy())            # ascending neuron index
        n = int(np.random.default_rng([a.seed, L]).permutation(alive)[rank])
        pilots.append(dict(id=neuron_id(L, n), layer=L, neuron=n, rank=rank, fineweb_max=float(mx[L, n]),
                           argmax_ctx=int(b["argmax_ctx"][L, n]), argmax_pos=int(b["argmax_pos"][L, n])))
run_dir = f"/workspace/results/pilots/{time.strftime('%Y%m%d_%H%M%S')}{'_' + a.tag if a.tag else ''}"
os.makedirs(run_dir, exist_ok=False)
cfg = dict(baseline_run=os.path.abspath(a.baseline), baseline_n_contexts=int(b["n_contexts"]),
           n_per_layer=a.n_per_layer, start_rank=a.start_rank, seed=a.seed, low_act_rel=a.low_act_rel,
           selection_rule=(f"per layer L: low-activation = (max <= 0) or (max < {a.low_act_rel} * median_n max[L, n]); "
                           f"perm_L = numpy.random.default_rng([{a.seed}, L]).permutation(indices of neurons that are not low-activation, "
                           f"in ascending order); pilots = perm_L[rank] for rank in [{a.start_rank}, {a.n_per_layer}), "
                           f"L = 0..31. Extending n_per_layer never changes lower ranks."),
           n_low_activation_per_layer=low.sum(1).tolist(), env=run_info())
json.dump(cfg, open(f"{run_dir}/config.json", "w"), indent=2)
json.dump(pilots, open(f"{run_dir}/pilots.json", "w"), indent=1)
json.dump(dict(n_pilots=len(pilots), ids=[p["id"] for p in pilots], n_low_activation_total=int(low.sum())),
          open(f"{run_dir}/summary.json", "w"), indent=1)
print("low-activation per layer:", low.sum(1).tolist())
print(f"{len(pilots)} pilots:", " ".join(p["id"] for p in pilots))
print("saved:", run_dir)
