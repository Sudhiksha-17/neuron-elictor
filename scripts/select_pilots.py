"""Select pilot neurons: n_per_layer per layer, sampled uniformly at random (fixed seed) among the
layer's non-dead neurons. Dead = FineWeb max <= 0, or FineWeb max < dead_rel * (median FineWeb max of
that layer). The baseline run, seed and rule are recorded in the output config.json.

Usage: python scripts/select_pilots.py --baseline results/phase2/<run> [--n_per_layer 1] [--seed 0]
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
ap.add_argument("--dead_rel", type=float, default=0.1)
ap.add_argument("--tag", default="")
a = ap.parse_args()

b = torch.load(f"{a.baseline}/baseline.pt")
mx = b["max"].float()                                     # [32, 9216]
med = mx.median(dim=1, keepdim=True).values
dead = (mx <= 0) | (mx < a.dead_rel * med)
rng = np.random.default_rng(a.seed)
pilots = []
for L in range(mx.shape[0]):
    alive = np.flatnonzero(~dead[L].numpy())               # ascending neuron index
    for n in sorted(rng.choice(alive, size=a.n_per_layer, replace=False).tolist()):
        pilots.append(dict(id=neuron_id(L, n), layer=L, neuron=n, fineweb_max=float(mx[L, n]),
                           argmax_ctx=int(b["argmax_ctx"][L, n]), argmax_pos=int(b["argmax_pos"][L, n])))
run_dir = f"/workspace/results/pilots/{time.strftime('%Y%m%d_%H%M%S')}{'_' + a.tag if a.tag else ''}"
os.makedirs(run_dir, exist_ok=False)
cfg = dict(baseline_run=os.path.abspath(a.baseline), baseline_n_contexts=int(b["n_contexts"]),
           n_per_layer=a.n_per_layer, seed=a.seed, dead_rel=a.dead_rel,
           selection_rule=(f"per layer L: dead = (max <= 0) or (max < {a.dead_rel} * median_n max[L, n]); "
                           f"sample {a.n_per_layer} neuron(s) uniformly without replacement from the non-dead "
                           f"neurons (ascending index order) with numpy default_rng({a.seed}), layers 0..31 in order"),
           n_dead_per_layer=dead.sum(1).tolist(), env=run_info())
json.dump(cfg, open(f"{run_dir}/config.json", "w"), indent=2)
json.dump(pilots, open(f"{run_dir}/pilots.json", "w"), indent=1)
json.dump(dict(n_pilots=len(pilots), ids=[p["id"] for p in pilots], n_dead_total=int(dead.sum())),
          open(f"{run_dir}/summary.json", "w"), indent=1)
print("dead per layer:", dead.sum(1).tolist())
print(f"{len(pilots)} pilots:", " ".join(p["id"] for p in pilots))
print("saved:", run_dir)
