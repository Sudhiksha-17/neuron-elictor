"""Phase B prompt check (pre-registered rule, decided 2026-11-17; written on a CPU-only pod: UNTESTED, run on GPU).

Before Phase B training, compare untrained samples on the four Phase B neurons for
  (a) "<|nK|>\\n"          (--phase_b_prompt nK, the default)
  (b) "<|nK|>\\nText:\\n"   (--phase_b_prompt nK_text)
Per neuron: best of 64 untrained samples (adapter and trainable tokens off = exactly the untrained elicitor;
the new <|nK|> rows are at their init), as a ratio to the neuron's FineWeb max on THIS GPU (fineweb_max.json from
scripts/grpo_fwmax.py). An all-empty neuron counts as ratio EMPTY_REWARD (-1.0), the GRPO empty reward.
Rule: use (b) if mean over the 4 neurons of the best-of-64 ratios is STRICTLY higher for (b) than for (a);
otherwise keep (a). Same seed for both prompts. Writes <dir>/prompt_check.json (comparison, choice, top texts)
and prints the line "CHOICE <nK|nK_text>".

Usage: python scripts/grpo_fwmax.py <dir> && python scripts/phase_b_prompt_check.py <dir> [--n 64] [--seed 0]
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import numpy as np, torch
from hermes.runinfo import run_info
from hermes.scoring import parse_neuron_id
from hermes.grpo import Elicitor, EMPTY_REWARD, PHASE_B_PROMPTS

NEURONS = ["L8N2027", "L9N5711", "L10N4358", "L11N3081"]     # = grpo_phase.py PHASES["B"]["neurons"]
ap = argparse.ArgumentParser()
ap.add_argument("dir", help="dir holding fineweb_max.json (scripts/grpo_fwmax.py)")
ap.add_argument("--n", type=int, default=64, help="untrained samples per neuron and prompt (best-of-N)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--batch", type=int, default=192)
a = ap.parse_args()
out = f"{a.dir}/prompt_check.json"
assert not os.path.exists(out), f"{out} exists (never overwrite results)"
fw = json.load(open(f"{a.dir}/fineweb_max.json"))
pairs = [parse_neuron_id(n) for n in NEURONS]
fwmax = [fw[n]["fw_max_this_gpu"] for n in NEURONS]

el = Elicitor(new_tokens=[f"<|n{k}|>" for k in range(4)], seed=a.seed)
res = dict(rule="use (b) nK_text iff mean_k(best-of-N ratio to FineWeb max this GPU) is strictly higher than for (a) nK; "
                "otherwise keep (a)", n=a.n, seed=a.seed, neurons=NEURONS, fw_max_this_gpu=dict(zip(NEURONS, fwmax)),
           variants={}, gpu=torch.cuda.get_device_name(0), env=run_info())
for key in ("nK", "nK_text"):
    prompts = [PHASE_B_PROMPTS[key].format(k=k) for k in range(4)]
    pids = [el.prompt_ids(p) for p in prompts]
    torch.manual_seed(a.seed); t0 = time.time()
    comps = el.generate(pids, a.n, batch=a.batch, untrained=True)
    per = {}
    for j, (n, pr, f) in enumerate(zip(NEURONS, pairs, fwmax)):
        texts = el.texts(comps[j]); s = el.score(texts, [pr] * len(texts)).numpy().astype(np.float64)
        ok = ~np.isnan(s)
        b = float(s[ok].max()) if ok.any() else None
        order = [int(i) for i in np.argsort(-np.where(ok, s, -np.inf), kind="stable")[:3] if ok[i]]
        per[n] = dict(prompt=prompts[j], prompt_token_ids=pids[j], best=b, ratio=b / f if b is not None else EMPTY_REWARD,
                      mean=float(s[ok].mean()) if ok.any() else None, n_empty=int((~ok).sum()),
                      top=[dict(text=texts[i], score=float(s[i])) for i in order])
    mean_ratio = float(np.mean([per[n]["ratio"] for n in NEURONS]))
    res["variants"][key] = dict(template=PHASE_B_PROMPTS[key], mean_best_ratio=mean_ratio, per_neuron=per,
                                seconds=time.time() - t0)
    print(f"({'a' if key == 'nK' else 'b'}) {key} {PHASE_B_PROMPTS[key]!r}: mean best-of-{a.n} ratio {mean_ratio:.4f} | "
          + " ".join(f"{n} {per[n]['best']} ({per[n]['ratio']:.3f})" for n in NEURONS), flush=True)
ra, rb = res["variants"]["nK"]["mean_best_ratio"], res["variants"]["nK_text"]["mean_best_ratio"]
res["choice"] = "nK_text" if rb > ra else "nK"
res["choice_reason"] = f"(b) {rb:.6f} {'>' if rb > ra else '<='} (a) {ra:.6f}"
json.dump(res, open(out, "w"), indent=1)
print(f"choice: {res['choice']} ({res['choice_reason']}); saved {out}")
print(f"CHOICE {res['choice']}")
