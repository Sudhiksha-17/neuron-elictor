"""Quick sanity checks on a new pod (no training):
  1. Rescore one text for L14N8949 three ways: text alone, "Text:\\n" + text, and text alone with position 0
     excluded from the max. Compared with the FineWeb max (3.0625) and an externally reported 8.56.
  2. Repeated-token check: a token repeated to 64 tokens, L14N8949 (" and") and L16N8927 (" retarded"),
     vs their FineWeb maxima (3.0625, 7.25).
  3. Cross-GPU check: rescore the stored top-10 FineWeb contexts of L14N8949 and L16N8927 (tokens from
     data/contexts.npz) on this GPU and compare with data/baseline.pt (computed on an A100). Reports max
     abs difference and whether any value differs by more than one bf16 step (ulp at that magnitude).

All scoring goes through hermes.scoring.Scorer (fixed [256, 64] shape, pads excluded), as in the baseline.
Writes results/quick_checks/<ts>/{config,results}.json and reports/<ts>_quick_checks.md.
"""
import os, sys, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from hermes.paths import RESULTS_DIR, REPORTS_DIR, BASELINE_DIR
import torch
from hermes.runinfo import run_info
from hermes.scoring import Scorer, CONTEXT_LEN, SCORE_BATCH, TOKENIZE_KWARGS, parse_neuron_id
from hermes.baseline import Baseline

TS = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = f"{RESULTS_DIR}/quick_checks/{TS}"
TEXT = ("The cultists have failled to spread our cult in Shalarok. We must find way to destroy the "
        "resistances and continue our")
EXTERNAL_TEXT_SCORE = 8.56          # reported by Chris (setup unknown)
REPEAT = {"L14N8949": " and", "L16N8927": " retarded"}
TOP10 = ["L14N8949", "L16N8927"]


def bf16_ulp(v):
    """Spacing of bf16 values at |v| (8 significant bits)."""
    v = abs(float(v))
    if v == 0:
        return 2.0 ** -133
    return 2.0 ** (math.floor(math.log2(v)) - 7)


def per_token(scorer, ids, lens, L, N):
    """Per-token activations of (L, N) for each row, via the same fixed-shape forward as scoring."""
    out = []
    for s in range(0, len(ids), SCORE_BATCH):
        b = min(SCORE_BATCH, len(ids) - s)
        x = torch.full((SCORE_BATCH, CONTEXT_LEN), scorer.pad_id, dtype=torch.long); x[:b] = ids[s:s + b]
        got = {}
        scorer._forward(x, [L], lambda l, a: got.__setitem__(l, a[:b, :, N].float().cpu()))
        out += [got[L][i, :int(lens[s + i])] for i in range(b)]
    return out


def main():
    os.makedirs(RUN_DIR, exist_ok=False)
    sc = Scorer()
    tok = sc.tok
    base = Baseline(BASELINE_DIR, tok)
    cfg = dict(text=TEXT, repeat=REPEAT, top10_neurons=TOP10, baseline_dir=BASELINE_DIR, kernels=sc.kernels,
               env=run_info())
    json.dump(cfg, open(f"{RUN_DIR}/config.json", "w"), indent=2)
    gpu, res, md = cfg["env"]["gpu"], {}, []

    # ---- 1. text variants ----
    L, N = parse_neuron_id("L14N8949"); fw = base.neuron(L, N)["max"]
    variants = {"text alone": TEXT, "\"Text:\\n\" prefix": "Text:\n" + TEXT}
    ids, lens = sc.encode(list(variants.values()))
    acts = per_token(sc, ids, lens, L, N)
    official = sc.score_pairs(ids, lens, [(L, N)] * len(variants))[0].tolist()
    rows = []
    for (name, t), a, off, row_ids, n in zip(variants.items(), acts, official, ids, lens):
        toks = [tok.decode([i]) for i in row_ids[:n].tolist()]
        p = int(a.argmax())
        assert float(a.max()) == off, (name, float(a.max()), off)          # per-token path == Scorer
        rows.append(dict(variant=name, n_tokens=int(n), score=off, argmax_pos=p, argmax_token=toks[p],
                         pos0_act=float(a[0]), pos0_token=toks[0]))
        if name == "text alone":
            a1 = a[1:]; p1 = int(a1.argmax()) + 1
            rows.append(dict(variant="text alone, position 0 excluded", n_tokens=int(n), score=float(a1.max()),
                             argmax_pos=p1, argmax_token=toks[p1], pos0_act=None, pos0_token=None))
            res["text_alone_per_token"] = [dict(pos=i, token=tk, act=float(v)) for i, (tk, v) in enumerate(zip(toks, a.tolist()))]
    res["text_variants"] = dict(neuron="L14N8949", fineweb_max=fw, external_score=EXTERNAL_TEXT_SCORE, rows=rows)
    md += ["## 1. Text rescore, L14N8949", "", f"Text: `{TEXT}`", "",
           f"FineWeb max 3.0625 (stored: {fw}); Chris's reported score {EXTERNAL_TEXT_SCORE}.", "",
           "| variant | tokens | score | score / FineWeb max | argmax pos (token) | pos 0 act (token) |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        p0 = "n/a" if r["pos0_act"] is None else f"{r['pos0_act']:.4g} ({r['pos0_token']!r})"
        md.append(f"| {r['variant']} | {r['n_tokens']} | {r['score']:.4g} | {r['score'] / fw:.3f} | "
                  f"{r['argmax_pos']} ({r['argmax_token']!r}) | {p0} |")

    # ---- 2. repeated token ----
    md += ["", "## 2. Repeated-token check (token repeated to 64)", "",
           "| neuron | token | token id(s) | FineWeb argmax token | score | FineWeb max | ratio |", "|---|---|---|---|---|---|---|"]
    res["repeated"] = []
    for nid, t in REPEAT.items():
        L, N = parse_neuron_id(nid); fw = base.neuron(L, N)["max"]
        tid = tok(t, **TOKENIZE_KWARGS).input_ids
        assert len(tid) == 1, (t, tid)
        ids = torch.full((1, CONTEXT_LEN), tid[0], dtype=torch.long); lens = torch.tensor([CONTEXT_LEN])
        s, p = sc.score_pairs(ids, lens, [(L, N)])
        am = base.argmax_token(L, N)
        r = dict(neuron=nid, token=t, token_id=tid[0], fineweb_argmax_token_id=am,
                 fineweb_argmax_token=tok.decode([am]), score=float(s[0]), pos=int(p[0]), fineweb_max=fw)
        res["repeated"].append(r)
        md.append(f"| {nid} | {t!r} | {tid[0]} | {r['fineweb_argmax_token']!r} ({am}) | {r['score']:.4g} (pos {r['pos']}) | "
                  f"{fw:.4g} | {r['score'] / fw:.3f} |")

    # ---- 3. top-10 contexts, this GPU vs A100 ----
    md += ["", f"## 3. Top-10 FineWeb contexts rescored on {gpu} vs stored (A100)", ""]
    res["top10"], worst = [], dict(diff=0.0, steps=0.0)
    for nid in TOP10:
        L, N = parse_neuron_id(nid); nb = base.neuron(L, N)
        ids = torch.tensor([base.context_tokens(c) for c in nb["top_ctx"]], dtype=torch.long)
        lens = torch.full((len(ids),), CONTEXT_LEN)
        s, p = sc.score_pairs(ids, lens, [(L, N)] * len(ids))
        md += [f"**{nid}**", "", "| rank | ctx | stored (A100) | this GPU | diff | bf16 steps | pos stored / now |",
               "|---|---|---|---|---|---|---|"]
        for k, (c, v, vp, new, newp) in enumerate(zip(nb["top_ctx"], nb["top_val"], nb["top_pos"], s.tolist(), p.tolist())):
            d = new - v; steps = abs(d) / bf16_ulp(max(abs(v), abs(new)))
            res["top10"].append(dict(neuron=nid, rank=k, ctx=c, stored=v, now=new, diff=d, bf16_steps=steps,
                                     pos_stored=vp, pos_now=newp))
            if abs(d) > worst["diff"]:
                worst = dict(diff=abs(d), steps=steps, neuron=nid, rank=k)
            md.append(f"| {k} | {c} | {v:.6g} | {new:.6g} | {d:+.4g} | {steps:.2f} | {vp} / {newp} |")
        md.append("")
    over = [r for r in res["top10"] if r["bf16_steps"] > 1.0 + 1e-9]
    n_exact = sum(r["diff"] == 0 for r in res["top10"])
    n_pos = sum(r["pos_stored"] != r["pos_now"] for r in res["top10"])
    res["top10_summary"] = dict(max_abs_diff=worst["diff"], worst=worst, n_exact=n_exact, n=len(res["top10"]),
                                n_more_than_one_bf16_step=len(over), n_argmax_pos_changed=n_pos)
    md += [f"- Max |diff|: **{worst['diff']:.4g}** ({worst['steps']:.2f} bf16 steps)"
           + (f", at {worst['neuron']} rank {worst['rank']}" if worst['diff'] else ""),
           f"- Bitwise identical: {n_exact}/{len(res['top10'])}",
           f"- More than one bf16 step: **{len(over)}**" + (f" ({', '.join(f'{r['neuron']} rank {r['rank']}' for r in over)})" if over else ""),
           f"- Argmax position changed: {n_pos}"]

    json.dump(res, open(f"{RUN_DIR}/results.json", "w"), indent=1)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    rep = f"{REPORTS_DIR}/{TS}_quick_checks.md"
    head = [f"# Quick checks on {gpu}", "",
            f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')}. Script `scripts/quick_checks.py`, results `{os.path.relpath(RUN_DIR, os.path.dirname(REPORTS_DIR))}`. "
            f"torch {torch.__version__}, model rev `{cfg['env']['model_revision'][:12]}`, kernels: "
            + ", ".join(f"{k.split('_', 1)[1]}->{v.split('.')[0]}" for k, v in sc.kernels.items()), ""]
    open(rep, "w").write("\n".join(head + md) + "\n")
    print("\n".join(md)); print("saved:", RUN_DIR); print("report:", rep)


if __name__ == "__main__":
    main()
