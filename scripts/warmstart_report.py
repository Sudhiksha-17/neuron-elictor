"""Report for the warm-start experiments (scripts/warmstart_run.sh).

  python scripts/warmstart_report.py <top_dir> [status_file]   -> reports/<ts>_warmstart.md (never raises)

<top_dir>/runs.json lists the runs (label, name, dir). Rep has phaseA (+ phaseB if its gate passed); the others phaseS.
ratio = (d) / FineWeb max (this GPU). (w) = best of n_fresh fresh samples from the frozen warm adapter.

Pre-registered criteria (user, 2026-11-18, fixed in this file before any result of these runs existed):
  1. run 1 replicates: in Rep, L14N8949 and L11N3081 both reach ratio >= 1.05
  2. warm start rescues: L9N5711 or L10N4358 reaches ratio >= 1.05 in both W0 seeds
  3. RL adds value: (d) > (w) wherever ratio >= 1.05 (warm-start runs, which have a (w))
  4. not copying: winning (d) texts (each (d) sample scoring >= 1.05 x FineWeb max, any run) have max token Jaccard
     < 0.5 to their neuron's top-10 FineWeb contexts
  5. KL comparison: WKL-L10 vs W0-L10 seed 0 ratios (descriptive, no direction predicted)
"""
import os, sys, json, time, math, traceback, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from hermes.paths import REPO, REPORTS_DIR
from grpo_report import jload, jl, key, f, md_text

WIN = 1.05
COPY_MAX = 0.5
RUN1 = os.path.join(REPO, "results/grpo/20261118_013749")


def load_phase(pd):
    return dict(pdir=pd, res=jload(f"{pd}/results.json"), cfg=jload(f"{pd}/config.json", {}), metrics=jl(f"{pd}/metrics.jsonl"),
                integrity=jload(f"{pd}/integrity.json"), warm=jload(f"{pd}/warm_start.json"))


def load_run(spec, top):
    d = spec["dir"] if os.path.isabs(spec["dir"]) else os.path.join(REPO, spec["dir"])
    if not os.path.isdir(d):
        d = os.path.join(top, spec["label"])
    phases = {ph: load_phase(f"{d}/phase{ph}") for ph in ("A", "B", "S") if os.path.isdir(f"{d}/phase{ph}")}
    return dict(spec, dir=d, phases=phases)


def row(ph, n):
    res = ph["res"]
    if not res or n not in res.get("neurons", {}):
        return None
    x = res["neurons"][n]
    _, B = key(x, "b_untrained"); _, D = key(x, "d_trained"); _, W = key(x, "w_warm")
    fw = x["fw_max_this_gpu"]
    d, b, w = (D or {}).get("best"), (B or {}).get("best"), (W or {}).get("best")
    ratio = d / fw if d is not None else None
    return dict(n=n, fw=fw, b=b, bN=(B or {}).get("n"), d=d, dN=(D or {}).get("n"), w=w, wN=(W or {}).get("n"), ratio=ratio,
                w105=None if ratio is None else ratio >= WIN, dgtb=None if (d is None or b is None) else d > b,
                dgtw=None if (d is None or w is None) else d > w, D=D or {}, W=W or {}, B=B or {},
                b_reused=(B or {}).get("reused_from"))


def yn(v):
    return "n/a" if v is None else ("**yes**" if v else "no")


def cc(E):
    c = (E or {}).get("copy_check")
    if not c:
        return "n/a"
    bt = c.get("best_text") or {}
    return (f"max J {f(c['max_jaccard'], 3)}, span {c['max_longest_span']} (best text: J {f(bt.get('max_jaccard'), 3)}, "
            f"span {bt.get('longest_span', 'n/a')})")


def div_str(ws):
    if not ws:
        return "n/a"
    out = []
    for at in ws.get("attempts", []):
        dv = at.get("diversity", {})
        out.append(f"{at.get('n_epochs')} ep: {dv.get('n_distinct_texts')}/{dv.get('n_samples')} distinct, pairwise J "
                   f"{f(dv.get('mean_pairwise_jaccard'), 3)}, max J to top-10 {f(dv.get('max_jaccard_to_top10_max'), 3)} "
                   f"(mean {f(dv.get('max_jaccard_to_top10_mean'), 3)}), near-copies {dv.get('n_near_copies')}")
    return "; ".join(out) + f" -> used {ws.get('used_epochs')} epoch(s)"


def bins(metrics, neurons, k=25):
    out = []
    for s in range(0, len(metrics), k):
        ch = metrics[s:s + k]
        kl = [m["mean_kl"] for m in ch if m.get("mean_kl") is not None]
        pn = {}
        for n in neurons:
            ms = [m["per_neuron"][n]["mean_score"] for m in ch if m["per_neuron"].get(n, {}).get("mean_score") is not None]
            mx = [m["per_neuron"][n]["max_score"] for m in ch if m["per_neuron"].get(n, {}).get("max_score") is not None]
            pn[n] = (sum(ms) / len(ms) if ms else None, max(mx) if mx else None)
        out.append(dict(lo=s + 1, hi=s + len(ch), flat=sum(m["frac_flat_groups"] for m in ch) / len(ch),
                        kl=sum(kl) / len(kl) if kl else None, maxkl=max(kl) if kl else None,
                        reward=sum(m["mean_reward"] for m in ch) / len(ch), pn=pn))
    return out


def final_stats(metrics, k=25):
    ch = metrics[-k:]
    if not ch:
        return dict(flat=None, kl=None, reward=None, maxkl=None)
    kl = [m["mean_kl"] for m in ch if m.get("mean_kl") is not None]
    allkl = [m["mean_kl"] for m in metrics if m.get("mean_kl") is not None]
    return dict(flat=sum(m["frac_flat_groups"] for m in ch) / len(ch), kl=sum(kl) / len(kl) if kl else None,
                maxkl=max(allkl) if allkl else None, reward=sum(m["mean_reward"] for m in ch) / len(ch))


def main(top, status_file=None):
    md = [f"# Warm-start experiments: report", ""]
    spec = jload(f"{top}/runs.json", {"runs": []})
    runs = [load_run(s, top) for s in spec["runs"]]
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    fwc = jload(f"{top}/fwmax_config.json", {})
    md += [f"Generated {time.strftime('%F %T %Z')}. Experiment dir `{os.path.relpath(top, REPO)}`, code commit `{git}`, "
           f"GPU {fwc.get('gpu', '?')}. Baseline: run 1 = `results/grpo/20261118_013749` (RTX 5090, report "
           "`reports/20261118_034919_grpo_final.md`).", "",
           "Design (user, 2026-11-18, following MAEM arXiv 2611.08465; NLA as comparison): cross-entropy warm start of a LoRA "
           "adapter \"warm\" on the neuron's top-10 FineWeb contexts (as completions of \"Text:\\n\"), then GRPO on adapter "
           "\"policy\" (initialized as an exact copy of warm). Main recipe no KL; comparison KL toward the frozen warm "
           "adapter (beta 0.01). The scorer is always all adapters disabled.", "",
           "All runs: 300 planned steps (no early stop), 4 groups x 16 completions per neuron per step, lr 2e-5 linear decay, "
           "LoRA r=32 alpha=64, unmerged generation, gen batch 192, micro-batch 32, "
           "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, all integrity checks (bitwise base weights every 50 steps + "
           "start/end, reference-text rescore, warm-adapter frozen check in warm-start runs).", ""]
    md += ["| run | phases | warm start | KL ref | kl_beta | seed | steps done | (a)(b)(c) |", "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        for ph, P in r["phases"].items():
            c, res = P["cfg"], P["res"] or {}
            ws = c.get("warm_start")
            reu = (res.get("reused_untrained") or {})
            md.append(f"| {r['name']} | {ph} / {', '.join(c.get('neurons', []))} | "
                      f"{'lr %g, %s ep (used %s)' % (ws['lr'], ws['epochs'], ws['used_epochs']) if ws else 'no'} | "
                      f"{c.get('kl_ref', 'base')} | {c.get('kl_beta')} | {c.get('seed')} | {res.get('steps_done', 'n/a')} | "
                      f"{'reused from `%s`' % next(iter((reu.get('sources') or {}).values()), '') if reu.get('reused') else ('computed' if res else 'n/a')} |")
        if not r["phases"]:
            md.append(f"| {r['name']} | (no output) | | | | | | |")
    md.append("")

    if status_file and os.path.exists(status_file):
        md += ["## Pipeline log", "", "```", open(status_file).read()[-12000:], "```", ""]

    fw = jload(f"{top}/fineweb_max.json", {})
    if fw:
        md += ["## FineWeb max (top-10 contexts rescored on this GPU)", "",
               "| neuron | A100 (stored) | this GPU | rank | run 1 (5090) |", "|---|---|---|---|---|"]
        r1fw = jload(f"{RUN1}/fineweb_max.json", {})
        for n, x in fw.items():
            md.append(f"| {n} | {x['fw_max_a100']} | {x['fw_max_this_gpu']} | {x['this_gpu_best_rank']} | "
                      f"{(r1fw.get(n) or {}).get('fw_max_this_gpu', 'n/a')} |")
        md.append("")

    # ---- per neuron and run ----
    allrows = []    # (run, phase, row)
    for r in runs:
        for ph, P in r["phases"].items():
            for n in (P["cfg"].get("neurons") or []):
                x = row(P, n)
                if x:
                    allrows.append((r, ph, P, x))
    md += ["## Per neuron and run", "",
           "(b) = compute-matched untrained best-of-N (N = 300 x 64 = 19,200); (w) = best of 8 fresh samples from the frozen "
           "warm adapter; (d) = best of 8 fresh samples from the trained policy; ratio = (d) / FineWeb max (this GPU). "
           "Copy check: max token Jaccard and longest shared token span of the texts vs the neuron's top-10 FineWeb "
           "contexts (over all 8 texts; and for the best text).", ""]
    neurons_order = []
    for _, _, _, x in allrows:
        if x["n"] not in neurons_order:
            neurons_order.append(x["n"])
    for n in neurons_order:
        md += [f"### {n}", "", "| run | FineWeb max (this GPU) | (b) | (w) | (d) | ratio | ratio >= 1.05 | (d) > (b) | (d) > (w) |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r, ph, P, x in allrows:
            if x["n"] != n:
                continue
            md.append(f"| {r['name']} ({ph}) | {f(x['fw'])} | {f(x['b'])} (N={x['bN']}) | "
                      f"{f(x['w']) + ' (N=%s)' % x['wN'] if x['w'] is not None else 'n/a'} | {f(x['d'])} (N={x['dN']}) | "
                      f"{f(x['ratio'], 3)} | {yn(x['w105'])} | {yn(x['dgtb'])} | {yn(x['dgtw'])} |")
        md += ["", "| run | copy check (d) | copy check (w) | diversity pre-check (64 warm samples) |", "|---|---|---|---|"]
        for r, ph, P, x in allrows:
            if x["n"] == n:
                md.append(f"| {r['name']} ({ph}) | {cc(x['D'])} | {cc(x['W'])} | {div_str(P['warm'])} |")
        md.append("")

    # ---- warm start details ----
    md += ["## Warm start details", "", "| run | eval CE on the 10 contexts (before -> after each epoch) | redone | (w) best | "
           "diversity-sample mean / max score |", "|---|---|---|---|---|"]
    for r in runs:
        P = r["phases"].get("S")
        if not P or not P["warm"]:
            continue
        for at in P["warm"]["attempts"]:
            dv = at.get("diversity", {})
            md.append(f"| {r['name']} ({at['n_epochs']} ep) | {' -> '.join(f(v, 4) for v in at['eval_ce'])} | "
                      f"{P['warm']['redone']} | {f((row(P, P['cfg']['neurons'][0]) or {}).get('w'))} | "
                      f"{f(dv.get('mean_score'))} / {f(dv.get('max_score'))} |")
    md.append("")

    # ---- training curves ----
    md += ["## Flat-group fraction and KL over training", "",
           "KL = mean per-token k3 KL(policy || reference) over all completions of a step; reference = frozen warm adapter "
           "in warm-start runs, all adapters off (untrained base) in Rep. Final = mean over the last 25 logged steps.", "",
           "| run | steps | final flat frac | final mean KL | max per-step KL | final mean reward |", "|---|---|---|---|---|---|"]
    for r in runs:
        for ph, P in r["phases"].items():
            s = final_stats(P["metrics"])
            md.append(f"| {r['name']} ({ph}) | {len(P['metrics'])} | {f(s['flat'], 3)} | {f(s['kl'], 3)} | {f(s['maxkl'], 3)} | "
                      f"{f(s['reward'], 3)} |")
    md.append("")
    for r in runs:
        for ph, P in r["phases"].items():
            ns = P["cfg"].get("neurons") or []
            b = bins(P["metrics"], ns)
            if not b:
                continue
            md += [f"### {r['name']} ({ph}): 25-step bins", "",
                   "| steps | flat-group frac | mean KL | mean reward | " + " | ".join(f"{n} mean / max score" for n in ns) + " |",
                   "|---|---|---|---|" + "---|" * len(ns)]
            for x in b:
                md.append(f"| {x['lo']}-{x['hi']} | {x['flat']:.3f} | {f(x['kl'], 3)} | {x['reward']:.3f} | "
                          + " | ".join(f"{f(x['pn'][n][0])} / {f(x['pn'][n][1])}" for n in ns) + " |")
            md.append("")

    # ---- seed comparison ----
    byl = {r["label"]: r for r in runs}
    def rr(label, n):
        r = byl.get(label)
        if not r:
            return None
        for P in r["phases"].values():
            x = row(P, n)
            if x:
                return x, final_stats(P["metrics"])
        return None
    md += ["## Seed comparison", "", "| neuron | run | seed 0 (d) / ratio / (w) / final flat / final KL | seed 1 (d) / ratio / (w) / final flat / final KL |",
           "|---|---|---|---|"]
    def sc(t):
        if not t:
            return "n/a"
        x, s = t
        return f"{f(x['d'])} / {f(x['ratio'], 3)} / {f(x['w'])} / {f(s['flat'], 3)} / {f(s['kl'], 3)}"
    md.append(f"| L9N5711 | W0-L9 | {sc(rr('W0-L9-s0', 'L9N5711'))} | {sc(rr('W0-L9-s1', 'L9N5711'))} |")
    md.append(f"| L10N4358 | W0-L10 | {sc(rr('W0-L10-s0', 'L10N4358'))} | {sc(rr('W0-L10-s1', 'L10N4358'))} |")
    r1 = {}
    for ph in ("A", "B"):
        rs = jload(f"{RUN1}/phase{ph}/results.json") or {}
        for n, x in rs.get("neurons", {}).items():
            _, D = key(x, "d_trained")
            if D and D.get("best") is not None:
                r1[n] = D["best"] / x["fw_max_this_gpu"]
    md += ["", "Rep (seed 1, this GPU) vs run 1 (seed 0, RTX 5090) ratios:", "", "| neuron | run 1 ratio | Rep ratio |", "|---|---|---|"]
    for n in ["L14N8949", "L8N2027", "L9N5711", "L10N4358", "L11N3081"]:
        t = rr("Rep", n)
        md.append(f"| {n} | {f(r1.get(n), 3)} | {f(t[0]['ratio'], 3) if t else 'n/a'} |")
    md.append("")

    # ---- top texts ----
    md += ["## Top 2 (d) texts per run", ""]
    for r, ph, P, x in allrows:
        md.append(f"**{r['name']} ({ph}), {x['n']}** (FineWeb max {f(x['fw'])}):")
        md.append("")
        for i, t in enumerate(x["D"].get("top", [])[:2], 1):
            c = t.get("copy") or {}
            md.append(f"{i}. **{f(t['score'])}** ({f(t['score'] / x['fw'], 3)}x; copy: max J {f(c.get('max_jaccard'), 3)} "
                      f"to rank {c.get('max_jaccard_rank')}, span {c.get('longest_span')}) {md_text(t['text'])}")
        if x["W"].get("top"):
            t = x["W"]["top"][0]; c = t.get("copy") or {}
            md.append(f"   - best (w) text: **{f(t['score'])}** (copy: max J {f(c.get('max_jaccard'), 3)}, span "
                      f"{c.get('longest_span')}) {md_text(t['text'])}")
        md.append("")

    # ---- integrity ----
    md += ["## Integrity checks", "", "| run | bitwise base weights | reference-text rescore | warm adapter frozen |", "|---|---|---|---|"]
    for r in runs:
        for ph, P in r["phases"].items():
            ig = P["integrity"] or {}
            def s_(k):
                v = ig.get(k) or []
                return f"{sum(x['ok'] for x in v)}/{len(v)} PASS" if v else "n/a"
            md.append(f"| {r['name']} ({ph}) | {s_('bitwise')} | {s_('reference')} | {s_('warm_frozen')} |")
    md.append("")

    # ---- criteria ----
    md += ["## Pre-registered criteria", ""]
    def verdict(v):
        return {True: "**MET**", False: "**NOT MET**"}.get(v, f"**{v}**")
    t14, t11 = rr("Rep", "L14N8949"), rr("Rep", "L11N3081")
    c1 = bool(t14 and t11 and t14[0]["w105"] and t11[0]["w105"])
    md.append(f"1. run 1 replicates (Rep: L14N8949 and L11N3081 both ratio >= 1.05): {verdict(c1)} "
              f"(L14N8949 {f(t14[0]['ratio'], 3) if t14 else 'n/a'}, L11N3081 {f(t11[0]['ratio'], 3) if t11 else 'n/a (Phase B not run?)'})")
    c2p = {}
    for n, labs in (("L9N5711", ("W0-L9-s0", "W0-L9-s1")), ("L10N4358", ("W0-L10-s0", "W0-L10-s1"))):
        ts = [rr(l, n) for l in labs]
        c2p[n] = ([f(t[0]["ratio"], 3) if t else "n/a" for t in ts], all(t and t[0]["w105"] for t in ts))
    c2 = any(v[1] for v in c2p.values())
    md.append(f"2. warm start rescues (L9N5711 or L10N4358 ratio >= 1.05 in both W0 seeds): {verdict(c2)} "
              f"(L9N5711 seeds 0/1: {', '.join(c2p['L9N5711'][0])}; L10N4358 seeds 0/1: {', '.join(c2p['L10N4358'][0])})")
    cases = [(r, x) for r, ph, P, x in allrows if x["w105"] and x["w"] is not None]
    if cases:
        c3 = all(x["dgtw"] for _, x in cases)
        md.append(f"3. RL adds value ((d) > (w) wherever ratio >= 1.05): {verdict(c3)} ("
                  + "; ".join(f"{r['name']} {x['n']}: (d) {f(x['d'])} vs (w) {f(x['w'])}" for r, x in cases) + ")")
    else:
        md.append("3. RL adds value ((d) > (w) wherever ratio >= 1.05): **NOT APPLICABLE** (no warm-start run reached "
                  "ratio >= 1.05, so there is no case to test)")
    win = [(r, x, t) for r, ph, P, x in allrows for t in x["D"].get("all", []) if not math.isnan(t["score"])
           and t["score"] >= WIN * x["fw"]]
    if win:
        bad = [(r, x, t) for r, x, t in win if (t.get("copy") or {}).get("max_jaccard", 1.0) >= COPY_MAX]
        md.append(f"4. not copying (winning (d) texts max token Jaccard < 0.5): {verdict(not bad)} ({len(win)} winning texts; "
                  f"max Jaccard over them {f(max((t.get('copy') or {}).get('max_jaccard', float('nan')) for _, _, t in win), 3)}"
                  + (f"; {len(bad)} at >= 0.5: " + ", ".join(f"{r['name']} {x['n']}" for r, x, _ in bad) if bad else "") + ")")
    else:
        md.append("4. not copying: **NOT APPLICABLE** (no (d) text reached 1.05 x FineWeb max)")
    tk, t0 = rr("WKL-L10", "L10N4358"), rr("W0-L10-s0", "L10N4358")
    md.append(f"5. KL comparison (descriptive): WKL-L10 ratio {f(tk[0]['ratio'], 3) if tk else 'n/a'} "
              f"((d) {f(tk[0]['d']) if tk else 'n/a'}, final flat {f(tk[1]['flat'], 3) if tk else 'n/a'}, final KL "
              f"{f(tk[1]['kl'], 3) if tk else 'n/a'}) vs W0-L10 seed 0 ratio {f(t0[0]['ratio'], 3) if t0 else 'n/a'} "
              f"((d) {f(t0[0]['d']) if t0 else 'n/a'}, final flat {f(t0[1]['flat'], 3) if t0 else 'n/a'}, final KL "
              f"{f(t0[1]['kl'], 3) if t0 else 'n/a'})")
    md.append("")

    # ---- timing ----
    md += ["## Timing", "", "| run | training h | eval s (by kind) | phase h |", "|---|---|---|---|"]
    for r in runs:
        for ph, P in r["phases"].items():
            res = P["res"] or {}
            md.append(f"| {r['name']} ({ph}) | {f((res.get('training_s') or 0) / 3600, 3)} | "
                      f"{', '.join(f'{k} {v:.0f}' for k, v in (res.get('eval_s') or {}).items())} | {f((res.get('phase_s') or 0) / 3600, 3)} |")
    md.append("")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = f"{REPORTS_DIR}/{time.strftime('%Y%m%d_%H%M%S')}_warmstart.md"
    open(out, "w").write("\n".join(md) + "\n")
    print("report:", out)
    return out


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    except Exception:
        traceback.print_exc()
        out = f"{REPORTS_DIR}/{time.strftime('%Y%m%d_%H%M%S')}_warmstart.md"
        open(out, "w").write("# Warm-start report FAILED\n\n```\n" + traceback.format_exc() + "\n```\n")
        print("report (failure):", out)
