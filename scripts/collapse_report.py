"""Comparison report for the collapse-fix experiments (scripts/collapse_run.sh) vs run 1.

  python scripts/collapse_report.py <top_dir> [status_file]   -> reports/<ts>_collapse_fix.md (never raises)

<top_dir>/runs.json lists the runs (label, name, dir, phase, args). Run 1 = results/grpo/20261118_013749/phaseB.

Pre-registered criteria (fixed in this file before any R1-R4 result existed, 2026-11-18):
  1. collapse fixed: R1's final flat-group fraction (mean over its last 25 logged training steps, the same definition
     as run 1's 0.475 = mean of steps 276-300) is "clearly below" 0.475, operationalized as <= 0.2375 (half of run 1).
  2. L10 rescued: L10N4358's (d) > its (b) (strict) in R1, R2 or R3 (which ones are listed).
  3. no damage: L11N3081's (d) / FineWeb max (this GPU) >= 1.05 in R1.
"""
import os, sys, json, time, math, traceback, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from hermes.paths import REPO, REPORTS_DIR
from grpo_report import jload, jl, key, f, md_text

RUN1 = os.path.join(REPO, "results/grpo/20261118_013749")
RUN1_FLAT_FINAL = 0.475
CLEAR_BELOW = RUN1_FLAT_FINAL / 2
WIN = 1.05
PHASE_B = ["L8N2027", "L9N5711", "L10N4358", "L11N3081"]


def phase_dir(run_dir):
    for ph in ("B", "S", "A"):
        if os.path.isdir(f"{run_dir}/phase{ph}"):
            return f"{run_dir}/phase{ph}"
    return None


def load_run(label, name, run_dir):
    pd = phase_dir(run_dir) if run_dir else None
    r = dict(label=label, name=name, dir=run_dir, pdir=pd, res=None, cfg={}, metrics=[], early=None)
    if pd:
        r["res"] = jload(f"{pd}/results.json"); r["cfg"] = jload(f"{pd}/config.json", {})
        r["metrics"] = jl(f"{pd}/metrics.jsonl"); r["early"] = jload(f"{pd}/early_stop.json")
        r["integrity"] = jload(f"{pd}/integrity.json")
    return r


def neuron_row(run, n):
    """-> dict(fw, b, bN, d, dN, ratio, w105, dgtb, top) or None if the run has no result for n."""
    res = run["res"]
    if not res or n not in res.get("neurons", {}):
        return None
    x = res["neurons"][n]
    _, B = key(x, "b_untrained"); _, D = key(x, "d_trained")
    fw = x["fw_max_this_gpu"]; d = D and D["best"]; b = B and B["best"]
    ratio = d / fw if d is not None else None
    return dict(fw=fw, b=b, bN=B and B["n"], d=d, dN=D and D["n"], ratio=ratio,
                w105=None if ratio is None else ratio >= WIN,
                dgtb=None if (d is None or b is None) else d > b, top=(D or {}).get("top", []),
                b_reused=(B or {}).get("reused_from"))


def final_flat(metrics, k=25):
    ch = metrics[-k:]
    return sum(m["frac_flat_groups"] for m in ch) / len(ch) if ch else None


def mean_of(metrics, field, k=None):
    ch = metrics[-k:] if k else metrics
    v = [m.get(field) for m in ch if m.get(field) is not None]
    return sum(v) / len(v) if v else None


def yn(v):
    return "n/a" if v is None else ("**yes**" if v else "no")


def met(v):
    return "**MET**" if v else "**NOT MET**"


def main(top, status_file=None):
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = f"{REPORTS_DIR}/{ts}_collapse_fix.md"
    md = []
    try:
        manifest = jload(f"{top}/runs.json", {"runs": []})
        runs = [load_run("run 1", "run 1 (Phase B, nK_text, beta 0, seed 0)", RUN1)]
        for x in manifest["runs"]:
            runs.append(load_run(x["label"], f"{x['label']} \"{x['name']}\"", os.path.join(REPO, x["dir"])))
        by = {r["label"]: r for r in runs}
        git = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        fw = jload(f"{top}/fineweb_max.json", {}); fw1 = jload(f"{RUN1}/fineweb_max.json", {})
        fcfg = jload(f"{top}/fwmax_config.json", {})
        md += ["# Mode-collapse fixes: KL penalty, single-neuron L10, seed check", "",
               f"Generated {time.strftime('%F %T %Z')}. Experiment dir `{os.path.relpath(top, REPO)}`, code commit `{git}` "
               f"(+ uncommitted run outputs), GPU {fcfg.get('gpu', 'n/a')}. Baseline: run 1 = `results/grpo/20261118_013749` "
               "(report `reports/20261118_034919_grpo_final.md`).", "",
               "All runs: 300 planned steps, 4 groups x 16 completions per neuron per step, lr 2e-5 linear decay, LoRA r=32 "
               "alpha=64, unmerged LoRA generation, gen batch 256, micro-batch 32, early stop if the flat-group fraction "
               "is > 0.9 for 25 consecutive steps, all integrity checks on (bitwise base weights every 50 steps + start/end, "
               "reference-text rescore at start / after training / end).", "",
               "| run | phase / neurons | prompt | kl_beta | seed | untrained (a), (b), (c) | steps done | early stop |",
               "|---|---|---|---|---|---|---|---|"]
        for r in runs:
            c, res = r["cfg"], r["res"] or {}
            ru = res.get("reused_untrained") or {}
            src = ("computed" if not ru.get("requested") else
                   (f"reused from `{os.path.relpath(next(iter(ru['sources'].values())), REPO)}`" if ru.get("reused")
                    else f"recomputed (reuse refused: {str(ru.get('reason_not_reused'))[:160]} …; full reason in results.json)"))
            if r["label"] == "run 1":
                src = "computed"
            e = r["early"] or {}
            md.append(f"| {r['name']} | {c.get('phase', '?')} / {', '.join(c.get('neurons', []))} | {md_text(', '.join(sorted(set(c.get('prompts', [])))), 40)} | "
                      f"{c.get('kl_beta', 0.0)} | {c.get('seed', '?')} | {src} | {len(r['metrics'])} of {c.get('steps', '?')} | "
                      f"{('step ' + str(e['stop_step'])) if e.get('stopped') else ('no' if r['metrics'] else 'n/a')} |")
        md.append("")
        missing = [r["name"] for r in runs if not r["res"]]
        if missing:
            md += [f"**No results.json for: {', '.join(missing)}** (run failed, skipped, or did not reach its evaluations; see the pipeline log).", ""]
        if status_file and os.path.exists(status_file):
            md += ["## Pipeline log", "", "```", *[l.rstrip("\n") for l in open(status_file)], "```", ""]

        # ---- FineWeb max ----
        if fw:
            md += ["## FineWeb max (top-10 contexts rescored on this GPU)", "", "| neuron | this pipeline | run 1 | equal |", "|---|---|---|---|"]
            for n in PHASE_B:
                a_, b_ = fw.get(n, {}).get("fw_max_this_gpu"), fw1.get(n, {}).get("fw_max_this_gpu")
                md.append(f"| {n} | {a_} | {b_} | {yn(a_ == b_)} |")
            md.append("")

        # ---- per-neuron tables ----
        md += ["## Per-neuron results", "",
               "(b) = compute-matched untrained best-of-N (N = 300 planned steps x 64 completions = 19,200, also for runs that stopped early, "
               "so for an early-stopped run (b) used more samples than training did). (d) = best of 8 fresh samples from the trained "
               "elicitor. ratio = (d) / FineWeb max (this GPU).", ""]
        for n in PHASE_B:
            md += [f"### {n}", "", "| run | FineWeb max (this GPU) | (b) | (d) | ratio | ratio >= 1.05 | (d) > (b) |", "|---|---|---|---|---|---|---|"]
            for r in runs:
                x = neuron_row(r, n)
                if x is None:
                    if n in r["cfg"].get("neurons", []) or (r["label"] in ("R1", "R4") and not r["res"]):
                        md.append(f"| {r['name']} | n/a | n/a | n/a | n/a | n/a | n/a |")
                    continue
                md.append(f"| {r['name']} | {x['fw']} | {f(x['b'])} (N={x['bN']:,}) | {f(x['d'])} (N={x['dN']}) | {f(x['ratio'], 3)} | "
                          f"{yn(x['w105'])} | {yn(x['dgtb'])} |")
            md.append("")

        # ---- training dynamics ----
        md += ["## Flat-group fraction and KL during training", "",
               "Final = mean over the last 25 logged steps (run 1's 0.475 is its steps 276-300). KL = mean per-token k3 "
               "KL(policy || LoRA-off reference) over all completions of a step (logged for beta 0 runs too; not logged in run 1).", "",
               "| run | steps done | early stop step | final flat frac | max flat frac (25-step bin) | final mean KL | max per-step KL | final mean reward |",
               "|---|---|---|---|---|---|---|---|"]
        for r in runs:
            m = r["metrics"]
            if not m:
                md.append(f"| {r['name']} | 0 | n/a | n/a | n/a | n/a | n/a | n/a |"); continue
            bins = [m[s:s + 25] for s in range(0, len(m), 25)]
            mxb = max(sum(x["frac_flat_groups"] for x in b) / len(b) for b in bins)
            kls = [x.get("mean_kl") for x in m if x.get("mean_kl") is not None]
            e = r["early"] or {}
            md.append(f"| {r['name']} | {len(m)} | {e['stop_step'] if e.get('stopped') else '-'} | {f(final_flat(m), 3)} | {f(mxb, 3)} | "
                      f"{f(mean_of(m, 'mean_kl', 25), 3)} | {f(max(kls) if kls else None, 3)} | {f(mean_of(m, 'mean_reward', 25), 3)} |")
        md.append("")
        for r in runs:
            m = r["metrics"]
            if not m:
                continue
            ns = r["cfg"].get("neurons", [])
            md += [f"### {r['name']}: 25-step bins", "",
                   "| steps | flat-group frac | mean KL | mean reward | " + " | ".join(f"{n} mean / max score" for n in ns) + " |",
                   "|---|---|---|---|" + "---|" * len(ns)]
            for s in range(0, len(m), 25):
                ch = m[s:s + 25]
                kl = [x.get("mean_kl") for x in ch if x.get("mean_kl") is not None]
                cells = []
                for n in ns:
                    ms = [x["per_neuron"][n]["mean_score"] for x in ch if x["per_neuron"][n]["mean_score"] is not None]
                    mx = [x["per_neuron"][n]["max_score"] for x in ch if x["per_neuron"][n]["max_score"] is not None]
                    cells.append(f"{f(sum(ms) / len(ms) if ms else None)} / {f(max(mx) if mx else None)}")
                md.append(f"| {ch[0]['step'] + 1}-{ch[-1]['step'] + 1} | {sum(x['frac_flat_groups'] for x in ch) / len(ch):.3f} | "
                          f"{f(sum(kl) / len(kl) if kl else None, 3)} | {sum(x['mean_reward'] for x in ch) / len(ch):.3f} | " + " | ".join(cells) + " |")
            md.append("")

        # ---- criteria ----
        md += ["## Pre-registered criteria", ""]
        R1 = by.get("R1")
        ff = final_flat(R1["metrics"]) if R1 and R1["metrics"] else None
        c1 = ff is not None and R1["res"] is not None and ff <= CLEAR_BELOW
        md.append(f"1. **Collapse fixed** (R1 final flat-group fraction clearly below run 1's {RUN1_FLAT_FINAL}; threshold set before the "
                  f"runs: <= {CLEAR_BELOW:.4g}, half of run 1): R1 final {f(ff, 3)}"
                  + (f", early stop at step {R1['early']['stop_step']}" if R1 and (R1["early"] or {}).get("stopped") else "")
                  + f". {met(c1)}")
        rescued = []
        for lab in ("R1", "R2", "R3"):
            x = neuron_row(by[lab], "L10N4358") if lab in by else None
            if x and x["dgtb"]:
                rescued.append(f"{lab} ((d) {f(x['d'])} > (b) {f(x['b'])})")
        detail = "; ".join(f"{lab}: (d) {f(x['d'])} vs (b) {f(x['b'])}" for lab in ("R1", "R2", "R3")
                           for x in [neuron_row(by[lab], "L10N4358") if lab in by else None] if x) or "no results"
        md.append(f"2. **L10 rescued** (L10N4358 (d) > (b) in R1, R2 or R3): {detail}. {met(bool(rescued))}"
                  + (f" in {', '.join(rescued)}." if rescued else ""))
        x11 = neuron_row(R1, "L11N3081") if R1 else None
        c3 = bool(x11 and x11["w105"])
        md.append(f"3. **No damage** (L11N3081 ratio >= 1.05 in R1): R1 ratio {f(x11 and x11['ratio'], 3)} "
                  f"((d) {f(x11 and x11['d'])}, FineWeb max {x11 and x11['fw']}). {met(c3)}")
        md.append("")

        # ---- seed stability ----
        R4 = by.get("R4")
        md += ["## Seed stability: R1 (seed 0) vs R4 (seed 1)", "",
               "| neuron | R1 (d) | R4 (d) | R1 ratio | R4 ratio | R4 - R1 ratio | ratio >= 1.05 agrees | (d) > (b) agrees |",
               "|---|---|---|---|---|---|---|---|"]
        for n in PHASE_B:
            a_ = neuron_row(R1, n) if R1 else None; b_ = neuron_row(R4, n) if R4 else None
            if not a_ or not b_:
                md.append(f"| {n} | {f(a_ and a_['d'])} | {f(b_ and b_['d'])} | n/a | n/a | n/a | n/a | n/a |"); continue
            md.append(f"| {n} | {f(a_['d'])} | {f(b_['d'])} | {f(a_['ratio'], 3)} | {f(b_['ratio'], 3)} | {f(b_['ratio'] - a_['ratio'], 3)} | "
                      f"{yn(a_['w105'] == b_['w105'])} | {yn(a_['dgtb'] == b_['dgtb'])} |")
        if R1 and R4 and R1["metrics"] and R4["metrics"]:
            md += ["", f"Final flat-group fraction: R1 {f(final_flat(R1['metrics']), 3)}, R4 {f(final_flat(R4['metrics']), 3)}; "
                   f"final mean KL: R1 {f(mean_of(R1['metrics'], 'mean_kl', 25), 3)}, R4 {f(mean_of(R4['metrics'], 'mean_kl', 25), 3)}."]
        md.append("")

        # ---- grids ----
        for lab in ("run 1", "R1", "R4"):
            r = by.get(lab)
            g = r and r["res"] and r["res"].get("grid")
            if not g:
                if lab != "run 1":
                    md += [f"## 4x4 grid {lab}: n/a", ""]
                continue
            md += [f"## 4x4 grid: {r['name']}", "",
                   "Rows: whose 8 fresh samples (prompt); columns: neuron scored on; ratio to the column's FineWeb max (this GPU). Diagonal in bold.", ""]
            for kind, rows in (("Mean ratio", g["mean_ratio"]), ("Max ratio", g["max_ratio"])):
                md += [f"{kind}:", "", "| samples of \\ scored on | " + " | ".join(g["neurons"]) + " |", "|---|" + "---|" * len(g["neurons"])]
                for n, row in zip(g["neurons"], rows):
                    md.append(f"| {n} | " + " | ".join(f"**{f(v, 3)}**" if n == c else f(v, 3) for c, v in zip(g["neurons"], row)) + " |")
                md.append("")
            md += [f"Diagonal by mean: **{'yes' if g['diag_by_mean']['diagonal'] else 'no'}**; by max: "
                   f"**{'yes' if g['diag_by_max']['diagonal'] else 'no'}**.", ""]

        # ---- top texts ----
        md += ["## Top 2 texts per neuron per run (from (d), 8 fresh samples)", ""]
        for n in PHASE_B:
            md += [f"### {n}", ""]
            for r in runs:
                x = neuron_row(r, n)
                if x is None:
                    continue
                md.append(f"- **{r['name']}**")
                for i, t in enumerate(x["top"][:2]):
                    md.append(f"  {i + 1}. **{f(t['score'])}** ({f(t['score'] / x['fw'], 3)}x) {md_text(t['text'], 300)}")
            md.append("")

        # ---- integrity ----
        md += ["## Integrity checks", "", "| run | bitwise base-weight checks | reference-text checks |", "|---|---|---|"]
        for r in runs:
            ig = r.get("integrity") or {}
            bw, rf = ig.get("bitwise", []), ig.get("reference", [])
            md.append(f"| {r['name']} | {sum(x['ok'] for x in bw)} / {len(bw)} PASS | {sum(x['ok'] for x in rf)} / {len(rf)} PASS |")
        md.append("")
    except Exception:
        md += ["", "**Report generation error:**", "", "```", traceback.format_exc(), "```"]
    os.makedirs(REPORTS_DIR, exist_ok=True)
    open(out, "w").write("\n".join(md) + "\n")
    print("report:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
