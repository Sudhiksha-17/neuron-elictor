"""GRPO pipeline gate + reports.

  python scripts/grpo_report.py gate  <run_dir>   exit 0 iff Phase A (d) > FineWeb max (this GPU) (strict)
  python scripts/grpo_report.py final <run_dir> [status_file]   writes reports/<ts>_grpo_final.md (never raises)
"""
import os, sys, json, time, math, glob, traceback, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from hermes.paths import REPO, REPORTS_DIR

CHRIS = 8.56
WIN = 1.05


def jload(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default


def jl(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except Exception:
        return []


def key(d, prefix):
    k = [x for x in d if x.startswith(prefix)]
    return (k[0], d[k[0]]) if k else (None, None)


def f(v, nd=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{nd}g}" if isinstance(v, float) else str(v)


def md_text(t, n=300):
    t = t if len(t) <= n else t[:n] + " …"
    return "`" + t.replace("`", "'").replace("\n", "\\n").replace("|", "\\|") + "`"


def gate(run_dir):
    r = jload(f"{run_dir}/phaseA/results.json")
    if not r:
        print("GATE FAIL: no Phase A results"); return 1
    n = r["neurons"]["L14N8949"]; _, d = key(n, "d_trained")
    ok = d is not None and d["best"] is not None and d["best"] > n["fw_max_this_gpu"]
    print(f"GATE {'PASS' if ok else 'FAIL'}: Phase A (d) {d and d['best']} vs FineWeb max (this GPU) {n['fw_max_this_gpu']}")
    json.dump(dict(passed=bool(ok), d_best=d and d["best"], fw_max_this_gpu=n["fw_max_this_gpu"], time=time.time()),
              open(f"{run_dir}/gate.json", "w"), indent=1)
    return 0 if ok else 1


def curve(metrics, neurons, bins=25):
    """Per bin of `bins` steps: mean of per-step mean score, max score, empty and flat fractions."""
    rows = []
    for s in range(0, len(metrics), bins):
        ch = metrics[s:s + bins]
        row = dict(steps=f"{ch[0]['step'] + 1}-{ch[-1]['step'] + 1}",
                   empty=sum(m["frac_empty"] for m in ch) / len(ch), flat=sum(m["frac_flat_groups"] for m in ch) / len(ch),
                   reward=sum(m["mean_reward"] for m in ch) / len(ch))
        for n in neurons:
            ms = [m["per_neuron"][n]["mean_score"] for m in ch if m["per_neuron"][n]["mean_score"] is not None]
            mx = [m["per_neuron"][n]["max_score"] for m in ch if m["per_neuron"][n]["max_score"] is not None]
            row[n] = (sum(ms) / len(ms) if ms else None, max(mx) if mx else None)
        rows.append(row)
    return rows


def closest_ctx(text, fwn, tok):
    """FineWeb top-10 context with the highest token-set Jaccard overlap with text."""
    a = set(tok(text, add_special_tokens=False).input_ids)
    best = None
    for c in fwn["top10"]:
        b = set(tok(c["text"], add_special_tokens=False).input_ids)
        j = len(a & b) / max(1, len(a | b))
        if best is None or j > best[0]:
            best = (j, c)
    return best


def phase_section(run_dir, ph, fw, tok, md, table):
    r = jload(f"{run_dir}/phase{ph}/results.json")
    cfg = jload(f"{run_dir}/phase{ph}/config.json", {})
    proj = jload(f"{run_dir}/phase{ph}/projection.json")
    metrics = jl(f"{run_dir}/phase{ph}/metrics.jsonl")
    md.append(f"## Phase {ph}")
    md.append("")
    if cfg:
        md.append(f"Neurons {', '.join(cfg['neurons'])}; prompts {', '.join(repr(p) for p in cfg['prompts'])}; "
                  f"{cfg['steps']} steps x {cfg['completions_per_step']} completions ({cfg['groups_per_step']} groups of "
                  f"{cfg['group_size']}); lr {cfg['lr']} linear decay; LoRA r={cfg['lora_r']} alpha={cfg['lora_alpha']}; "
                  f"{cfg['trainable_params']:,} trainable params; micro-batch {cfg['micro_bs']}, generation batch {cfg['gen_batch']}.")
    md.append(f"Training steps logged: {len(metrics)}" + (f" of {cfg.get('steps')}" if cfg else "") + ".")
    if proj:
        md.append(f"Projected after {proj['after_steps']} steps: phase {proj['projected_phase_s'] / 3600:.2f} h "
                  f"(training {proj['projected_training_s'] / 3600:.2f} h).")
    if r and r.get("phase_s"):
        md.append(f"Actual: phase {r['phase_s'] / 3600:.2f} h (training {r['training_s'] / 3600:.2f} h).")
    md.append("")
    if not r:
        md += ["**No results.json: the phase did not reach its evaluations.**", ""]
    neurons = cfg.get("neurons", [])
    if metrics:
        md += ["### Training curve (25-step bins: mean of per-step mean score / max score per neuron)", "",
               "| steps | " + " | ".join(neurons) + " | mean reward | empty frac | flat-group frac |",
               "|---|" + "---|" * (len(neurons) + 3)]
        for row in curve(metrics, neurons):
            md.append(f"| {row['steps']} | " + " | ".join(f"{f(row[n][0])} / {f(row[n][1])}" for n in neurons)
                      + f" | {row['reward']:.3f} | {row['empty']:.3f} | {row['flat']:.3f} |")
        md.append("")
    if not r:
        return r
    for n, d in r["neurons"].items():
        fa, fg = d["fw_max_a100"], d["fw_max_this_gpu"]
        ka, A = key(d, "a_untrained"); kb, B = key(d, "b_untrained"); C = d.get("c_repeated_token"); kd, D = key(d, "d_trained")
        db = D and D["best"]; ratio = db / fg if db is not None else None
        table.append([n, fa, fg, A and A["best"], (B and B["best"], B and B["n"]), C and C["score"], db, ratio,
                      None if ratio is None else ratio >= WIN, None if db is None else db > fg,
                      None if (db is None or not B or B["best"] is None) else db > B["best"]])
        md += [f"### {n}", "", f"FineWeb max: A100 {fa}, this GPU {fg}" + (f"; Chris's reported score {CHRIS}" if n == "L14N8949" else "") + ".", ""]
        rows = [("(a) untrained best of " + (ka.split('_')[-1] if ka else "?"), A and A["best"]),
                (f"(b) compute-matched untrained best of {B['n']:,}" if B else "(b) compute-matched", B and B["best"]),
                (f"(c) {C['token']!r} repeated x64" if C else "(c) repeated token", C and C["score"]),
                ("(d) trained best of " + (kd.split('_')[-1] if kd else "?"), db)]
        md += ["| eval | score | / FineWeb (this GPU) | / FineWeb (A100)" + (" | / Chris 8.56" if n == "L14N8949" else "") + " |",
               "|---|---|---|---|" + ("---|" if n == "L14N8949" else "")]
        for name, v in rows:
            md.append(f"| {name} | {f(v)} | {f(v / fg if v is not None else None, 3)} | {f(v / fa if v is not None else None, 3)}"
                      + (f" | {f(v / CHRIS if v is not None else None, 3)}" if n == "L14N8949" else "") + " |")
        if B and B.get("quantiles"):
            md.append("")
            md.append(f"(b) distribution: median {f(B['quantiles']['0.5'] if '0.5' in B['quantiles'] else B['quantiles'].get(0.5))}, "
                      f"p99 {f(B['quantiles'].get('0.99', B['quantiles'].get(0.99)))}, p99.9 {f(B['quantiles'].get('0.999', B['quantiles'].get(0.999)))}; "
                      f"{B['n_ge_fw']:,} of {B['n']:,} untrained samples beat FineWeb (this GPU), {B['n_ge_1_05']:,} reach ratio >= 1.05; "
                      f"empty {B['n_empty']:,}.")
        if D:
            md += ["", f"Top texts from (d) (8 fresh samples, {D['n_empty']} empty):", ""]
            for i, t in enumerate(D["top"][:3 if n == "L14N8949" else 2]):
                md.append(f"{i + 1}. **{f(t['score'])}** ({f(t['score'] / fg, 3)}x) {md_text(t['text'])}")
                if tok is not None and n in fw:
                    try:
                        j, c = closest_ctx(t["text"], fw[n], tok)
                        md.append(f"   - closest FineWeb top-10 context (token Jaccard {j:.2f}): rank {c['rank']}, "
                                  f"score {c['this_gpu']} (A100 {c['stored']}): {md_text(c['text'], 200)}")
                    except Exception as e:
                        md.append(f"   - closest context: error {e}")
            if D.get("per_token_best"):
                md += ["", "Per-token activations of the best (d) text:", "", "| pos | token | act |", "|---|---|---|"]
                for i, (tk, v) in enumerate(D["per_token_best"]):
                    md.append(f"| {i} | {md_text(tk, 30)} | {v:.4g} |")
        if A and A.get("top"):
            md += ["", f"Best untrained (a) text: {md_text(A['top'][0]['text'], 200)} ({f(A['top'][0]['score'])})"]
        if B and B.get("top"):
            md += [f"Best untrained (b) text: {md_text(B['top'][0]['text'], 200)} ({f(B['top'][0]['score'])})"]
        md.append("")
    if r.get("grid"):
        g = r["grid"]
        md += ["### 4x4 cross-neuron grid (rows: whose 8 fresh samples; columns: neuron scored on; ratio to the column's FineWeb max, this GPU)", "",
               "Mean ratio:", "", "| samples of \\ scored on | " + " | ".join(g["neurons"]) + " |", "|---|" + "---|" * len(g["neurons"])]
        for n, row in zip(g["neurons"], g["mean_ratio"]):
            md.append(f"| {n} | " + " | ".join(f"**{f(v, 3)}**" if n == c else f(v, 3) for c, v in zip(g["neurons"], row)) + " |")
        md += ["", "Max ratio:", "", "| samples of \\ scored on | " + " | ".join(g["neurons"]) + " |", "|---|" + "---|" * len(g["neurons"])]
        for n, row in zip(g["neurons"], g["max_ratio"]):
            md.append(f"| {n} | " + " | ".join(f"**{f(v, 3)}**" if n == c else f(v, 3) for c, v in zip(g["neurons"], row)) + " |")
        dm, dx = g["diag_by_mean"], g["diag_by_max"]
        md += ["", f"Diagonal by mean ratio: **{'yes' if dm['diagonal'] else 'no'}** (column max on diagonal: {dm['column_max_on_diagonal']}, "
               f"row max on diagonal: {dm['row_max_on_diagonal']}). By max ratio: **{'yes' if dx['diagonal'] else 'no'}** "
               f"(columns {dx['column_max_on_diagonal']}, rows {dx['row_max_on_diagonal']}).", ""]
    return r


def samples_section(run_dir, ph, md):
    s = jl(f"{run_dir}/phase{ph}/samples.jsonl")
    if not s:
        return
    md += [f"### Phase {ph} sample texts during training (4 per neuron every 50 steps; first sample shown per step)", ""]
    seen = set()
    for x in s:
        k = (x["step"], x["neuron"])
        if k in seen:
            continue
        seen.add(k)
        md.append(f"- step {x['step'] + 1}, {x['neuron']}: {f(x['score'])} {md_text(x['text'], 160)}")
    md.append("")


def final(run_dir, status_file=None):
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = f"{REPORTS_DIR}/{ts}_grpo_final.md"
    md = []
    try:
        fw = jload(f"{run_dir}/fineweb_max.json", {})
        fcfg = jload(f"{run_dir}/fwmax_config.json", {})
        status = [l.rstrip("\n") for l in open(status_file)] if status_file and os.path.exists(status_file) else []
        gate_r = jload(f"{run_dir}/gate.json")
        git = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        tok = None
        try:
            from transformers import AutoTokenizer
            from hermes.runinfo import MODEL_ID, pinned_revisions
            tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=pinned_revisions()["model_revision"])
        except Exception as e:
            md.append(f"(tokenizer unavailable for context matching: {e})")
        md = [f"# GRPO elicitor: final report", "",
              f"Generated {time.strftime('%F %T %Z')}. Run dir `{os.path.relpath(run_dir, REPO)}`, code commit `{git}`, "
              f"GPU {fcfg.get('gpu', 'n/a')}.", ""] + md
        if status:
            md += ["## Pipeline steps", "", "```", *status, "```", ""]
        if fw:
            md += ["## FineWeb max per neuron (top-10 contexts rescored on this GPU)", "",
                   "| neuron | A100 (stored) | this GPU (max of rescored top-10) | rank of this-GPU max |", "|---|---|---|---|"]
            for n, d in fw.items():
                md.append(f"| {n} | {d['fw_max_a100']} | {d['fw_max_this_gpu']} | {d['this_gpu_best_rank']} |")
            md.append("")
        table = []
        sections = []
        for ph in ("A", "B"):
            if os.path.isdir(f"{run_dir}/phase{ph}"):
                sec = []
                phase_section(run_dir, ph, fw, tok, sec, table)
                sections.append((ph, sec))
        if gate_r is not None:
            md += [f"## Gate: **{'PASS' if gate_r['passed'] else 'FAIL'}** (Phase A (d) {gate_r['d_best']} vs FineWeb max this GPU "
                   f"{gate_r['fw_max_this_gpu']}, strict >)", ""]
        if table:
            md += ["## Summary table", "",
                   "Main success test: ratio (d) / FineWeb max (this GPU) >= 1.05. Strict greater-than is secondary. "
                   "(d) > (b) is the test of whether RL adds anything beyond sampling the untrained model as often as training did.", "",
                   "| neuron | FineWeb max (A100) | FineWeb max (this GPU) | (a) untrained best of 64 | (b) compute-matched best-of-N | (c) repeated token | (d) trained best of 8 | (d) / FineWeb (this GPU) | ratio >= 1.05 | strict win | (d) > (b) |",
                   "|---|---|---|---|---|---|---|---|---|---|---|"]
            yn = lambda v: "n/a" if v is None else ("**yes**" if v else "no")
            for n, fa, fg, A, (B, BN), C, D, ratio, w105, strict, dgtb in table:
                md.append(f"| {n} | {fa} | {fg} | {f(A)} | {f(B)} (N={BN:,}) | {f(C)} | {f(D)} | {f(ratio, 3)} | {yn(w105)} | {yn(strict)} | {yn(dgtb)} |"
                          if BN else f"| {n} | {fa} | {fg} | {f(A)} | n/a | {f(C)} | {f(D)} | {f(ratio, 3)} | {yn(w105)} | {yn(strict)} | {yn(dgtb)} |")
            md.append("")
        for ph, sec in sections:
            md += sec
            if ph == "A" and gate_r is not None and not gate_r["passed"]:
                md += ["### Why the gate failed", "",
                       "Phase A's trained elicitor did not strictly beat the FineWeb max on this GPU, so Phase B was not run. "
                       "The training curve above shows the score per 25-step bin, the empty-output fraction and the flat-group "
                       "fraction; sample texts during training follow.", ""]
                samples_section(run_dir, ph, md)
        for ph, _ in sections:
            if not (ph == "A" and gate_r is not None and not gate_r["passed"]):
                samples_section(run_dir, ph, md)
        dev = os.path.join(REPO, "scripts", "grpo_deviations.md")
        if os.path.exists(dev):
            md += [open(dev).read(), ""]
    except Exception:
        md += ["", "**Report generation error:**", "", "```", traceback.format_exc(), "```"]
    os.makedirs(REPORTS_DIR, exist_ok=True)
    open(out, "w").write("\n".join(md) + "\n")
    print("report:", out)
    return 0


if __name__ == "__main__":
    cmd, run_dir = sys.argv[1], sys.argv[2]
    if cmd == "gate":
        sys.exit(gate(run_dir))
    sys.exit(final(run_dir, sys.argv[3] if len(sys.argv) > 3 else None))
