"""Final phase 2 report, written by scripts/watch_and_stop.sh when the full run exits (success or failure).
Selects the 32 pilot neurons (seed 0, 1 per layer) from the finished baseline and writes
/workspace/reports/<timestamp>_phase2_final.md. Never raises: on missing outputs it reports what exists.

Usage: python scripts/final_report.py <run_dir> <exit_code> <run_log> [report_dir] [pilot_tag]
"""
import sys, os, json, time, glob, subprocess, traceback
run_dir, code, run_log = sys.argv[1], sys.argv[2], sys.argv[3]
W = "/workspace"
report_dir = sys.argv[4] if len(sys.argv) > 4 else f"{W}/reports"
pilot_tag = sys.argv[5] if len(sys.argv) > 5 else "full1M_pilots"
os.makedirs(report_dir, exist_ok=True)
out = f"{report_dir}/{time.strftime('%Y%m%d_%H%M%S')}_phase2_final.md"
md = [f"# Phase 2 full baseline: final report", "",
      f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')} by watch_and_stop.sh. Run dir `{run_dir}`, log `{run_log}`.", "",
      f"- **Run exit code: {code}**"]
try:
    cfg = json.load(open(f"{run_dir}/config.json"))
    md += [f"- Config: {cfg['n_contexts']:,} contexts requested, layers 0-31 ({cfg.get('n_neurons_total')} neurons), "
           f"max_chunks_per_doc={cfg['max_chunks_per_doc']}, model rev `{cfg['env']['model_revision'][:12]}`, "
           f"FineWeb rev `{cfg['env']['fineweb_revision'][:12]}`, started {cfg['started']}"]
    if not os.path.exists(f"{run_dir}/summary.json"):
        md += ["", "**The run did not finish: no summary.json.** Last log lines:", "", "```",
               *open(run_log, errors="replace").read().splitlines()[-30:], "```"]
        st = glob.glob(f"{run_dir}/state.pt")
        md.append(f"\nCheckpoint state present: {bool(st)} (resume with `python scripts/phase2_baseline.py --resume {run_dir}`).")
    else:
        s = json.load(open(f"{run_dir}/summary.json")); pl = s["per_layer"]
        import torch
        b = torch.load(f"{run_dir}/baseline.pt")
        mx = b["max"].float(); med = mx.median(1, keepdim=True).values
        low_rel = 0.1
        low = (mx <= 0) | (mx < low_rel * med)
        md += [f"- Finished {s['finished']}; **time taken {s['elapsed_s'] / 60:.1f} min** this session "
               f"({s['n_contexts_this_session']:,} contexts; steady {s['steady_ctx_per_s']:.0f} ctx/s; startup {s['first_batch_s']:.0f} s; "
               f"data wait {s['data_wait_s']:.0f} s)",
               f"- Contexts: {s['n_contexts']:,}. **Documents used: {s['n_docs_contributing']:,}** contributing "
               f"({s['n_docs_streamed']:,} streamed incl. docs < 64 tokens); chunks/doc histogram (0..4): {s['chunks_per_doc_hist']}",
               f"- GPU memory peak: {s['gpu_peak_allocated_gib']:.1f} GiB allocated / {s['gpu_peak_reserved_gib']:.1f} GiB reserved",
               f"- Output: " + ", ".join(f"{k} {v / 1e6:.1f} MB" for k, v in s["output_bytes"].items()), "",
               "## Max activation distribution per layer (quantiles over the layer's 9216 neuron maxima)", "",
               "| Layer | min | 1% | 10% | median | 90% | 99% | max | low-act | frac argmax@pos0 | >=5 top-10 same doc |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        q = pl["max_quantiles"]
        for L in range(32):
            md.append(f"| {L} | " + " | ".join(f"{q[k][L]:.3g}" for k in ("0", "0.01", "0.1", "0.5", "0.9", "0.99", "1")) +
                      f" | {int(low[L].sum())} | {pl['frac_argmax_pos0'][L]:.3f} | {pl['n_neurons_top10_same_doc_ge5'][L]} |")
        md += ["", f"## Low-activation neurons", "",
               f"Rule (relative): low-activation = FineWeb max <= 0 or max < {low_rel} x median neuron max of its layer. "
               f"**Total low-activation: {int(low.sum())}** of 294,912 (never positive: {int((mx <= 0).sum())}). Per layer: {low.sum(1).tolist()}", "",
               "## Argmax-position histogram (number of neurons whose max is at each position, buckets)", "",
               "| Layer | pos 0 | 1-3 | 4-15 | 16-31 | 32-63 |", "|---|---|---|---|---|---|"]
        for L in range(32):
            h = pl["argmax_pos_hist"][L]
            md.append(f"| {L} | {h[0]} | {sum(h[1:4])} | {sum(h[4:16])} | {sum(h[16:32])} | {sum(h[32:64])} |")
        md += ["", "Full 64-bin histograms: summary.json `per_layer.argmax_pos_hist`.", "",
               "## Top-10 clustering", "",
               f"Neurons with >= 5 of their top-10 contexts from one document: **{sum(pl['n_neurons_top10_same_doc_ge5'])}** "
               f"(0 by construction with the 4-chunk cap). Distribution of the largest same-document count in a top-10, "
               f"summed over layers (count 1..10): {[sum(r[k] for r in pl['top10_max_same_doc_hist']) for k in range(1, 11)]}", ""]
        # ---- pilots ----
        r = subprocess.run([sys.executable, f"{W}/scripts/select_pilots.py", "--baseline", run_dir, "--seed", "0",
                            "--n_per_layer", "1", "--tag", pilot_tag], capture_output=True, text=True)
        pdir = [l.split("saved: ")[1] for l in r.stdout.splitlines() if l.startswith("saved: ")]
        if r.returncode == 0 and pdir:
            pil = json.load(open(f"{pdir[0]}/pilots.json")); pc = json.load(open(f"{pdir[0]}/config.json"))
            md += ["## The 32 pilot neurons", "", f"Seed {pc['seed']}, {pc['n_per_layer']} per layer. Rule: {pc['selection_rule']}",
                   f"Saved: `{pdir[0]}`", "", "| Neuron | FineWeb max | argmax ctx | argmax pos |", "|---|---|---|---|"]
            md += [f"| {p['id']} | {p['fineweb_max']:.3f} | {p['argmax_ctx']} | {p['argmax_pos']} |" for p in pil]
        else:
            md += ["## Pilot selection FAILED", "", "```", r.stdout[-2000:], r.stderr[-3000:], "```"]
except Exception:
    md += ["", "## Report generation error", "", "```", traceback.format_exc(), "```"]
open(out, "w").write("\n".join(md) + "\n")
print("final report:", out)
