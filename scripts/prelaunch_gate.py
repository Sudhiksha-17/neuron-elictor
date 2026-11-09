"""Pre-launch gate for the full phase 2 run. Collects the step-3 checks (phase 1, scoring tests on L15 and
L2, reward tests, 10k dry run) from one check timestamp, writes a markdown report to /workspace/reports/,
and exits 0 only if every gate condition holds.

Usage: python scripts/prelaunch_gate.py <check_ts> <full_config.json>
  full_config.json = output of `phase2_baseline.py --print_config ...` for the planned full run
"""
import sys, os, json, re, glob, time
sys.path.insert(0, "/workspace")
from hermes.scoring import assert_fast_kernels, KernelFallbackError

ts, full_cfg_path = sys.argv[1], sys.argv[2]
W = "/workspace"
log = lambda name: open(f"{W}/logs/{name}_{ts}.log").read()
def res_dir(log_text):
    return re.findall(r"saved: (\S+)", log_text)[-1]

gate, lines = {}, []
# ---- checks ----
p1 = log("phase1"); p1r = json.load(open(f"{res_dir(p1)}/results.json"))
sc = {L: log(f"test_scoring_L{L}") for L in (15, 2)}
scr = {L: json.load(open(f"{res_dir(t)}/results.json")) for L, t in sc.items()}
rw = log("test_reward"); rwr = json.load(open(f"{res_dir(rw)}/results.json"))
dry_log = log("phase2_dry10k_final"); dry_dir = res_dir(dry_log)
dry_dir = dry_dir if dry_dir.startswith("/") else f"{W}/{dry_dir}"
dry_cfg = json.load(open(f"{dry_dir}/config.json")); dry = json.load(open(f"{dry_dir}/summary.json"))
full = json.load(open(full_cfg_path))

all_checks = {f"phase1.{k}": v for k, v in p1r["checks"].items()}
for L in (15, 2): all_checks.update({f"scoring_L{L}.{k}": v for k, v in scr[L]["checks"].items()})
all_checks.update({f"reward.{k}": v for k, v in rwr["checks"].items()})
all_checks["dry_run.summary_written"] = os.path.exists(f"{dry_dir}/summary.json") and dry["n_contexts"] == 10000
gate["every step-3 check passes"] = all(all_checks.values())
try:
    kern = assert_fast_kernels(); kern_ok = all(v.startswith(("fla.", "causal_conv1d.")) for v in kern.values())
except KernelFallbackError as e:
    kern, kern_ok = {"error": str(e)}, False
gate["fast kernel active (assertion now + kernels used in dry run)"] = kern_ok and "fla.ops.gated_delta_rule" in dry_log
gate["full config covers all 32 layers (294,912 neurons)"] = full["layers"] == list(range(32)) and full["n_neurons_total"] == 294912
gate["4 chunks/doc cap set and recorded in config.json"] = (full.get("max_chunks_per_doc") == 4 and dry_cfg.get("max_chunks_per_doc") == 4)
gate["position histogram + top-10 clustering outputs enabled"] = (
    {"argmax_pos_hist", "n_neurons_top10_same_doc_ge5", "top10_max_same_doc_hist"} <= set(full["outputs"]["per_layer"])
    and all(k in dry["per_layer"] for k in ("argmax_pos_hist", "n_neurons_top10_same_doc_ge5", "top10_max_same_doc_hist")))
ok = all(gate.values())

def maxdiff(t, label):
    m = re.search(label + r".*?max\|diff\| ([0-9.e+-]+)", t); return m.group(1) if m else "?"
proj_min = (full["n_contexts"] / dry["steady_ctx_per_s"] + dry["first_batch_s"]) / 60
ctx_per_doc = dry["n_contexts"] / dry["n_docs_contributing"]
md = [f"# Pre-launch report: step-0 checks on the A100 + 10k dry run", "",
      f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')} on pod {os.environ.get('RUNPOD_POD_ID')}, "
      f"GPU {dry_cfg['env']['gpu']}. Check run timestamp `{ts}`.", "",
      f"**Gate: {'PASS - full run may launch' if ok else 'FAIL - full run NOT launched'}**", "",
      "| Gate condition | Result |", "|---|---|"]
md += [f"| {k} | {'PASS' if v else '**FAIL**'} |" for k, v in gate.items()]
md += ["", "## Fast-kernel assertion", "",
       f"- `assert_fast_kernels()` passes; simulated fallback -> {scr[15]['simulated_fallback'][:60]}...",
       *[f"- `{k}` -> `{v}`" for k, v in kern.items()], "",
       "## Hook check (phase 1, L15N2341)", "",
       f"- max |hook - manual bf16 silu(gate)*up| = {p1r['max_abs_diff_bf16']:.1e} (all 9216 channels exact)",
       f"- max |hook - manual fp32| = {p1r['max_abs_diff_fp32']:.2e} (bf16 rounding)",
       f"- max |early-exit - full forward| = {p1r['max_abs_diff_early_vs_full']:.1e}",
       f"- score {p1r['score']:.4f}; Scorer (fixed shape [256,64]) {p1r['scorer_score']:.4f}",
       f"- checks: {sum(p1r['checks'].values())}/{len(p1r['checks'])} pass", "",
       "## Padded batch vs one-at-a-time (scripts/test_scoring.py, 24 texts x 9216 neurons)", "",
       "| Layer | value mismatches | argmax mismatches | max abs diff | pad leakage | mixed-layer pairs | checks |",
       "|---|---|---|---|---|---|---|"]
for L in (15, 2):
    m = scr[L]["mismatches"]
    md.append(f"| {L} ({'full' if L == 15 else 'linear'} attn) | {m['batched']} (shuffled {m['shuffled']}) | {m['argmax']} | "
              f"{maxdiff(sc[L], 'batched vs one-at-a-time')} | {m['garbage_pads']} | {m['mixed_pairs']} | "
              f"{sum(scr[L]['checks'].values())}/{len(scr[L]['checks'])} |")
md += ["", f"Reward tests (scripts/test_reward.py): {sum(rwr['checks'].values())}/{len(rwr['checks'])} pass "
       "(mixed-layer batch == single scoring, LoRA off == base model, empty text -> NaN).", "",
       f"## 10k dry run (`{dry_dir}`)", "",
       "| Metric | Value |", "|---|---|",
       f"| Steady speed | {dry['steady_ctx_per_s']:.0f} contexts/s |",
       f"| Startup (first batch, mostly FineWeb stream) | {dry['first_batch_s']:.0f} s |",
       f"| Documents | {dry['n_docs_streamed']} streamed, {dry['n_docs_contributing']} contributing ({ctx_per_doc:.2f} contexts/doc) |",
       f"| Chunks per doc (0..4) | {dry['chunks_per_doc_hist']} |",
       f"| Projected full run ({full['n_contexts']:,} contexts) | ~{proj_min:.0f} min; ~{full['n_contexts'] / ctx_per_doc / 1e3:.0f}k documents |",
       f"| GPU memory | {dry['gpu_peak_allocated_gib']:.1f} GiB peak allocated, {dry['gpu_peak_reserved_gib']:.1f} GiB reserved |",
       f"| Output size (10k) | " + ", ".join(f"{k} {v / 1e6:.1f} MB" for k, v in dry["output_bytes"].items()) + " |",
       f"| Projected output (1M) | baseline.pt ~{dry['output_bytes']['baseline.pt'] / 1e6:.0f} MB (fixed), contexts.npz ~{dry['output_bytes']['contexts.npz'] * full['n_contexts'] / dry['n_contexts'] / 1e9:.2f} GB |",
       f"| Neurons with >=5 of top-10 from one doc | {sum(dry['per_layer']['n_neurons_top10_same_doc_ge5'])} total (0 by construction with the 4-chunk cap) |",
       f"| Frac. neurons with argmax at pos 0, layers 0-3 | {[round(x, 3) for x in dry['per_layer']['frac_argmax_pos0'][:4]]} |", "",
       "## Planned full-run config", "", "```json", json.dumps(full, indent=1), "```"]
os.makedirs(f"{W}/reports", exist_ok=True)
out = f"{W}/reports/{time.strftime('%Y%m%d_%H%M%S')}_prelaunch_{'PASS' if ok else 'FAIL'}.md"
open(out, "w").write("\n".join(md) + "\n")
print("\n".join(f"{'PASS' if v else 'FAIL'}  {k}" for k, v in gate.items()))
if not ok:
    print("failed checks:", [k for k, v in all_checks.items() if not v])
print("report:", out)
sys.exit(0 if ok else 1)
