"""GPU-session benchmarks for the GRPO elicitor (written 2026-11-16 on a CPU-only pod: UNTESTED, run on GPU).

  python scripts/gpu_bench.py env   environment check: GPU, versions, pinned model loads from the local HF cache
                                    (run with HF_HUB_OFFLINE=1), fast kernels, data files present
  python scripts/gpu_bench.py gen   generation benchmark. FIRST the KV-cache check (is the cache actually used
                                    when model.config.use_cache=False and generation_config.use_cache=True; cost
                                    of use_cache on/off; effect of gradient checkpointing in eval mode), then one
                                    real training step (so optimizer state and grads are resident as in training),
                                    then a sweep: mode {untrained (LoRA off), merged, lora} x batch {64,128,192,256},
                                    64 new tokens forced, each config run twice (first = warm-up/autotune). Also
                                    merge/unmerge/restore timing, scoring time, bitwise check after the sweep, and a
                                    time projection for the real run (defaults: 300 steps, 4 groups x 16 per neuron,
                                    compute-matched (b) = 19,200 samples per neuron).
  python scripts/gpu_bench.py pg    pg_backward micro-batch check: micro_bs {16,32,48,64} on 256 completions, each
                                    run twice (first = warm-up), peak memory and time.

Writes results/gpu_bench/<ts>_<what>/results.json (+ config.json, summary.md). Exit code 1 if a hard check fails
(env: kernels/cache/data; gen: bitwise check after the sweep).
"""
import os, sys, json, time, math, copy, contextlib, platform, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
from hermes.paths import RESULTS_DIR, BASELINE_DIR

WHAT = sys.argv[1] if len(sys.argv) > 1 else ""
assert WHAT in ("env", "gen", "pg"), __doc__
OUT = f"{RESULTS_DIR}/gpu_bench/{time.strftime('%Y%m%d_%H%M%S')}_{WHAT}"
os.makedirs(OUT, exist_ok=False)
res, md, hard_fail = dict(what=WHAT), [f"## gpu_bench {WHAT}", ""], []


def log(*x):
    print(f"[{time.strftime('%F %T')}]", *x, flush=True)


def save():
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1, default=str)
    open(f"{OUT}/summary.md", "w").write("\n".join(md) + "\n")


import torch
from hermes.runinfo import run_info, MODEL_ID, pinned_revisions
json.dump(dict(what=WHAT, env=run_info()), open(f"{OUT}/config.json", "w"), indent=1, default=str)

# ------------------------------------------------------------------ env
if WHAT == "env":
    rev = pinned_revisions()["model_revision"]
    snap = f"{os.environ['HF_HOME']}/hub/models--{MODEL_ID.replace('/', '--')}/snapshots/{rev}"
    res["hf_hub_offline"] = os.environ.get("HF_HUB_OFFLINE")
    res["snapshot_dir_exists"] = os.path.isdir(snap)
    res["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    res["gpu_mem_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None
    import transformers, peft
    res["versions"] = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                           transformers=transformers.__version__, peft=peft.__version__)
    res["data_files"] = {f: os.path.exists(f"{BASELINE_DIR}/{f}") for f in ("baseline.pt", "contexts.npz", "pilots.json")}
    t0 = time.time()
    try:
        from hermes.scoring import Scorer
        sc = Scorer()
        res["model_load_s"] = time.time() - t0
        res["kernels"] = sc.kernels
        res["score_check"] = sc.score_texts(["The quick brown fox jumps over the lazy dog."], (15, 2341))
        res["model_loaded"] = True
    except Exception as e:
        res["model_loaded"] = False; res["model_error"] = repr(e)
    for k, ok in [("gpu present", res["gpu"] is not None), ("pinned snapshot in cache", res["snapshot_dir_exists"]),
                  ("model loads offline at pinned revision + fast kernels", res["model_loaded"]),
                  ("data/ files present", all(res["data_files"].values()))]:
        md.append(f"- {'PASS' if ok else 'FAIL'} {k}")
        if not ok:
            hard_fail.append(k)
    md += [f"- GPU {res['gpu']} ({res['gpu_mem_gb'] and round(res['gpu_mem_gb'], 1)} GB), versions {res['versions']}",
           f"- model load {res.get('model_load_s', float('nan')):.1f}s, HF_HUB_OFFLINE={res['hf_hub_offline']}", ""]
    save(); print("\n".join(md)); sys.exit(1 if hard_fail else 0)

# ------------------------------------------------------------------ gen / pg share an elicitor
from hermes.grpo import Elicitor, group_advantages, to_reward
from hermes.scoring import find_lm_layers
el = Elicitor(new_tokens=[f"<|n{k}|>" for k in range(4)], log=log)
pids = [el.prompt_ids(f"<|n{k}|>\n") for k in range(4)]
P = len(pids[0])
dev = el.device


def batch_x(b):
    return torch.tensor([pids[k % 4] for k in range(b)], device=dev)


def gcfg(n_tok, use_cache=True):
    g = copy.deepcopy(el.gen_config)
    g.max_new_tokens = n_tok; g.min_new_tokens = n_tok; g.use_cache = use_cache
    return g


def ctx_for(mode):
    return {"untrained": el.adapter_off, "merged": el.merged, "lora": contextlib.nullcontext}[mode]()


def timed_gen(mode, b, n_tok=64, use_cache=True):
    """One generate call of b sequences x n_tok forced new tokens. OOM -> dict(oom=True)."""
    x = batch_x(b); el.model.eval()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    try:
        with ctx_for(mode):
            torch.cuda.synchronize(); t1 = time.time()
            with torch.no_grad():
                g = el.model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=gcfg(n_tok, use_cache))
            torch.cuda.synchronize(); t2 = time.time()
    except torch.OutOfMemoryError:
        del x; torch.cuda.empty_cache()
        return dict(mode=mode, batch=b, oom=True)
    new = g.shape[1] - P
    return dict(mode=mode, batch=b, oom=False, new_tokens=new, gen_s=t2 - t1, total_s=time.time() - t0,
                ms_per_decode_step=(t2 - t1) / new * 1000, seq_per_s=b / (t2 - t1),
                peak_gb=torch.cuda.max_memory_allocated() / 1e9)


def one_batch(n_per_prompt=64):
    """256 untrained completions (4 prompts x 64), their scores and advantages (groups of 16)."""
    comps = el.generate(pids, n_per_prompt, batch=128, untrained=True)
    flat = [c for per in comps for c in per]
    texts = el.texts(flat)
    pairs = [(8, 2027)] * n_per_prompt + [(9, 5711)] * n_per_prompt + [(10, 4358)] * n_per_prompt + [(11, 3081)] * n_per_prompt
    t0 = time.time(); sc = el.score(texts, pairs).tolist(); t_sc = time.time() - t0
    rew = [to_reward(s, 1.0) for s in sc]
    adv, _ = group_advantages(rew, 16)
    return flat, adv, [pids[i // n_per_prompt] for i in range(len(flat))], t_sc


if WHAT == "gen":
    # ---------- 1. KV-cache check FIRST ----------
    kv = dict(model_config_use_cache=getattr(el.model.config, "use_cache", None),
              text_config_use_cache=getattr(getattr(el.model.config, "text_config", None), "use_cache", None),
              generation_config_use_cache=el.gen_config.use_cache,
              gradient_checkpointing=bool(getattr(el.model, "is_gradient_checkpointing", None)))
    seen = []
    def pre(mod, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        seen.append(int(h.shape[1]))
    hk = find_lm_layers(el.model)[0].register_forward_pre_hook(pre, with_kwargs=True)
    el.model.eval()
    with torch.no_grad():
        x = batch_x(1)
        el.model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=gcfg(8))
    hk.remove()
    kv["layer0_input_seq_lens_8_new_tokens"] = seen
    kv["kv_cache_used"] = len(seen) == 8 and seen[0] == P and all(l == 1 for l in seen[1:])
    log("KV cache check:", kv)
    for uc in (True, False):
        timed_gen("lora", 64, 32, use_cache=uc)                 # warm-up
        kv[f"lora_b64_32tok_use_cache_{uc}"] = timed_gen("lora", 64, 32, use_cache=uc)
    gc_kwargs = {"use_reentrant": False}
    el.model.gradient_checkpointing_disable()
    timed_gen("lora", 64, 32)
    kv["lora_b64_32tok_checkpointing_off"] = timed_gen("lora", 64, 32)
    el.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs); el.model.config.use_cache = False
    kv["lora_b64_32tok_checkpointing_on"] = timed_gen("lora", 64, 32)
    res["kv_cache"] = kv; save()
    ms = lambda d: "OOM" if d.get("oom") else f"{d['ms_per_decode_step']:.0f} ms/step"
    md += ["### KV cache (checked first)", "",
           f"- {'PASS' if kv['kv_cache_used'] else '**FAIL**'} KV cache used during generate "
           f"(layer-0 input lengths for 8 new tokens: {seen}; expected [{P}, 1, 1, ...])",
           f"- config use_cache: model {kv['model_config_use_cache']}, text {kv['text_config_use_cache']}, "
           f"generation {kv['generation_config_use_cache']}; gradient checkpointing enabled: {kv['gradient_checkpointing']}",
           f"- LoRA, batch 64, 32 tokens: use_cache=True {ms(kv['lora_b64_32tok_use_cache_True'])}, "
           f"use_cache=False {ms(kv['lora_b64_32tok_use_cache_False'])}; checkpointing off "
           f"{ms(kv['lora_b64_32tok_checkpointing_off'])}, on {ms(kv['lora_b64_32tok_checkpointing_on'])}", ""]

    # ---------- 2. one real training step, so optimizer state + grads are resident as during training ----------
    flat, adv, pid_per, t_score = one_batch()
    params = [p for _, p in el.trainable]
    opt = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.0)
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    n_tr, _ = el.pg_backward(pid_per, flat, adv, len(flat), 32); torch.cuda.synchronize(); t_pg = time.time() - t0
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()      # grads kept (the loop zeroes them after generating)
    res["train_step"] = dict(n_trained=n_tr, pg_backward_s=t_pg, peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                             score_256_s=t_score)
    log("train step:", res["train_step"]); save()

    # ---------- 3. sweep ----------
    sweep = []
    for mode in ("untrained", "merged", "lora"):
        for b in (64, 128, 192, 256):
            first = timed_gen(mode, b)
            if first.get("oom"):
                sweep.append(dict(first)); log("sweep", first); break
            second = timed_gen(mode, b)
            second["first_run_gen_s"] = first["gen_s"]
            sweep.append(second); log("sweep", second); save()
            if second.get("oom"):
                break
    res["sweep"] = sweep
    ml = [r for r in el.merge_log if r["restore_s"] is not None]
    res["merge_timing"] = dict(calls=len(el.merge_log), records=el.merge_log,
                               mean_merge_s=sum(r["merge_s"] or 0 for r in ml) / max(1, len(ml)),
                               mean_unmerge_s=sum(r["unmerge_s"] for r in ml) / max(1, len(ml)),
                               mean_restore_s=sum(r["restore_s"] for r in ml) / max(1, len(ml)))
    bw = el.check_base_bitwise(); res["bitwise_after_sweep"] = bw
    if not bw["ok"]:
        hard_fail.append("bitwise base-weight check after the sweep")
    md += ["### Generation sweep (64 new tokens forced; second run of each config; optimizer state resident)", "",
           "| mode | batch | gen s | ms / decode step | seq / s | peak GB | first-run gen s |", "|---|---|---|---|---|---|---|"]
    for d in sweep:
        md.append(f"| {d['mode']} | {d['batch']} | OOM | | | | |" if d.get("oom") else
                  f"| {d['mode']} | {d['batch']} | {d['gen_s']:.1f} | {d['ms_per_decode_step']:.0f} | {d['seq_per_s']:.2f} | "
                  f"{d['peak_gb']:.1f} | {d.get('first_run_gen_s', float('nan')):.1f} |")
    mt = res["merge_timing"]
    md += ["", f"- merged(): {mt['calls']} calls, mean merge {mt['mean_merge_s']:.2f}s, unmerge {mt['mean_unmerge_s']:.2f}s, "
           f"exact restore copy {mt['mean_restore_s']:.2f}s; backup {el.backup_bytes / 1e9:.2f} GB pinned",
           f"- {'PASS' if bw['ok'] else '**FAIL**'} bitwise base-weight check after the sweep ({bw['n_checked']} tensors, {bw['seconds']:.1f}s)",
           f"- train step (256 completions, micro_bs 32): pg_backward {t_pg:.1f}s, peak {res['train_step']['peak_gb']:.1f} GB; "
           f"scoring 256 texts {t_score:.1f}s", ""]

    # ---------- 4. projection for the real run (defaults) ----------
    def best(mode, need):
        """Largest measured batch <= need that fit (else the smallest measured one)."""
        ok = [d for d in sweep if d["mode"] == mode and not d.get("oom")]
        if not ok:
            return None
        cand = [d for d in ok if d["batch"] <= need]
        return max(cand, key=lambda d: d["batch"]) if cand else min(ok, key=lambda d: d["batch"])
    def gen_time(mode, n):
        d = best(mode, n)
        if d is None:
            return float("nan")
        extra = (mt["mean_merge_s"] + mt["mean_unmerge_s"] + mt["mean_restore_s"]) if mode == "merged" else 0.0
        return math.ceil(n / d["batch"]) * d["gen_s"] + extra
    train_per = t_pg / max(1, n_tr)
    score_per = t_score / 256
    proj = {}
    for ph, n_nrn in (("A", 1), ("B", 4)):
        n_step = 64 * n_nrn
        for gm in ("merged", "lora"):
            step_s = gen_time(gm, n_step) + n_step * (train_per + score_per)
            proj[f"{ph}_{gm}_step_s"] = step_s
        d_un = best("untrained", 2048)
        un_per = (d_un["gen_s"] / d_un["batch"]) if d_un else float("nan")
        n_eval_un = n_nrn * (64 + 19200)
        proj[f"{ph}_eval_untrained_s"] = n_eval_un * (un_per + score_per)
        proj[f"{ph}_total_h_merged"] = (300 * proj[f"{ph}_merged_step_s"] + proj[f"{ph}_eval_untrained_s"] + 300) / 3600
    res["projection"] = proj
    md += ["### Projection for the real run (300 steps, 64 completions per neuron per step, (b) = 19,200 per neuron)", "",
           f"- Phase A: step {proj['A_merged_step_s']:.0f}s merged ({proj['A_lora_step_s']:.0f}s unmerged LoRA); "
           f"untrained evals {proj['A_eval_untrained_s'] / 3600:.2f} h; total ~{proj['A_total_h_merged']:.1f} h",
           f"- Phase B: step {proj['B_merged_step_s']:.0f}s merged ({proj['B_lora_step_s']:.0f}s unmerged LoRA); "
           f"untrained evals {proj['B_eval_untrained_s'] / 3600:.2f} h; total ~{proj['B_total_h_merged']:.1f} h",
           "- Assumes every completion runs the full 64 tokens (upper bound for generation) and the step's other costs "
           "scale linearly with completions.", ""]
    save(); print("\n".join(md)); sys.exit(1 if hard_fail else 0)

if WHAT == "pg":
    flat, adv, pid_per, _ = one_batch()
    params = [p for _, p in el.trainable]
    rows = []
    for mb in (16, 32, 48, 64):
        r = dict(micro_bs=mb)
        for k in ("warmup", "measured"):
            for p in params:
                p.grad = None
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            try:
                n_tr, _ = el.pg_backward(pid_per, flat, adv, len(flat), mb); torch.cuda.synchronize()
                r[f"{k}_s"] = time.time() - t0; r[f"{k}_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            except torch.OutOfMemoryError:
                r["oom"] = True; torch.cuda.empty_cache(); break
        rows.append(r); log("pg", r); res["pg_backward"] = rows; save()
        if r.get("oom"):
            break
    md += ["### pg_backward micro-batch check (256 completions; second run measured)", "",
           "| micro_bs | warm-up s | measured s | peak GB |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['micro_bs']} | OOM | | |" if r.get("oom") and "measured_s" not in r else
                  f"| {r['micro_bs']} | {r.get('warmup_s', float('nan')):.1f} | {r.get('measured_s', float('nan')):.1f} | "
                  f"{r.get('measured_peak_gb', float('nan')):.1f} |")
    md.append("")
    save(); print("\n".join(md)); sys.exit(0)
