"""One GRPO phase (train + evaluations) with our own loop (hermes/grpo.py).

Phase A: single neuron L14N8949, prompt "Text:\\n", 4 groups x 16 completions per step.
Phase B: shared elicitor for L8N2027, L9N5711, L10N4358, L11N3081, one new special token per neuron
         (<|n0|>..<|n3|>, embedding rows trained via PEFT trainable_token_indices), prompt "<|nK|>\\n",
         16 groups x 16 per step = exactly 4 groups per neuron.

Evaluations (written to results.json after each one, so a crash keeps what finished):
  (d) best of 8 fresh samples from the trained elicitor (+ top 3 texts, per-token activations of the best)
  (e) Phase B only: 4x4 grid, each neuron's 8 fresh samples scored on all 4 neurons (ratio to column FineWeb max)
  (a) best of 64 untrained samples (LoRA and trainable tokens disabled = exactly the untrained model)
  (c) the neuron's FineWeb argmax token repeated to 64 tokens
  (b) compute-matched best-of-N: best of as many untrained samples per neuron as training generated for it
The untrained evaluations run after training with the adapter disabled (bit-identical to the untrained
model: LoRA B = 0 at init, trainable-token deltas start at the original rows; checked by scripts/test_grpo.py).

Integrity checks (UNTESTED, run on GPU): bitwise base-weight check (GPU weights vs the exact CPU backup) at the
start, every --check_every steps, after training and at the end; reference-text check (each phase neuron's best
FineWeb context rescored LoRA-off must equal fineweb_max.json's this-GPU value exactly) at the start, after
training and at the end. Any failure is logged and the phase exits with code 3.

Gate: a non-dry Phase B refuses to start unless <run_dir>/gate.json says passed and Phase A was not a dry run.
There is no override outside --dry (the pipeline's --force_phase_b requires --dry).

Scope flags (defaults unchanged): --steps, --groups_per_step, --n_matched (compute-matched best-of-N,
default steps x groups_per_neuron x group_size, also in --dry), --n_untrained_best, --n_fresh.

Run defaults for the real run (decided 2026-11-17): unmerged LoRA generation (no --merged_gen; merge-path test
cosine 0.843 between merged and LoRA logit changes, unmerged costs ~20 extra min on the 5090), --gen_batch 192 (2026-11-18, RTX 4090; was 256 on the 5090),
--micro_bs 32. --phase_b_prompt nK | nK_text selects the Phase B prompt "<|nK|>\\n" (default) or "<|nK|>\\nText:\\n";
scripts/real_run.sh picks it with the pre-registered rule in scripts/phase_b_prompt_check.py.

Collapse-fix flags (added 2026-11-18; defaults reproduce run 1 exactly):
  --kl_beta B            KL penalty, per-token k3 against the LoRA-off reference (hermes/grpo.py). Default 0. Mean
                         per-token KL over all completions is logged every step in either case (metrics.jsonl mean_kl).
  --stop_flat_frac F --stop_patience K
                         early stop: end training once the flat-group fraction has been > F for K consecutive steps
                         (stop step logged, early_stop.json). Off unless --stop_flat_frac is given.
  --reuse_untrained_from <run dir>
                         reuse (a), (b), (c) from a previous run dir (searched: <dir>/phase*/) for the same neurons, only if
                         its GPU name equals this GPU's and neuron, prompt, new tokens, model revision, (a)/(b) sizes and
                         FineWeb max (this GPU) all match; otherwise recompute and log why (results.json reused_untrained).
  --phase S --neuron LxNy [--prompt 'Text:\n']
                         single-neuron run (Phase A layout: 4 groups x 16 per step) on any neuron in fineweb_max.json.
  --gate_from <run dir>  non-dry Phase B in a run dir without its own Phase A: read gate.json and phaseA/config.json from
                         that run dir instead (the gate itself is unchanged: passed and not dry).

Usage: python scripts/grpo_phase.py --phase A --run_dir results/grpo/<ts> [--steps 300] [--dry]
"""
import os, sys, json, time, math, argparse, glob, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import numpy as np, torch
from hermes.runinfo import run_info
from hermes.scoring import parse_neuron_id, CONTEXT_LEN
from hermes.grpo import (Elicitor, group_advantages, to_reward, EMPTY_REWARD, FLAT_STD, ADV_EPS, LORA_TARGET_RE,
                         PHASE_B_PROMPTS)

PHASES = {
    "A": dict(neurons=["L14N8949"], prompts=["Text:\n"], tokens=[], groups_per_neuron=4),
    "B": dict(neurons=["L8N2027", "L9N5711", "L10N4358", "L11N3081"], prompts=[f"<|n{k}|>\n" for k in range(4)],
              tokens=[f"<|n{k}|>" for k in range(4)], groups_per_neuron=4),
    "S": dict(neurons=None, prompts=None, tokens=[], groups_per_neuron=4),   # single neuron, set by --neuron/--prompt
}
ap = argparse.ArgumentParser()
ap.add_argument("--phase", choices=list(PHASES), required=True)
ap.add_argument("--run_dir", required=True, help="pipeline run dir holding fineweb_max.json")
ap.add_argument("--steps", type=int, default=None, help="default 300 (3 with --dry)")
ap.add_argument("--groups_per_step", type=int, default=None,
                help="groups per step over all neurons (multiple of the number of neurons); default 4 per neuron")
ap.add_argument("--group_size", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--lora_r", type=int, default=32)
ap.add_argument("--lora_alpha", type=int, default=64)
ap.add_argument("--max_grad_norm", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--micro_bs", type=int, default=32)
ap.add_argument("--phase_b_prompt", choices=list(PHASE_B_PROMPTS), default="nK",
                help="Phase B prompt template: nK = '<|nK|>\\n' (a), nK_text = '<|nK|>\\nText:\\n' (b)")
ap.add_argument("--gen_batch", type=int, default=192, help="generation batch during training and for (d)")
ap.add_argument("--eval_batch", type=int, default=192, help="generation batch for untrained (adapter off) evals (a), (b)")
ap.add_argument("--merged_gen", action="store_true", help="generate with LoRA merged into bf16 weights (faster, not bitwise)")
ap.add_argument("--n_untrained_best", type=int, default=None, help="(a) best-of-N size; default 64 (8 with --dry)")
ap.add_argument("--n_matched", type=int, default=None,
                help="(b) best-of-N size; default steps x groups_per_neuron x group_size (compute-matched, also in --dry)")
ap.add_argument("--n_fresh", type=int, default=8)
ap.add_argument("--sample_every", type=int, default=None, help="default 50 (2 with --dry)")
ap.add_argument("--check_every", type=int, default=50, help="bitwise base-weight check every this many steps")
ap.add_argument("--dry", action="store_true", help="tiny smoke run (for testing the pipeline)")
ap.add_argument("--kl_beta", type=float, default=0.0, help="KL penalty (k3 vs LoRA-off reference); 0 = run 1 loss")
ap.add_argument("--stop_flat_frac", type=float, default=None, help="early stop threshold on the flat-group fraction (off)")
ap.add_argument("--stop_patience", type=int, default=25, help="consecutive steps above --stop_flat_frac before stopping")
ap.add_argument("--reuse_untrained_from", default=None, help="run dir whose (a), (b), (c) to reuse if compatible")
ap.add_argument("--neuron", default=None, help="--phase S: the neuron, e.g. L10N4358")
ap.add_argument("--prompt", default="Text:\n", help="--phase S: the prompt (default 'Text:\\n')")
ap.add_argument("--gate_from", default=None, help="non-dry Phase B: read the gate from this run dir")
# warm start (added 2026-11-18; all off by default, so earlier runs reproduce exactly)
ap.add_argument("--warm_start", action="store_true",
                help="cross-entropy warm start of adapter 'warm' on the neuron's top-10 FineWeb contexts, then GRPO on "
                     "adapter 'policy' (initialized as a copy of warm)")
ap.add_argument("--warm_lr", type=float, default=1e-4)
ap.add_argument("--warm_epochs", type=int, default=2)
ap.add_argument("--warm_bs", type=int, default=1, help="warm-start micro-batch = optimizer batch (contexts per step)")
ap.add_argument("--kl_ref", choices=["base", "warm"], default=None,
                help="reference policy of the k3 KL term (and of the logged KL): base = all adapters off (run 1), "
                     "warm = frozen warm adapter. Default: warm with --warm_start, else base")
ap.add_argument("--n_diversity", type=int, default=64, help="warm-start diversity pre-check sample count")
ap.add_argument("--test_force_redo", action="store_true", help="TESTING ONLY (needs --dry): treat the first warm start as near-copies so the 1-epoch redo path runs")
a = ap.parse_args()
if a.kl_ref is None:
    a.kl_ref = "warm" if a.warm_start else "base"
if a.kl_ref == "warm" and not a.warm_start:
    sys.exit("--kl_ref warm needs --warm_start")
if a.warm_start and a.phase == "B":
    sys.exit("--warm_start is only implemented for single-neuron phases (A, S)")
if a.test_force_redo and not a.dry:
    sys.exit("--test_force_redo needs --dry")
if a.warm_start and a.merged_gen:
    sys.exit("--warm_start uses unmerged generation only")
P = PHASES[a.phase]
if a.phase == "S":
    if not a.neuron:
        sys.exit("--phase S needs --neuron")
    P["neurons"], P["prompts"] = [a.neuron], [a.prompt]
elif a.neuron:
    sys.exit("--neuron only applies to --phase S")
if a.phase == "B":
    P["prompts"] = [PHASE_B_PROMPTS[a.phase_b_prompt].format(k=k) for k in range(len(P["neurons"]))]
elif a.phase_b_prompt != "nK":
    sys.exit("--phase_b_prompt only applies to --phase B")
DEF = dict(steps=3, n_untrained_best=8, sample_every=2) if a.dry else dict(steps=300, n_untrained_best=64, sample_every=50)
for k, v in DEF.items():          # explicit flags always win, also with --dry
    if getattr(a, k) is None:
        setattr(a, k, v)
n_nrn = len(P["neurons"])
if a.groups_per_step is None:
    a.groups_per_step = P["groups_per_neuron"] * n_nrn
if a.groups_per_step % n_nrn or a.groups_per_step <= 0:
    sys.exit(f"--groups_per_step {a.groups_per_step} must be a positive multiple of the {n_nrn} neurons")
P["groups_per_neuron"] = a.groups_per_step // n_nrn
n_matched = a.n_matched or a.steps * P["groups_per_neuron"] * a.group_size

if a.phase == "B" and not a.dry:   # the gate cannot be bypassed in a real run (UNTESTED, run on GPU)
    gdir = a.gate_from or a.run_dir
    try:
        gate_r = json.load(open(f"{gdir}/gate.json"))
        cfg_a = json.load(open(f"{gdir}/phaseA/config.json"))
    except Exception as e:
        sys.exit(f"REFUSING Phase B: no gate.json / Phase A config in {gdir} ({e})")
    if not gate_r.get("passed") or cfg_a.get("dry"):
        sys.exit(f"REFUSING Phase B: gate passed={gate_r.get('passed')}, Phase A dry={cfg_a.get('dry')}")
out_dir = f"{a.run_dir}/phase{a.phase}"
os.makedirs(out_dir, exist_ok=False)
LOG = open(f"{out_dir}/train.log", "a")
def log(*x):
    s = f"[{time.strftime('%F %T')}] " + " ".join(str(v) for v in x)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

fw = json.load(open(f"{a.run_dir}/fineweb_max.json"))
neurons = P["neurons"]; pairs_n = [parse_neuron_id(n) for n in neurons]
fwmax = [fw[n]["fw_max_this_gpu"] for n in neurons]

def ref_contexts():
    """Each phase neuron's best FineWeb context (tokens) and its exact this-GPU score from fineweb_max.json."""
    if all("ref_tokens" in fw[n] for n in neurons):
        toks = {n: fw[n]["ref_tokens"] for n in neurons}
    else:   # run dir written by an older grpo_fwmax.py: read the tokens from the baseline
        from hermes.paths import BASELINE_DIR
        from hermes.baseline import Baseline
        b = Baseline(BASELINE_DIR)
        toks = {n: b.context_tokens(fw[n]["top10"][fw[n]["this_gpu_best_rank"]]["ctx"]) for n in neurons}
        del b
    return [dict(neuron=list(pr), tokens=toks[n], expected=fw[n]["fw_max_this_gpu"]) for n, pr in zip(neurons, pairs_n)]

integrity = dict(bitwise=[], reference=[])
def fail(msg):
    log("INTEGRITY FAIL: " + msg)
    json.dump(integrity, open(f"{out_dir}/integrity.json", "w"), indent=1)
    sys.exit(3)

def check_bitwise(where):
    r = el.check_base_bitwise(); r["where"] = where; integrity["bitwise"].append(r)
    json.dump(integrity, open(f"{out_dir}/integrity.json", "w"), indent=1)
    log(f"bitwise base-weight check ({where}): {'PASS' if r['ok'] else 'FAIL'} "
        f"({r['n_checked']} tensors, {r['n_mismatch']} mismatched {r['mismatches']}, {r['seconds']:.1f}s)")
    if not r["ok"]:
        fail(f"base weights differ from the exact backup at {where}: {r['mismatches']}")

def check_reference(where):
    r = el.reference_scores(REFS); r["where"] = where; integrity["reference"].append(r)
    json.dump(integrity, open(f"{out_dir}/integrity.json", "w"), indent=1)
    log(f"reference-text check ({where}): {'PASS' if r['ok'] else 'FAIL'} "
        + " ".join(f"L{x['neuron'][0]}N{x['neuron'][1]} {x['got']} (expected {x['expected']})" for x in r["rows"]))
    if not r["ok"]:
        fail(f"reference contexts no longer score their stored values at {where}")
G, gpn = a.group_size, P["groups_per_neuron"]
per_neuron = G * gpn; n_step = per_neuron * len(neurons)

el = Elicitor(new_tokens=P["tokens"], lora_r=a.lora_r, lora_alpha=a.lora_alpha, seed=a.seed, log=log,
              warm_start=a.warm_start)
REFS = ref_contexts()
pids = [el.prompt_ids(p) for p in P["prompts"]]

# ---------------- warm start (added 2026-11-18) ----------------
from hermes.copycheck import copy_stats, diversity, NEAR_COPY_JACCARD
TOP10 = {}          # neuron -> its stored top-10 FineWeb contexts (token lists, rank order), for warm start + copy check
if a.warm_start:
    from hermes.paths import BASELINE_DIR
    from hermes.baseline import Baseline
    _b = Baseline(BASELINE_DIR)
    TOP10 = {n: [_b.context_tokens(c["ctx"]) for c in fw[n]["top10"]] for n in neurons}
    del _b

def text_tokens(texts):
    ids, lens = el.scorer.encode(texts)
    return [ids[i, :int(lens[i])].tolist() for i in range(len(texts))]

def diversity_check(n_samples, tag):
    """n_samples from the active adapter (warm) for the (single) neuron: diversity metrics + their scores."""
    comps = el.generate(pids, n_samples, batch=a.gen_batch)[0]
    t = el.texts(comps); s = el.score(t, [pairs_n[0]] * len(t)).tolist()
    d = diversity(text_tokens(t), t, TOP10[neurons[0]])
    sv = [v for v in s if not math.isnan(v)]
    d.update(tag=tag, mean_score=float(np.mean(sv)) if sv else None, max_score=max(sv) if sv else None,
             n_empty=sum(math.isnan(v) for v in s), samples=[dict(text=x, score=v, **c) for x, v, c in zip(t, s, d.pop("per_sample"))])
    log(f"diversity pre-check ({tag}): {d['n_distinct_texts']}/{n_samples} distinct, mean pairwise Jaccard "
        f"{d['mean_pairwise_jaccard']:.3f}, max Jaccard to top-10 mean {d['max_jaccard_to_top10_mean']:.3f} / max "
        f"{d['max_jaccard_to_top10_max']:.3f}, near-copies (> {NEAR_COPY_JACCARD}) {d['n_near_copies']}/{n_samples}; "
        f"score mean {d['mean_score']} max {d['max_score']}")
    return d

warm_info = None
if a.warm_start:
    check_bitwise("before warm start"); check_reference("before warm start")
    init_state = el.adapter_state("warm")
    attempts = []
    for ep in ([a.warm_epochs, 1] if a.warm_epochs > 1 else [a.warm_epochs]):
        if attempts:
            log(f"more than half of the diversity samples are near-copies: redoing the warm start with {ep} epoch")
            el.load_adapter_state(init_state)
        h = el.warm_start_train(pids[0], TOP10[neurons[0]], lr=a.warm_lr, epochs=ep, bs=a.warm_bs, seed=a.seed)
        torch.manual_seed(a.seed + 10 + len(attempts))
        h["diversity"] = diversity_check(a.n_diversity, f"{ep} epoch(s)")
        attempts.append(dict(n_epochs=ep, **h))
        if h["diversity"]["frac_near_copies"] <= 0.5 and not (a.test_force_redo and len(attempts) == 1):
            break
    warm_info = dict(lr=a.warm_lr, bs=a.warm_bs, planned_epochs=a.warm_epochs, used_epochs=attempts[-1]["n_epochs"],
                     redone=len(attempts) > 1, attempts=attempts, contexts=f"top-10 FineWeb contexts of {neurons[0]} "
                     "(data/contexts.npz), as completions of the prompt", near_copy_threshold=NEAR_COPY_JACCARD)
    json.dump(warm_info, open(f"{out_dir}/warm_start.json", "w"), indent=1)
    el.make_policy()
    WARM_STATE = el.adapter_state("warm")
    torch.manual_seed(a.seed)      # training RNG stream starts from the seed after the warm start
el.kl_ref = a.kl_ref

def check_warm_frozen(where):
    if not a.warm_start:
        return
    ok, n, bad = el.adapter_equal(WARM_STATE)
    integrity.setdefault("warm_frozen", []).append(dict(where=where, ok=ok, n_checked=n, mismatches=bad))
    json.dump(integrity, open(f"{out_dir}/integrity.json", "w"), indent=1)
    log(f"warm-adapter frozen check ({where}): {'PASS' if ok else 'FAIL'} ({n} tensors, mismatched {bad})")
    if not ok:
        fail(f"warm adapter weights changed at {where}: {bad}")
hp = dict(phase=a.phase, neurons=neurons, prompts=P["prompts"],
          phase_b_prompt=a.phase_b_prompt if a.phase == "B" else None, prompt_token_ids=pids, new_tokens=P["tokens"],
          new_token_ids=el.new_ids, steps=a.steps, group_size=G, groups_per_neuron=gpn, groups_per_step=gpn * len(neurons),
          completions_per_step=n_step, lr=a.lr, lr_schedule="linear decay to 0 over all steps, no warmup",
          optimizer="AdamW(betas=(0.9,0.999), eps=1e-8, weight_decay=0)", max_grad_norm=a.max_grad_norm,
          kl_beta=a.kl_beta, kl_ref=a.kl_ref, kl_estimator="k3 per token: exp(ref-logp)-(ref-logp)-1, ref = " + ("all adapters off (untrained base)" if a.kl_ref == "base" else "frozen warm-start adapter"),
          warm_start=(dict(lr=a.warm_lr, epochs=a.warm_epochs, bs=a.warm_bs, used_epochs=warm_info["used_epochs"],
                           redone=warm_info["redone"], n_diversity=a.n_diversity, optimizer="AdamW(0.9,0.999,1e-8,wd 0), constant lr",
                           max_grad_norm=1.0, loss="mean CE over completion tokens of the top-10 FineWeb contexts")
                      if a.warm_start else None),
          adapters=(["warm (frozen)", "policy (trained)"] if a.warm_start else ["default (trained)"]), lora_r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, lora_target_modules=LORA_TARGET_RE,
          trainable_token_indices=bool(el.new_ids), trainable_params=el.n_trainable, adapter_dtype="float32",
          new_token_init="mean of the real-vocabulary embedding rows" if el.new_ids else None,
          generation=dict(mode="raw (no chat template), thinking off", max_new_tokens=64, temperature=1.0, top_k=0,
                          top_p=1.0, eos_token_ids=el.eos_ids, suppress_tokens=el.new_ids),
          reward="Scorer score of cleaned completion (position 0 included) / FineWeb max (this GPU); empty -> -1.0",
          empty_reward=EMPTY_REWARD, advantage=f"(r - mean_g) / (std_g + {ADV_EPS}), std unbiased; 0 if std_g < {FLAT_STD}",
          loss=("-(1/N_step) sum_i A_i * mean_t logp(o_i,t) (GRPO, beta 0, one update per batch: ratio 1, no clipping)"
                if a.kl_beta == 0 else f"-(1/N_step) sum_i mean_t [A_i logp(o_i,t) - {a.kl_beta} k3_i,t] (GRPO, one update "
                "per batch: ratio 1, no clipping; all completions trained)"),
          early_stop=dict(flat_frac=a.stop_flat_frac, patience=a.stop_patience) if a.stop_flat_frac is not None else None,
          reuse_untrained_from=a.reuse_untrained_from, gate_from=a.gate_from,
          micro_bs=a.micro_bs, gen_batch=a.gen_batch, eval_batch=a.eval_batch, merged_generation=a.merged_gen, gradient_checkpointing=True, seed=a.seed,
          fw_max_this_gpu=dict(zip(neurons, fwmax)), fw_max_a100={n: fw[n]["fw_max_a100"] for n in neurons},
          eval=dict(n_untrained_best=a.n_untrained_best, n_matched=n_matched, n_fresh=a.n_fresh), dry=a.dry,
          check_every=a.check_every, base_backup_bytes=el.backup_bytes,
          reference_contexts=[dict(neuron=r["neuron"], expected=r["expected"]) for r in REFS],
          kernels=el.kernels, model_revision=el.revision)
json.dump(dict(**hp, env=run_info()), open(f"{out_dir}/config.json", "w"), indent=2)
log(f"phase {a.phase}: neurons {neurons}, FineWeb max this GPU {fwmax}; {n_step} completions/step; "
    f"{el.n_trainable:,} trainable params; micro_bs {a.micro_bs}, gen_batch {a.gen_batch}")

check_bitwise("start"); check_reference("start"); check_warm_frozen("start")

params = [p for _, p in el.trainable]
opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: max(0.0, 1.0 - k / a.steps))
pid_flat = [pids[j] for j in range(len(neurons)) for _ in range(per_neuron)]
pair_flat = [pairs_n[j] for j in range(len(neurons)) for _ in range(per_neuron)]
metrics_f = open(f"{out_dir}/metrics.jsonl", "a"); samples_f = open(f"{out_dir}/samples.jsonl", "a")
t_phase = time.time(); step_times, gen_rate = [], []
flat_streak = 0
early_stop = dict(stopped=False, stop_step=None, planned_steps=a.steps, flat_frac=a.stop_flat_frac, patience=a.stop_patience)

for step in range(a.steps):
    t0 = time.time(); torch.cuda.reset_peak_memory_stats(); n_merge0 = len(el.merge_log)
    comps = el.generate(pids, per_neuron, batch=a.gen_batch, merged=a.merged_gen)
    mrec = el.merge_log[n_merge0:]
    t_merge = sum(r["merge_s"] or 0 for r in mrec); t_unmerge = sum(r["unmerge_s"] or 0 for r in mrec)
    t_restore = sum(r["restore_s"] or 0 for r in mrec)
    flat = [c for per in comps for c in per]; t_gen = time.time() - t0
    texts = el.texts(flat)
    t1 = time.time(); scores = el.score(texts, pair_flat).tolist(); t_score = time.time() - t1
    fm = [fwmax[i // per_neuron] for i in range(n_step)]
    rewards = [to_reward(s, f) for s, f in zip(scores, fm)]
    adv, flat_g = group_advantages(rewards, G)
    t2 = time.time(); opt.zero_grad(set_to_none=True)
    n_tr, mlp_, pgs = el.pg_backward(pid_flat, flat, adv, n_step, a.micro_bs, kl_beta=a.kl_beta)
    gn = float(torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)) if n_tr else 0.0
    lr = sched.get_last_lr()[0]
    if n_tr:
        opt.step()
    sched.step(); t_train = time.time() - t2
    dt = time.time() - t0; step_times.append(dt); gen_rate.append(t_gen / n_step)
    m = dict(step=step, completions_generated=n_step, completions_trained=n_tr, lr=lr, grad_norm=gn, mean_logp=mlp_,
             mean_kl=pgs["mean_kl"], loss=pgs["loss"], kl_beta=a.kl_beta,
             mean_reward=float(np.mean(rewards)), frac_empty=float(np.mean([math.isnan(s) for s in scores])),
             frac_flat_groups=float(np.mean(flat_g)), mean_completion_tokens=float(np.mean([len(c) for c in flat])),
             time_s=dt, gen_s=t_gen, score_s=t_score, train_s=t_train,
             merge_calls=len(mrec), merge_s=t_merge, unmerge_s=t_unmerge, restore_s=t_restore, gen_cap=dict(el.gen_cap),
             peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9, per_neuron={})
    for j, n in enumerate(neurons):
        s = [v for v in scores[j * per_neuron:(j + 1) * per_neuron] if not math.isnan(v)]
        r = rewards[j * per_neuron:(j + 1) * per_neuron]
        m["per_neuron"][n] = dict(mean_score=float(np.mean(s)) if s else None, max_score=max(s) if s else None,
                                  mean_reward=float(np.mean(r)), frac_flat=float(np.mean(flat_g[j * gpn:(j + 1) * gpn])))
    metrics_f.write(json.dumps(m) + "\n"); metrics_f.flush()
    pn = " ".join(f"{n}: mean {d['mean_score'] if d['mean_score'] is None else round(d['mean_score'], 3)} "
                  f"max {d['max_score']}" for n, d in m["per_neuron"].items())
    log(f"step {step + 1}/{a.steps} {dt:.1f}s (gen {t_gen:.1f} score {t_score:.1f} train {t_train:.1f}) trained {n_tr}/{n_step} "
        f"reward {m['mean_reward']:.3f} empty {m['frac_empty']:.3f} flat {m['frac_flat_groups']:.3f} "
        f"kl {pgs['mean_kl'] if pgs['mean_kl'] is None else round(pgs['mean_kl'], 5)} "
        f"gn {gn:.3f} lr {lr:.2e} mem {m['peak_mem_gb']:.1f}GB"
        + (f" merge {t_merge:.2f}s unmerge {t_unmerge:.2f}s restore {t_restore:.2f}s ({len(mrec)}x)" if mrec else "")
        + f" | {pn}")
    if a.stop_flat_frac is not None:
        flat_streak = flat_streak + 1 if m["frac_flat_groups"] > a.stop_flat_frac else 0
        if flat_streak >= a.stop_patience and step < a.steps - 1:
            early_stop = dict(stopped=True, stop_step=step + 1, planned_steps=a.steps, flat_frac=a.stop_flat_frac,
                              patience=a.stop_patience, streak_start_step=step + 2 - flat_streak)
            log(f"EARLY STOP after step {step + 1}/{a.steps}: flat-group fraction > {a.stop_flat_frac} for "
                f"{flat_streak} consecutive steps (steps {step + 2 - flat_streak}-{step + 1})")
    last = step == a.steps - 1 or early_stop["stopped"]
    if (step + 1) % a.check_every == 0 or last:
        check_bitwise(f"after step {step + 1}" + (" (early stop)" if early_stop["stopped"] else ""))
        check_warm_frozen(f"after step {step + 1}")
    if step % a.sample_every == 0 or last:
        for j, n in enumerate(neurons):
            for i in range(4):
                k = j * per_neuron + i
                samples_f.write(json.dumps(dict(step=step, neuron=n, text=texts[k], score=scores[k], reward=rewards[k])) + "\n")
        samples_f.flush()
    if step + 1 == min(10, a.steps):
        st = float(np.mean(step_times[1:] if len(step_times) > 1 else step_times))
        g = float(np.mean(gen_rate))
        n_eval = len(neurons) * (a.n_fresh * (2 if a.warm_start else 1) + a.n_untrained_best + n_matched)
        eval_s = n_eval * g * 1.15        # + scoring overhead
        train_total = sum(step_times) + st * (a.steps - step - 1)
        proj = dict(after_steps=step + 1, mean_step_s=st, projected_training_s=train_total,
                    projected_eval_s=eval_s, projected_phase_s=train_total + eval_s + 120)
        json.dump(proj, open(f"{out_dir}/projection.json", "w"), indent=1)
        log(f"PROJECTED phase {a.phase}: training {train_total / 3600:.2f} h ({st:.1f} s/step x {a.steps}), "
            f"evaluations ~{eval_s / 3600:.2f} h ({n_eval:,} samples at {g * 1000:.0f} ms/sample), "
            f"total ~{(train_total + eval_s + 120) / 3600:.2f} h; ends ~{time.strftime('%F %T', time.localtime(t_phase + train_total + eval_s + 120))}")
    if early_stop["stopped"]:
        break
json.dump(early_stop, open(f"{out_dir}/early_stop.json", "w"), indent=1)

t_train_total = time.time() - t_phase
el.model.save_pretrained(f"{out_dir}/adapter")
for f in ("README.md",):
    if os.path.exists(f"{out_dir}/adapter/{f}"):
        os.remove(f"{out_dir}/adapter/{f}")
log(f"training done in {t_train_total / 3600:.2f} h; adapter saved to {out_dir}/adapter")
check_reference("after training")

# ---------------- evaluations ----------------
res = dict(phase=a.phase, neurons={}, training_s=t_train_total, eval_s={}, integrity=integrity, early_stop=early_stop,
           steps_done=len(step_times), kl_beta=a.kl_beta, kl_ref=a.kl_ref, seed=a.seed, warm_start=warm_info)
for n, (L, N), f in zip(neurons, pairs_n, fwmax):
    res["neurons"][n] = dict(layer=L, neuron=N, prompt=P["prompts"][neurons.index(n)], fw_max_a100=fw[n]["fw_max_a100"],
                             fw_max_this_gpu=f)
def save():
    json.dump(res, open(f"{out_dir}/results.json", "w"), indent=1)

def sample_eval(n_per, untrained, chunk=None):
    """-> per neuron: (scores np.array, texts list) for n_per samples each (own neuron)."""
    chunk = chunk or n_per
    sc_all, tx_all = [[] for _ in neurons], [[] for _ in neurons]
    done = 0
    while done < n_per:
        k = min(chunk, n_per - done)
        comps = el.generate(pids, k, batch=a.eval_batch if untrained else a.gen_batch, untrained=untrained,
                            merged=a.merged_gen and not untrained)
        for j in range(len(neurons)):
            t = el.texts(comps[j]); s = el.score(t, [pairs_n[j]] * len(t)).tolist()
            sc_all[j] += s; tx_all[j] += t
        done += k
        if n_per > 1000:
            log(f"  {done:,}/{n_per:,} samples per neuron")
    return [np.array(s, dtype=np.float64) for s in sc_all], tx_all

def best(scores, texts, top=3):
    s = np.where(np.isnan(scores), -np.inf, scores); order = np.argsort(-s, kind="stable")[:top]
    return dict(best=float(s[order[0]]) if np.isfinite(s[order[0]]) else None, n=len(scores),
                n_empty=int(np.isnan(scores).sum()), mean=float(np.nanmean(scores)) if (~np.isnan(scores)).any() else None,
                top=[dict(text=texts[i], score=float(scores[i])) for i in order if np.isfinite(s[i])])

def add_copy_check(d, n):
    """Copy check on every text of an eval dict (d["all"]) against the neuron's top-10 FineWeb contexts."""
    if n not in TOP10:
        from hermes.paths import BASELINE_DIR
        from hermes.baseline import Baseline
        _b = Baseline(BASELINE_DIR); TOP10[n] = [_b.context_tokens(c["ctx"]) for c in fw[n]["top10"]]; del _b
    toks = text_tokens([x["text"] for x in d["all"]])
    for x, t in zip(d["all"], toks):
        x["copy"] = copy_stats(t, TOP10[n])
    for x in d["top"]:
        x["copy"] = copy_stats(text_tokens([x["text"]])[0], TOP10[n])
    cs = [x["copy"] for x in d["all"]]
    d["copy_check"] = dict(max_jaccard=max(c["max_jaccard"] for c in cs), max_longest_span=max(c["longest_span"] for c in cs),
                           best_text=d["top"][0]["copy"] if d["top"] else None)

torch.manual_seed(a.seed + 1)
t0 = time.time()
fresh_s, fresh_t = sample_eval(a.n_fresh, untrained=False)
for j, (n, (L, N)) in enumerate(zip(neurons, pairs_n)):
    d = best(fresh_s[j], fresh_t[j], top=3)
    d["all"] = [dict(text=t, score=float(s)) for t, s in zip(fresh_t[j], fresh_s[j])]
    d["per_token_best"] = el.token_acts(d["top"][0]["text"], L, N) if d["top"] else None
    add_copy_check(d, n)
    res["neurons"][n]["d_trained_best_of_%d" % a.n_fresh] = d
res["eval_s"]["d"] = time.time() - t0; save()

if a.warm_start:   # (w) best of n_fresh fresh samples from the frozen warm adapter, scored exactly like (d)
    torch.manual_seed(a.seed + 4); t0 = time.time()
    with el.use_adapter("warm"):
        w_s, w_t = sample_eval(a.n_fresh, untrained=False)
    for j, n in enumerate(neurons):
        d = best(w_s[j], w_t[j], top=3)
        d["all"] = [dict(text=t, score=float(s)) for t, s in zip(w_t[j], w_s[j])]
        add_copy_check(d, n)
        res["neurons"][n]["w_warm_best_of_%d" % a.n_fresh] = d
    res["eval_s"]["w"] = time.time() - t0; save()
    log("(w) warm-start best of", a.n_fresh, {n: res["neurons"][n][f"w_warm_best_of_{a.n_fresh}"]["best"] for n in neurons})
log("(d) trained best of", a.n_fresh, {n: res["neurons"][n][f"d_trained_best_of_{a.n_fresh}"]["best"] for n in neurons})

if len(neurons) > 1:   # (e) cross-neuron grid
    grid_mean, grid_max = [], []
    for j in range(len(neurons)):
        rm, rx = [], []
        for c, (n_c, f_c) in enumerate(zip(neurons, fwmax)):
            s = el.score(fresh_t[j], [pairs_n[c]] * len(fresh_t[j])).numpy().astype(np.float64)
            r = np.where(np.isnan(s), np.nan, s / f_c)
            rm.append(float(np.nanmean(r)) if (~np.isnan(r)).any() else None)
            rx.append(float(np.nanmax(r)) if (~np.isnan(r)).any() else None)
        grid_mean.append(rm); grid_max.append(rx)
    def diag(g):
        col = [max(range(len(g)), key=lambda r: -1e9 if g[r][c] is None else g[r][c]) == c for c in range(len(g))]
        row = [max(range(len(g)), key=lambda c: -1e9 if g[r][c] is None else g[r][c]) == r for r in range(len(g))]
        return dict(column_max_on_diagonal=col, row_max_on_diagonal=row, diagonal=all(col) and all(row))
    res["grid"] = dict(rows="samples of neuron (prompt)", cols="scored on neuron, ratio to its FineWeb max (this GPU)",
                       neurons=neurons, mean_ratio=grid_mean, max_ratio=grid_max,
                       diag_by_mean=diag(grid_mean), diag_by_max=diag(grid_max))
    save(); log("(e) grid mean ratio", grid_mean, "diagonal by mean:", res["grid"]["diag_by_mean"]["diagonal"])

def reuse_check(src):
    """-> (dict(neuron -> source phase dir), None) if (a), (b), (c) can be reused for every neuron, else (None, reason)."""
    my_gpu = torch.cuda.get_device_name(0)
    ka, kb = f"a_untrained_best_of_{a.n_untrained_best}", f"b_untrained_best_of_{n_matched}"
    found, why = {}, []
    for d in sorted(glob.glob(f"{src}/phase*/")):
        try:
            r = json.load(open(f"{d}results.json")); c = json.load(open(f"{d}config.json"))
        except Exception as e:
            why.append(f"{d}: unreadable ({e})"); continue
        g = c.get("env", {}).get("gpu")
        if g != my_gpu:
            why.append(f"{d}: GPU {g!r} != this GPU {my_gpu!r}"); continue
        if c.get("model_revision") != el.revision:
            why.append(f"{d}: model revision {c.get('model_revision')} != {el.revision}"); continue
        if list(c.get("new_tokens", [])) != list(P["tokens"]):
            why.append(f"{d}: new tokens {c.get('new_tokens')} != {P['tokens']}"); continue
        for n, pr, f_ in zip(neurons, P["prompts"], fwmax):
            x = r.get("neurons", {}).get(n)
            if x is None or n in found:
                continue
            miss = [k for k in (ka, kb, "c_repeated_token") if k not in x]
            if x.get("prompt") != pr:
                why.append(f"{d}: {n} prompt {x.get('prompt')!r} != {pr!r}")
            elif miss:
                why.append(f"{d}: {n} has no {miss} (sizes must match: a {a.n_untrained_best}, b {n_matched})")
            elif x.get("fw_max_this_gpu") != f_:
                why.append(f"{d}: {n} FineWeb max this GPU {x.get('fw_max_this_gpu')} != {f_}")
            elif not os.path.exists(f"{d}b_untrained_scores_{n}.npy"):
                why.append(f"{d}: {n} b_untrained_scores npy missing")
            else:
                found[n] = d
    if all(n in found for n in neurons):
        return found, None
    return None, f"no compatible source for {[n for n in neurons if n not in found]}: " + "; ".join(why or ["no phase dirs"])

reuse_src, reuse_why = (reuse_check(a.reuse_untrained_from) if a.reuse_untrained_from else (None, "not requested"))
res["reused_untrained"] = dict(requested=a.reuse_untrained_from, reused=reuse_src is not None, gpu=torch.cuda.get_device_name(0),
                               sources=reuse_src, reason_not_reused=reuse_why)
if reuse_src:
    for n in neurons:
        src = json.load(open(f"{reuse_src[n]}results.json"))["neurons"][n]
        for k in (f"a_untrained_best_of_{a.n_untrained_best}", f"b_untrained_best_of_{n_matched}", "c_repeated_token"):
            res["neurons"][n][k] = dict(src[k], reused_from=reuse_src[n])
        shutil.copy2(f"{reuse_src[n]}b_untrained_scores_{n}.npy", f"{out_dir}/b_untrained_scores_{n}.npy")
    res["eval_s"].update(a=0.0, b=0.0)
    log(f"REUSED untrained (a), (b), (c) from {reuse_src} (same GPU {torch.cuda.get_device_name(0)!r}, prompts, sizes)")
    log("(a)", {n: res["neurons"][n][f"a_untrained_best_of_{a.n_untrained_best}"]["best"] for n in neurons},
        "(b)", {n: res["neurons"][n][f"b_untrained_best_of_{n_matched}"]["best"] for n in neurons},
        "(c)", {n: res["neurons"][n]["c_repeated_token"]["score"] for n in neurons})
    res["phase_s"] = time.time() - t_phase
    check_bitwise("end"); check_reference("end"); check_warm_frozen("end")
    save(); log(f"phase {a.phase} done in {res['phase_s'] / 3600:.2f} h; results {out_dir}/results.json")
    sys.exit(0)
if a.reuse_untrained_from:
    log(f"NOT reusing untrained baselines from {a.reuse_untrained_from}: {reuse_why}; recomputing (a), (b), (c)")

torch.manual_seed(a.seed + 2); t0 = time.time()
un_s, un_t = sample_eval(a.n_untrained_best, untrained=True)
for j, n in enumerate(neurons):
    res["neurons"][n][f"a_untrained_best_of_{a.n_untrained_best}"] = best(un_s[j], un_t[j], top=3)
res["eval_s"]["a"] = time.time() - t0; save()
log("(a) untrained best of", a.n_untrained_best, {n: res["neurons"][n][f"a_untrained_best_of_{a.n_untrained_best}"]["best"] for n in neurons})

for j, (n, (L, N)) in enumerate(zip(neurons, pairs_n)):   # (c) repeated FineWeb argmax token
    tid = fw[n]["argmax_token_id"]
    ids = torch.full((1, CONTEXT_LEN), tid, dtype=torch.long)
    el.model.eval()
    with el.adapter_off():
        s, p = el.scorer.score_pairs(ids, torch.tensor([CONTEXT_LEN]), [(L, N)])
    res["neurons"][n]["c_repeated_token"] = dict(token=fw[n]["argmax_token"], token_id=tid, score=float(s[0]), pos=int(p[0]))
save(); log("(c) repeated token", {n: res["neurons"][n]["c_repeated_token"] for n in neurons})

torch.manual_seed(a.seed + 3); t0 = time.time()
log(f"(b) compute-matched: {n_matched:,} untrained samples per neuron")
bm_s, bm_t = sample_eval(n_matched, untrained=True, chunk=max(1, 2048 // len(neurons)))
for j, n in enumerate(neurons):
    d = best(bm_s[j], bm_t[j], top=5)
    s = bm_s[j]; f = fwmax[j]
    d.update(n_ge_fw=int(np.sum(s > f)), n_ge_1_05=int(np.sum(s >= 1.05 * f)),
             quantiles={q: float(np.nanquantile(s, q)) for q in (0.5, 0.9, 0.99, 0.999)})
    np.save(f"{out_dir}/b_untrained_scores_{n}.npy", s)
    res["neurons"][n][f"b_untrained_best_of_{n_matched}"] = d
res["eval_s"]["b"] = time.time() - t0
res["phase_s"] = time.time() - t_phase
check_bitwise("end"); check_reference("end"); check_warm_frozen("end")
ml = [r for r in el.merge_log if r["restore_s"] is not None]
res["merge_timing"] = dict(calls=len(el.merge_log), mean_merge_s=float(np.mean([r["merge_s"] or 0 for r in ml])) if ml else None,
                           mean_unmerge_s=float(np.mean([r["unmerge_s"] for r in ml])) if ml else None,
                           mean_restore_s=float(np.mean([r["restore_s"] for r in ml])) if ml else None,
                           failed_calls=sum(not r["ok"] for r in el.merge_log), gen_cap=dict(el.gen_cap))
save(); log("(b) compute-matched best", {n: res["neurons"][n][f"b_untrained_best_of_{n_matched}"]["best"] for n in neurons})
log(f"phase {a.phase} done in {res['phase_s'] / 3600:.2f} h; results {out_dir}/results.json")
