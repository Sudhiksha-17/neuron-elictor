"""Checks for hermes/grpo.py before a long run (GPU). Phase-B-style elicitor (4 new neuron tokens).
  1. trainable params: LoRA only on language-model attention + MLP projections, trainable token rows, all fp32
  2. scoring through the PEFT model (LoRA off) == stored rescore of L14N8949's top FineWeb context
  3. after a real optimizer step: LoRA-off scores unchanged (bitwise), disable_adapter restores the original
     neuron-token embeddings, LoRA-on embeddings of the neuron tokens changed, other rows unchanged
  4. generation: neuron tokens never generated (suppressed), EOS cut, timing for batch 256 x 64 tokens
  5. pg_backward peak memory / time per micro-batch size
  6. merge path (added 2026-11-16, UNTESTED, run on GPU), after a real optimizer step (non-trivial merge):
     base weights bitwise equal to the init backup, LoRA-on logits bitwise unchanged, LoRA-off score unchanged
     and neuron-token rows restored after (i) a merged forward, (ii) an exception inside merged(), (iii) an
     exception in merge_adapter itself, (iv) merged generation; merged vs unmerged LoRA logits (top-1
     agreement >= 95%; delta cosine reported); merge/unmerge/restore timing; simulated OOM -> smaller batch
     remembered across calls; reference_scores() exact.
  7. KL penalty (added 2026-11-18): k3 KL(policy || LoRA-off reference) is 0 at init (LoRA off == reference) and grows
     after an optimizer step; kl_beta=0 reproduces the old (run 1) pg_backward loss and gradients exactly (old code
     copied below; compared bitwise, or within run-to-run noise if two old runs already differ); kl_beta>0 changes the
     gradient after a step; memory/time of pg_backward with kl_beta 0.04.
  8. warm start (added 2026-11-18): defaults (no warm start, kl_ref base, kl_beta 0) = one "default" adapter and the
     old loss exactly (7b); copy-check helpers; on a single-neuron L10N4358 elicitor: warm-start CE decreases over its
     epochs, scorer (all adapters off) bitwise identical before/after warm start and after RL steps, policy = exact
     copy of warm, kl_ref=warm KL is 0 at RL step 0 and > 0 after one optimizer step, warm adapter bitwise unchanged
     after 3 RL steps, base weights bitwise, reference-text score exact; a 3-step grpo_phase.py --dry warm-start solo
     run end to end (WKL-L10 config: (w), diversity pre-check, copy check, integrity); the 1-epoch redo path
     (--test_force_redo).
Usage: python scripts/test_grpo.py [--micro_bs 16 32]
"""
import os, sys, json, time, argparse, subprocess, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from hermes.paths import RESULTS_DIR, BASELINE_DIR
import torch
from hermes.runinfo import run_info
from hermes.grpo import Elicitor, group_advantages
from hermes.baseline import Baseline

ap = argparse.ArgumentParser(); ap.add_argument("--micro_bs", type=int, nargs="+", default=[16, 32])
ap.add_argument("--gen_batch", type=int, default=192); a = ap.parse_args()
RUN_DIR = f"{RESULTS_DIR}/grpo_tests/{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(RUN_DIR, exist_ok=False)
checks, info = {}, {}
# Expected score of L14N8949's top FineWeb context on THIS GPU, from a plain (non-PEFT) Scorer in a separate process
# (scripts/grpo_fwmax.py). Replaces the constant 3.046875 measured on an earlier 5090 pod (changed 2026-11-18: this pod
# scores 3.03125, one bf16 step lower, equal to run 1's FineWeb max this GPU).
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "grpo_fwmax.py"), f"{RUN_DIR}/fwmax"],
               check=True, stdout=subprocess.DEVNULL)
EXPECTED = json.load(open(f"{RUN_DIR}/fwmax/fineweb_max.json"))["L14N8949"]["top10"][0]["this_gpu"]
info["expected_L14N8949_top_ctx_this_gpu"] = EXPECTED
print("expected L14N8949 top ctx score (plain Scorer, this GPU):", EXPECTED)

el = Elicitor(new_tokens=[f"<|n{k}|>" for k in range(4)])
names = [n for n, _ in el.trainable]
info["n_trainable"] = el.n_trainable; info["trainable_examples"] = names[:3] + names[-3:]
print("trainable params:", el.n_trainable, "tensors:", len(names)); print("\n".join(names[:4] + ["..."] + names[-4:]))
checks["no_vision_tower_lora"] = not any("visual" in n for n in names)
checks["lora_on_attn_linear_attn_mlp"] = all(any(k in n for n in names) for k in
    ("self_attn.q_proj", "linear_attn.in_proj_qkv", "linear_attn.out_proj", "mlp.down_proj"))
tt = [n for n in names if "token" in n]
print("trainable-token params:", tt); info["trainable_token_params"] = tt
checks["trainable_token_params_present"] = len(tt) > 0
checks["all_trainable_fp32"] = all(p.dtype == torch.float32 for _, p in el.trainable)

base = Baseline(BASELINE_DIR, el.tok)
L, N = 14, 8949; nb = base.neuron(L, N)
ctx_text = base.context_text(nb["top_ctx"][0])
ids = torch.tensor([base.context_tokens(nb["top_ctx"][0])]); lens = torch.tensor([64])
def score_ctx():
    el.model.eval()
    with el.adapter_off():
        return float(el.scorer.score_pairs(ids, lens, [(L, N)])[0][0])
s0 = score_ctx(); print("L14N8949 top ctx, LoRA off:", s0)
checks["peft_scorer_matches_plain_scorer_this_gpu"] = s0 == EXPECTED

emb = el.model.get_input_embeddings()
def emb_rows(off):
    with (el.adapter_off() if off else torch.no_grad()):
        with torch.no_grad():
            return emb(torch.tensor(el.new_ids + [100, 1000], device=el.device)).float().cpu()
e_on0, e_off0 = emb_rows(False), emb_rows(True)
checks["init_on_equals_off"] = torch.equal(e_on0, e_off0)

# ---- real step on random-ish advantages ----
pids = [el.prompt_ids(f"<|n{k}|>\n") for k in range(4)]
t0 = time.time(); comps = el.generate(pids, 64, batch=a.gen_batch); torch.cuda.synchronize(); tg = time.time() - t0
flat_c = [c for per in comps for c in per]
print(f"generate 4 x 64 = {len(flat_c)} completions: {tg:.1f}s"); info["gen_256_s"] = tg
gen_tokens = {t for c in flat_c for t in c}
checks["neuron_tokens_never_generated"] = not (gen_tokens & set(el.new_ids))
checks["eos_cut"] = all(all(t not in el.eos_ids for t in c[:-1]) for c in flat_c)
info["mean_comp_len"] = sum(map(len, flat_c)) / len(flat_c)
info["frac_ending_eos"] = sum(c[-1] in el.eos_ids for c in flat_c) / len(flat_c)
texts = el.texts(flat_c)
print("examples:", [t[:80] for t in texts[:3]])
pairs = [(8, 2027)] * 64 + [(9, 5711)] * 64 + [(10, 4358)] * 64 + [(11, 3081)] * 64
t0 = time.time(); sc = el.score(texts, pairs); ts = time.time() - t0; info["score_256_s"] = ts
rew = [-1.0 if torch.isnan(v) else float(v) for v in sc]
adv, flat = group_advantages(rew, 16)
pid_per = [pids[i // 64] for i in range(256)]
opt = torch.optim.AdamW([p for _, p in el.trainable], lr=2e-5, weight_decay=0.0)
# ---- 7a. KL at init (before any optimizer step: LoRA B = 0, token deltas = original rows -> policy == reference) ----
kl0 = el.kl_stats(pid_per, flat_c, 32); info["kl_init"] = kl0
opt.zero_grad(set_to_none=True)
_, _, st0 = el.pg_backward(pid_per, flat_c, adv, 256, 32, kl_beta=0.04); info["kl_init_from_pg_backward_beta_0.04"] = st0
opt.zero_grad(set_to_none=True)
print(f"KL at init: kl_stats {kl0!r}, pg_backward(beta 0.04) {st0}")
checks["kl_zero_at_init"] = abs(kl0) <= 1e-6 and abs(st0["mean_kl"]) <= 1e-6
mem = {}
for mb in a.micro_bs:
    torch.cuda.reset_peak_memory_stats(); opt.zero_grad(set_to_none=True)
    t0 = time.time(); n_tr, lp, _ = el.pg_backward(pid_per, flat_c, adv, 256, mb); torch.cuda.synchronize()
    mem[mb] = dict(peak_gb=torch.cuda.max_memory_allocated() / 1e9, s=time.time() - t0, n_trained=n_tr, mean_logp=lp)
    print(f"micro_bs {mb}: peak {mem[mb]['peak_gb']:.1f} GB, backward {mem[mb]['s']:.1f}s, trained {n_tr}, mean logp {lp:.3f}")
info["pg_backward"] = mem
gn = torch.nn.utils.clip_grad_norm_([p for _, p in el.trainable], 1.0); info["grad_norm"] = float(gn)
tt_grads = [float(p.grad.norm()) if p.grad is not None else None for n, p in el.trainable if "token" in n]
print("grad norm", float(gn), "trainable-token grad norms", tt_grads)
checks["trainable_token_grad_nonzero"] = all(g is not None and g > 0 for g in tt_grads)
opt.step()
s1 = score_ctx()
checks["lora_off_score_unchanged_after_step"] = s1 == s0
e_on1, e_off1 = emb_rows(False), emb_rows(True)
checks["disable_adapter_restores_neuron_token_rows"] = torch.equal(e_off1, e_off0)
checks["neuron_token_rows_changed_with_adapter"] = not torch.equal(e_on1[:4], e_on0[:4])
checks["other_rows_unchanged_with_adapter"] = torch.equal(e_on1[4:], e_on0[4:])
el.model.eval()
with torch.no_grad():
    x = torch.tensor([pids[0] + flat_c[0]], device=el.device)
    lo_on = el.model(input_ids=x).logits.float()
    with el.adapter_off():
        lo_off = el.model(input_ids=x).logits.float()
checks["lora_on_changes_policy_logits"] = not torch.equal(lo_on, lo_off)

# ---- 7b. KL after the optimizer step; kl_beta=0 == old loss ----
kl1 = el.kl_stats(pid_per, flat_c, 32); info["kl_after_step"] = kl1
print(f"KL after one step: {kl1!r} (init {kl0!r})")
checks["kl_grows_after_step"] = kl1 > kl0 and kl1 > 0

def old_pg_backward(prompt_ids, comps, adv, n_total, micro_bs):
    """hermes/grpo.py pg_backward as of commit 36445a2 (run 1), verbatim apart from the return value (+ loss)."""
    el.model.train()
    idx = [i for i in range(len(comps)) if adv[i] != 0.0]
    lp_sum, lp_n, loss_sum = 0.0, 0, 0.0
    for s in range(0, len(idx), micro_bs):
        b = idx[s:s + micro_bs]
        P = len(prompt_ids[b[0]]); assert all(len(prompt_ids[i]) == P for i in b)
        T = max(len(comps[i]) for i in b)
        x = torch.full((len(b), P + T), el.tok.pad_token_id, dtype=torch.long)
        m = torch.zeros((len(b), T))
        for r, i in enumerate(b):
            x[r, :P] = torch.tensor(prompt_ids[i]); x[r, P:P + len(comps[i])] = torch.tensor(comps[i])
            m[r, :len(comps[i])] = 1
        x, m = x.to(el.device), m.to(el.device)
        a_ = torch.tensor([adv[i] for i in b], device=el.device)
        logits = el.model(input_ids=x, use_cache=False, logits_to_keep=T + 1).logits[:, :T].float()
        if el.new_ids:
            logits[:, :, el.new_ids] = float("-inf")
        logp = torch.log_softmax(logits, -1).gather(-1, x[:, P:P + T, None]).squeeze(-1)
        seq = (logp * m).sum(1) / m.sum(1)
        loss = -(a_ * seq).sum() / n_total
        loss.backward(); loss_sum += float(loss)
        lp_sum += float((logp * m).sum()); lp_n += int(m.sum())
        del logits, logp, loss
    return len(idx), lp_sum / lp_n, loss_sum

tparams = [p for _, p in el.trainable]
def grads_of(fn):
    opt.zero_grad(set_to_none=True); out = fn(); torch.cuda.synchronize()
    g = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in tparams]
    opt.zero_grad(set_to_none=True)
    return out, g
def maxdiff(g1, g2):
    return max(float((u - v).abs().max()) for u, v in zip(g1, g2))
def reldiff(g1, g2):   # ||g1 - g2|| / ||g1|| over all trainable params
    return float(torch.sqrt(sum(((u - v).float() ** 2).sum() for u, v in zip(g1, g2)))
                 / torch.sqrt(sum((u.float() ** 2).sum() for u in g1)))
# The backward is NOT bitwise reproducible run to run on this stack (measured 2026-11-18: two runs of the old code differ
# by ~0.1% relative grad norm). So: forward quantities (loss, trained set, mean logp) must match exactly; grads must be
# bitwise equal if the old code is itself bitwise reproducible, else within 2x the old-vs-old spread (3 old, 2 new runs).
olds = [grads_of(lambda: old_pg_backward(pid_per, flat_c, adv, 256, 32)) for _ in range(3)]
news = [grads_of(lambda: el.pg_backward(pid_per, flat_c, adv, 256, 32, kl_beta=0.0)) for _ in range(2)]
o_kl, g_kl = grads_of(lambda: el.pg_backward(pid_per, flat_c, adv, 256, 32, kl_beta=0.04))
(o_old1, g_old1), (o_new, g_new) = olds[0], news[0]
oo = [reldiff(olds[i][1], olds[j][1]) for i, j in ((0, 1), (0, 2), (1, 2))]
on = [reldiff(go, gn) for _, go in olds for _, gn in news]
old_bitwise = all(all(torch.equal(u, v) for u, v in zip(olds[0][1], g)) for _, g in olds[1:])
new_bitwise = all(all(torch.equal(u, v) for u, v in zip(olds[0][1], g)) for _, g in news)
info["kl0_vs_old"] = dict(old_loss=[o[2] for o, _ in olds], new_loss=[o[2]["loss"] for o, _ in news], old_n=o_old1[0], new_n=o_new[0],
                          old_mean_logp=o_old1[1], new_mean_logp=o_new[1], old_bitwise_reproducible=old_bitwise,
                          old_vs_old_rel=oo, old_vs_new_rel=on, old_vs_new_bitwise=new_bitwise,
                          old_vs_old_maxabs=maxdiff(olds[0][1], olds[1][1]), old_vs_new_maxabs=maxdiff(g_old1, g_new),
                          new_mean_kl_beta0=o_new[2]["mean_kl"], beta_004=dict(n=o_kl[0], **o_kl[2]),
                          beta_004_vs_0_rel=reldiff(g_new, g_kl), beta_004_vs_0_grad_maxdiff=maxdiff(g_kl, g_new))
print("kl_beta=0 vs old pg_backward:", info["kl0_vs_old"])
checks["kl_beta0_same_trained_set_and_logp"] = all(o[0] == o_old1[0] and o[1] == o_old1[1] for o, _ in news)
checks["kl_beta0_loss_equals_old_exactly"] = all(o[2]["loss"] == o_old1[2] for o, _ in news) and len(set(o[2] for o, _ in olds)) == 1
checks["kl_beta0_grads_equal_old"] = new_bitwise if old_bitwise else max(on) <= 2 * max(oo)
checks["kl_beta0_logged_kl_matches_kl_stats"] = abs(o_new[2]["mean_kl"] - kl1) <= 1e-6 + 0.05 * abs(kl1)
checks["kl_beta_positive_changes_grads"] = info["kl0_vs_old"]["beta_004_vs_0_rel"] > 2 * max(oo)
torch.cuda.reset_peak_memory_stats(); t0 = time.time()
opt.zero_grad(set_to_none=True); el.pg_backward(pid_per, flat_c, adv, 256, 32, kl_beta=0.04); torch.cuda.synchronize()
info["pg_backward_kl0.04_mb32"] = dict(peak_gb=torch.cuda.max_memory_allocated() / 1e9, s=time.time() - t0)
opt.zero_grad(set_to_none=True)
print("pg_backward kl_beta 0.04, micro_bs 32:", info["pg_backward_kl0.04_mb32"])

# ---- 6. merge path (added 2026-11-16, UNTESTED, run on GPU) ----
# Runs after the real optimizer step above, so LoRA B != 0 and the neuron-token deltas != 0 (a non-trivial merge).
def logits(merged=False):
    el.model.eval()
    with torch.no_grad():
        if merged:
            with el.merged():
                return el.model(input_ids=x).logits.float()
        return el.model(input_ids=x).logits.float()
def merge_state_ok(tag):
    bw = el.check_base_bitwise()
    checks[f"{tag}_base_weights_bitwise"] = bw["ok"]
    checks[f"{tag}_lora_on_logits_bitwise_unchanged"] = torch.equal(logits(), lo_on)
    checks[f"{tag}_lora_off_score_unchanged"] = score_ctx() == s0
    e_on, e_off = emb_rows(False), emb_rows(True)
    checks[f"{tag}_neuron_token_rows_lora_off_restored"] = torch.equal(e_off, e_off0)
    checks[f"{tag}_neuron_token_rows_lora_on_unchanged"] = torch.equal(e_on, e_on1)
    return bw

checks["bitwise_before_any_merge"] = el.check_base_bitwise()["ok"]
lo_m = logits(merged=True)
merge_state_ok("after_merged_forward")
checks["merged_logits_differ_from_base"] = not torch.equal(lo_m, lo_off)
lp_on, lp_m, lp_off = (torch.log_softmax(t, -1) for t in (lo_on, lo_m, lo_off))
d_lora = (lo_on - lo_off).flatten(); d_merged = (lo_m - lo_off).flatten()
info["merged_vs_lora_on"] = dict(
    max_abs_logprob_diff=float((lp_m - lp_on).abs().max()), mean_abs_logprob_diff=float((lp_m - lp_on).abs().mean()),
    top1_agreement=float((lo_m.argmax(-1) == lo_on.argmax(-1)).float().mean()),
    lora_delta_norm=float(d_lora.norm()), merged_delta_norm=float(d_merged.norm()),
    delta_cosine=float(torch.nn.functional.cosine_similarity(d_lora, d_merged, dim=0)),
    base_vs_lora_on_mean_abs_logprob_diff=float((lp_off - lp_on).abs().mean()))
print("merged vs unmerged LoRA:", info["merged_vs_lora_on"])
checks["merged_top1_agrees_with_lora_on_95pct"] = info["merged_vs_lora_on"]["top1_agreement"] >= 0.95
# Not a pass/fail check: after one step at lr 2e-5 the LoRA delta can be below bf16 rounding of the base weights,
# so the merged policy may not show it yet. A low cosine means generation samples from a policy that lags the trained one.
if info["merged_vs_lora_on"]["delta_cosine"] < 0.9:
    print("WARN: merged model reproduces the LoRA logit change poorly (cosine "
          f"{info['merged_vs_lora_on']['delta_cosine']:.3f}); check again later in training")

# exception inside the merged block -> exact restore
try:
    with el.merged():
        raise RuntimeError("injected failure inside merged()")
except RuntimeError:
    pass
merge_state_ok("after_exception_in_body")

# exception in merge_adapter itself (after it merged) -> exact restore
_orig_merge = el.model.merge_adapter
def _bad_merge(*args, **kw):
    _orig_merge(*args, **kw); raise RuntimeError("injected failure in merge_adapter")
el.model.merge_adapter = _bad_merge
try:
    with el.merged():
        pass
    checks["merge_failure_raised"] = False
except RuntimeError:
    checks["merge_failure_raised"] = True
finally:
    del el.model.merge_adapter
merge_state_ok("after_merge_failure")

# merged generation end to end
n_log0 = len(el.merge_log)
t0 = time.time(); comps_m = el.generate(pids, 16, batch=64, merged=True); info["merged_gen_64_s"] = time.time() - t0
fm = [c for per in comps_m for c in per]
checks["merged_gen_neuron_tokens_never_generated"] = not ({t for c in fm for t in c} & set(el.new_ids))
merge_state_ok("after_merged_generate")
info["merge_timing"] = el.merge_log[n_log0:] + el.merge_log[:n_log0]
print("merge timing (s):", [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in el.merge_log])

# OOM fallback remembers the smaller batch (simulated OOM above 4 sequences)
_orig_gen = el.model.generate; calls = []
def _fake_gen(*args, **kw):
    calls.append(int(kw["input_ids"].shape[0]))
    if calls[-1] > 4:
        raise torch.OutOfMemoryError("simulated OOM")
    return _orig_gen(*args, **kw)
el.model.generate = _fake_gen; el.gen_cap.pop("lora", None)
try:
    out1 = el.generate(pids[:1], 8, batch=16); calls1 = list(calls); calls.clear()
    out2 = el.generate(pids[:1], 8, batch=16); calls2 = list(calls)
finally:
    del el.model.generate
info["oom_sim_calls"] = dict(first=calls1, second=calls2, gen_cap=dict(el.gen_cap))
checks["oom_fallback_first_call"] = calls1 == [8, 4, 4] and len(out1[0]) == 8
checks["oom_fallback_remembered"] = calls2 == [4, 4] and el.gen_cap.get("lora") == 4 and len(out2[0]) == 8

# reference-score helper used by grpo_phase.py
rs = el.reference_scores([dict(neuron=(L, N), tokens=ids[0].tolist(), expected=EXPECTED)])
checks["reference_scores_helper_exact"] = rs["ok"]
info["reference_scores"] = rs

# ---- 8. warm start (added 2026-11-18) ----
# 8a. defaults: no warm start, kl_ref base, kl_beta 0 reproduce the old (run 1) loss exactly (section 7b on this seed)
checks["defaults_single_default_adapter_kl_ref_base"] = list(el.model.peft_config) == ["default"] and el.kl_ref == "base" and not el.warm
checks["defaults_reproduce_old_loss_exactly"] = (checks["kl_beta0_loss_equals_old_exactly"]
                                                 and checks["kl_beta0_same_trained_set_and_logp"] and checks["kl_beta0_grads_equal_old"])
# 8b. copy-check helpers on known inputs
from hermes.copycheck import jaccard, longest_shared_span, copy_stats, diversity
checks["copycheck_helpers"] = (jaccard([1, 2, 3], [3, 2, 1]) == 1.0 and jaccard([1, 2], [3, 4]) == 0.0
                               and abs(jaccard([1, 2, 3], [2, 3, 4]) - 0.5) < 1e-12
                               and longest_shared_span([5, 1, 2, 3, 9], [7, 1, 2, 3, 8]) == 3 and longest_shared_span([1], []) == 0
                               and copy_stats([1, 2, 3], [[9], [1, 2, 3]])["max_jaccard_rank"] == 1
                               and diversity([[1, 2], [1, 2], [3]], ["a", "a", "b"], [[1, 2]])["n_near_copies"] == 2)
# free the Phase-B-style model before loading the warm-start one (24 GB GPU)

del el, opt, olds, news, g_kl, g_old1, g_new, tparams, emb, lo_on, lo_off, lo_m, lp_on, lp_m, lp_off, d_lora, d_merged
del _orig_merge, _bad_merge, _orig_gen, _fake_gen, x, ids
import gc; gc.collect(); torch.cuda.empty_cache()
info["gpu_allocated_gb_before_warm_model"] = torch.cuda.memory_allocated() / 1e9
print("GPU allocated before the warm-start model:", info["gpu_allocated_gb_before_warm_model"], "GB")

fwj = json.load(open(f"{RUN_DIR}/fwmax/fineweb_max.json"))
WN = "L10N4358"; WL, WNN = 10, 4358
w = Elicitor(seed=0, warm_start=True)
checks["warm_model_has_only_warm_adapter"] = list(w.model.peft_config) == ["warm"] and w.model.active_adapter == "warm"
ctxs = [base.context_tokens(c["ctx"]) for c in fwj[WN]["top10"]]
wpid = w.prompt_ids("Text:\n")
probe_texts = [base.context_text(c["ctx"]) for c in fwj[WN]["top10"]] + ["The quick brown fox.", "Hello world, this is a test."]
def probe():
    return w.score(probe_texts, [(WL, WNN)] * len(probe_texts))
s_pre = probe()
init_warm = w.adapter_state("warm")
h = w.warm_start_train(wpid, ctxs, lr=1e-4, epochs=2, bs=1, seed=0)
info["warm_start"] = dict(eval_ce=h["eval_ce"], mean_train_loss=[e["mean_train_loss"] for e in h["epochs"]], seconds=h["seconds"])
print("warm start:", info["warm_start"])
checks["warm_ce_decreases_over_epochs"] = h["eval_ce"][0] > h["eval_ce"][1] > h["eval_ce"][2]
checks["warm_train_loss_epoch2_below_epoch1"] = h["epochs"][1]["mean_train_loss"] < h["epochs"][0]["mean_train_loss"]
checks["warm_adapter_changed_by_training"] = not w.adapter_equal(init_warm)[0]
checks["scorer_bitwise_after_warm_start"] = torch.equal(probe(), s_pre)
checks["base_bitwise_after_warm_start"] = w.check_base_bitwise()["ok"]
w.make_policy(); w.kl_ref = "warm"
WARM = w.adapter_state("warm")
checks["policy_only_trainable"] = (all(".policy." in n for n, _ in w.trainable)
                                   and not any(p.requires_grad for _, p in w._adapter_params("warm")))
xw = torch.tensor([wpid + ctxs[0]], device=w.device)
def lg():
    w.model.eval()
    with torch.no_grad():
        return w.model(input_ids=xw).logits.float()
lg_pol = lg()
with w.use_adapter("warm"):
    lg_warm = lg()
checks["policy_logits_equal_warm_at_copy"] = torch.equal(lg_pol, lg_warm)
checks["use_adapter_restores_requires_grad"] = (all(p.requires_grad for _, p in w.trainable)
                                                and not any(p.requires_grad for _, p in w._adapter_params("warm"))
                                                and w.model.active_adapter == "policy")
wpairs = [(WL, WNN)] * 64; wpid_per = [wpid] * 64
def rl_step(opt_):
    comps = w.generate([wpid], 64, batch=a.gen_batch)[0]
    sc_ = w.score(w.texts(comps), wpairs)
    adv_, _ = group_advantages([-1.0 if torch.isnan(v) else float(v) for v in sc_], 16)
    opt_.zero_grad(set_to_none=True)
    n_, _, st_ = w.pg_backward(wpid_per, comps, adv_, 64, 32, kl_beta=0.01)
    torch.nn.utils.clip_grad_norm_([p for _, p in w.trainable], 1.0); opt_.step(); opt_.zero_grad(set_to_none=True)
    return comps, st_
wopt = torch.optim.AdamW([p for _, p in w.trainable], lr=2e-5, weight_decay=0.0)
comps0 = w.generate([wpid], 64, batch=a.gen_batch)[0]
kl_w0 = w.kl_stats(wpid_per, comps0, 32)
w.kl_ref = "base"; kl_b0 = w.kl_stats(wpid_per, comps0, 32); w.kl_ref = "warm"
info["kl_ref_warm_step0"] = kl_w0; info["kl_ref_base_after_warm_start"] = kl_b0
checks["kl_ref_warm_zero_at_rl_step0"] = kl_w0 == 0.0
checks["kl_ref_base_positive_after_warm_start"] = kl_b0 > 0
_, st1 = rl_step(wopt)
kl_w1 = w.kl_stats(wpid_per, comps0, 32); info["kl_ref_warm_after_1_step"] = kl_w1; info["pg_step1_mean_kl"] = st1["mean_kl"]
print(f"KL vs warm: step 0 {kl_w0!r}, after one optimizer step {kl_w1!r}; KL vs base after warm start {kl_b0!r}")
checks["kl_ref_warm_zero_in_first_pg_backward"] = st1["mean_kl"] == 0.0
checks["kl_ref_warm_positive_after_one_step"] = kl_w1 > 0
for _ in range(2):
    rl_step(wopt)
ok_w, n_w, bad_w = w.adapter_equal(WARM); info["warm_frozen_check"] = dict(ok=ok_w, n=n_w, bad=bad_w)
checks["warm_adapter_unchanged_after_3_rl_steps"] = ok_w
checks["policy_changed_after_rl"] = not torch.equal(lg(), lg_pol)
checks["scorer_bitwise_after_rl_steps"] = torch.equal(probe(), s_pre)
checks["base_bitwise_after_rl_steps"] = w.check_base_bitwise()["ok"]
rs_w = w.reference_scores([dict(neuron=(WL, WNN), tokens=fwj[WN]["ref_tokens"], expected=fwj[WN]["fw_max_this_gpu"])])
checks["reference_scores_exact_after_rl_steps"] = rs_w["ok"]; info["reference_scores_warm"] = rs_w
del w, wopt, lg_pol, lg_warm; gc.collect(); torch.cuda.empty_cache()

# 8c. 3-step dry run of one warm-start solo run end to end (separate process: grpo_phase.py), incl. (w), the diversity
#     pre-check and the copy check. Config = WKL-L10 (exercises kl_ref warm + kl_beta > 0).
DRY = f"{RUN_DIR}/dry_warm"; os.makedirs(DRY)
for f in ("fineweb_max.json", "fwmax_config.json"):
    shutil.copy2(f"{RUN_DIR}/fwmax/{f}", f"{DRY}/{f}")
cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "grpo_phase.py"), "--phase", "S", "--neuron", WN,
       "--warm_start", "--kl_ref", "warm", "--kl_beta", "0.01", "--seed", "0", "--dry", "--steps", "3",
       "--gen_batch", str(a.gen_batch), "--micro_bs", "32", "--run_dir", DRY]
t0 = time.time(); pr = subprocess.run(cmd, stdout=open(f"{DRY}/stdout.log", "w"), stderr=subprocess.STDOUT)
info["dry_run"] = dict(cmd=" ".join(cmd), exit=pr.returncode, seconds=time.time() - t0)
print("dry run:", info["dry_run"])
try:
    rj = json.load(open(f"{DRY}/phaseS/results.json"))["neurons"][WN]
    ws = json.load(open(f"{DRY}/phaseS/warm_start.json")); integ = json.load(open(f"{DRY}/phaseS/integrity.json"))
    met = [json.loads(l) for l in open(f"{DRY}/phaseS/metrics.jsonl")]
    dkey, wkey = "d_trained_best_of_8", "w_warm_best_of_8"
    info["dry_run"].update(d=rj[dkey]["best"], w=rj[wkey]["best"], b=rj.get("b_untrained_best_of_192", {}).get("best"),
                           diversity=ws["attempts"][-1]["diversity"]["n_distinct_texts"], used_epochs=ws["used_epochs"],
                           kl=[m["mean_kl"] for m in met], d_copy=rj[dkey]["copy_check"], w_copy=rj[wkey]["copy_check"])
    checks["dry_run_exit_0"] = pr.returncode == 0
    checks["dry_run_3_steps_logged"] = len(met) == 3
    checks["dry_run_has_w_and_d"] = rj[wkey]["n"] == 8 and rj[dkey]["n"] == 8
    checks["dry_run_diversity_precheck"] = all(k in ws["attempts"][-1]["diversity"] for k in
                                               ("n_distinct_texts", "mean_pairwise_jaccard", "max_jaccard_to_top10_max"))
    checks["dry_run_copy_check_every_text"] = all("copy" in x for k in (dkey, wkey) for x in rj[k]["all"])
    checks["dry_run_kl_zero_first_step"] = met[0]["mean_kl"] == 0.0
    checks["dry_run_integrity_all_pass"] = (all(r["ok"] for r in integ["bitwise"]) and all(r["ok"] for r in integ["reference"])
                                            and len(integ.get("warm_frozen", [])) >= 2 and all(r["ok"] for r in integ["warm_frozen"]))
except Exception as e:
    print("dry run results unreadable:", repr(e)); checks["dry_run_results_readable"] = False

# 8d. redo path: a dry run with --test_force_redo must warm-start twice (2 epochs, then 1) and report both
DRY2 = f"{RUN_DIR}/dry_warm_redo"; os.makedirs(DRY2)
for f in ("fineweb_max.json", "fwmax_config.json"):
    shutil.copy2(f"{RUN_DIR}/fwmax/{f}", f"{DRY2}/{f}")
cmd2 = [c if c != DRY else DRY2 for c in cmd] + ["--test_force_redo", "--steps", "1"]
pr2 = subprocess.run(cmd2, stdout=open(f"{DRY2}/stdout.log", "w"), stderr=subprocess.STDOUT)
try:
    ws2 = json.load(open(f"{DRY2}/phaseS/warm_start.json"))
    info["dry_run_redo"] = dict(exit=pr2.returncode, attempts=[x["n_epochs"] for x in ws2["attempts"]], used=ws2["used_epochs"])
    checks["warm_redo_path_two_attempts_reported"] = (pr2.returncode == 0 and [x["n_epochs"] for x in ws2["attempts"]] == [2, 1]
                                                       and ws2["used_epochs"] == 1 and ws2["redone"]
                                                       and all("diversity" in x for x in ws2["attempts"]))
except Exception as e:
    print("redo dry run unreadable:", repr(e)); checks["warm_redo_path_two_attempts_reported"] = False

print("\nchecks:"); [print(f"  {'PASS' if v else 'FAIL'}  {k}") for k, v in checks.items()]
json.dump(dict(checks=checks, info=info, env=run_info()), open(f"{RUN_DIR}/results.json", "w"), indent=1)
print("saved:", RUN_DIR)
sys.exit(0 if all(checks.values()) else 1)
