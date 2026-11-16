"""Checks for hermes/grpo.py before a long run (GPU). Phase-B-style elicitor (4 new neuron tokens).
  1. trainable params: LoRA only on language-model attention + MLP projections, trainable token rows, all fp32
  2. scoring through the PEFT model (LoRA off) == stored rescore of L14N8949's top FineWeb context
  3. after a real optimizer step: LoRA-off scores unchanged (bitwise), disable_adapter restores the original
     neuron-token embeddings, LoRA-on embeddings of the neuron tokens changed, other rows unchanged
  4. generation: neuron tokens never generated (suppressed), EOS cut, timing for batch 256 x 64 tokens
  5. pg_backward peak memory / time per micro-batch size
Usage: python scripts/test_grpo.py [--micro_bs 16 32]
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from hermes.paths import RESULTS_DIR, BASELINE_DIR
import torch
from hermes.runinfo import run_info
from hermes.grpo import Elicitor, group_advantages
from hermes.baseline import Baseline

ap = argparse.ArgumentParser(); ap.add_argument("--micro_bs", type=int, nargs="+", default=[16, 32])
ap.add_argument("--gen_batch", type=int, default=256); a = ap.parse_args()
RUN_DIR = f"{RESULTS_DIR}/grpo_tests/{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(RUN_DIR, exist_ok=False)
checks, info = {}, {}

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
checks["peft_scorer_matches_quickcheck_rescore_3.046875"] = s0 == 3.046875

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
mem = {}
for mb in a.micro_bs:
    torch.cuda.reset_peak_memory_stats(); opt.zero_grad(set_to_none=True)
    t0 = time.time(); n_tr, lp = el.pg_backward(pid_per, flat_c, adv, 256, mb); torch.cuda.synchronize()
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

print("\nchecks:"); [print(f"  {'PASS' if v else 'FAIL'}  {k}") for k, v in checks.items()]
json.dump(dict(checks=checks, info=info, env=run_info()), open(f"{RUN_DIR}/results.json", "w"), indent=1)
print("saved:", RUN_DIR)
sys.exit(0 if all(checks.values()) else 1)
