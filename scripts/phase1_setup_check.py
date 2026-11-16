"""Phase 1: setup check.
Load Qwen3.5-4B (bf16), find the 32 language-model MLP layers, read neuron
L{layer}N{neuron} (default L15N2341; any layer/neuron via --layer/--neuron) per token on a test sentence via a hook on mlp.down_proj's input,
verify against manual silu(gate_proj(x)) * up_proj(x), and check that an
early-exit forward (stop after layer L) gives identical values, and that hermes.scoring.Scorer
(fixed-shape, multi-layer) gives the same score.
"""
import os, sys, json, time, re, random, platform, argparse
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from hermes.paths import RESULTS_DIR, BASELINE_DIR
import torch.nn.functional as F
from hermes.runinfo import run_info
import transformers
from transformers import AutoTokenizer, AutoModelForImageTextToText

_ap = argparse.ArgumentParser()
_ap.add_argument("--layer", type=int, default=15)
_ap.add_argument("--neuron", type=int, default=2341)
_args = _ap.parse_args()
CFG = dict(
    model_id="Qwen/Qwen3.5-4B", dtype="bfloat16", layer=_args.layer, neuron=_args.neuron, seed=0,
    text="The Eiffel Tower in Paris was completed in 1889 and is one of the most visited monuments in the world.",
)
RUN_DIR = f"{RESULTS_DIR}/phase1/{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(RUN_DIR, exist_ok=False)

class EarlyExit(Exception):
    pass

def main():
    random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
    env = dict(torch=torch.__version__, transformers=transformers.__version__,
               python=platform.python_version(), gpu=torch.cuda.get_device_name(0))
    for pkg in ("fla", "causal_conv1d"):
        try:
            m = __import__(pkg); env[pkg] = getattr(m, "__version__", "installed")
        except Exception as e:
            env[pkg] = f"NOT AVAILABLE ({e})"
    env.update(run_info())
    json.dump(dict(config=CFG, env=env), open(f"{RUN_DIR}/config.json", "w"), indent=2)
    print("run dir:", RUN_DIR); print("config:", CFG); print("env:", env)

    rev = env["model_revision"]
    tok = AutoTokenizer.from_pretrained(CFG["model_id"], revision=rev)
    model = AutoModelForImageTextToText.from_pretrained(
        CFG["model_id"], revision=rev, dtype=torch.bfloat16, device_map="cuda").eval()
    print("model class:", type(model).__name__)

    # --- find language-model MLP layers (exclude vision tower / MTP) ---
    pat = re.compile(r"^(.*language_model\.layers)\.(\d+)\.mlp$")
    lm_mlps = {}
    for name, mod in model.named_modules():
        m = pat.match(name)
        if m:
            lm_mlps[int(m.group(2))] = (name, mod)
    all_mlp_names = [n for n, _ in model.named_modules() if n.endswith(".mlp")]
    print(f"\nall modules ending in '.mlp': {len(all_mlp_names)} "
          f"(language-model: {len(lm_mlps)}, other: {len(all_mlp_names) - len(lm_mlps)})")
    assert sorted(lm_mlps) == list(range(32)), sorted(lm_mlps)
    layers_prefix = pat.match(lm_mlps[0][0]).group(1)
    print("language-model layers path:", layers_prefix)
    print("example:", lm_mlps[0][0], "->", lm_mlps[0][1])
    text_cfg = model.config.text_config
    print("layer types (L0-L31):", "".join("F" if t == "full_attention" else "l" for t in text_cfg.layer_types),
          "(F=full attn, l=linear attn)")

    L, N = CFG["layer"], CFG["neuron"]
    mlp = lm_mlps[L][1]
    decoder_layer = model.get_submodule(f"{layers_prefix}.{L}")
    print(f"\nlayer {L} type: {text_cfg.layer_types[L]}; intermediate_size={mlp.gate_proj.out_features}")

    # --- hooks ---
    cap = {}
    def pre_mlp(mod, args):          # x = input to MLP (post-attention RMSNorm output)
        cap["x"] = args[0].detach()
    def pre_down(mod, args):         # input to down_proj = silu(gate(x)) * up(x)
        cap["act"] = args[0].detach()
    def post_layer(mod, args, out):  # early exit
        if cap.get("early_exit"):
            raise EarlyExit
    hs = [mlp.register_forward_pre_hook(pre_mlp), mlp.down_proj.register_forward_pre_hook(pre_down),
          decoder_layer.register_forward_hook(post_layer)]

    enc = tok(CFG["text"], return_tensors="pt").to("cuda")
    ids = enc.input_ids[0].tolist()
    print(f"\ntest sentence: {CFG['text']!r}  ({len(ids)} tokens)")

    with torch.no_grad():
        # (a) full forward, no early exit
        cap["early_exit"] = False
        t0 = time.time(); model(**enc, use_cache=False); t_full = time.time() - t0
        act_full = cap["act"][0, :, N].float().clone()
        x = cap["x"]; act_all_full = cap["act"].clone()
        # (b) early-exit forward
        cap.clear(); cap["early_exit"] = True
        t0 = time.time()
        try:
            model(**enc, use_cache=False); exited = False
        except EarlyExit:
            exited = True
        t_early = time.time() - t0
        act_early = cap["act"][0, :, N].float()

        # (c) manual calculation from the captured MLP input x
        manual_bf16 = (F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))[0, :, N].float()
        xf = x.float()
        g = xf @ mlp.gate_proj.weight[N].float(); u = xf @ mlp.up_proj.weight[N].float()
        manual_fp32 = (F.silu(g) * u)[0]
    for h in hs: h.remove()

    # (d) hermes.scoring.Scorer on the same model: fixed shape [256, 64], early exit at L and at 31
    from hermes.scoring import Scorer, TOKENIZE_KWARGS
    sc = Scorer(revision=rev, model=model, tok=tok)
    sids, slens = sc.encode([CFG["text"]])
    same_tokens = sids[0, :slens[0]].tolist() == tok(CFG["text"], **TOKENIZE_KWARGS).input_ids[:64]
    sc_score = sc.score_pairs(sids, slens, [(L, N)])[0].item()
    sc_deep = sc.score_ids(sids, slens, [L, 31])[0][0, 0, N].item()
    ref = tok(CFG["text"], **TOKENIZE_KWARGS).input_ids == ids  # phase-1 encoding has no special tokens either
    print(f"\nScorer: score {sc_score:.4f} (exit at L{L}), {sc_deep:.4f} (exit at L31); kernels {list(sc.kernels.values())}")

    print(f"\nper-token activation of L{L}N{N} (hook) vs manual:")
    print(f"{'pos':>3} {'token':<16} {'hook':>10} {'manual_bf16':>12} {'manual_fp32':>12}")
    for i, t in enumerate(ids):
        print(f"{i:>3} {tok.decode([t])!r:<16} {act_early[i].item():>10.4f} "
              f"{manual_bf16[i].item():>12.4f} {manual_fp32[i].item():>12.4f}")

    d_bf16 = (act_early - manual_bf16).abs().max().item()
    d_fp32 = (act_early - manual_fp32).abs().max().item()
    d_early = (act_early - act_full).abs().max().item()
    rel_fp32 = d_fp32 / act_early.abs().max().item()
    full_mat_diff = (act_all_full - (F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))).abs().max().item()
    score, argmax = act_early.max().item(), act_early.argmax().item()

    checks = {
        "found_32_lm_mlp_layers": len(lm_mlps) == 32,
        "hook_equals_manual_bf16_exact": d_bf16 == 0.0,
        "hook_equals_manual_all_9216_channels_exact": full_mat_diff == 0.0,
        "hook_close_to_manual_fp32 (rel<2%)": rel_fp32 < 0.02,
        "early_exit_raised": exited,
        "early_exit_equals_full_forward": d_early == 0.0,
        "scorer_exit_L_equals_exit_31": sc_score == sc_deep,
        "scorer_close_to_hook_max (rel<2%, fixed shape differs)": abs(sc_score - act_early.max().item()) <= 0.02 * abs(act_early.max().item()),
        "same_tokenization_as_scorer": bool(ref) and same_tokens,
    }
    print(f"\nmax |hook - manual_bf16|            = {d_bf16:.3e}")
    print(f"max |hook - manual_bf16| (all chans) = {full_mat_diff:.3e}")
    print(f"max |hook - manual_fp32|            = {d_fp32:.3e}  (relative to max |act|: {rel_fp32:.2%})")
    print(f"max |early-exit - full forward|     = {d_early:.3e}")
    print(f"forward time: full {t_full*1000:.0f} ms, early-exit {t_early*1000:.0f} ms")
    print(f"score (max over tokens) = {score:.4f} at pos {argmax} ({tok.decode([ids[argmax]])!r})")
    print("\nchecks:"); [print(f"  {'PASS' if v else 'FAIL'}  {k}") for k, v in checks.items()]

    json.dump(dict(tokens=[tok.decode([t]) for t in ids], hook=act_early.tolist(),
                   manual_bf16=manual_bf16.tolist(), manual_fp32=manual_fp32.tolist(),
                   score=score, argmax=argmax, scorer_score=sc_score, max_abs_diff_bf16=d_bf16, max_abs_diff_fp32=d_fp32,
                   max_abs_diff_early_vs_full=d_early, checks=checks),
              open(f"{RUN_DIR}/results.json", "w"), indent=2)
    print("\nsaved:", RUN_DIR)
    sys.exit(0 if all(checks.values()) else 1)

if __name__ == "__main__":
    main()
