"""Tests for hermes/reward.py (GRPO reward):
 1. cleaning: <think> blocks, chat-template tokens and role headers are stripped (raw vs cleaned printed)
 2. one batch with prompts for neurons in 8 different layers -> each reward == Scorer score of the
    cleaned text for that neuron alone (bitwise)
 3. with a LoRA adapter (perturbed so it changes the model) on the shared weights: reward with
    adapter_off == base-model reward (bitwise), while scoring with the adapter on differs
 4. conversational (TRL chat-format) prompts/completions give the same rewards as plain strings
"""
import os, sys, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from hermes.paths import RESULTS_DIR, BASELINE_DIR
import torch
from hermes.runinfo import run_info
from hermes.scoring import Scorer
from hermes.reward import NeuronReward, Cleaner, make_prompt, parse_prompt, EMPTY_REWARD

RUN_DIR = f"{RESULTS_DIR}/reward_tests/{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(RUN_DIR, exist_ok=False)
CFG = dict(seed=0, lora=dict(r=16, alpha=32, target_modules="all-linear (language model)"), env=run_info())
json.dump(CFG, open(f"{RUN_DIR}/config.json", "w"), indent=2)
torch.manual_seed(0)

NEURONS = [(0, 17), (3, 9000), (7, 1234), (15, 2341), (16, 5), (22, 777), (28, 4321), (31, 9215)]
COMPLETIONS = [
    "<think>\nhmm, which words?\n</think>\n\nThe Eiffel Tower in Paris was completed in 1889.",
    "<think>\n\n</think>\n\nSELECT name FROM users WHERE id = 3;<|im_end|>",
    "Once upon a time there was a dragon.<|im_end|>\n<|endoftext|>",
    "<|im_start|>assistant\nBuy now! Limited offer! Free shipping!<|im_end|>",
    "def f(x):\n    return x * 2</think>",
    "plain text with no tags at all, repeated, repeated, repeated",
    "<think>unterminated reasoning that never closes",
    "日本語のテキストと English mixed 123",
]

def nmis(a, b): return sum(not (x == y or (math.isnan(x) and math.isnan(y))) for x, y in zip(a, b))   # NaN == NaN here

def main():
    sc = Scorer()
    cl = Cleaner(sc.tok)
    prompts = [make_prompt(L, N) for L, N in NEURONS]
    assert [parse_prompt(p) for p in prompts] == NEURONS
    print("prompt example:", prompts[3])
    # ---- 1. cleaning ----
    cleaned = [cl(c) for c in COMPLETIONS]
    print("\nraw -> cleaned:")
    for r, c in zip(COMPLETIONS, cleaned):
        print(f"  {r!r}\n    -> {c!r}")
    bad = [c for c in cleaned if any(t in c for t in ("<think>", "</think>", "<|im_end|>", "<|im_start|>", "<|endoftext|>"))]
    # ---- 2. mixed layers ----
    rw = NeuronReward(sc, log_path=f"{RUN_DIR}/reward_log.jsonl", log_n=8)
    r_mixed = rw(prompts, COMPLETIONS)
    r_single = [sc.score_texts([c], (L, N))[0] for c, (L, N) in zip(cleaned, NEURONS)]
    v2 = nmis(r_mixed, r_single)
    print(f"\nmixed-layer batch ({len(set(L for L, _ in NEURONS))} layers) vs one-at-a-time: mismatches {v2}")
    print("rewards:", [round(x, 4) for x in r_mixed])
    # ---- 4. conversational format ----
    conv_p = [[{"role": "user", "content": p}] for p in prompts]
    conv_c = [[{"role": "assistant", "content": c}] for c in COMPLETIONS]
    v4 = nmis(rw(conv_p, conv_c), r_mixed)
    print(f"conversational format vs plain strings: mismatches {v4}")
    # ---- 3. LoRA on shared weights ----
    from peft import LoraConfig, get_peft_model
    pm = get_peft_model(sc.model, LoraConfig(r=16, lora_alpha=32, target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)"))
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n: p.normal_(0, 0.02)       # B=0 at init would make LoRA a no-op
    sc2 = Scorer(model=pm, tok=sc.tok)
    rw2 = NeuronReward(sc2, adapter_off=pm.disable_adapter)
    r_off = rw2(prompts, COMPLETIONS)
    r_on = NeuronReward(sc2)(prompts, COMPLETIONS)
    v3 = nmis(r_off, r_mixed); d_on = nmis(r_on, r_mixed)
    print(f"\nLoRA attached ({sum(p.numel() for n, p in pm.named_parameters() if 'lora_' in n)/1e6:.1f}M params): "
          f"adapter_off vs base mismatches {v3}; adapter ON differs on {d_on}/{len(r_on)}")
    checks = {
        "no_tags_left_after_cleaning": not bad,
        "think_content_removed": "hmm" not in cleaned[0] and cleaned[6] == "",
        "stray_close_tag_keeps_answer": cleaned[4] == "def f(x):\n    return x * 2",
        "mixed_layers_equal_single_scoring": v2 == 0,
        "conversational_equals_plain": v4 == 0,
        "adapter_off_equals_base_model": v3 == 0,
        "adapter_on_changes_scores": d_on > 0,
        "empty_completion_scores_nan": math.isnan(r_mixed[6]) and math.isnan(sc.score_texts([""], (15, 2341))[0]),
        "nonempty_completions_finite": all(math.isfinite(r) for i, r in enumerate(r_mixed) if i != 6),
    }
    print("\nchecks:"); [print(f"  {'PASS' if v else 'FAIL'}  {k}") for k, v in checks.items()]
    json.dump(dict(cleaned=cleaned, rewards=r_mixed, rewards_lora_on=r_on, checks=checks),
              open(f"{RUN_DIR}/results.json", "w"), indent=2)
    print("saved:", RUN_DIR)
    sys.exit(0 if all(checks.values()) else 1)

main()
