"""Tests for hermes/scoring.py (layer chosen with --layer, default 15):
 1. fast-kernel assertion passes, and fails loudly on a simulated fallback
 2. padded-batch scores == one-text-at-a-time scores (bitwise), all 9216 neurons of layer 15,
    on 24 texts of different lengths, in shuffled order / different batch neighbours
 3. pad positions are excluded from the max, and pad content cannot leak into real positions
 4. full 64-token texts score identically in an all-full batch and in a mixed padded batch
    (phase 2 batches have no padding, phase 3 batches do)
 5. for reference: size of the bf16 shape-dependence the fixed shape removes
 6. multi-layer: reading layer L together with other layers (deeper early exit) == reading L alone,
    and score_pairs with neurons from different layers in one batch == per-layer scoring (bitwise)
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from hermes.paths import RESULTS_DIR, BASELINE_DIR
import torch
from hermes.runinfo import run_info
from hermes import scoring
from hermes.scoring import Scorer, assert_fast_kernels, KernelFallbackError

RUN_DIR = f"{RESULTS_DIR}/scoring_tests/{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(RUN_DIR, exist_ok=False)
_ap = argparse.ArgumentParser(); _ap.add_argument("--layer", type=int, default=15); _args = _ap.parse_args()
CFG = dict(layer=_args.layer, seed=0, context_len=scoring.CONTEXT_LEN, score_batch=scoring.SCORE_BATCH,
           padding="right, no attention mask passed, pads excluded from max")
torch.manual_seed(CFG["seed"])

TEXTS = [
    "Hi.", "The cat sat.", "Paris is the capital of France.", "def f(x):\n    return x * 2",
    "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12",
    "The Eiffel Tower in Paris was completed in 1889 and is one of the most visited monuments in the world.",
    "In 2019, researchers at the university published a study showing that regular exercise improves memory in older adults.",
    "SELECT name, email FROM users WHERE created_at > '2024-01-01' ORDER BY name;",
    "Ingredients: 2 cups flour, 1 cup sugar, 3 eggs, 1 tsp vanilla. Preheat oven to 350F.",
    "¿Dónde está la biblioteca? Je ne sais pas. 我不知道。",
    "<|endoftext|> literal special-token string inside text",
    "Terms and conditions apply. See store for details. Offer valid while supplies last. No purchase necessary.",
    "Once upon a time, in a kingdom far away, there lived a young princess who loved to read books about the stars and the planets beyond.",
    "The mitochondria is the powerhouse of the cell. " * 4,
    "Click here to subscribe to our newsletter and get 10% off your first order!",
    "He said, \"I'll be there by noon,\" but he never showed up, and nobody heard from him again for three years.",
    "a",
    "x = np.linspace(0, 10, 100)\ny = np.sin(x)\nplt.plot(x, y)\nplt.show()",
    "The quick brown fox jumps over the lazy dog. " * 6,   # > 64 tokens -> truncated
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " * 3,
    "Q: What is 2+2?\nA: 4",
    "Breaking news: the stock market fell sharply on Monday amid fears of rising interest rates and slowing growth.",
    "🙂🙂🙂 emoji test",
    "Section 3.2(b) of the Agreement shall survive termination.",
]

def nmis(a, b): return int((a != b).sum())

def main():
    out = {"config": CFG}
    out["env"] = run_info()
    json.dump(out, open(f"{RUN_DIR}/config.json", "w"), indent=2)
    # ---- 1. kernel assertion ----
    rep = assert_fast_kernels()
    print("fast kernels:"); [print(f"  {k} -> {v}") for k, v in rep.items()]
    fn = scoring.qwen35_modeling.torch_chunk_gated_delta_rule
    cell = fn.__closure__[fn.__code__.co_freevars.index("is_new_implementation")]
    cell.cell_contents = False                     # simulate a fallback to PyTorch
    try:
        assert_fast_kernels(); neg = "DID NOT RAISE"
    except KernelFallbackError as e:
        neg = f"raised KernelFallbackError ({str(e)[:80]}...)"
    finally:
        cell.cell_contents = True
    print("simulated fallback ->", neg)

    sc = Scorer()
    L = CFG["layer"]
    _score_all = sc.score_ids
    def score_one_layer(ids_, lens_):        # [n, N] for layer L
        m, p = _score_all(ids_, lens_, [L]); return m[:, 0], p[:, 0]
    sc.score_ids = score_one_layer
    act_fn = sc.activations
    sc.activations = lambda x: act_fn(x, L)
    ids, lens = sc.encode(TEXTS)
    n, N = len(TEXTS), sc.n_neurons
    print(f"\n{n} texts, token lengths: {lens.tolist()}")

    # ---- 2. batched vs one at a time ----
    b_max, b_pos = sc.score_ids(ids, lens)                       # all 24 in one batch
    s = [sc.score_ids(ids[i:i+1], lens[i:i+1]) for i in range(n)]  # one text per call
    s_max = torch.cat([x[0] for x in s]); s_pos = torch.cat([x[1] for x in s])
    perm = torch.randperm(n)
    p_max, p_pos = sc.score_ids(ids[perm], lens[perm])            # shuffled order
    inv = torch.argsort(perm.cuda())
    v1, a1 = nmis(b_max, s_max), nmis(b_pos, s_pos)
    v2, a2 = nmis(p_max[inv], s_max), nmis(p_pos[inv], s_pos)
    print(f"\nbatched vs one-at-a-time ({n*N} scores): value mismatches {v1}, argmax mismatches {a1}, max|diff| {(b_max-s_max).abs().max().item():.1e}")
    print(f"shuffled batch vs one-at-a-time:         value mismatches {v2}, argmax mismatches {a2}")

    # ---- 3. pad exclusion and no leakage ----
    pos_ok = bool((b_pos < lens.cuda()[:, None]).all())
    padded = ids.clone(); padded[ids == sc.tok.pad_token_id] = 0
    garbage = ids.clone()
    for i in range(n): garbage[i, lens[i]:] = torch.randint(1000, 100000, (64 - lens[i],))
    g_max, g_pos = sc.score_ids(garbage, lens)
    leak = nmis(g_max, b_max)
    act = sc.activations(torch.nn.functional.pad(ids, (0, 0, 0, scoring.SCORE_BATCH - n), value=sc.tok.pad_token_id).cuda())[:n].float()
    would_win = sum(int((act[i, lens[i]:].max(0).values > act[i, :lens[i]].max(0).values).sum()) for i in range(n) if lens[i] < 64)
    print(f"\nall argmax positions inside real tokens: {pos_ok}")
    print(f"without exclusion, a pad position would have been the max for {would_win} text-neuron pairs")
    print(f"random tokens instead of pad after each text -> value mismatches {leak} (0 = no leakage)")

    # ---- 4. full 64-token rows: all-full batch vs mixed batch ----
    full_ids, full_lens = sc.encode([(t + " ") * 80 for t in TEXTS])
    assert (full_lens == 64).all()
    f_alone = sc.score_ids(full_ids, full_lens)[0]
    mixed_ids = torch.cat([full_ids, ids]); mixed_lens = torch.cat([full_lens, lens])
    f_mixed = sc.score_ids(mixed_ids, mixed_lens)[0][:n]
    v4 = nmis(f_alone, f_mixed)
    print(f"\nfull 64-token texts: all-full batch vs mixed padded batch -> value mismatches {v4}")

    # ---- 5. reference: what the fixed shape removes ----
    def unfixed(x):    # forward at natural shape [b, 64] instead of [256, 64]
        return sc.activations(x.cuda()).max(1).values.float()
    un_b = unfixed(full_ids); un_s = torch.cat([unfixed(full_ids[i:i+1]) for i in range(n)])
    d = (un_b - un_s).abs(); big = un_s.abs() > 1
    print(f"\nreference, WITHOUT fixed shape (batch of {n} vs batch of 1): mismatches {nmis(un_b, un_s)}/{n*N}, "
          f"max|diff| {d.max().item():.3f}, median rel diff for |score|>1: {(d[big]/un_s.abs()[big]).median().item():.2%}")

    # ---- 6. multi-layer ----
    sc.score_ids = _score_all
    others = sorted({0, max(L - 1, 0), L, min(L + 1, 31), 31})
    m_multi, p_multi = _score_all(ids, lens, others)
    j = others.index(L)
    v6a = nmis(m_multi[:, j], b_max) + nmis(p_multi[:, j], b_pos.to(p_multi.dtype))
    g = torch.Generator().manual_seed(1)
    pairs = [(others[int(torch.randint(len(others), (1,), generator=g))], int(torch.randint(N, (1,), generator=g))) for _ in range(n)]
    sp, spos = sc.score_pairs(ids, lens, pairs)
    ref = torch.stack([_score_all(ids[i:i+1], lens[i:i+1], [pl])[0][0, 0, pn] for i, (pl, pn) in enumerate(pairs)])
    v6b = nmis(sp, ref)
    print(f"\nmulti-layer read {others} vs layer {L} alone: mismatches {v6a}")
    print(f"score_pairs, {n} rows with neurons from layers {sorted({p[0] for p in pairs})}: mismatches vs per-row single-layer scoring {v6b}")

    checks = {
        "fast_kernel_assert_passes": True,
        "fast_kernel_assert_fails_on_simulated_fallback": neg.startswith("raised"),
        "batched_equals_single_exact_values": v1 == 0 and v2 == 0,
        "batched_equals_single_exact_argmax": a1 == 0 and a2 == 0,
        "argmax_never_on_pad": pos_ok,
        "pad_content_cannot_leak": leak == 0,
        "full_rows_same_in_full_and_mixed_batches": v4 == 0,
        "at_least_20_texts_varied_lengths": n >= 20 and len(set(lens.tolist())) >= 15,
        "multi_layer_read_equals_single_layer": v6a == 0,
        "mixed_layer_pairs_equal_single_scoring": v6b == 0,
    }
    print("\nchecks:"); [print(f"  {'PASS' if v else 'FAIL'}  {k}") for k, v in checks.items()]
    out.update(kernels=rep, simulated_fallback=neg, lens=lens.tolist(), mismatches=dict(batched=v1, shuffled=v2,
               argmax=a1, argmax_shuffled=a2, garbage_pads=leak, full_vs_mixed=v4, multi_layer=v6a, mixed_pairs=v6b), pad_would_win=would_win, checks=checks)
    json.dump(out, open(f"{RUN_DIR}/results.json", "w"), indent=2)
    print("saved:", RUN_DIR)
    sys.exit(0 if all(checks.values()) else 1)

main()
