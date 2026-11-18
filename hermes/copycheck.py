"""Copy check and diversity metrics for warm-started elicitors (added 2026-11-18).

All on token ids of the target tokenizer (texts re-tokenized with Scorer.encode, i.e. truncated to 64 tokens, the
same tokens that are scored). Token Jaccard = |set(a) & set(b)| / |set(a) | set(b)| (0 if both empty).
Longest shared span = length of the longest contiguous token run present in both sequences.
"""
import itertools
import numpy as np

NEAR_COPY_JACCARD = 0.8


def jaccard(a, b):
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


def longest_shared_span(a, b):
    """Longest common contiguous substring length (O(len(a) * len(b)) DP; sequences are <= 64 tokens)."""
    if not a or not b:
        return 0
    best, prev = 0, [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b):
            if x == y:
                cur[j + 1] = prev[j] + 1
                if cur[j + 1] > best:
                    best = cur[j + 1]
        prev = cur
    return best


def copy_stats(tokens, contexts):
    """tokens: one text's token ids; contexts: list of token lists (the neuron's top-10 FineWeb contexts, rank order).
    -> dict(max_jaccard, max_jaccard_rank, longest_span, longest_span_rank)."""
    js = [jaccard(tokens, c) for c in contexts]
    sp = [longest_shared_span(tokens, c) for c in contexts]
    kj, ks = int(np.argmax(js)), int(np.argmax(sp))
    return dict(max_jaccard=float(js[kj]), max_jaccard_rank=kj, longest_span=int(sp[ks]), longest_span_rank=ks)


def diversity(token_lists, texts, contexts, thresh=NEAR_COPY_JACCARD):
    """Diversity pre-check over samples: distinct texts, mean pairwise token Jaccard, per-sample max token Jaccard to
    the contexts, number / fraction of near-copies (max Jaccard > thresh)."""
    n = len(token_lists)
    pair = [jaccard(token_lists[i], token_lists[j]) for i, j in itertools.combinations(range(n), 2)]
    cs = [copy_stats(t, contexts) for t in token_lists]
    mj = [c["max_jaccard"] for c in cs]
    n_copy = sum(v > thresh for v in mj)
    return dict(n_samples=n, n_distinct_texts=len(set(texts)), mean_pairwise_jaccard=float(np.mean(pair)) if pair else None,
                max_jaccard_to_top10_mean=float(np.mean(mj)) if mj else None,
                max_jaccard_to_top10_max=float(np.max(mj)) if mj else None,
                max_jaccard_to_top10_median=float(np.median(mj)) if mj else None,
                longest_span_max=max((c["longest_span"] for c in cs), default=0),
                near_copy_threshold=thresh, n_near_copies=int(n_copy), frac_near_copies=n_copy / n if n else None,
                per_sample=cs)
