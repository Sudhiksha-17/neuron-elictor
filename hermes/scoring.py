"""Neuron scoring for Qwen3.5-4B (the frozen target).

Activation of neuron (L, N) = channel N of the input to layers[L].mlp.down_proj,
i.e. silu(gate_proj(x)) * up_proj(x), read with a forward pre-hook.
Score of a text = max of that activation over its (<= 64) token positions.

Determinism: every forward runs at the fixed shape [SCORE_BATCH, CONTEXT_LEN]. Shorter texts are
right-padded with the pad token and no attention mask is passed (all layers are causal, so pads only
follow real tokens and cannot influence them); pad positions are excluded from the max. Unused batch
rows are filled with pad rows. With a fixed shape, a text's scores are bitwise identical whatever else
is in the batch (checked by scripts/test_scoring.py). The forward stops right after the deepest layer
that is needed (early exit), so the target never runs the final norm / LM head.

Several layers can be read in one forward (phase 2 reads all 32; the GRPO reward reads whichever
layers the neurons in its batch belong to).
"""
import os, re
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForImageTextToText
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35_modeling
from hermes.runinfo import MODEL_ID, pinned_revisions

CONTEXT_LEN = 64
SCORE_BATCH = 256
N_LAYERS = 32
# no BOS/EOS/chat template; literal special-token strings in text are tokenized as ordinary text
TOKENIZE_KWARGS = dict(add_special_tokens=False, split_special_tokens=True)
FAST_KERNEL_FUNCS = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                     "causal_conv1d_fn", "causal_conv1d_update")
LAYER_RE = re.compile(r"^(.*language_model\.layers)\.(\d+)$")


class KernelFallbackError(RuntimeError):
    pass


class EarlyExit(Exception):
    pass


def _closure_var(fn, name):
    if name not in fn.__code__.co_freevars:
        raise KernelFallbackError(f"cannot inspect kernel wrapper {fn.__name__}: no closure var {name!r} "
                                  f"(transformers changed?); refusing to run unverified")
    return fn.__closure__[fn.__code__.co_freevars.index(name)].cell_contents


def assert_fast_kernels():
    """Raise KernelFallbackError unless FLA + causal-conv1d kernels are used (not the slow torch path)."""
    rep = {}
    for name in FAST_KERNEL_FUNCS:
        fn = getattr(qwen35_modeling, name)
        if not _closure_var(fn, "is_new_implementation"):
            raise KernelFallbackError(f"{name} falls back to the reference PyTorch implementation "
                                      f"(flash-linear-attention / causal-conv1d missing or failing to import)")
        impl = _closure_var(fn, "implementation")
        rep[name] = f"{impl.__module__}.{getattr(impl, '__qualname__', impl)}"
    return rep


def find_lm_layers(model):
    """The 32 language-model decoder layers (excludes vision tower / MTP), by index."""
    layers = {}
    for name, mod in model.named_modules():
        m = LAYER_RE.match(name)
        if m:
            layers[int(m.group(2))] = mod
    assert sorted(layers) == list(range(N_LAYERS)), sorted(layers)
    return [layers[i] for i in range(N_LAYERS)]


def parse_neuron_id(s):
    """'L15N2341' -> (15, 2341)."""
    m = re.fullmatch(r"L(\d+)N(\d+)", s)
    if not m:
        raise ValueError(f"bad neuron id {s!r}")
    return int(m.group(1)), int(m.group(2))


def neuron_id(layer, neuron):
    return f"L{layer}N{neuron}"


class Scorer:
    """Target scorer. Pass `model`/`tok` to share weights with the elicitor (call inside
    `peft_model.disable_adapter()` so the target is the base model)."""

    def __init__(self, revision=None, model=None, tok=None):
        self.kernels = assert_fast_kernels()
        revision = revision or pinned_revisions()["model_revision"]
        self.revision = revision
        self.tok = tok or AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
        if model is None:
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID, revision=revision, dtype=torch.bfloat16, device_map="cuda").eval()
        self.model = model
        self.layers = find_lm_layers(model)
        self.n_neurons = self.layers[0].mlp.gate_proj.out_features
        self.device = next(model.parameters()).device
        self.pad_id = self.tok.pad_token_id

    # ---------- tokenization ----------
    def encode(self, texts):
        """-> ids [n, 64] (right-padded with pad id, truncated to 64), lens [n]."""
        enc = self.tok(list(texts), **TOKENIZE_KWARGS).input_ids
        ids = torch.full((len(enc), CONTEXT_LEN), self.pad_id, dtype=torch.long)
        lens = torch.zeros(len(enc), dtype=torch.long)
        for i, t in enumerate(enc):
            t = t[:CONTEXT_LEN]
            ids[i, :len(t)] = torch.tensor(t, dtype=torch.long); lens[i] = len(t)
        return ids, lens

    # ---------- forward ----------
    def _forward(self, ids, layers, on_act, fixed_shape=True):
        """One fixed-shape forward [SCORE_BATCH, 64]; on_act(layer, act[B,64,N]) is called for each
        requested layer; stops after max(layers)."""
        if fixed_shape:
            assert ids.shape == (SCORE_BATCH, CONTEXT_LEN), ids.shape
        layers = sorted(set(layers))
        hooks = []
        for L in layers:
            hooks.append(self.layers[L].mlp.down_proj.register_forward_pre_hook(
                lambda mod, args, L=L: on_act(L, args[0])))
        def stop(mod, args, out):
            raise EarlyExit
        hooks.append(self.layers[layers[-1]].register_forward_hook(stop))
        try:
            with torch.no_grad():
                self.model(input_ids=ids.to(self.device), use_cache=False)
            raise RuntimeError("early exit did not trigger")
        except EarlyExit:
            pass
        finally:
            for h in hooks:
                h.remove()

    def activations(self, ids, layer):
        """Raw activations [B, 64, N] of one layer (testing only; any batch shape, no padding/masking)."""
        out = {}
        self._forward(ids, [layer], lambda L, a: out.__setitem__(L, a.detach().clone()), fixed_shape=False)
        return out[layer]

    def score_ids(self, ids, lens, layers):
        """Scores of every neuron of the given layers.
        -> max [n, len(layers), N] (bf16 values as float32), pos [n, len(layers), N] (int16: first
        position attaining the max). Pad positions (>= lens) are excluded."""
        layers = list(layers)
        n = len(ids)
        mx_out = torch.empty((n, len(layers), self.n_neurons), dtype=torch.float32, device=self.device)
        pos_out = torch.empty((n, len(layers), self.n_neurons), dtype=torch.int16, device=self.device)
        ar = torch.arange(CONTEXT_LEN, device=self.device)
        for s in range(0, n, SCORE_BATCH):
            b = min(SCORE_BATCH, n - s)
            x = torch.full((SCORE_BATCH, CONTEXT_LEN), self.pad_id, dtype=torch.long)
            x[:b] = ids[s:s + b]
            ln = torch.zeros(SCORE_BATCH, dtype=torch.long, device=self.device)
            ln[:b] = lens[s:s + b].to(self.device)
            pad = (ar[None, :] >= ln[:, None])[:, :, None]            # [B, 64, 1]
            def on_act(L, a):
                a = a.masked_fill(pad, float("-inf"))
                m = a.max(dim=1).values                                # [B, N]
                p = torch.where(a == m[:, None, :], ar[None, :, None], CONTEXT_LEN).amin(dim=1)
                j = layers.index(L)
                mx_out[s:s + b, j] = m[:b].float(); pos_out[s:s + b, j] = p[:b].to(torch.int16)
            self._forward(x, layers, on_act)
        return mx_out, pos_out

    def score_pairs(self, ids, lens, pairs):
        """Score row i for neuron pairs[i] = (layer, neuron); layers may differ between rows.
        -> scores [n] float32, pos [n]. An empty text (lens == 0) scores NaN (pos -1): it has no
        activation, and any number (e.g. 0.0) could beat real attempts since activations can be
        negative. RL code MUST penalize NaN scores explicitly (see hermes/reward.py)."""
        layers = sorted({L for L, _ in pairs})
        mx, pos = self.score_ids(ids, lens, layers)
        r = torch.arange(len(pairs), device=self.device)
        li = torch.tensor([layers.index(L) for L, _ in pairs], device=self.device)
        ni = torch.tensor([N for _, N in pairs], device=self.device)
        empty = lens.to(self.device) == 0
        sc, ps = mx[r, li, ni], pos[r, li, ni]
        return sc.masked_fill(empty, float("nan")), ps.masked_fill(empty, -1)

    def score_texts(self, texts, pairs):
        """Convenience: texts[i] scored for neuron pairs[i] (or one (layer, neuron) for all)."""
        if isinstance(pairs, tuple):
            pairs = [pairs] * len(texts)
        ids, lens = self.encode(texts)
        return self.score_pairs(ids, lens, pairs)[0].tolist()
