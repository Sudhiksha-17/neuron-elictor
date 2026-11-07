"""Shared neuron-scoring module (used by phase 2 baseline and phase 3 DPO scoring).

Activation of neuron (L, N) = channel N of the input to layer L's mlp.down_proj,
i.e. silu(gate_proj(x)) * up_proj(x). Score of a text = max over its (non-pad)
token positions. The forward pass stops right after decoder layer L.
"""
import os, inspect, logging
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForImageTextToText
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35_modeling

MODEL_ID = "Qwen/Qwen3.5-4B"
CONTEXT_LEN = 64
SCORE_BATCH = 256   # fixed forward shape [SCORE_BATCH, CONTEXT_LEN] -> bitwise-reproducible scores
# Tokenize as plain text: no BOS/EOS added, and literal strings like "<|endoftext|>"
# inside the text are NOT converted to special tokens.
TOKENIZE_KWARGS = dict(add_special_tokens=False, split_special_tokens=True)

# kernel-dispatching functions in the Qwen3.5 modeling code -> required implementation package
_FAST_KERNELS = {
    "torch_chunk_gated_delta_rule": "fla.",       # gated delta rule, prefill/scoring path
    "torch_recurrent_gated_delta_rule": "fla.",   # gated delta rule, single-token decode path
    "causal_conv1d_fn": "causal_conv1d.",
    "causal_conv1d_update": "causal_conv1d.",
}


class KernelFallbackError(RuntimeError):
    pass


class _FallbackWarningIsError(logging.Filter):
    def filter(self, record):
        if "falling back to its reference PyTorch implementation" in record.getMessage():
            raise KernelFallbackError(record.getMessage())
        return True


def assert_fast_kernels():
    """Fail loudly unless every linear-attention / conv kernel dispatches to its fast package."""
    report = {}
    for fname, prefix in _FAST_KERNELS.items():
        fn = getattr(qwen35_modeling, fname)
        nl = inspect.getclosurevars(fn).nonlocals
        impl = nl.get("implementation")
        where = f"{getattr(impl, '__module__', '?')}.{getattr(impl, '__name__', '?')}"
        if not nl.get("is_new_implementation") or not where.startswith(prefix):
            raise KernelFallbackError(
                f"{fname} dispatches to {where}, expected a '{prefix}*' fast kernel. "
                f"The model would run on the slow PyTorch fallback. Check that flash-linear-attention "
                f"and causal-conv1d import cleanly with the installed triton/torch.")
        report[fname] = where
    # belt and braces: turn transformers' runtime fallback warning into an exception
    logging.getLogger(qwen35_modeling.__name__).addFilter(_FallbackWarningIsError())
    transformers.utils.logging.get_logger("transformers.integrations.hub_kernels").addFilter(_FallbackWarningIsError())
    return report


class _EarlyExit(Exception):
    pass


class NeuronScorer:
    """Holds the model and returns per-text max activations for every neuron of one layer."""

    def __init__(self, layer: int, model=None, tokenizer=None, device="cuda", revision=None):
        self.kernels = assert_fast_kernels()
        self.tok = tokenizer or AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
        self.model = model or AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, revision=revision, dtype=torch.bfloat16, device_map=device).eval()
        self.layer = layer
        layers = self.model.model.language_model.layers
        assert len(layers) == 32, len(layers)
        self._decoder_layer = layers[layer]
        self.mlp = layers[layer].mlp
        self.n_neurons = self.mlp.down_proj.in_features
        self._act = None
        self._active = False
        self.mlp.down_proj.register_forward_pre_hook(self._pre_down)
        self._decoder_layer.register_forward_hook(self._post_layer)

    def _pre_down(self, mod, args):
        if self._active:
            self._act = args[0]

    def _post_layer(self, mod, args, out):
        if self._active:
            raise _EarlyExit

    @torch.no_grad()
    def activations(self, input_ids):
        """Raw [B, T, n_neurons] activations of layer `layer` (bf16, GPU) for exactly this input
        shape. No attention mask is passed: with right padding and a causal model, pad tokens
        come after the real tokens and cannot influence them."""
        self._act, self._active = None, True
        try:
            self.model(input_ids=input_ids, use_cache=False)
            raise RuntimeError("early exit did not trigger")
        except _EarlyExit:
            pass
        finally:
            self._active = False
        act, self._act = self._act, None
        return act

    @torch.no_grad()
    def score_ids(self, input_ids, lengths):
        """input_ids: [n, <=64] right-padded token ids; lengths: [n] number of real tokens.
        Returns (max [n, N] float32, argmax position [n, N] int64) over real tokens only.

        Reproducibility: bf16 GEMMs round differently for different shapes, so every forward
        pass uses the same fixed shape [SCORE_BATCH, CONTEXT_LEN] (short batches are filled with
        dummy rows). With a fixed shape a text's score is bitwise identical regardless of which
        other texts share its batch or which row it lands in."""
        dev = self.model.device
        input_ids = torch.as_tensor(input_ids, device=dev)
        lengths = torch.as_tensor(lengths, device=dev)
        n, T = input_ids.shape
        assert T <= CONTEXT_LEN and int(lengths.min()) >= 1 and int(lengths.max()) <= T
        out_max, out_pos = [], []
        for s in range(0, n, SCORE_BATCH):
            ids = input_ids[s:s + SCORE_BATCH]; ln = lengths[s:s + SCORE_BATCH]; b = ids.shape[0]
            x = torch.full((SCORE_BATCH, CONTEXT_LEN), self.tok.pad_token_id, dtype=torch.long, device=dev)
            x[:b, :T] = ids
            act = self.activations(x)[:b]
            valid = torch.arange(CONTEXT_LEN, device=dev)[None, :] < ln[:, None]
            act = act.masked_fill(~valid[:, :, None], float("-inf"))
            mx, pos = act.max(dim=1)
            out_max.append(mx.float()); out_pos.append(pos)
        return torch.cat(out_max), torch.cat(out_pos)

    def encode(self, texts, max_len=CONTEXT_LEN):
        """Tokenize texts as plain text, truncate to max_len tokens, right-pad.
        Returns (input_ids [n, max_len] long, lengths [n] long) on CPU."""
        ids = [self.tok(t, **TOKENIZE_KWARGS).input_ids[:max_len] for t in texts]
        input_ids = torch.full((len(ids), max_len), self.tok.pad_token_id, dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        return input_ids, torch.tensor([len(x) for x in ids])

    def score_texts(self, texts, max_len=CONTEXT_LEN):
        return self.score_ids(*self.encode(texts, max_len))
