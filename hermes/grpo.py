"""Our own GRPO elicitor (no TRL trainer, no team RL code).

One copy of Qwen3.5-4B (bf16, pinned revision). LoRA on = elicitor (policy); LoRA off
(peft disable_adapter) = scorer / target and the "untrained" model. Optional new special tokens (one per
neuron) whose embedding rows are trained with PEFT trainable_token_indices.

GRPO as used here (KL beta = 0, one optimizer step per generated batch, so the PPO ratio is exactly 1
and clipping never acts):
    A_i  = (r_i - mean_g) / (std_g + 1e-4)  within each group g (one neuron, G completions);
           A_i = 0 for a flat group (std_g < 1e-6)
    loss = -(1 / N_step) * sum_i A_i * (1/|o_i|) * sum_t log pi(o_i,t | prompt, o_i,<t)
N_step = all completions generated in the step (flat-group completions contribute 0 and are skipped in the
backward pass, which leaves the gradient unchanged). The step is split into micro-batches with gradient
accumulation, so the effective batch is always the full step.

Reward = Scorer score of the cleaned completion (prompt never scored, position 0 included) / FineWeb max of
the neuron rescored on this GPU. Empty after cleaning (score NaN) -> reward -1.0.
"""
import os, re, math, json, time, contextlib
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForImageTextToText, GenerationConfig
from peft import LoraConfig, get_peft_model
from hermes.runinfo import MODEL_ID, pinned_revisions
from hermes.scoring import Scorer, CONTEXT_LEN, SCORE_BATCH, assert_fast_kernels
from hermes.reward import Cleaner

EMPTY_REWARD = -1.0
FLAT_STD = 1e-6
ADV_EPS = 1e-4
# LoRA on the language model's attention (full + linear/GatedDeltaNet) and MLP projections; not the vision tower
LORA_TARGET_RE = (r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
                  r"|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)"
                  r"|mlp\.(gate_proj|up_proj|down_proj))")


def bf16_ulp(v):
    v = abs(float(v))
    return 2.0 ** (math.floor(math.log2(v)) - 7) if v else 2.0 ** -133


class Elicitor:
    def __init__(self, new_tokens=(), lora_r=32, lora_alpha=64, seed=0, log=print):
        self.log = log
        torch.manual_seed(seed); np.random.seed(seed)
        self.kernels = assert_fast_kernels()
        rev = pinned_revisions()["model_revision"]
        self.revision = rev
        tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=rev)
        base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=rev, dtype=torch.bfloat16,
                                                           device_map="cuda")
        n_vocab0 = len(tok)
        self.new_tokens, self.new_ids = list(new_tokens), []
        if self.new_tokens:
            tok.add_tokens(self.new_tokens, special_tokens=True)
            self.new_ids = tok.convert_tokens_to_ids(self.new_tokens)
            emb = base.get_input_embeddings().weight
            assert max(self.new_ids) < emb.shape[0], "embedding has no free rows; resize needed"
            assert all(len(tok(t + "\n", add_special_tokens=False).input_ids) == 2 for t in self.new_tokens)
            with torch.no_grad():   # deterministic init: every new row = mean of the real vocabulary rows
                emb[self.new_ids] = emb[:n_vocab0].float().mean(0).to(emb.dtype)
        self.eos_ids = sorted({tok.pad_token_id, tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>")})
        cfg = dict(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none", target_modules=LORA_TARGET_RE)
        if self.new_ids:
            cfg["trainable_token_indices"] = {"embed_tokens": self.new_ids}
        self.lora_config = LoraConfig(**cfg)
        self.model = get_peft_model(base, self.lora_config)
        # adapters / trainable token deltas in fp32 (bf16 would round away lr-2e-5 updates)
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
        self.trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        self.n_trainable = sum(p.numel() for _, p in self.trainable)
        self.tok = tok
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.config.use_cache = False
        self.scorer = Scorer(model=self.model, tok=tok, revision=rev)
        self.clean = Cleaner(tok)        # built after adding tokens: strips them too
        self.gen_config = GenerationConfig(
            do_sample=True, temperature=1.0, top_k=0, top_p=1.0, min_p=None, repetition_penalty=1.0,
            max_new_tokens=64, eos_token_id=self.eos_ids, pad_token_id=tok.pad_token_id, use_cache=True,
            suppress_tokens=self.new_ids or None)
        self.device = self.scorer.device

    # ---------- helpers ----------
    def adapter_off(self):
        return self.model.disable_adapter()

    @contextlib.contextmanager
    def merged(self):
        """LoRA (and trainable tokens) merged into the bf16 weights, for faster generation. PEFT's unmerge is a
        lossy bf16 round trip, so the LoRA base weights are restored bitwise from a pinned CPU copy afterwards
        (trainable-token unmerge restores the stored original rows exactly). The merged policy is NOT bitwise the
        unmerged one: merged weights are rounded to bf16."""
        if not hasattr(self, "_lora_mods"):
            self._lora_mods = [m for _, m in self.model.named_modules() if hasattr(m, "lora_A") and hasattr(m, "base_layer")]
            self._orig = [m.base_layer.weight.detach().to("cpu", copy=True).pin_memory() for m in self._lora_mods]
        self.model.merge_adapter()
        try:
            yield
        finally:
            self.model.unmerge_adapter()
            with torch.no_grad():
                for m, o in zip(self._lora_mods, self._orig):
                    m.base_layer.weight.copy_(o, non_blocking=True)
            torch.cuda.synchronize()

    def prompt_ids(self, prompt):
        return self.tok(prompt, add_special_tokens=False).input_ids

    @torch.no_grad()
    def generate(self, prompt_ids, n, batch=256, untrained=False, merged=False):
        """n completions for each prompt in prompt_ids (list of token lists, all the same length).
        -> list (per prompt) of lists of completion token lists (cut after the first EOS, EOS kept)."""
        P = len(prompt_ids[0]); assert all(len(p) == P for p in prompt_ids)
        self.model.eval()
        out = [[] for _ in prompt_ids]
        per = max(1, batch // len(prompt_ids))
        ctx = self.adapter_off() if untrained else (self.merged() if merged else contextlib.nullcontext())
        with ctx:
            done = 0
            while done < n:
                k = min(per, n - done)
                x = torch.tensor([p for p in prompt_ids for _ in range(k)], device=self.device)
                try:
                    g = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=self.gen_config)
                except torch.OutOfMemoryError:
                    if per == 1:
                        raise
                    del x; torch.cuda.empty_cache(); per = max(1, per // 2)
                    self.log(f"generate: CUDA OOM at {k * len(prompt_ids)} sequences; retrying with {per * len(prompt_ids)}")
                    continue
                comp = g[:, P:].tolist()
                for j, c in enumerate(comp):
                    cut = next((i for i, t in enumerate(c) if t in self.eos_ids), None)
                    out[j // k].append(c[:cut + 1] if cut is not None else c)
                done += k
        return out

    def texts(self, comps):
        """completion token lists -> cleaned texts (special tokens, incl. neuron tokens, stripped)."""
        return [self.clean(self.tok.decode(c, skip_special_tokens=False)) for c in comps]

    @torch.no_grad()
    def score(self, texts, pairs):
        """Raw target scores (NaN if empty after cleaning), LoRA off, fixed-shape Scorer."""
        self.model.eval()
        ids, lens = self.scorer.encode(texts)
        with self.adapter_off():
            s, _ = self.scorer.score_pairs(ids, lens, pairs)
        return s.float().cpu()

    @torch.no_grad()
    def token_acts(self, text, layer, neuron):
        """Per-token activations of (layer, neuron) on one text (same fixed-shape forward as scoring)."""
        self.model.eval()
        ids, lens = self.scorer.encode([text])
        x = torch.full((SCORE_BATCH, CONTEXT_LEN), self.scorer.pad_id, dtype=torch.long); x[:1] = ids
        got = {}
        with self.adapter_off():
            self.scorer._forward(x, [layer], lambda L, a: got.__setitem__(L, a[0, :, neuron].float().cpu()))
        n = int(lens[0])
        return [(self.tok.decode([t]), float(v)) for t, v in zip(ids[0, :n].tolist(), got[layer][:n].tolist())]

    # ---------- policy gradient ----------
    def pg_backward(self, prompt_ids, comps, adv, n_total, micro_bs):
        """Accumulate grads of -(1/n_total) sum_i adv_i * mean_t logp(o_i,t). prompt_ids[i] per completion.
        Skips adv == 0. Returns (n_trained, mean logp per token)."""
        self.model.train()
        idx = [i for i in range(len(comps)) if adv[i] != 0.0]
        lp_sum, lp_n = 0.0, 0
        for s in range(0, len(idx), micro_bs):
            b = idx[s:s + micro_bs]
            P = len(prompt_ids[b[0]]); assert all(len(prompt_ids[i]) == P for i in b)
            T = max(len(comps[i]) for i in b)
            x = torch.full((len(b), P + T), self.tok.pad_token_id, dtype=torch.long)
            m = torch.zeros((len(b), T))
            for r, i in enumerate(b):
                x[r, :P] = torch.tensor(prompt_ids[i]); x[r, P:P + len(comps[i])] = torch.tensor(comps[i])
                m[r, :len(comps[i])] = 1
            x, m = x.to(self.device), m.to(self.device)
            a = torch.tensor([adv[i] for i in b], device=self.device)
            # no attention mask: right padding only follows real tokens and every layer is causal
            logits = self.model(input_ids=x, use_cache=False, logits_to_keep=T + 1).logits[:, :T].float()
            if self.new_ids:   # same distribution as sampling: neuron tokens suppressed
                logits[:, :, self.new_ids] = float("-inf")
            logp = torch.log_softmax(logits, -1).gather(-1, x[:, P:P + T, None]).squeeze(-1)
            seq = (logp * m).sum(1) / m.sum(1)
            loss = -(a * seq).sum() / n_total
            loss.backward()
            lp_sum += float((logp * m).sum()); lp_n += int(m.sum())
            del logits, logp, loss
        return len(idx), (lp_sum / lp_n if lp_n else float("nan"))


def group_advantages(rewards, group_size):
    """rewards [n] (grouped contiguously) -> advantages [n], flat mask [n_groups]."""
    r = torch.tensor(rewards, dtype=torch.float64).view(-1, group_size)
    std = r.std(dim=1, keepdim=True)            # unbiased, as in TRL
    flat = (std < FLAT_STD).squeeze(1)
    adv = (r - r.mean(1, keepdim=True)) / (std + ADV_EPS)
    adv[flat] = 0.0
    return adv.view(-1).tolist(), flat.tolist()


def to_reward(score, fw_max):
    return EMPTY_REWARD if math.isnan(score) else score / fw_max
