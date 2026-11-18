"""Our own GRPO elicitor (no TRL trainer, no team RL code).

One copy of Qwen3.5-4B (bf16, pinned revision). LoRA on = elicitor (policy); LoRA off
(peft disable_adapter) = scorer / target and the "untrained" model. Optional new special tokens (one per
neuron) whose embedding rows are trained with PEFT trainable_token_indices.

GRPO as used here (one optimizer step per generated batch, so the PPO ratio is exactly 1 and clipping never acts):
    A_i  = (r_i - mean_g) / (std_g + 1e-4)  within each group g (one neuron, G completions);
           A_i = 0 for a flat group (std_g < 1e-6)
    loss = -(1 / N_step) * sum_i (1/|o_i|) * sum_t [ A_i * log pi(o_i,t | prompt, o_i,<t) - beta * k3_i,t ]
    k3_i,t = exp(ref - logp) - (ref - logp) - 1, ref = log pi_ref(o_i,t | ...), pi_ref = LoRA off (added 2026-11-18,
    --kl_beta, default 0 = the run 1 loss exactly; with beta > 0 flat-group completions are trained for their KL term).
N_step = all completions generated in the step (flat-group completions contribute 0 and are skipped in the
backward pass, which leaves the gradient unchanged). The step is split into micro-batches with gradient
accumulation, so the effective batch is always the full step.

Reward = Scorer score of the cleaned completion (prompt never scored, position 0 included) / FineWeb max of
the neuron rescored on this GPU. Empty after cleaning (score NaN) -> reward -1.0.

Base-weight integrity (UNTESTED, run on GPU: scripts/test_grpo.py): at init an exact pinned-CPU copy is taken of
every LoRA base weight and of the neuron-token embedding rows. merged() always restores both from that copy
(also when merge/generation fails), check_base_bitwise() compares the GPU weights with it, and
reference_scores() rescores fixed FineWeb contexts LoRA-off (must equal the stored this-GPU values exactly).
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
# Phase B prompt templates (scripts/grpo_phase.py --phase_b_prompt, scripts/phase_b_prompt_check.py): (a) and (b)
PHASE_B_PROMPTS = {"nK": "<|n{k}|>\n", "nK_text": "<|n{k}|>\nText:\n"}
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
        self._emb_mod = base.get_input_embeddings()     # original nn.Embedding (see _emb_weight)
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
        self.gen_cap = {}          # mode -> max sequences per generate call that fit (remembered after an OOM)
        self.merge_log = []        # one dict per merged() call: merge_s, unmerge_s, restore_s, ok
        self._backup_base()

    # ---------- exact base-weight backup (UNTESTED, run on GPU) ----------
    def _emb_weight(self):
        """The real (base) input-embedding weight. self._emb_mod is the original nn.Embedding captured before
        get_peft_model; PEFT's trainable-token layer keeps that same module as its base layer and merges into its
        weight. .weight is looked up each time in case PEFT replaces the Parameter object."""
        return self._emb_mod.weight

    def _backup_base(self):
        """Pinned CPU copy of every LoRA base weight + the neuron-token embedding rows, taken once at init
        (before any merge or optimizer step). This copy is the reference for restore and the bitwise check."""
        t0 = time.time()
        self._lora_mods = [(n, m) for n, m in self.model.named_modules() if hasattr(m, "lora_A") and hasattr(m, "base_layer")]
        self._orig = [m.base_layer.weight.detach().to("cpu", copy=True).pin_memory() for _, m in self._lora_mods]
        self._ids_t = torch.tensor(self.new_ids, dtype=torch.long, device=self.device) if self.new_ids else None
        self._orig_emb_rows = (self._emb_weight().detach()[self._ids_t].to("cpu", copy=True).pin_memory()
                               if self.new_ids else None)
        self.backup_bytes = sum(o.numel() * o.element_size() for o in self._orig)
        self.log(f"base-weight backup: {len(self._orig)} LoRA base weights + {len(self.new_ids)} embedding rows, "
                 f"{self.backup_bytes / 1e9:.2f} GB pinned CPU, {time.time() - t0:.1f}s")

    @torch.no_grad()
    def restore_base(self):
        """Copy the exact backup back into the GPU weights (LoRA base weights and neuron-token rows)."""
        for (_, m), o in zip(self._lora_mods, self._orig):
            m.base_layer.weight.copy_(o, non_blocking=True)
        if self.new_ids:
            self._emb_weight().index_copy_(0, self._ids_t, self._orig_emb_rows.to(self.device, non_blocking=True))
        torch.cuda.synchronize()

    @torch.no_grad()
    def check_base_bitwise(self):
        """GPU base weights vs the exact CPU backup. -> dict(ok, n_checked, mismatches (first 5 names), seconds).
        Must be called outside merged() (inside it the weights are merged on purpose)."""
        t0 = time.time(); bad = []
        for (n, m), o in zip(self._lora_mods, self._orig):
            if not torch.equal(m.base_layer.weight, o.to(self.device, non_blocking=True)):
                bad.append(n)
        if self.new_ids and not torch.equal(self._emb_weight()[self._ids_t], self._orig_emb_rows.to(self.device)):
            bad.append("embed_tokens[neuron-token rows]")
        torch.cuda.synchronize()
        return dict(ok=not bad, n_checked=len(self._orig) + bool(self.new_ids), n_mismatch=len(bad),
                    mismatches=bad[:5], seconds=time.time() - t0)

    @torch.no_grad()
    def reference_scores(self, refs):
        """refs: list of dict(neuron=(L, N), tokens=[64 ids], expected=float). Scores each fixed context LoRA-off
        in one fixed-shape forward. -> dict(ok, rows=[dict(neuron, expected, got, equal)])."""
        self.model.eval()
        ids = torch.tensor([r["tokens"] for r in refs], dtype=torch.long)
        lens = torch.tensor([len(r["tokens"]) for r in refs], dtype=torch.long)
        with self.adapter_off():
            s, _ = self.scorer.score_pairs(ids, lens, [tuple(r["neuron"]) for r in refs])
        rows = [dict(neuron=list(r["neuron"]), expected=r["expected"], got=float(v), equal=float(v) == r["expected"])
                for r, v in zip(refs, s.float().cpu().tolist())]
        return dict(ok=all(r["equal"] for r in rows), rows=rows)

    # ---------- helpers ----------
    def adapter_off(self):
        return self.model.disable_adapter()

    @contextlib.contextmanager
    def merged(self):
        """LoRA (and trainable tokens) merged into the bf16 weights, for faster generation. PEFT's unmerge is a
        lossy bf16 round trip, so afterwards the LoRA base weights AND the neuron-token embedding rows are
        restored bitwise from the exact CPU backup taken at init. merge_adapter() runs inside the try, so a
        failure in the merge itself, in generation, or in unmerge always ends with the exact restore.
        The merged policy is NOT bitwise the unmerged one: merged weights are rounded to bf16.
        UNTESTED, run on GPU."""
        rec = dict(merge_s=None, unmerge_s=None, restore_s=None, ok=False)
        try:
            torch.cuda.synchronize(); t0 = time.time()
            self.model.merge_adapter()
            torch.cuda.synchronize(); rec["merge_s"] = time.time() - t0
            yield
            rec["ok"] = True
        finally:
            t1 = time.time()
            try:
                self.model.unmerge_adapter()     # resets PEFT's merged flags; its weight values are overwritten below
            except Exception as e:
                self.log(f"merged(): unmerge_adapter failed ({e!r}); restoring from the exact backup anyway")
                rec["ok"] = False
            finally:
                torch.cuda.synchronize(); t2 = time.time(); rec["unmerge_s"] = t2 - t1
                self.restore_base()
                rec["restore_s"] = time.time() - t2
                self.merge_log.append(rec)

    def prompt_ids(self, prompt):
        return self.tok(prompt, add_special_tokens=False).input_ids

    @torch.no_grad()
    def generate(self, prompt_ids, n, batch=256, untrained=False, merged=False):
        """n completions for each prompt in prompt_ids (list of token lists, all the same length).
        -> list (per prompt) of lists of completion token lists (cut after the first EOS, EOS kept)."""
        P = len(prompt_ids[0]); assert all(len(p) == P for p in prompt_ids)
        self.model.eval()
        out = [[] for _ in prompt_ids]
        mode = "untrained" if untrained else ("merged" if merged else "lora")
        # an OOM lowers gen_cap[mode] for the rest of the run, so later calls do not OOM again (UNTESTED, run on GPU)
        cap = min(batch, self.gen_cap.get(mode, batch))
        per = max(1, cap // len(prompt_ids))
        ctx = self.adapter_off() if untrained else (self.merged() if merged else contextlib.nullcontext())
        with ctx:
            done = 0
            while done < n:
                k = min(per, n - done)
                x = torch.tensor([p for p in prompt_ids for _ in range(k)], device=self.device)
                try:
                    g = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=self.gen_config)
                except torch.OutOfMemoryError:
                    if k == 1:
                        raise
                    del x; torch.cuda.empty_cache(); per = max(1, k // 2)
                    self.gen_cap[mode] = per * len(prompt_ids)
                    self.log(f"generate ({mode}): CUDA OOM at {k * len(prompt_ids)} sequences; "
                             f"using {per * len(prompt_ids)} from now on")
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
    def _batch(self, prompt_ids, comps, b):
        P = len(prompt_ids[b[0]]); assert all(len(prompt_ids[i]) == P for i in b)
        T = max(len(comps[i]) for i in b)
        x = torch.full((len(b), P + T), self.tok.pad_token_id, dtype=torch.long)
        m = torch.zeros((len(b), T))
        for r, i in enumerate(b):
            x[r, :P] = torch.tensor(prompt_ids[i]); x[r, P:P + len(comps[i])] = torch.tensor(comps[i])
            m[r, :len(comps[i])] = 1
        return x.to(self.device), m.to(self.device), P, T

    def _logp(self, x, P, T):
        """Per-token log pi(o_t | prompt, o_<t) of the current adapter state, [B, T] fp32 (grad if enabled)."""
        # no attention mask: right padding only follows real tokens and every layer is causal
        logits = self.model(input_ids=x, use_cache=False, logits_to_keep=T + 1).logits[:, :T].float()
        if self.new_ids:   # same distribution as sampling: neuron tokens suppressed
            logits[:, :, self.new_ids] = float("-inf")
        return torch.log_softmax(logits, -1).gather(-1, x[:, P:P + T, None]).squeeze(-1)

    @torch.no_grad()
    def _ref_logp(self, x, P, T):
        """Reference policy = same model with LoRA (and trainable-token deltas) disabled = the scorer weights."""
        with self.adapter_off():
            return self._logp(x, P, T)

    @staticmethod
    def k3(logp, ref_logp):
        """Per-token k3 KL estimator (GRPO paper, Schulman 2020): exp(ref - logp) - (ref - logp) - 1 >= 0."""
        d = ref_logp - logp
        return torch.exp(d) - d - 1.0

    @torch.no_grad()
    def kl_stats(self, prompt_ids, comps, micro_bs):
        """Mean per-token k3 KL(policy || reference) over all completions (diagnostic only, no grads)."""
        self.model.train()   # same mode as pg_backward (no dropout anywhere, so only a bookkeeping choice)
        s, n = 0.0, 0
        for st in range(0, len(comps), micro_bs):
            b = list(range(st, min(st + micro_bs, len(comps))))
            x, m, P, T = self._batch(prompt_ids, comps, b)
            kl = self.k3(self._logp(x, P, T), self._ref_logp(x, P, T))
            s += float((kl * m).sum()); n += int(m.sum())
        return s / n if n else float("nan")

    def pg_backward(self, prompt_ids, comps, adv, n_total, micro_bs, kl_beta=0.0, log_kl=True):
        """Accumulate grads of
             -(1/n_total) sum_i [ adv_i * mean_t logp(o_i,t) - kl_beta * mean_t k3_i,t ]
        (GRPO with one update per batch, so the ratio is 1; k3 against the LoRA-off reference).
        kl_beta == 0: exactly the old loss and code path (adv == 0 completions skipped; same micro-batches).
        kl_beta > 0: every completion is trained (flat-group completions still carry the KL term).
        log_kl: also measure mean per-token k3 over ALL completions (no-grad extra forwards; grads unchanged).
        Returns (n_trained, mean logp per token over trained, dict(mean_kl, loss))."""
        self.model.train()
        idx = [i for i in range(len(comps)) if kl_beta > 0 or adv[i] != 0.0]
        lp_sum, lp_n, kl_sum, kl_n, loss_sum = 0.0, 0, 0.0, 0, 0.0
        for s in range(0, len(idx), micro_bs):
            b = idx[s:s + micro_bs]
            x, m, P, T = self._batch(prompt_ids, comps, b)
            a = torch.tensor([adv[i] for i in b], device=self.device)
            ref = self._ref_logp(x, P, T) if (kl_beta > 0 or log_kl) else None
            logp = self._logp(x, P, T)
            seq = (logp * m).sum(1) / m.sum(1)
            if kl_beta > 0:
                kl = self.k3(logp, ref)
                loss = -(a * seq - kl_beta * ((kl * m).sum(1) / m.sum(1))).sum() / n_total
            else:
                loss = -(a * seq).sum() / n_total
            loss.backward()
            loss_sum += float(loss.detach())
            lp_sum += float((logp * m).sum()); lp_n += int(m.sum())
            if ref is not None:
                with torch.no_grad():
                    kl_sum += float((self.k3(logp.detach(), ref) * m).sum()); kl_n += int(m.sum())
            del logp, loss, ref
        if log_kl and kl_beta == 0:   # completions skipped above (flat groups): KL measured without grads
            rest = sorted(set(range(len(comps))) - set(idx))
            for s in range(0, len(rest), micro_bs):
                b = rest[s:s + micro_bs]
                x, m, P, T = self._batch(prompt_ids, comps, b)
                with torch.no_grad():
                    kl = self.k3(self._logp(x, P, T), self._ref_logp(x, P, T))
                kl_sum += float((kl * m).sum()); kl_n += int(m.sum())
        stats = dict(mean_kl=(kl_sum / kl_n if kl_n else None), loss=loss_sum)
        return len(idx), (lp_sum / lp_n if lp_n else float("nan")), stats


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
