"""GRPO reward: target score of the cleaned elicitor text, for whichever neuron each prompt names.

Prompts in one batch may name neurons in different layers; all of them are scored in one fixed-shape
forward that reads every needed layer and stops after the deepest one (hermes.scoring.Scorer.score_pairs).
Works with TRL's GRPOTrainer reward-function signature: reward(prompts, completions, **kwargs).
"""
import json, re, contextlib, time
from hermes.scoring import parse_neuron_id

PROMPT_TEMPLATE = "Write text that maximally activates neuron L{layer}N{neuron}."
PROMPT_RE = re.compile(r"neuron (L\d+N\d+)\.")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
EMPTY_REWARD = 0.0     # reward for a completion that is empty after cleaning (TO BE APPROVED in phase 3)


def make_prompt(layer, neuron):
    return PROMPT_TEMPLATE.format(layer=layer, neuron=neuron)


def _text(x):
    """str, or TRL conversational format [{'role':..., 'content':...}, ...] -> text of the last message."""
    if isinstance(x, str):
        return x
    c = x[-1]["content"]
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in c)


def parse_prompt(prompt):
    ids = PROMPT_RE.findall(_text(prompt))
    if not ids:
        raise ValueError(f"no neuron id in prompt {prompt!r}")
    return parse_neuron_id(ids[-1])


class Cleaner:
    """Keep only the elicitor's generated text: drop <think>...</think> blocks and everything after an
    unterminated <think>, chat-template role headers and every special/added token string (incl. a
    stray </think>)."""

    def __init__(self, tok):
        specials = set(tok.all_special_tokens) | {t.content for t in tok.added_tokens_decoder.values()}
        specials |= {"<think>", "</think>"}
        self.special_re = re.compile("|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True)))
        self.role_re = re.compile(r"<\|im_start\|>(system|user|assistant)\n?")

    def __call__(self, text):
        t = THINK_RE.sub("", text)
        t = t.split("<think>")[0]          # unterminated block: everything after <think> is reasoning
        # a stray </think> is only removed as a tag (thinking is off: the prompt already holds the
        # closed empty think block, so text before a stray </think> is the answer, not reasoning)
        t = self.role_re.sub("", t)
        t = self.special_re.sub("", t)
        return t.strip()


class NeuronReward:
    """reward(prompts, completions) -> list of float target scores.

    scorer: hermes.scoring.Scorer sharing weights with the elicitor.
    adapter_off: context-manager factory that disables the LoRA adapter (e.g. peft_model.disable_adapter),
                 so the target is the base model. None if the scorer's model has no adapter.
    log_path: jsonl file; every call appends raw vs cleaned examples (first `log_n` of each call)."""

    __name__ = "neuron_reward"

    def __init__(self, scorer, adapter_off=None, log_path=None, log_n=4):
        self.sc, self.adapter_off, self.log_path, self.log_n = scorer, adapter_off, log_path, log_n
        self.clean = Cleaner(scorer.tok)
        self.calls = 0

    def __call__(self, prompts, completions, **kwargs):
        pairs = [parse_prompt(p) for p in prompts]
        raw = [_text(c) for c in completions]
        cleaned = [self.clean(r) for r in raw]
        ids, lens = self.sc.encode(cleaned)
        ctx = self.adapter_off() if self.adapter_off else contextlib.nullcontext()
        with ctx:
            scores, pos = self.sc.score_pairs(ids, lens, pairs)
        rewards = [EMPTY_REWARD if int(l) == 0 else float(s) for s, l in zip(scores.tolist(), lens)]
        if self.log_path:
            with open(self.log_path, "a") as f:
                for i in range(min(self.log_n, len(raw))):
                    f.write(json.dumps(dict(call=self.calls, time=time.time(), neuron=f"L{pairs[i][0]}N{pairs[i][1]}",
                                            raw=raw[i], cleaned=cleaned[i], n_tokens=int(lens[i]),
                                            reward=rewards[i], argmax_pos=int(pos[i]))) + "\n")
        self.calls += 1
        return rewards
