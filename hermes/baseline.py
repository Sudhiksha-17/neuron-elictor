"""Load a phase 2 baseline run and decode its contexts (text is not stored, only tokens)."""
import numpy as np, torch


class Baseline:
    def __init__(self, run_dir, tok=None):
        self.run_dir = run_dir
        self.b = torch.load(f"{run_dir}/baseline.pt")
        self.ctx = np.load(f"{run_dir}/contexts.npz")
        self.tokens = self.ctx["tokens"]
        self.tok = tok

    def __getitem__(self, k):
        return self.b[k]

    def neuron(self, layer, neuron):
        b = self.b
        return dict(max=float(b["max"][layer, neuron]), argmax_ctx=int(b["argmax_ctx"][layer, neuron]),
                    argmax_pos=int(b["argmax_pos"][layer, neuron]), mean=float(b["mean_score"][layer, neuron]),
                    std=float(b["std_score"][layer, neuron]), top_ctx=b["top_ctx"][layer, neuron].tolist(),
                    top_val=b["top_val"][layer, neuron].tolist(), top_pos=b["top_pos"][layer, neuron].tolist())

    def context_tokens(self, ctx_id):
        return self.tokens[ctx_id].tolist()

    def context_text(self, ctx_id):
        return self.tok.decode(self.context_tokens(ctx_id))

    def argmax_token(self, layer, neuron):
        """Token id at the FineWeb argmax of (layer, neuron) (for the repeated-token trivial check)."""
        c, p = int(self.b["argmax_ctx"][layer, neuron]), int(self.b["argmax_pos"][layer, neuron])
        return int(self.tokens[c][p])
