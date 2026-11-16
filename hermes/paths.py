"""Repo-relative paths. The repo used to be /workspace itself; it now lives at /workspace/neuron-elictor,
so nothing hardcodes /workspace except HF_HOME (/workspace/hf_cache, shared cache outside the repo).

Inputs (gitignored, from https://huggingface.co/datasets/gray171/hermes-baseline, see data/DATASET_README.md):
  data/baseline.pt, data/contexts.npz  - full 1M phase 2 baseline (was results/phase2/20261109_204110_full1M)
  data/pilots.json                     - the 32 pilot neurons (was results/pilots/20261109_213918_full1M_pilots)
"""
import json, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
BASELINE_DIR = DATA_DIR                      # hermes.baseline.Baseline(BASELINE_DIR) reads baseline.pt + contexts.npz
PILOTS_FILE = os.path.join(DATA_DIR, "pilots.json")
RESULTS_DIR = os.path.join(REPO, "results")
REPORTS_DIR = os.path.join(REPO, "reports")
LOGS_DIR = os.path.join(REPO, "logs")


def load_pilots(path=PILOTS_FILE):
    """The pilot neuron list: [{id, layer, neuron, rank, fineweb_max, argmax_ctx, argmax_pos}, ...]."""
    with open(path) as f:
        return json.load(f)
