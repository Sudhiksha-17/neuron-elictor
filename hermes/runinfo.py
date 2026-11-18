"""Run metadata recorded in every run's config.json: pip freeze, GPU, model + dataset revisions.

Revisions are pinned in hermes/revisions.json (tracked in git). The first call resolves the current
Hub commit hashes and writes the file; later calls (all phases, all runs) reuse the pinned hashes, so
data and model are always loaded at the recorded revisions. Delete the file only on purpose.
"""
import json, os, platform, subprocess, sys

MODEL_ID = "Qwen/Qwen3.5-4B"
FINEWEB_ID = "HuggingFaceFW/fineweb"
REV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "revisions.json")


def pinned_revisions():
    if os.path.exists(REV_FILE):
        return json.load(open(REV_FILE))
    from huggingface_hub import HfApi
    api = HfApi()
    rev = dict(model_id=MODEL_ID, model_revision=api.model_info(MODEL_ID).sha,
               fineweb_id=FINEWEB_ID, fineweb_revision=api.dataset_info(FINEWEB_ID).sha)
    json.dump(rev, open(REV_FILE, "w"), indent=2)
    return rev


def run_info():
    import torch
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return dict(**pinned_revisions(), gpu=gpu, python=platform.python_version(),
                runpod_pod_id=os.environ.get("RUNPOD_POD_ID"),
                pytorch_cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), pip_freeze=freeze)
