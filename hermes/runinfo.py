"""Run metadata recorded in every run's config.json: pip freeze, GPU, data/model revisions."""
import os, subprocess, sys, platform, functools
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

FINEWEB_REPO = "HuggingFaceFW/fineweb"
MODEL_REPO = "Qwen/Qwen3.5-4B"


@functools.lru_cache(None)
def hub_revision(repo_id, repo_type):
    """Current commit sha of a Hub repo (resolved once per process)."""
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.dataset_info(repo_id) if repo_type == "dataset" else api.model_info(repo_id)
    return info.sha


def run_info(fineweb_revision=None, model_revision=None):
    import torch
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    return dict(
        gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        cuda=torch.version.cuda, python=platform.python_version(), platform=platform.platform(),
        fineweb_repo=FINEWEB_REPO,
        fineweb_revision=fineweb_revision or hub_revision(FINEWEB_REPO, "dataset"),
        model_repo=MODEL_REPO,
        model_revision=model_revision or hub_revision(MODEL_REPO, "model"),
        pip_freeze=freeze.strip().splitlines(),
    )
