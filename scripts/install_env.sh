#!/bin/bash
# Install the pinned Python environment on a freshly wiped container and download the pinned model.
# Needs network. Called as step 0 of scripts/gpu_session.sh (with HF_HUB_OFFLINE unset); safe to re-run.
#   - torch 2.8.0+cu129 (the version recorded in every run config since 2026-11-16; cu129 wheel index),
#     held fixed by a constraints file so no later install can swap it
#   - transformers>=5.9, accelerate, peft, trl, datasets
#   - flash-linear-attention==0.5.2 (+ fla-core==0.5.2), --no-build-isolation
#     (0.6.0 fails to import with triton 3.4, see reports/20261116_162500_pod_setup.md)
#   - causal-conv1d from git at v1.7.0 = cd81f04 (the commit recorded in earlier pip freezes), --no-build-isolation;
#     its setup.py fetches the prebuilt cu12/torch2.8/cp312 wheel from the GitHub release
#   - Qwen/Qwen3.5-4B at the revision pinned in hermes/revisions.json, into HF_HOME=/workspace/hf_cache
# Exits non-zero on the first failure.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export HF_HOME=/workspace/hf_cache PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p "$HF_HOME"
TORCH_INDEX=https://download.pytorch.org/whl/cu129
CAUSAL_CONV1D_REF=cd81f0413cad2fc1e6f17e785ac39f59aae690cd   # tag v1.7.0
CONS=$(mktemp); trap 'rm -f "$CONS"' EXIT
printf '%s\n' "torch==2.8.0+cu129" "flash-linear-attention==0.5.2" "fla-core==0.5.2" > "$CONS"
say() { echo "[install_env $(date '+%F %T')] $*"; }
retry() { local i; for i in 1 2 3; do "$@" && return 0; say "attempt $i failed: $*"; sleep 20; done; return 1; }

say "python $(python -V 2>&1), pip $(pip -V | cut -d' ' -f2)"
say "1/6 torch 2.8.0+cu129"
retry pip install "torch==2.8.0+cu129" --index-url "$TORCH_INDEX"
say "2/6 build tools"
retry pip install -c "$CONS" packaging ninja wheel setuptools
say "3/6 transformers>=5.9 accelerate peft trl datasets"
retry pip install -c "$CONS" "transformers>=5.9" accelerate peft trl datasets
say "4/6 flash-linear-attention==0.5.2"
retry pip install -c "$CONS" --no-build-isolation "flash-linear-attention==0.5.2" "fla-core==0.5.2"
say "5/6 causal-conv1d @ $CAUSAL_CONV1D_REF"
retry pip install -c "$CONS" --no-build-isolation \
  "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d@$CAUSAL_CONV1D_REF"
python - <<'EOF'
import torch, transformers, accelerate, peft, trl, datasets, fla, causal_conv1d
assert torch.__version__ == "2.8.0+cu129", torch.__version__
from packaging.version import Version
assert Version(transformers.__version__) >= Version("5.9"), transformers.__version__
for m in (torch, transformers, accelerate, peft, trl, datasets, fla, causal_conv1d):
    print(f"  {m.__name__:14s} {getattr(m, '__version__', '?')}")
print("  cuda available:", torch.cuda.is_available(), torch.version.cuda)
EOF
say "6/6 model download (pinned revision)"
retry python -c '
import json, os
from huggingface_hub import snapshot_download
rev = json.load(open("hermes/revisions.json"))
p = snapshot_download(rev["model_id"], revision=rev["model_revision"])
print("  model snapshot:", p, sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p)) // 2**20, "MiB")
'
say "pip freeze:"; pip freeze
say "done"
