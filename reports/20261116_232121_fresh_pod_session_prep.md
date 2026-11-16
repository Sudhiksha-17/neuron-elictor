# Fresh pod prep + GPU session launch (20261116_232121)

GPU: NVIDIA GeForce RTX 5090, 32607 MiB. Pod v6sebqua04e1qe.

## Data
- Downloaded from HF `gray171/hermes-baseline` (dataset revision 06aa544) into `data/` with curl.
- sha256 verified: `baseline.pt` 70cde47f…649817 OK, `contexts.npz` d519c23a…7143ff2f OK.
- `pilots.json` is not in the dataset; copied from `results/pilots/20261109_213918_full1M_pilots/pilots.json` (cmp identical).

## Script changes
- New `scripts/install_env.sh` (step 0 of gpu_session.sh, network allowed): torch==2.8.0+cu129 (cu129 index; fixed via
  constraints), transformers>=5.9, accelerate, peft, trl, datasets, flash-linear-attention==0.5.2 + fla-core==0.5.2
  (--no-build-isolation), causal-conv1d from git at v1.7.0 = cd81f04 (the commit in earlier pip freezes;
  --no-build-isolation; a prebuilt cu12/torch2.8/cp312 wheel exists for it, so the pod's CUDA 13.2 nvcc should not be
  needed), then the pinned model (851bf6e) into HF_HOME=/workspace/hf_cache. Each install retried 3x. On failure the
  session writes its report and stops the pod. HF_HUB_OFFLINE=1 is exported only after step 0.
- New `scripts/ensure_watchdog.sh`: deletes `logs/watchdog.disable`, starts `idle_watchdog.sh` (setsid nohup) unless
  one is running. Called at the start of `gpu_session.sh` and `grpo_pipeline.sh`.
- `grpo_pipeline.sh` real-run stop path no longer touches `logs/watchdog.disable`.
- Note: install/model download is GPU-idle time; the watchdog only stops after 60 min idle.
