"""Hermes package. Sets the CUDA allocator config before torch touches the GPU (decided 2026-11-18 for the
RTX 4090: without expandable segments the micro_bs-32 training step OOMs after generation). setdefault, so an
explicit PYTORCH_CUDA_ALLOC_CONF from the caller wins; the value in effect is recorded by hermes.runinfo.run_info()."""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
