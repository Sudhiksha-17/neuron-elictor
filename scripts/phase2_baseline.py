"""Phase 2: FineWeb baseline for all neurons of all 32 layers (32 x 9216 = 294,912 neurons).

Streams HuggingFaceFW/fineweb (sample-10BT) at the pinned revision, splits each document into
non-overlapping 64-token chunks (at most --max_chunks_per_doc per document), scores every chunk with
hermes.scoring (one forward, all 32 layers read, early exit after layer 31) and keeps, per neuron:
max, context id + token position of the max, mean and std of the per-context score, and the top-10
context ids (with values and positions). Context tokens are saved, so any context's text can be
decoded (hermes/baseline.py). Per layer: histogram of argmax positions, fraction of neurons with the
max at position 0, and how many neurons have >= 5 of their top-10 from one document.

Usage:
  python scripts/phase2_baseline.py --n_contexts 10000 --tag dry10k    # new run (own timestamped dir)
  python scripts/phase2_baseline.py --resume results/phase2/<dir>      # continue an interrupted run
"""
import os, sys, json, time, argparse, threading, queue, glob
sys.path.insert(0, "/workspace")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
import numpy as np, torch
import datasets, transformers
from hermes import scoring
from hermes.scoring import Scorer, CONTEXT_LEN, TOKENIZE_KWARGS, SCORE_BATCH, N_LAYERS
from hermes.runinfo import run_info

TOPK = 10
SAME_DOC_MIN = 5
DATASET = dict(path="HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)


def chunking_text(cap):
    return (f"Each FineWeb document's 'text' is tokenized with the Qwen/Qwen3.5-4B tokenizer using "
            f"{TOKENIZE_KWARGS} (no BOS/EOS or other special tokens added; literal special-token strings "
            f"in the text are tokenized as ordinary text; no chat template). The token sequence is split into "
            f"non-overlapping {CONTEXT_LEN}-token chunks starting at token 0: tokens[64k:64k+64] for k = 0.."
            f"min(floor(len/64), {cap})-1, i.e. at most {cap} chunks per document (the first {cap}). The trailing "
            f"remainder (< 64 tokens) is dropped; documents shorter than 64 tokens contribute no chunks. "
            f"Documents are taken in stream order, chunks in document order; context_id = running index over "
            f"all chunks. Each context's (doc_idx, fineweb id, chunk_idx) is saved.")


def doc_stream(tok, start_doc, start_chunk, revision, cap, doc_batch=256):
    """Yields (doc_idx, fineweb_id, chunk_idx, tokens[64]) in deterministic stream order."""
    ds = datasets.load_dataset(**DATASET, revision=revision)
    if start_doc:
        ds = ds.skip(start_doc)
    doc_idx = start_doc
    buf = []
    def flush(buf, doc_idx0):
        enc = tok([d["text"] for d in buf], **TOKENIZE_KWARGS).input_ids
        for j, (d, ids) in enumerate(zip(buf, enc)):
            di = doc_idx0 + j
            for k in range(min(len(ids) // CONTEXT_LEN, cap)):
                if di == start_doc and k < start_chunk:
                    continue
                yield di, d["id"], k, ids[k * CONTEXT_LEN:(k + 1) * CONTEXT_LEN]
    for d in ds:
        buf.append(d)
        if len(buf) == doc_batch:
            yield from flush(buf, doc_idx); doc_idx += len(buf); buf = []
    if buf:
        yield from flush(buf, doc_idx)


def producer(tok, start_doc, start_chunk, revision, cap, q, stop):
    try:
        batch = []
        for item in doc_stream(tok, start_doc, start_chunk, revision, cap):
            if stop.is_set(): return
            batch.append(item)
            if len(batch) == SCORE_BATCH:
                q.put(batch); batch = []
                if stop.is_set(): return
        if batch and not stop.is_set(): q.put(batch)
        q.put(None)
    except Exception as e:
        q.put(e)


def sortable_key(vals_f32, n_cols):
    """bf16-valued scores [M, C] -> unique int32 keys: higher score first, ties -> lower column
    (columns are in context-id order, see merge_topk)."""
    b = vals_f32.to(torch.bfloat16).view(torch.int16).int()
    s = torch.where(b < 0, b ^ 0x7FFF, b)                      # monotone in the float value
    col = torch.arange(n_cols, device=vals_f32.device, dtype=torch.int32)
    return (s << 16) | (0xFFFF - col)[None, :]


def merge_topk(top_val, top_ctx, top_pos, mx, pos, ctx0):
    """Merge one batch (mx/pos [M, b], contexts ctx0..ctx0+b-1) into the running top-K [M, K].
    Running entries all have lower context ids than the batch and are sorted (value desc, ctx asc),
    so column order [running K | batch b] is a valid tie-break order."""
    M, b = mx.shape
    v = torch.cat([top_val, mx], 1)
    key = sortable_key(v, v.shape[1])
    _, idx = key.topk(TOPK, dim=1, sorted=True)
    c = torch.cat([top_ctx, (ctx0 + torch.arange(b, device=mx.device))[None, :].expand(M, -1)], 1)
    p = torch.cat([top_pos, pos.to(top_pos.dtype)], 1)
    return v.gather(1, idx), c.gather(1, idx), p.gather(1, idx)


def save_state(run_dir, st):
    tmp = f"{run_dir}/state.pt.tmp"
    torch.save(st, tmp); os.replace(tmp, f"{run_dir}/state.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_contexts", type=int, default=1_000_000)
    ap.add_argument("--max_chunks_per_doc", type=int, default=4)
    ap.add_argument("--checkpoint_every", type=int, default=50_000)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out_dir", default=None, help="exact run dir (must not exist); default results/phase2/<timestamp>_<tag>")
    ap.add_argument("--print_config", action="store_true", help="print the config this run would use (minus env) and exit")
    ap.add_argument("--stop_after", type=int, default=None, help="testing only: exit after the checkpoint at this count (simulates an interruption)")
    a = ap.parse_args()

    if a.resume:
        run_dir = a.resume
        cfg = json.load(open(f"{run_dir}/config.json"))
        st = torch.load(f"{run_dir}/state.pt")
        print(f"resuming {run_dir} at context {st['n_done']} (doc {st['next_doc']}, chunk {st['next_chunk']})")
    else:
        run_dir = a.out_dir or f"/workspace/results/phase2/{time.strftime('%Y%m%d_%H%M%S')}{'_' + a.tag if a.tag else ''}"
        cfg = dict(phase=2, model_id=scoring.MODEL_ID, dtype="bfloat16", layers=list(range(N_LAYERS)),
                   n_neurons_total=N_LAYERS * 9216,
                   outputs=dict(per_neuron=["max", "argmax_ctx", "argmax_pos", "mean_score", "std_score",
                                            "top_val", "top_ctx", "top_pos", "top_max_same_doc"],
                                per_layer=["argmax_pos_hist", "frac_argmax_pos0",
                                           "n_neurons_top10_same_doc_ge5", "top10_max_same_doc_hist"]),
                   n_contexts=a.n_contexts, max_chunks_per_doc=a.max_chunks_per_doc, context_len=CONTEXT_LEN,
                   topk=TOPK, same_doc_min=SAME_DOC_MIN, checkpoint_every=a.checkpoint_every, seed=0,
                   dataset=DATASET, chunking=chunking_text(a.max_chunks_per_doc),
                   tokenize_kwargs=TOKENIZE_KWARGS, score_batch=SCORE_BATCH,
                   activation="channel N of input to layers[L].mlp.down_proj = silu(gate_proj(x))*up_proj(x)",
                   score="max over the 64 token positions; argmax position = first position attaining it",
                   topk_tiebreak="higher value first, ties -> lower context id",
                   spread="mean and std (population) of the per-context score over all contexts",
                   started=time.strftime("%Y-%m-%d %H:%M:%S"))
        if a.print_config:
            print(json.dumps(dict(run_dir=run_dir, **cfg), indent=1)); return
        cfg["env"] = dict(torch=torch.__version__, transformers=transformers.__version__,
                          datasets=datasets.__version__, **run_info())
        os.makedirs(run_dir, exist_ok=False)
        json.dump(cfg, open(f"{run_dir}/config.json", "w"), indent=2)
        st = None
    print("run dir:", run_dir)
    print(json.dumps({k: v for k, v in cfg.items() if k not in ("chunking", "env")}))
    print("env:", {k: v for k, v in cfg["env"].items() if k != "pip_freeze"}, f"+ pip freeze ({len(cfg['env']['pip_freeze'])} pkgs)")
    cap = cfg["max_chunks_per_doc"]

    t_load = time.time()
    sc = Scorer(revision=cfg["env"]["model_revision"])
    NL, NN = N_LAYERS, sc.n_neurons
    M, dev = NL * NN, sc.device
    t_load = time.time() - t_load
    print(f"model loaded in {t_load:.1f}s; kernels: {list(sc.kernels.values())}")

    if st is None:
        st = dict(n_done=0, next_doc=0, next_chunk=0, shard=0,
                  top_val=torch.full((M, TOPK), float("-inf")), top_ctx=torch.full((M, TOPK), -1, dtype=torch.long),
                  top_pos=torch.full((M, TOPK), -1, dtype=torch.int16),
                  sum=torch.zeros(M, dtype=torch.float64), sumsq=torch.zeros(M, dtype=torch.float64),
                  timing=[])
    top_val, top_ctx, top_pos = st["top_val"].to(dev), st["top_ctx"].to(dev), st["top_pos"].to(dev)
    s1, s2 = st["sum"].to(dev), st["sumsq"].to(dev)
    n_done = n_start = st["n_done"]

    q, stop = queue.Queue(maxsize=16), threading.Event()
    producer_thread = threading.Thread(target=producer, args=(sc.tok, st["next_doc"], st["next_chunk"],
                                       cfg["env"]["fineweb_revision"], cap, q, stop), daemon=True)
    producer_thread.start()

    shard = dict(tokens=[], doc_idx=[], doc_id=[], chunk_idx=[])
    t0 = time.time(); t_first = None; t_wait = 0.0; seg_start = n_done
    lengths_full = torch.full((SCORE_BATCH,), CONTEXT_LEN, dtype=torch.long)
    torch.cuda.reset_peak_memory_stats()

    def checkpoint(final=False):
        nonlocal shard, seg_start
        if shard["tokens"]:
            np.savez(f"{run_dir}/shard_{st['shard']:03d}.npz", context_start=seg_start,
                     tokens=np.array(shard["tokens"], dtype=np.uint32), doc_idx=np.array(shard["doc_idx"], dtype=np.int64),
                     chunk_idx=np.array(shard["chunk_idx"], dtype=np.int32), doc_id=np.array(shard["doc_id"]))
            st["shard"] += 1
        st.update(n_done=n_done, top_val=top_val.cpu(), top_ctx=top_ctx.cpu(), top_pos=top_pos.cpu(),
                  sum=s1.cpu(), sumsq=s2.cpu())
        st["timing"].append(dict(n_done=n_done, wall=time.time(), elapsed_this_session=time.time() - t0))
        save_state(run_dir, st)
        shard = dict(tokens=[], doc_idx=[], doc_id=[], chunk_idx=[]); seg_start = n_done
        print(f"  checkpoint: {n_done} contexts saved{' (final)' if final else ''}", flush=True)

    leftover = []
    while n_done < cfg["n_contexts"]:
        tw = time.time()
        batch = leftover or q.get()
        leftover = []
        t_wait += time.time() - tw
        if batch is None: print("dataset exhausted"); break
        if isinstance(batch, Exception): raise batch
        # never cross a checkpoint boundary or the target inside one batch
        next_ck = (n_done // cfg["checkpoint_every"] + 1) * cfg["checkpoint_every"]
        take = min(len(batch), cfg["n_contexts"] - n_done, next_ck - n_done)
        batch, leftover = batch[:take], batch[take:]
        ids = torch.tensor([b[3] for b in batch], dtype=torch.long)
        mx, pos = sc.score_ids(ids, lengths_full[:len(batch)], range(NL))     # [b, 32, 9216]
        mx = mx.reshape(len(batch), M).T; pos = pos.reshape(len(batch), M).T  # [M, b]
        top_val, top_ctx, top_pos = merge_topk(top_val, top_ctx, top_pos, mx, pos, n_done)
        md = mx.double(); s1 += md.sum(1); s2 += (md ** 2).sum(1); del md
        for b in batch:
            shard["doc_idx"].append(b[0]); shard["doc_id"].append(b[1]); shard["chunk_idx"].append(b[2]); shard["tokens"].append(b[3])
        n_done += len(batch)
        last = batch[-1]
        st["next_doc"], st["next_chunk"] = last[0], last[2] + 1
        if leftover:
            st["next_doc"], st["next_chunk"] = leftover[0][0], leftover[0][2]
        if t_first is None:
            torch.cuda.synchronize(); t_first = time.time() - t0
        if n_done % cfg["checkpoint_every"] == 0 or n_done >= cfg["n_contexts"]:
            checkpoint(final=n_done >= cfg["n_contexts"])
            if a.stop_after and n_done >= a.stop_after:
                print(f"--stop_after {a.stop_after}: exiting to simulate an interruption"); os._exit(3)
        elif (n_done // SCORE_BATCH) % 20 == 0:
            el = time.time() - t0
            print(f"  {n_done} contexts, {(n_done - n_start) / el:.0f} ctx/s, data wait {t_wait:.1f}s, "
                  f"peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    # stop the data thread cleanly (otherwise it is still streaming at interpreter shutdown -> abort)
    stop.set()
    while producer_thread.is_alive():
        try: q.get(timeout=0.5)
        except queue.Empty: pass
    if n_done % cfg["checkpoint_every"] != 0 and n_done < cfg["n_contexts"]:
        checkpoint(final=True)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak_alloc, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()

    # ---- final outputs ----
    print("writing final outputs ...", flush=True)
    shards = sorted(glob.glob(f"{run_dir}/shard_*.npz"))
    toks, didx, dids, cidx = [], [], [], []
    for f in shards:
        z = np.load(f); toks.append(z["tokens"]); didx.append(z["doc_idx"]); dids.append(z["doc_id"]); cidx.append(z["chunk_idx"])
    toks, didx, dids, cidx = np.concatenate(toks), np.concatenate(didx), np.concatenate(dids), np.concatenate(cidx)
    assert len(toks) == n_done, (len(toks), n_done)
    np.savez(f"{run_dir}/contexts.npz", tokens=toks, doc_idx=didx, doc_id=dids, chunk_idx=cidx)
    for f in shards: os.remove(f)

    tv, tc, tp = top_val.cpu(), top_ctx.cpu(), top_pos.cpu()
    mean = (s1 / n_done).cpu(); std = ((s2 / n_done).cpu() - mean ** 2).clamp_min(0).sqrt()
    R = lambda x: x.reshape(NL, NN, *x.shape[1:])                         # -> [layer, neuron, ...]
    # per-layer statistics
    amax_pos = R(tp[:, 0].long())                                         # [32, 9216]
    pos_hist = torch.stack([torch.bincount(amax_pos[L], minlength=CONTEXT_LEN) for L in range(NL)])
    frac_pos0 = (amax_pos == 0).double().mean(1)
    top_docs = torch.from_numpy(didx)[tc.clamp_min(0)]                    # [M, 10] doc index of each top-10 context
    srt = top_docs.sort(1).values
    run = torch.ones_like(srt)
    for k in range(1, TOPK):                                              # longest run of equal doc ids
        run[:, k] = torch.where(srt[:, k] == srt[:, k - 1], run[:, k - 1] + 1, 1)
    best = run.max(1).values
    same_doc_hist = torch.stack([torch.bincount(R(best)[L], minlength=TOPK + 1) for L in range(NL)])  # [32, 11]
    n_same_doc = (R(best) >= SAME_DOC_MIN).sum(1)
    torch.save(dict(layers=list(range(NL)), n_neurons=NN, n_contexts=n_done,
                    max=R(tv[:, 0]).clone(), argmax_ctx=R(tc[:, 0]).clone(), argmax_pos=R(tp[:, 0]).clone(),
                    mean_score=R(mean).float(), std_score=R(std).float(),
                    top_val=R(tv), top_ctx=R(tc), top_pos=R(tp), top_max_same_doc=R(best).to(torch.int8),
                    argmax_pos_hist=pos_hist, frac_argmax_pos0=frac_pos0.float(),
                    n_neurons_top10_same_doc_ge5=n_same_doc), f"{run_dir}/baseline.pt")
    os.remove(f"{run_dir}/state.pt")
    vmax = R(tv[:, 0])
    _, chunks_per_doc = np.unique(didx, return_counts=True)
    summary = dict(n_contexts=n_done, n_docs_contributing=int(len(chunks_per_doc)),
                   n_docs_streamed=int(didx.max()) + 1,   # incl. docs < 64 tokens that contribute no chunk
                   chunks_per_doc_hist=np.bincount(chunks_per_doc, minlength=cap + 1).tolist(),
                   elapsed_s=elapsed, model_load_s=t_load, first_batch_s=t_first, data_wait_s=t_wait,
                   steady_ctx_per_s=(n_done - n_start - SCORE_BATCH) / max(elapsed - t_first, 1e-9),
                   n_contexts_this_session=n_done - n_start,
                   gpu_peak_allocated_gib=peak_alloc / 2**30, gpu_peak_reserved_gib=peak_reserved / 2**30,
                   output_bytes={os.path.basename(f): os.path.getsize(f) for f in glob.glob(f"{run_dir}/*")},
                   per_layer=dict(
                       max_quantiles={str(q_): vmax.quantile(q_, dim=1).tolist() for q_ in (0, .01, .1, .5, .9, .99, 1)},
                       frac_argmax_pos0=frac_pos0.tolist(),
                       n_neurons_top10_same_doc_ge5=n_same_doc.tolist(),
                       top10_max_same_doc_hist=same_doc_hist.tolist(),
                       argmax_pos_hist=pos_hist.tolist()),
                   finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(summary, open(f"{run_dir}/summary.json", "w"), indent=1)
    s = {k: v for k, v in summary.items() if k != "per_layer"}
    print(json.dumps(s, indent=1))
    print("frac argmax at pos 0 per layer:", [round(x, 3) for x in frac_pos0.tolist()])
    print("neurons with >=5 of top-10 from one doc, per layer:", n_same_doc.tolist())
    print("median neuron max per layer:", [round(x, 2) for x in vmax.median(1).values.tolist()])
    print("saved:", run_dir)


if __name__ == "__main__":
    main()
    # All outputs are written and closed. Exit hard: the HF streaming client can still have a prefetch
    # thread alive, which otherwise aborts the interpreter at shutdown (exit 134) after a successful run.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
