"""Phase 2: FineWeb baseline for all 9,216 neurons of layer 15.

Streams HuggingFaceFW/fineweb (sample-10BT), splits each document into non-overlapping
64-token chunks, scores every chunk with hermes.scoring (early exit after layer 15) and keeps,
per neuron: max activation, the context id + token position of the max, and the top-10 contexts.

Usage:
  python scripts/phase2_baseline.py --n_contexts 10000            # new run (own timestamped dir)
  python scripts/phase2_baseline.py --resume results/phase2/<dir> # continue an interrupted run
"""
import os, sys, json, time, argparse, threading, queue, platform, glob
sys.path.insert(0, "/workspace")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
import numpy as np, torch
import datasets, transformers
from hermes import scoring
from hermes.scoring import NeuronScorer, CONTEXT_LEN, TOKENIZE_KWARGS, SCORE_BATCH
from hermes.runinfo import run_info

TOPK = 10
DATASET = dict(path="HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
CHUNKING = (f"Each FineWeb document's 'text' is tokenized with the Qwen/Qwen3.5-4B tokenizer using "
            f"{TOKENIZE_KWARGS} (no BOS/EOS or other special tokens added; literal special-token strings "
            f"in the text are tokenized as ordinary text; no chat template). The token sequence is split into "
            f"non-overlapping {CONTEXT_LEN}-token chunks starting at token 0: tokens[64k:64k+64] for k = 0.."
            f"floor(len/64)-1. The trailing remainder (< 64 tokens) is dropped; documents shorter than 64 "
            f"tokens contribute no chunks. Documents are taken in stream order, chunks in document order; "
            f"context_id = running index over all chunks. Each context's (doc_idx, fineweb id, chunk_idx) is saved.")


def doc_stream(tok, start_doc, start_chunk, revision, doc_batch=256):
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
            for k in range(len(ids) // CONTEXT_LEN):
                if di == start_doc and k < start_chunk:
                    continue
                yield di, d["id"], k, ids[k * CONTEXT_LEN:(k + 1) * CONTEXT_LEN]
    for d in ds:
        buf.append(d)
        if len(buf) == doc_batch:
            yield from flush(buf, doc_idx); doc_idx += len(buf); buf = []
    if buf:
        yield from flush(buf, doc_idx)


def producer(tok, start_doc, start_chunk, revision, q, stop):
    try:
        batch = []
        for item in doc_stream(tok, start_doc, start_chunk, revision):
            if stop.is_set(): return
            batch.append(item)
            if len(batch) == SCORE_BATCH:
                q.put(batch); batch = []
                if stop.is_set(): return
        if batch and not stop.is_set(): q.put(batch)
        q.put(None)
    except Exception as e:
        q.put(e)


def save_state(run_dir, st):
    tmp = f"{run_dir}/state.pt.tmp"
    torch.save(st, tmp); os.replace(tmp, f"{run_dir}/state.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_contexts", type=int, default=1_000_000)
    ap.add_argument("--checkpoint_every", type=int, default=50_000)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--stop_after", type=int, default=None, help="testing only: exit after the checkpoint at this count (simulates an interruption)")
    a = ap.parse_args()

    if a.resume:
        run_dir = a.resume
        cfg = json.load(open(f"{run_dir}/config.json"))
        st = torch.load(f"{run_dir}/state.pt")
        print(f"resuming {run_dir} at context {st['n_done']} (doc {st['next_doc']}, chunk {st['next_chunk']})")
    else:
        run_dir = f"/workspace/results/phase2/{time.strftime('%Y%m%d_%H%M%S')}{'_' + a.tag if a.tag else ''}"
        os.makedirs(run_dir, exist_ok=False)
        cfg = dict(phase=2, model_id=scoring.MODEL_ID, dtype="bfloat16", layer=a.layer,
                   n_contexts=a.n_contexts, context_len=CONTEXT_LEN, topk=TOPK,
                   checkpoint_every=a.checkpoint_every, seed=0, dataset=DATASET, chunking=CHUNKING,
                   tokenize_kwargs=TOKENIZE_KWARGS, score_batch=SCORE_BATCH,
                   activation="channel N of input to mlp.down_proj = silu(gate_proj(x))*up_proj(x)",
                   score="max over the 64 token positions",
                   env=dict(torch=torch.__version__, transformers=transformers.__version__,
                            datasets=datasets.__version__, **run_info()),
                   started=time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump(cfg, open(f"{run_dir}/config.json", "w"), indent=2)
        st = None
    print("run dir:", run_dir)
    print(json.dumps({k: v for k, v in cfg.items() if k not in ("chunking", "env")}, indent=1))
    print("env:", {k: v for k, v in cfg["env"].items() if k != "pip_freeze"}, f"+ pip freeze ({len(cfg['env']['pip_freeze'])} pkgs)")

    t_load = time.time()
    sc = NeuronScorer(cfg["layer"], revision=cfg["env"]["model_revision"])
    cfg_kernels = sc.kernels
    N, dev = sc.n_neurons, sc.model.device
    t_load = time.time() - t_load
    print(f"model loaded in {t_load:.1f}s; kernels: {list(cfg_kernels.values())}")

    if st is None:
        st = dict(n_done=0, next_doc=0, next_chunk=0, shard=0,
                  top_val=torch.full((N, TOPK), float("-inf")), top_ctx=torch.full((N, TOPK), -1, dtype=torch.long),
                  top_pos=torch.full((N, TOPK), -1, dtype=torch.long),
                  sum=torch.zeros(N, dtype=torch.float64), sumsq=torch.zeros(N, dtype=torch.float64),
                  timing=[])
    top_val, top_ctx, top_pos = st["top_val"].to(dev), st["top_ctx"].to(dev), st["top_pos"].to(dev)
    s1, s2 = st["sum"].to(dev), st["sumsq"].to(dev)
    n_done = n_start = st["n_done"]

    q, stop = queue.Queue(maxsize=16), threading.Event()
    producer_thread = threading.Thread(target=producer, args=(sc.tok, st["next_doc"], st["next_chunk"], cfg["env"]["fineweb_revision"], q, stop), daemon=True)
    producer_thread.start()

    shard = dict(tokens=[], doc_idx=[], doc_id=[], chunk_idx=[])
    t0 = time.time(); t_first = None; t_wait = 0.0; seg_start = n_done
    lengths_full = torch.full((SCORE_BATCH,), CONTEXT_LEN, dtype=torch.long)

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
        mx, pos = sc.score_ids(ids, lengths_full[:len(batch)])          # [b, N]
        ctx = torch.arange(n_done, n_done + len(batch), device=dev)
        v = torch.cat([top_val, mx.T], 1)
        c = torch.cat([top_ctx, ctx[None, :].expand(N, -1)], 1)
        p = torch.cat([top_pos, pos.T], 1)
        # deterministic top-k: highest value first, ties broken by lowest context id
        o1 = c.argsort(dim=1, stable=True)
        v, c, p = v.gather(1, o1), c.gather(1, o1), p.gather(1, o1)
        o2 = v.argsort(dim=1, descending=True, stable=True)[:, :TOPK]
        top_val, top_ctx, top_pos = v.gather(1, o2), c.gather(1, o2), p.gather(1, o2)
        s1 += mx.double().sum(0); s2 += (mx.double() ** 2).sum(0)
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
        elif (n_done // SCORE_BATCH) % 40 == 0:
            el = time.time() - t0
            print(f"  {n_done} contexts, {(n_done - n_start) / el:.0f} ctx/s, data wait {t_wait:.1f}s", flush=True)
    # stop the data thread cleanly (otherwise it is still streaming at interpreter shutdown -> abort)
    stop.set()
    while producer_thread.is_alive():
        try: q.get(timeout=0.5)
        except queue.Empty: pass
    if n_done % cfg["checkpoint_every"] != 0 and n_done < cfg["n_contexts"]:
        checkpoint(final=True)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    # ---- final outputs ----
    print("writing final outputs ...", flush=True)
    shards = sorted(glob.glob(f"{run_dir}/shard_*.npz"))
    toks, didx, dids, cidx = [], [], [], []
    for f in shards:
        z = np.load(f); toks.append(z["tokens"]); didx.append(z["doc_idx"]); dids.append(z["doc_id"]); cidx.append(z["chunk_idx"])
    toks, didx, dids, cidx = np.concatenate(toks), np.concatenate(didx), np.concatenate(dids), np.concatenate(cidx)
    assert len(toks) == n_done, (len(toks), n_done)
    tv, tc, tp = top_val.cpu(), top_ctx.cpu(), top_pos.cpu()
    mean = (s1 / n_done).cpu(); std = ((s2 / n_done).cpu() - mean ** 2).clamp_min(0).sqrt()
    torch.save(dict(max=tv[:, 0].clone(), argmax_ctx=tc[:, 0].clone(), argmax_pos=tp[:, 0].clone(),
                    top_val=tv, top_ctx=tc, top_pos=tp, mean_score=mean.float(), std_score=std.float(),
                    n_contexts=n_done, layer=cfg["layer"]), f"{run_dir}/baseline.pt")
    tok = sc.tok
    with open(f"{run_dir}/top_contexts.jsonl", "w") as fh:
        for n in range(N):
            tops = []
            for r in range(TOPK):
                c, pz = int(tc[n, r]), int(tp[n, r])
                t = toks[c].tolist()
                tops.append(dict(rank=r, context_id=c, act=float(tv[n, r]), pos=pz,
                                 token_at_max=tok.decode([t[pz]]), doc_idx=int(didx[c]), doc_id=str(dids[c]),
                                 chunk_idx=int(cidx[c]), text=tok.decode(t)))
            fh.write(json.dumps(dict(neuron=n, max=float(tv[n, 0]), argmax_ctx=int(tc[n, 0]), argmax_pos=int(tp[n, 0]),
                                     mean_score=float(mean[n]), std_score=float(std[n]), top10=tops)) + "\n")
    summary = dict(n_contexts=n_done, n_docs_touched=int(didx.max()) + 1, elapsed_s=elapsed, model_load_s=t_load,
                   first_batch_s=t_first, data_wait_s=t_wait,
                   steady_ctx_per_s=(n_done - n_start - SCORE_BATCH) / max(elapsed - t_first, 1e-9),
                   n_contexts_this_session=n_done - n_start,
                   max_quantiles={q_: float(tv[:, 0].quantile(q_)) for q_ in (0, .01, .1, .5, .9, .99, 1)},
                   finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(summary, open(f"{run_dir}/summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=1))
    print("saved:", run_dir)


if __name__ == "__main__":
    main()
