import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from sources import ROOT, SOURCES, batch_texts, load_config, source_files

BATCH_ROWS = {"gutenberg": 16}

_tok = None
_counts = None
_targets = None


def init_worker(tok_path, counts, targets):
    global _tok, _counts, _targets
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from tokenizers import Tokenizer
    _tok = Tokenizer.from_file(tok_path)
    _tok.encode_special_tokens = True
    _counts = counts
    _targets = targets


def encode_file(job):
    si, name, path, out_dir, val_tokens = job
    if _counts[si] >= _targets[si]:
        return si, None, 0, 0
    eot = _tok.token_to_id("<|endoftext|>")
    p = Path(path)
    stem = ("_".join(p.parts[-3:]) if name == "dclm" else p.name).replace(".parquet", "")
    train_path = out_dir / f"{stem}.bin"
    val_path = out_dir / "val.bin"
    ft = open(str(train_path) + ".tmp", "wb")
    fv = open(str(val_path) + ".tmp", "wb") if val_tokens else None
    n_train = n_val = 0
    seen = set()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS.get(name, 512), columns=SOURCES[name][1]):
        train_docs = []
        for text in batch_texts(name, batch, seen):
            ids = _tok.encode(text, add_special_tokens=False).ids
            ids.append(eot)
            ids = np.asarray(ids, dtype=np.uint16)
            if n_val < val_tokens:
                ids.tofile(fv)
                n_val += len(ids)
            else:
                train_docs.append(ids)
        if train_docs:
            arr = np.concatenate(train_docs)
            arr.tofile(ft)
            n_train += len(arr)
            with _counts.get_lock():
                _counts[si] += len(arr)
            if _counts[si] >= _targets[si]:
                break
    ft.close()
    os.replace(str(train_path) + ".tmp", train_path)
    if fv:
        fv.close()
        os.replace(str(val_path) + ".tmp", val_path)
    return si, train_path.name, n_train, n_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "Train" / "config.toml"))
    ap.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer" / "tokenizer.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "bin"))
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--margin", type=float, default=1.03)
    ap.add_argument("--val_tokens", type=int, default=1_000_000)
    ap.add_argument("--total_tokens", type=float, default=None)
    ap.add_argument("--max_files", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    mix = cfg["data"]["mix"]
    total = args.total_tokens or cfg["data"]["total_tokens"]
    names = list(mix)
    out = Path(args.out)
    counts = mp.Array("q", len(names))
    targets = mp.Array("q", [int(total * mix[n] * args.margin) for n in names], lock=False)
    per_source = []
    for si, name in enumerate(names):
        files = source_files(name, cfg["data"]["seed"])
        if args.max_files:
            files = files[:args.max_files]
        assert files, f"no files for {name}"
        (out / name).mkdir(parents=True, exist_ok=True)
        per_source.append([(si, name, f, out / name, args.val_tokens if i == 0 else 0) for i, f in enumerate(files)])
    jobs = []
    for i in range(max(map(len, per_source))):
        for js in per_source:
            if i < len(js):
                jobs.append(js[i])
    meta = {"tokenizer": args.tokenizer, "dtype": "uint16", "sources": {n: {"train": [], "val": None} for n in names}}
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers, initializer=init_worker, initargs=(args.tokenizer, counts, targets)) as pool:
        for si, fname, n_train, n_val in pool.imap_unordered(encode_file, jobs):
            if fname is None:
                continue
            m = meta["sources"][names[si]]
            m["train"].append([fname, n_train])
            if n_val:
                m["val"] = ["val.bin", n_val]
            done = ", ".join(f"{n}={counts[i] / 1e9:.2f}/{targets[i] / 1e9:.2f}B" for i, n in enumerate(names))
            print(f"[{time.time() - t0:.0f}s] {names[si]}/{fname} {n_train / 1e6:.1f}M | {done}", flush=True)
    for n in names:
        meta["sources"][n]["train"].sort()
        meta["sources"][n]["train_tokens"] = sum(c for _, c in meta["sources"][n]["train"])
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    meta["eot"] = tok.token_to_id("<|endoftext|>")
    meta["vocab_size"] = tok.get_vocab_size()
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
