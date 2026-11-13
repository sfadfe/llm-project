import glob
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "smol-smoltalk" / "data"
OUT = ROOT / "data" / "sft"
ROW = 2049
N_VAL = 2000
OPEN = 64
RL_SOURCES = {"smol-magpie-ultra-short", "everyday-conversations", "openhermes-50k"}


class Chat:
    def __init__(self, tok_path=ROOT / "data" / "tokenizer" / "tokenizer.json"):
        self.tok = Tokenizer.from_file(str(tok_path))
        self.tok.encode_special_tokens = True
        sid = self.tok.token_to_id
        self.eot, self.pad, self.start, self.end = sid("<|endoftext|>"), sid("<|pad|>"), sid("<|im_start|>"), sid("<|im_end|>")
        self.nl = self.enc("\n")
        self.role = {r: self.enc(r + "\n") for r in ("system", "user", "assistant")}

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False).ids

    def render(self, messages, contents=None, generation=False):
        ids, flags = [], []
        for i, m in enumerate(messages):
            c = contents[i] if contents is not None else self.enc(m["content"])
            head = [self.start] + self.role[m["role"]]
            ids += head
            flags += [0] * len(head)
            body = c + [self.end]
            ids += body
            flags += [int(m["role"] == "assistant")] * len(body)
            ids += self.nl
            flags += [0] * len(self.nl)
        if generation:
            head = [self.start] + self.role["assistant"]
            ids += head
            flags += [0] * len(head)
        return ids, flags

    def decode(self, ids):
        return self.tok.decode(ids)


def encode_convs(chat, rows):
    flat = [m["content"] for r in rows for m in r["messages"]]
    enc = [e.ids for e in chat.tok.encode_batch(flat, add_special_tokens=False)]
    out, j = [], 0
    for r in rows:
        n = len(r["messages"])
        ids, flags = chat.render(r["messages"], enc[j:j + n])
        j += n
        out.append((ids + [chat.eot], flags + [0]))
    return out


def pack(convs, pad):
    rows, open_ = [], []
    for ids, flags in convs:
        n = len(ids)
        for r in open_:
            if len(r[0]) + n <= ROW:
                r[0].extend(ids)
                r[1].extend(flags)
                break
        else:
            open_.append((list(ids), list(flags)))
            if len(open_) > OPEN:
                open_.sort(key=lambda r: len(r[0]))
                rows.append(open_.pop())
    rows += open_
    tok = np.full((len(rows), ROW), pad, dtype=np.uint16)
    flg = np.zeros((len(rows), ROW), dtype=np.uint8)
    for i, (a, b) in enumerate(rows):
        tok[i, :len(a)] = a
        flg[i, :len(b)] = b
    return tok, flg


def valid(r):
    roles = [m["role"] for m in r["messages"]]
    return "assistant" in roles and all(m["content"] is not None for m in r["messages"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    chat = Chat()
    rng = random.Random(0)
    train = [r for f in sorted(glob.glob(str(RAW / "train-*.parquet"))) for r in pq.read_table(f).to_pylist()]
    test = pq.read_table(str(RAW / "test-00000-of-00001.parquet")).to_pylist()
    rng.shuffle(train)
    rng.shuffle(test)
    val, rest = test[:N_VAL], test[N_VAL:]
    stats = {}
    for name, rows in (("train", train), ("val", val)):
        rows = [r for r in rows if valid(r)]
        convs = []
        for i in range(0, len(rows), 20000):
            convs += encode_convs(chat, rows[i:i + 20000])
        kept = [c for c in convs if len(c[0]) <= ROW]
        tok, flg = pack(kept, chat.pad)
        tok.tofile(OUT / f"{name}.bin")
        flg.tofile(OUT / f"{name}_mask.bin")
        stats[name] = {"convs": len(rows), "dropped_long": len(convs) - len(kept), "rows": len(tok),
                       "tokens": int(sum(len(c[0]) for c in kept)), "trained_tokens": int(flg.sum()),
                       "fill": float(sum(len(c[0]) for c in kept) / tok.size)}
        print(name, stats[name], flush=True)
    n_rl = 0
    with open(OUT / "rl_prompts.jsonl", "w") as f:
        for r in rest:
            msgs = r["messages"]
            if r["source"] not in RL_SOURCES or not msgs:
                continue
            sys = msgs[0]["content"] if msgs[0]["role"] == "system" else None
            user = msgs[1 if sys is not None else 0]
            if user["role"] != "user" or len(chat.enc(user["content"])) > 160:
                continue
            f.write(json.dumps({"system": sys, "user": user["content"], "source": r["source"]}) + "\n")
            n_rl += 1
    stats["rl_prompts"] = n_rl
    stats["row"] = ROW
    stats["pad"], stats["eot"], stats["im_end"] = chat.pad, chat.eot, chat.end
    print("rl_prompts", n_rl)
    json.dump(stats, open(OUT / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
