import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from sources import ROOT, SOURCES, SPECIALS, batch_texts, load_config, source_files

PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


def sample_texts(mix, total_chars, max_doc_chars, files_per_source, seed):
    for name, w in mix.items():
        budget = int(total_chars * w)
        per_file = budget // files_per_source + 1
        used = 0
        t0 = time.time()
        for path in source_files(name, seed):
            if used >= budget:
                break
            got = 0
            seen = set()
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=SOURCES[name][1]):
                texts = [t[:max_doc_chars] for t in batch_texts(name, batch, seen)]
                n = sum(map(len, texts))
                got += n
                used += n
                yield texts
                if got >= per_file or used >= budget:
                    break
        print(f"{name}: {used / 1e6:.1f}M chars in {time.time() - t0:.0f}s", flush=True)


def build(vocab_size):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    return tok, trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "Train" / "config.toml"))
    ap.add_argument("--chars", type=float, default=1.5e9)
    ap.add_argument("--max_doc_chars", type=int, default=20000)
    ap.add_argument("--files_per_source", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "data" / "tokenizer" / "tokenizer.json"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    tok, trainer = build(cfg["model"]["vocab_size"])
    t0 = time.time()
    it = sample_texts(cfg["data"]["mix"], args.chars, args.max_doc_chars, args.files_per_source, cfg["data"]["seed"])
    tok.train_from_iterator(it, trainer=trainer)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    print(f"vocab {tok.get_vocab_size()} saved {out} in {time.time() - t0:.0f}s", flush=True)
    sample = "The quick brown fox didn't jump over 12345 lazy dogs.\n\nIt was, however, rather tired."
    enc = tok.encode(sample)
    print(len(sample), len(enc.ids), enc.tokens)
    assert tok.decode(enc.ids) == sample


if __name__ == "__main__":
    main()
