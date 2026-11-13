import glob
import random
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

SOURCES = {
    "dclm": ("dclm/**/*.parquet", ["text"]),
    "fineweb_edu": ("fineweb-edu/sample/10BT/*.parquet", ["text"]),
    "wikipedia": ("wikipedia/20231101.en/*.parquet", ["title", "text"]),
    "gutenberg": ("gutenberg/data/en-*.parquet", ["id", "text"]),
    "cosmopedia": ("smollm-corpus/cosmopedia-v2/*.parquet", ["text"]),
}

SPECIALS = ["<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>"] + [f"<|reserved_{i}|>" for i in range(12)]

PG_START = re.compile(r"\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*", re.I)
PG_END = re.compile(r"\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG|End of (?:the )?Project Gutenberg", re.I)
PARA = re.compile(r"\n[ \t]*\n+")


def load_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def source_files(name, seed=0):
    pattern, _ = SOURCES[name]
    files = sorted(glob.glob(str(RAW / pattern), recursive=True))
    random.Random(f"{seed}-{name}").shuffle(files)
    return files


def clean_gutenberg(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    m = PG_START.search(text)
    if m:
        text = text[m.end():]
    m = PG_END.search(text)
    if m:
        text = text[:m.start()]
    out = []
    for para in PARA.split(text):
        lines = [l.strip() for l in para.split("\n") if l.strip()]
        if not lines:
            continue
        body = lines[:-1]
        if body and sum(len(l) >= 55 for l in body) * 2 >= len(body):
            out.append(" ".join(lines))
        else:
            out.append("\n".join(lines))
    return "\n\n".join(out)


def batch_texts(name, batch, seen):
    cols = batch.to_pydict()
    if name == "wikipedia":
        texts = [f"{t}\n\n{x}" for t, x in zip(cols["title"], cols["text"])]
    elif name == "gutenberg":
        texts = []
        for i, x in zip(cols["id"], cols["text"]):
            book = str(i).split("-")[0]
            if book not in seen:
                seen.add(book)
                texts.append(clean_gutenberg(x))
    else:
        texts = cols["text"]
    return [t.strip() for t in texts if t and t.strip()]
