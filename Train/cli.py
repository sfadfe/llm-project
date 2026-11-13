import argparse
import os
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HIPBLASLT_TENSILE_LIBPATH", str(ROOT / "hipblaslt-lib" / "gfx1201"))
sys.path.insert(0, str(ROOT / "Train"))
sys.path.insert(0, str(ROOT / "Prep"))

import readline  # noqa: E402,F401
import torch  # noqa: E402

import decode  # noqa: E402
from model import LM, ModelConfig  # noqa: E402
from sft_data import Chat  # noqa: E402

SEQ_LEN = 2048
DIM, RESET = "\033[2m", "\033[0m"


def fit(chat, messages, max_new):
    first = 1 if messages and messages[0]["role"] == "system" else 0
    k = first
    while True:
        ids = chat.render(messages[:first] + messages[k:], generation=True)[0]
        if len(ids) + max_new <= SEQ_LEN:
            del messages[first:k]
            return ids
        if len(messages) - k <= 1:
            return None
        k += 2


def reply(model, chat, ids, args):
    out, shown = [], ""
    t0 = time.time()
    gen = decode.stream(model, ids, args.max_new, {chat.end, chat.eot}, temperature=args.temperature, top_p=args.top_p)
    try:
        for tok in gen:
            out.append(tok)
            text = chat.decode(out)
            if not text.endswith("�") and len(text) > len(shown):
                print(text[len(shown):], end="", flush=True)
                shown = text
    except KeyboardInterrupt:
        print(" [stopped]", end="")
    finally:
        gen.close()
    dt = time.time() - t0
    text = chat.decode(out)
    if len(text) > len(shown):
        print(text[len(shown):], end="")
    print(f"\n{DIM}[{len(out)} tok, {len(out) / max(dt, 1e-9):.1f} tok/s]{RESET}", flush=True)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=str(ROOT / "model" / "hyb200m_rl_20261109_035245"))
    ap.add_argument("--ckpt", default="latest.pt")
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--system", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    with open(run / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    model = LM(ModelConfig(**cfg["model"])).cuda().eval()
    model.load_state_dict(torch.load(run / args.ckpt, map_location="cuda", weights_only=False)["model"])
    chat = Chat()
    base = [{"role": "system", "content": args.system}] if args.system else []
    messages = list(base)
    print(f"{DIM}{run.name}/{args.ckpt} | /reset clears history, /exit or Ctrl-D quits{RESET}")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "/exit":
            break
        if line == "/reset":
            messages = list(base)
            print(f"{DIM}[history cleared]{RESET}")
            continue
        messages.append({"role": "user", "content": line})
        ids = fit(chat, messages, args.max_new)
        if ids is None:
            messages.pop()
            print(f"{DIM}[message too long for {SEQ_LEN}-token context with max_new={args.max_new}]{RESET}")
            continue
        messages.append({"role": "assistant", "content": reply(model, chat, ids, args)})


if __name__ == "__main__":
    main()
