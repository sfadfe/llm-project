import argparse
import tomllib
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import LM, ModelConfig

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="latest.pt")
    ap.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer" / "tokenizer.json"))
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run = Path(args.run)
    with open(run / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    model = LM(ModelConfig(**cfg["model"])).cuda().eval()
    model.load_state_dict(torch.load(run / args.ckpt, map_location="cuda", weights_only=False)["model"])
    tok = Tokenizer.from_file(args.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    torch.manual_seed(args.seed)
    for prompt in args.prompt or ["The meaning of a word often depends on", "Once upon a time, in a small village,"]:
        ids = tok.encode(prompt).ids
        n0 = len(ids)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(args.max_new):
                logits = model(torch.tensor([ids], device="cuda"))[0, -1].float()
                if args.temperature <= 0:
                    nxt = int(logits.argmax())
                else:
                    probs = torch.softmax(logits / args.temperature, -1)
                    sp, si = probs.sort(descending=True)
                    keep = sp.cumsum(-1) - sp < args.top_p
                    sp = sp * keep
                    nxt = int(si[torch.multinomial(sp / sp.sum(), 1)])
                if nxt == eot:
                    break
                ids.append(nxt)
        print(f"--- {prompt!r}\n{tok.decode(ids[:n0])}|{tok.decode(ids[n0:])}\n", flush=True)


if __name__ == "__main__":
    main()
