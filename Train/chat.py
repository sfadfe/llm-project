import argparse
import sys
import tomllib
from pathlib import Path

import torch

import decode
from model import LM, ModelConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Prep"))
from sft_data import Chat  # noqa: E402

PROMPTS = [
    "What is the difference between weather and climate?",
    "Write a short story about a lost dog finding its way home.",
    "Fix the grammar in this sentence: 'Me and him goes to the store yesterday.'",
    "Give me three tips for learning a new language. Answer in all lowercase letters.",
    "Explain why the sky is blue in less than 50 words.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="latest.pt")
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    run = Path(args.run)
    with open(run / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    model = LM(ModelConfig(**cfg["model"])).cuda().eval()
    model.load_state_dict(torch.load(run / args.ckpt, map_location="cuda", weights_only=False)["model"])
    chat = Chat()
    prompts = args.prompt or PROMPTS
    ids = [chat.render([{"role": "user", "content": p}], generation=True)[0] for p in prompts]
    outs, done = decode.generate(model, ids, args.max_new, {chat.end, chat.eot}, temperature=args.temperature)
    for p, o, d in zip(prompts, outs, done):
        print(f"### {p}\n{chat.decode([t for t in o if t not in (chat.end, chat.eot)])}\n[{'end' if d else 'cut'}]\n", flush=True)


if __name__ == "__main__":
    main()
