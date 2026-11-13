import argparse
import faulthandler
import json
import os
import signal
import sys
import time
import tomllib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import constraints
import decode
from model import LM, ModelConfig
from sft import dump_toml
from train import apply_grads, save

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Prep"))
from sft_data import Chat  # noqa: E402

STOP = False


def on_signal(signum, frame):
    global STOP
    STOP = True


def hidden(model, ids, cu, cu_cpu):
    x = model.embed(ids[None])
    for blk in model.blocks:
        x = blk(x, cu, cu_cpu)
    return model.norm(x)[0]


def token_logp(model, seqs, plens, device):
    lens = [len(s) for s in seqs]
    cu_cpu = torch.zeros(len(seqs) + 1, dtype=torch.long)
    cu_cpu[1:] = torch.tensor(lens).cumsum(0)
    ids = torch.tensor([t for s in seqs for t in s], device=device)
    pos = torch.cat([torch.arange(cu_cpu[i] + p - 1, cu_cpu[i + 1] - 1) for i, p in enumerate(plens)]).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = hidden(model, ids, cu_cpu.to(device), cu_cpu)
        logits = F.linear(h[pos], model.embed.weight).float()
    return logits.log_softmax(-1).gather(-1, ids[pos + 1][:, None]).squeeze(-1)


def micro_batches(idx, lens, budget):
    out, cur, n = [], [], 0
    for i in idx:
        if cur and n + lens[i] > budget:
            out.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += lens[i]
    if cur:
        out.append(cur)
    return out


def score(text, checks):
    return [bool(c(text)) for c in checks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "Train" / "rl.toml"))
    ap.add_argument("--init", default=None, help="SFT run dir (overrides config)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", default=str(ROOT / "model"))
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if ddp:
        dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank() if ddp else 0
    world = dist.get_world_size() if ddp else 1
    master = rank == 0

    if args.resume:
        run = Path(args.resume)
        with open(run / "config.toml", "rb") as f:
            cfg = tomllib.load(f)
    else:
        with open(args.config, "rb") as f:
            cfg = tomllib.load(f)
        if args.init:
            cfg["init"]["run"] = str(Path(args.init).resolve())
        with open(ROOT / cfg["init"]["run"] / "config.toml", "rb") as f:
            cfg["model"] = tomllib.load(f)["model"]
        name = [f"{cfg['train']['run_name']}_{datetime.now():%Y%m%d_%H%M%S}"]
        if ddp:
            dist.broadcast_object_list(name, src=0)
        run = Path(args.out) / name[0]
        if master:
            run.mkdir(parents=True, exist_ok=True)
            (run / "config.toml").write_text(dump_toml(cfg))
        if ddp:
            dist.barrier()
    tc, rc = cfg["train"], cfg["rl"]

    chat = Chat()
    stop_ids = {chat.end, chat.eot}
    raw = [json.loads(l) for l in open(ROOT / rc["prompts"])]
    pool = constraints.build_pool(raw, rc["seed"])
    pool = [p for p in pool if len(chat.render(p["messages"], generation=True)[0]) <= rc["max_prompt"]]
    held, train_pool = pool[-rc["eval_prompts"]:], pool[:-rc["eval_prompts"]]
    order = np.random.default_rng(rc["seed"]).permutation(len(train_pool))

    mc = ModelConfig(**cfg["model"])
    model = LM(mc).to(device)
    ref = LM(mc).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    init = torch.load(ROOT / cfg["init"]["run"] / cfg["init"]["ckpt"], map_location=device, weights_only=False)["model"]
    ref.load_state_dict(init)
    opt = torch.optim.AdamW(model.param_groups(tc["weight_decay"]), lr=tc["lr"], betas=(tc["beta1"], tc["beta2"]),
                            eps=1e-8, fused=True)
    step = 0
    if args.resume:
        ck = torch.load(run / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        step = ck["step"]
        del ck
    else:
        model.load_state_dict(init)
    del init

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    P, G = rc["prompts_per_step"], rc["group"]
    assert P % world == 0
    Pl = P // world
    total = rc["steps"]
    log = open(run / "log.jsonl", "a") if master else None
    if master:
        print(f"run {run}", flush=True)
        print(f"rl init {cfg['init']}  pool {len(train_pool)}  held {len(held)}  world {world}  "
              f"prompts/step {P} group {G}  steps {step}->{total}", flush=True)

    def evaluate(tag):
        model.eval()
        mine = held[rank::world]
        outs, done = decode.generate(model, [chat.render(p["messages"], generation=True)[0] for p in mine],
                                     rc["max_new"], stop_ids, temperature=0.0)
        acc = torch.zeros(3, device=device)
        for p, o, d in zip(mine, outs, done):
            ok = score(chat.decode([t for t in o if t not in stop_ids]), p["checks"]) if d else [False] * len(p["checks"])
            acc += torch.tensor([float(all(ok)), sum(ok) / len(ok), float(d)], device=device)
        acc = torch.cat([acc, torch.tensor([float(len(mine))], device=device)])
        if ddp:
            dist.all_reduce(acc)
        model.train()
        n = acc[3].item()
        rec = {"step": step, "eval": tag, "strict": acc[0].item() / n, "loose": acc[1].item() / n, "finished": acc[2].item() / n}
        if master:
            print("eval", json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if master and tag == "final" or master and step == 0:
            for p, o in list(zip(mine, outs))[:3]:
                print("--- sample", p["kinds"], repr(p["messages"][-1]["content"][-200:]), "\n>>>",
                      repr(chat.decode([t for t in o if t not in stop_ids])[:600]), flush=True)

    if step == 0:
        evaluate("init")
    gen = torch.Generator(device=device)
    t_start = time.time()
    while step < total:
        t0 = time.time()
        lr = tc["lr"] * min(1.0, (step + 1) / tc["warmup_steps"])
        for g in opt.param_groups:
            g["lr"] = lr
        base = step * P + rank * Pl
        items = [train_pool[order[(base + j) % len(order)]] for j in range(Pl)]
        prompts = [chat.render(p["messages"], generation=True)[0] for p in items]
        gen.manual_seed(rc["seed"] * 7919 + step * 101 + rank)
        model.eval()
        outs, done = decode.generate(model, [p for p in prompts for _ in range(G)], rc["max_new"], stop_ids,
                                     temperature=rc["temperature"], top_p=1.0, gen=gen)
        model.train()
        t_gen = time.time() - t0

        rewards, per_kind = [], defaultdict(list)
        for j, p in enumerate(items):
            for g_ in range(G):
                k = j * G + g_
                if done[k]:
                    ok = score(chat.decode([t for t in outs[k] if t not in stop_ids]), p["checks"])
                else:
                    ok = [False] * len(p["checks"])
                rewards.append(sum(ok) / len(ok))
                for kind, o in zip(p["kinds"], ok):
                    per_kind[kind].append(o)
        r = torch.tensor(rewards, device=device).view(Pl, G)
        adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
        adv = adv.flatten()

        seqs = [prompts[k // G] + outs[k] for k in range(Pl * G)]
        plens = [len(prompts[k // G]) for k in range(Pl * G)]
        use = [k for k in range(Pl * G) if adv[k].abs() > 1e-6]
        ntok = torch.tensor(float(sum(len(outs[k]) for k in use)), device=device)
        if ddp:
            dist.all_reduce(ntok)
        ntok = ntok.clamp(min=1.0)
        kl_acc = torch.zeros((), device=device)
        for mbi in micro_batches(use, [len(s) for s in seqs], rc["token_budget"]):
            ss, pl = [seqs[k] for k in mbi], [plens[k] for k in mbi]
            a = torch.cat([adv[k].expand(len(outs[k])) for k in mbi])
            with torch.no_grad():
                lr_ref = token_logp(ref, ss, pl, device)
            lp = token_logp(model, ss, pl, device)
            d = lr_ref - lp
            kl = d.exp() - d - 1
            loss = (-(a * torch.exp(lp - lp.detach())) + rc["kl_beta"] * kl).sum() / ntok
            loss.backward()
            kl_acc += kl.detach().sum()
        for p_ in model.parameters():
            if p_.grad is None:
                p_.grad = torch.zeros_like(p_)
            if ddp:
                dist.all_reduce(p_.grad)
        gnorm, bad = apply_grads(model, opt, tc["grad_clip"])
        if bad and master:
            print(f"skip step {step + 1} gnorm {gnorm.item()} top {bad}", flush=True)
        opt.zero_grad(set_to_none=True)
        step += 1

        stats = torch.tensor([sum(rewards), float(len(rewards)), float(sum(done)), float(sum(len(o) for o in outs)),
                              float(len(use)), kl_acc.item()], device=device)
        kinds = constraints.KINDS
        ks = torch.tensor([[sum(per_kind[k]), len(per_kind[k])] for k in kinds], device=device, dtype=torch.float)
        stop = torch.tensor(float(STOP), device=device)
        if ddp:
            dist.all_reduce(stats)
            dist.all_reduce(ks)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        stop = bool(stop.item())
        if master:
            n = stats[1].item()
            rec = {"step": step, "reward": stats[0].item() / n, "finished": stats[2].item() / n,
                   "len": stats[3].item() / n, "used": stats[4].item() / n, "kl": stats[5].item() / ntok.item(),
                   "lr": lr, "gnorm": gnorm.item(), "t_gen": t_gen, "t_step": time.time() - t0,
                   "elapsed_h": (time.time() - t_start) / 3600,
                   "kinds": {k: round(ks[i, 0].item() / ks[i, 1].item(), 3) for i, k in enumerate(kinds) if ks[i, 1] > 0}}
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items() if k != "kinds"),
                  flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % rc["eval_every"] == 0 and step != total:
            evaluate("periodic")
        if step % tc["save_every"] == 0 or step == total or stop:
            if master:
                save(run, model, opt, step, keep=step % tc["keep_every"] == 0 or step == total)
                print(f"saved step {step}", flush=True)
            if ddp:
                dist.barrier()
        if stop:
            break
    if step == total:
        evaluate("final")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
