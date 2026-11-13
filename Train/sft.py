import argparse
import faulthandler
import json
import os
import signal
import time
import tomllib
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import LM, ModelConfig
from train import apply_grads, lr_at, save

ROOT = Path(__file__).resolve().parent.parent
STOP = False


def on_signal(signum, frame):
    global STOP
    STOP = True


def dump_toml(cfg):
    lines = []
    for sec, kv in cfg.items():
        lines.append(f"[{sec}]")
        for k, v in kv.items():
            lines.append(f"{k} = {json.dumps(v)}")
        lines.append("")
    return "\n".join(lines)


class SFTData:
    def __init__(self, d, epochs, seed, index_path, load, write):
        meta = json.load(open(d / "meta.json"))
        self.R, self.eot = meta["row"], meta["eot"]
        self.tok = {s: np.memmap(d / f"{s}.bin", dtype=np.uint16, mode="r").reshape(-1, self.R) for s in ("train", "val")}
        self.flg = {s: np.memmap(d / f"{s}_mask.bin", dtype=np.uint8, mode="r").reshape(-1, self.R) for s in ("train", "val")}
        if load:
            self.index = np.load(index_path)
        else:
            rng = np.random.default_rng(seed)
            self.index = np.concatenate([rng.permutation(len(self.tok["train"])) for _ in range(epochs)])
            if write:
                np.save(str(index_path) + ".tmp.npy", self.index)
                Path(str(index_path) + ".tmp.npy").replace(index_path)

    def batch(self, split, rows, device):
        rows = np.sort(rows)
        buf = self.tok[split][rows].astype(np.int64)
        flg = self.flg[split][rows]
        x = buf[:, :-1]
        y = np.where(flg[:, 1:] > 0, buf[:, 1:], -100)
        B, T = x.shape
        pos = np.flatnonzero(x.reshape(-1) == self.eot) + 1
        cu_cpu = torch.from_numpy(np.union1d(np.arange(0, B * T + 1, T), pos))
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).pin_memory().to(device, non_blocking=True)
        return to(x), to(y), cu_cpu.to(device), cu_cpu


@torch.no_grad()
def evaluate(model, data, mb, n_batches, rank, world, device):
    model.eval()
    acc = torch.zeros(2, device=device)
    n = min(len(data.tok["val"]), n_batches * mb * world)
    rows = np.arange(n)[rank::world]
    for i in range(0, len(rows) - mb + 1, mb):
        x, y, cu, cu_cpu = data.batch("val", rows[i:i + mb], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y, cu, cu_cpu)
        w = (y != -100).sum()
        acc += torch.stack([loss.float() * w, w.float()])
    if world > 1:
        dist.all_reduce(acc)
    model.train()
    return (acc[0] / acc[1].clamp(min=1)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "Train" / "sft.toml"))
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
    tc, dc = cfg["train"], cfg["data"]

    data = SFTData(ROOT / dc["dir"], dc["epochs"], dc["seed"], run / "index.npy", load=bool(args.resume),
                   write=master and not args.resume)
    mb = tc["micro_batch"]
    accum = tc["global_batch"] // (mb * world)
    assert accum * mb * world == tc["global_batch"]
    total = len(data.index) // tc["global_batch"]

    torch.manual_seed(dc["seed"])
    mc = ModelConfig(**cfg["model"])
    model = LM(mc).to(device)
    opt = torch.optim.AdamW(model.param_groups(tc["weight_decay"]), lr=tc["lr"], betas=(tc["beta1"], tc["beta2"]),
                            eps=1e-8, fused=True)
    step = 0
    if args.resume:
        ck = torch.load(run / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        step = ck["step"]
    else:
        ck = torch.load(ROOT / cfg["init"]["run"] / cfg["init"]["ckpt"], map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
    del ck
    if tc["compile"]:
        model.compile_mlps()
    net = DDP(model, device_ids=[local]) if ddp else model

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    tok_step = tc["global_batch"] * (data.R - 1)
    log = open(run / "log.jsonl", "a") if master else None
    if master:
        print(f"run {run}", flush=True)
        print(f"sft init {cfg['init']}  rows {len(data.tok['train'])}  world {world}  micro {mb}  accum {accum}  "
              f"steps {step}->{total}  tokens/step {tok_step}", flush=True)
        if step == 0:
            ev = evaluate(model, data, mb, tc["eval_batches"], rank, world, device)
            print("eval", json.dumps({"step": 0, "val": ev}), flush=True)
            log.write(json.dumps({"step": 0, "val": ev}) + "\n")
    elif step == 0:
        evaluate(model, data, mb, tc["eval_batches"], rank, world, device)

    loss_acc = torch.zeros((), device=device)
    n_acc = 0
    t_log, step_log, t_start = time.time(), step, time.time()
    G = tc["global_batch"]
    while step < total:
        lr = lr_at(step, total, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        rows = data.index[step * G:(step + 1) * G]
        for i in range(accum):
            off = (i * world + rank) * mb
            x, y, cu, cu_cpu = data.batch("train", rows[off:off + mb], device)
            with net.no_sync() if ddp and i < accum - 1 else nullcontext():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = net(x, y, cu, cu_cpu)
                (loss / accum).backward()
            loss_acc += loss.detach()
            n_acc += 1
        gnorm, bad = apply_grads(model, opt, tc["grad_clip"])
        if bad and master:
            print(f"skip step {step + 1} gnorm {gnorm.item()} top {bad}", flush=True)
        opt.zero_grad(set_to_none=True)
        step += 1

        stop = torch.tensor(float(STOP), device=device)
        if ddp:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        stop = bool(stop.item())

        if step % tc["log_every"] == 0 or step == total or stop:
            avg = loss_acc / max(n_acc, 1)
            if ddp:
                dist.all_reduce(avg, op=dist.ReduceOp.AVG)
            now = time.time()
            tps = tok_step * (step - step_log) / (now - t_log)
            rec = {"step": step, "loss": avg.item(), "lr": lr, "gnorm": gnorm.item(), "tok_s": tps,
                   "mem_gb": torch.cuda.max_memory_allocated() / 2**30, "elapsed_h": (now - t_start) / 3600,
                   "eta_h": (total - step) * tok_step / max(tps, 1) / 3600}
            if master:
                print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            loss_acc.zero_()
            n_acc = 0
            t_log, step_log = now, step

        if step % tc["eval_every"] == 0 or step == total:
            ev = evaluate(model, data, mb, tc["eval_batches"], rank, world, device)
            if master:
                print("eval", json.dumps({"step": step, "val": ev}), flush=True)
                log.write(json.dumps({"step": step, "val": ev}) + "\n")
                log.flush()
            t_log = time.time()

        if step % tc["save_every"] == 0 or step == total or stop:
            if master:
                save(run, model, opt, step, keep=step % tc["keep_every"] == 0 or step == total)
                print(f"saved step {step}", flush=True)
            if ddp:
                dist.barrier()
        if stop:
            break

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
