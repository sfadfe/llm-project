import argparse
import os
import zlib
from pathlib import Path

import msgpack

NEED = ["_Ailk_Bjlk_BBS_BH_", "_MT32x32x128_", "_LBSPPA512_", "_PGR2_", "_SS0_", "_SVW8_", "_TLDS0_"]


def tables(node):
    if isinstance(node, dict):
        if node.get("type") == "Matching" and "table" in node:
            yield node
        for k, v in node.items():
            if k != "table":
                yield from tables(v)
    elif isinstance(node, list):
        for v in node:
            yield from tables(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/hipblaslt/library")
    ap.add_argument("--out", default="hipblaslt-lib")
    ap.add_argument("--arch", default="gfx1201")
    ap.add_argument("--match", default="TensileLibrary_BB_BB_*_Ailk_Bjlk_*.dat.zlib")
    args = ap.parse_args()

    src = Path(args.src).resolve() / args.arch
    out = Path(args.out).resolve() / args.arch
    out.mkdir(parents=True, exist_ok=True)
    targets = set(src.glob(args.match))
    changed = 0
    for f in sorted(src.iterdir()):
        dst = out / f.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if f not in targets:
            os.symlink(f, dst)
            continue
        d = msgpack.unpackb(zlib.decompress(f.read_bytes()), raw=False, strict_map_key=False)
        drop = {s["index"] for s in d["solutions"] if all(n in s.get("kernelName", "") for n in NEED)}
        removed = 0
        kept = 0
        for t in tables(d["library"]) if drop else ():
            before = len(t["table"])
            t["table"] = [e for e in t["table"] if e.get("index") not in drop]
            removed += before - len(t["table"])
            kept += len(t["table"])
        if not removed:
            os.symlink(f, dst)
            continue
        assert kept > 0
        dst.write_bytes(zlib.compress(msgpack.packb(d, use_bin_type=True)))
        print(f"{f.name}: solutions {sorted(drop)} removed {removed} kept {kept}")
        changed += 1
    print(f"rewritten {changed} of {len(targets)} matched, out {out.parent}")


if __name__ == "__main__":
    main()
