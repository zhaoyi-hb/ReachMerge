"""Cache each expert's Gram matrices and linear weights (used by BMM and the diagnostics)."""
import argparse
import json
import os
import random

import torch

from reachmerge import common as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=list(C.BACKBONES), required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--n_seq", type=int, default=2000)
    p.add_argument("--n_item", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    a = p.parse_args()

    cache = os.path.abspath(a.cache)
    os.makedirs(cache, exist_ok=True)
    C.enter_mergerec()
    torch.manual_seed(0)
    random.seed(0)
    base_m = C.build_model(a.backbone)
    names = C.linear_names(base_m)
    for d in a.domains:
        if os.path.exists(f"{cache}/{d}.pt"):
            continue
        sd = C.load_expert(a.ckpt_dir, d)
        m = C.build_model(a.backbone)
        m.load_state_dict(sd, strict=False)
        m.to("cuda").eval()
        dm = C.data_module(a.backbone, d, base_m.tokenizer, a.batch_size)
        G = C.collect_grams(m, dm, names, a.n_seq, a.n_item)
        torch.save({"G": G, "W": {n: sd[f"{n}.weight"].double() for n in names}}, f"{cache}/{d}.pt")
        del m, G
        torch.cuda.empty_cache()
        print(f"[gram] {d} done", flush=True)
    json.dump(names, open(f"{cache}/lin_names.json", "w"))


if __name__ == "__main__":
    main()
