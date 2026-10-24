"""RegMean (Jin et al., ICLR 2023) on the per-domain experts.

Every expert's Gram statistics are collected on its own inputs, all layers in one pass.
"""
import argparse
import os
import random

import torch

from reachmerge import common as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=list(C.BACKBONES), required=True)
    p.add_argument("--ckpt_dir", required=True, help="checkpoints/<run>, one sub-folder per domain")
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_seq", type=int, default=2000)
    p.add_argument("--n_item", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=32)
    a = p.parse_args()

    a.out = os.path.abspath(a.out)
    C.enter_mergerec()
    torch.manual_seed(0)
    random.seed(0)
    base_m = C.build_model(a.backbone)
    base = {k: v.detach().float().clone() for k, v in base_m.state_dict().items()}
    names = C.linear_names(base_m)
    K = len(a.domains)

    Ws = {n: [] for n in names}
    Gs = {n: [] for n in names}
    others = {}
    for d in a.domains:
        sd = C.load_expert(a.ckpt_dir, d)
        m = C.build_model(a.backbone)
        _, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected[:3]
        m.to("cuda").eval()
        dm = C.data_module(a.backbone, d, base_m.tokenizer, a.batch_size)
        G = C.collect_grams(m, dm, names, a.n_seq, a.n_item)
        del m
        torch.cuda.empty_cache()
        for n in names:
            Ws[n].append(sd[f"{n}.weight"].double())
            Gs[n].append(G[n])
        lin_keys = {f"{n}.weight" for n in names}
        for k, v in sd.items():
            if k not in lin_keys:
                others[k] = others.get(k, 0) + v.float()
        print(f"[regmean] {d} done", flush=True)

    merged = {}
    fallback = 0
    for n in names:
        W, fb = C.solve_layer(Ws[n], Gs[n], base[f"{n}.weight"].double(), a.alpha, a.ridge)
        merged[f"{n}.weight"] = W.float()
        fallback += fb
    for k, v in others.items():
        merged[k] = v / K
    for k in base:
        merged.setdefault(k, base[k])
    C.save_state(merged, base_m, a.out)
    print(f"[regmean] saved {a.out} (fallback layers: {fallback}/{len(names)})", flush=True)


if __name__ == "__main__":
    main()
