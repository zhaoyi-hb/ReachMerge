"""Depth-ordered RegMean (RegMean++).

Layers are solved block by block. The Gram statistics of block l are collected on a network
whose embeddings and blocks 0..l-1 are already merged, while block l onwards still belongs
to expert d. Non-linear parameters (embeddings, LayerNorm, biases) are averaged.
"""
import argparse
import os
import time

import torch

from reachmerge import common as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=list(C.BACKBONES), required=True)
    p.add_argument("--ckpt_dir", required=True)
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
    base_m = C.build_model(a.backbone)
    base = {k: v.detach().float().clone() for k, v in base_m.state_dict().items()}
    names = C.linear_names(base_m)
    model = base_m.to("cuda").eval()

    experts = {d: C.load_expert(a.ckpt_dir, d) for d in a.domains}
    K = len(a.domains)
    keys = list(base)
    first = experts[a.domains[0]]
    merged = {k: (sum(experts[d][k].float() for d in a.domains) / K if k in first else base[k]) for k in keys}
    dms = {d: C.data_module(a.backbone, d, base_m.tokenizer, a.batch_size) for d in a.domains}

    n_layers = max(C.layer_index(n) for n in names if C.layer_index(n) is not None) + 1
    groups = [[n for n in names if C.layer_index(n) == l] for l in range(n_layers)]
    groups.append([n for n in names if C.layer_index(n) is None])

    def in_prefix(k, l):
        if "embeddings" in k:
            return True
        j = C.layer_index(k)
        return j is not None and j < l

    fallback = 0
    t0 = time.time()
    for l, block in enumerate(groups):
        if not block:
            continue
        Gs = {}
        for d in a.domains:
            sd = dict(experts[d])
            sd.update({k: v for k, v in base.items() if k not in sd})
            for k in keys:
                if in_prefix(k, l):
                    sd[k] = merged[k]
            model.load_state_dict(sd, strict=False)
            Gs[d] = C.collect_grams(model, dms[d], block, a.n_seq, a.n_item, seed=0)
        for n in block:
            Ws = [experts[d][f"{n}.weight"].double() for d in a.domains]
            W, fb = C.solve_layer(Ws, [Gs[d][n] for d in a.domains], base[f"{n}.weight"].double(), a.alpha, a.ridge)
            merged[f"{n}.weight"] = W.float()
            fallback += fb
        print(f"[regmean++] block {l + 1}/{len(groups)} ({len(block)} layers) {time.time() - t0:.0f}s", flush=True)

    C.save_state(merged, base_m, a.out)
    print(f"[regmean++] saved {a.out} (fallback layers: {fallback}/{len(names)})", flush=True)


if __name__ == "__main__":
    main()
