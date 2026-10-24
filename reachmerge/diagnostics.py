"""Reproduction ratio a_d and pooled least-squares scale s_LS of a merged model.

For domain d and linear layer n, with Gram matrix G of expert d on its own inputs,
  a_{d,n} = <D_d, D*>_G / ||D_d||_G^2,   <A, B>_G = tr(A G B^T),
where D_d and D* are the expert and merged updates. Layers are pooled with weights ||D_d||_G^2,
and the result is averaged over domains without weighting.
"""
import argparse
import json
import os

import torch

from reachmerge import common as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=list(C.BACKBONES), required=True)
    p.add_argument("--cache", required=True, help="output folder of gram_cache.py")
    p.add_argument("--merged", required=True)
    a = p.parse_args()

    cache = os.path.abspath(a.cache)
    merged = torch.load(os.path.abspath(a.merged), map_location="cpu")
    C.enter_mergerec()
    base = C.build_model(a.backbone).state_dict()
    names = json.load(open(f"{cache}/lin_names.json"))
    domains = sorted(f[:-3] for f in os.listdir(cache) if f.endswith(".pt"))

    ratio = {}
    num_ls = den_ls = 0.0
    for d in domains:
        c = torch.load(f"{cache}/{d}.pt")
        cross = own = 0.0
        for n in names:
            G = c["G"][n].cuda()
            w0 = base[f"{n}.weight"].double().cuda()
            Dd = c["W"][n].cuda() - w0
            Ds = merged[f"{n}.weight"].double().cuda() - w0
            own += float(torch.trace(Dd @ G @ Dd.T))
            cr = float(torch.trace(Dd @ G @ Ds.T))
            cross += cr
            num_ls += cr
            den_ls += float(torch.trace(Ds @ G @ Ds.T))
        ratio[d] = cross / own
        del c
    mean_a = sum(ratio.values()) / len(ratio)
    print(json.dumps({"a_d": ratio, "mean_a": mean_a, "inv_mean_a": 1 / mean_a, "s_ls": num_ls / den_ls}, indent=1))


if __name__ == "__main__":
    main()
