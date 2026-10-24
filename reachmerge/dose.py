"""Dose calibration: theta_0 + s * (theta_merged - theta_0), with s chosen on validation data.

The score of a scale is the mean over domains of validation R@10 relative to s = 1, so every
domain counts equally. Ties (to four decimals) go to the scale closest to one.
"""
import argparse
import json
import os

import torch

from reachmerge import common as C
from reachmerge.fasteval import FastEval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=list(C.BACKBONES), required=True)
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--merged", required=True, help="state_dict of the merge to be scaled")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--grid", type=float, nargs="+", default=[1.0, 1.1, 1.2, 1.3, 1.45])
    p.add_argument("--n_val", type=int, default=10 ** 9, help="validation users per domain (default: all)")
    a = p.parse_args()

    out_dir = os.path.abspath(a.out_dir)
    merged_path = os.path.abspath(a.merged)
    fe = FastEval(a.backbone, a.domains, n_val=a.n_val)
    base = {k: v.detach().cpu().clone() for k, v in fe.model.state_dict().items()}
    M = torch.load(merged_path, map_location="cpu")

    def scaled(s):
        return {k: ((base[k].float() + s * (M[k].float() - base[k].float())).to(M[k].dtype)
                    if (k in base and M[k].is_floating_point()) else M[k]) for k in M}

    rel, ref = {}, None
    for s in a.grid:
        fe.load(scaled(s))
        r = fe.recall10("val")
        if ref is None:
            ref = r
        rel[s] = sum(r[d] / max(ref[d], 1e-9) for d in a.domains) / len(a.domains)
        print(f"s={s} val_rel={rel[s]:.4f} " + " ".join(f"{d}={r[d]:.4f}" for d in a.domains), flush=True)
    best = max(a.grid, key=lambda s: (round(rel[s], 4), -abs(s - 1)))

    os.makedirs(out_dir, exist_ok=True)
    torch.save(scaled(best), f"{out_dir}/state_dict.pt")
    json.dump({"val_rel": rel, "best": best}, open(f"{out_dir}/choice.json", "w"), indent=1)
    print(f"selected s={best}", flush=True)


if __name__ == "__main__":
    main()
