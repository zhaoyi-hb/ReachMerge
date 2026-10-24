"""Coefficient-based merges: weight averaging, Task Arithmetic, NAN and DARE.

NAN coefficients: lambda_d = mean_d ||tau_d|| / (K * ||tau_d||), i.e. every task vector is
rescaled to the same length and the lengths add up to the mean task-vector norm.
DARE uses the MergeRec implementation; note that its `density` argument is the dropout
probability, so density=0.1 drops 10% of each update.
"""
import argparse
import os

import torch

from reachmerge import common as C


def nan_coefficients(T, domains):
    norms = {d: sum(float((v * v).sum()) for v in T[d].values()) ** 0.5 for d in domains}
    mean = sum(norms.values()) / len(domains)
    return [mean / len(domains) / norms[d] for d in domains]


def linear_merge(base, T, domains, lam):
    return {k: base[k] + sum(l * T[d][k] for l, d in zip(lam, domains)) for k in base}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=list(C.BACKBONES), required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ta_weight", type=float, default=0.2)
    p.add_argument("--dare_weight", type=float, default=0.2)
    p.add_argument("--dare_density", type=float, default=0.1)
    a = p.parse_args()

    out = os.path.abspath(a.out_dir)
    C.enter_mergerec()
    m0 = C.build_model(a.backbone)
    base = {k: v.float() for k, v in m0.state_dict().items()}
    T = C.task_vectors(base, a.ckpt_dir, a.domains)
    K = len(a.domains)

    C.save_state(linear_merge(base, T, a.domains, [1.0 / K] * K), m0, f"{out}/wa/state_dict.pt")
    C.save_state(linear_merge(base, T, a.domains, [a.ta_weight] * K), m0, f"{out}/ta/state_dict.pt")
    C.save_state(linear_merge(base, T, a.domains, nan_coefficients(T, a.domains)), m0, f"{out}/nan/state_dict.pt")

    from rec_retrieval.merger import ModelMerger
    raw = [C.load_expert(a.ckpt_dir, d) for d in a.domains]
    mm = ModelMerger(models=raw, base_model={k: v.clone() for k, v in m0.state_dict().items()}, align_key_order=True)
    torch.manual_seed(0)
    merged = mm.merge(merge_type="dare", weights=[a.dare_weight] * K, density=a.dare_density)
    C.save_state({k: v for k, v in merged.items() if torch.is_tensor(v)}, m0, f"{out}/dare/state_dict.pt")
    print("saved wa, ta, nan, dare to", out, flush=True)


if __name__ == "__main__":
    main()
