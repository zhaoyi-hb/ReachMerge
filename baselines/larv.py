"""LARV: layer-wise adaptive rescaling of task vectors.

Every weight matrix gets a score z = z(effective-rank contrast) - z(commutator conflict),
averaged over tasks; the bottom, middle and top thirds are scaled by 0.8, 1.0 and 1.2 before
the task vectors are combined with the coefficients of the base merge (TA or NAN).
"""
import argparse
import collections
import os

import torch

from reachmerge import common as C
from baselines.coefficient import nan_coefficients


def effective_rank(A):
    try:
        s = torch.linalg.svdvals(A.float())
    except Exception:
        return None
    s = s[s > 0]
    if s.numel() < 2:
        return None
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p)).sum()))


def layer_scales(base, T, domains):
    raw = {}
    for k, B in base.items():
        if B.ndim < 2:
            continue
        e_vals, c_vals = [], []
        for d in domains:
            Tm = T[d][k]
            r0, r1 = effective_rank(B), effective_rank(Tm)
            if r0 and r1:
                e_vals.append(r0 / r1)
            if B.shape[0] == B.shape[1] and Tm.shape == B.shape:
                nb, nt = float(B.norm()), float(Tm.norm())
                if nb > 0 and nt > 0:
                    c_vals.append(float((Tm @ B - B @ Tm).norm()) / (nb * nt))
        if e_vals:
            raw[k] = (sum(e_vals) / len(e_vals), sum(c_vals) / len(c_vals) if c_vals else 0.0)

    def mean_sd(v):
        m = sum(v) / len(v)
        return m, (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5 or 1.0
    em, esd = mean_sd([v[0] for v in raw.values()])
    cm, csd = mean_sd([v[1] for v in raw.values()])
    score = {k: (e - em) / esd - (c - cm) / csd for k, (e, c) in raw.items()}
    order = sorted(score.values())
    t1, t2 = order[len(order) // 3], order[2 * len(order) // 3]
    return {k: 0.8 if s <= t1 else (1.0 if s <= t2 else 1.2) for k, s in score.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=list(C.BACKBONES), required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ta_weight", type=float, default=0.2)
    a = p.parse_args()

    out = os.path.abspath(a.out_dir)
    C.enter_mergerec()
    m0 = C.build_model(a.backbone)
    base = {k: v.float().clone() for k, v in m0.state_dict().items()}
    T = C.task_vectors(base, a.ckpt_dir, a.domains)
    scale = layer_scales(base, T, a.domains)
    print("tiers:", dict(collections.Counter(scale.values())), flush=True)
    K = len(a.domains)
    for name, lam in (("larv_ta", [a.ta_weight] * K), ("larv_nan", nan_coefficients(T, a.domains))):
        sd = {k: base[k] + scale.get(k, 1.0) * sum(l * T[d][k] for l, d in zip(lam, a.domains)) for k in base}
        C.save_state(sd, m0, f"{out}/{name}/state_dict.pt")
        print("saved", name, flush=True)


if __name__ == "__main__":
    main()
