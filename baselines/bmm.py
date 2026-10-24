"""BMM (Bayesian model merging) around a RegMean anchor.

Inner problem, per linear layer: D* = (sum_d D_d G_d + L D_anchor)(sum_d G_d + L I)^{-1},
L = lam[b, g] * mean(diag(sum_d G_d)); merged weight W0 + alpha[b] * D*. Layers are split into
`n_blocks` depth blocks and four groups (attention in/out, MLP in/out), giving 5 parameters per
block. The outer search is Gaussian-process Bayesian optimisation of the mean relative
validation R@10. Non-linear parameters are taken from the anchor.
"""
import argparse
import json
import os
import re
import time

import optuna
import torch

from reachmerge import common as C
from reachmerge.fasteval import FastEval

GROUPS = ["attn_in", "attn_out", "mlp_in", "mlp_out"]


def group_of(name):
    if re.search(r"attention\.self\.(query|key|value)", name):
        return "attn_in"
    if "attention.output.dense" in name:
        return "attn_out"
    if "intermediate.dense" in name:
        return "mlp_in"
    return "mlp_out"


class BMM:
    def __init__(self, cache, domains, base, anchor, n_layers, n_blocks=3, alpha=0.9, device="cuda"):
        self.lin = json.load(open(f"{cache}/lin_names.json"))
        self.base, self.anchor, self.nb, self.nl, self.dev = base, anchor, n_blocks, n_layers, device
        SG, SA = {}, {}
        for d in domains:
            c = torch.load(f"{cache}/{d}.pt")
            for n in self.lin:
                Gt = C.shrink(c["G"][n], alpha)
                SG[n] = SG.get(n, 0) + Gt
                SA[n] = SA.get(n, 0) + (c["W"][n] - base[f"{n}.weight"].double()) @ Gt
            del c
        self.SG = {n: v.float().to(device) for n, v in SG.items()}
        self.SA = {n: v.float().to(device) for n, v in SA.items()}
        self.md = {n: float(torch.diag(self.SG[n]).mean()) for n in self.lin}
        self.DA = {n: (anchor[f"{n}.weight"].float() - base[f"{n}.weight"].float()).to(device) for n in self.lin}

    def block_of(self, n):
        l = C.layer_index(n)
        return self.nb - 1 if l is None else min(l * self.nb // self.nl, self.nb - 1)

    @torch.no_grad()
    def build(self, lam, alpha):
        out = dict(self.anchor)
        for n in self.lin:
            b = self.block_of(n)
            L = lam[(b, group_of(n))] * self.md[n]
            S = self.SG[n] + L * torch.eye(self.SG[n].shape[0], device=self.dev)
            D = torch.linalg.solve(S.T, (self.SA[n] + L * self.DA[n]).T).T
            w0 = self.base[f"{n}.weight"]
            out[f"{n}.weight"] = (w0.float() + alpha[b] * D.cpu()).to(w0.dtype)
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=list(C.BACKBONES), required=True)
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--cache", required=True, help="output folder of reachmerge.gram_cache")
    p.add_argument("--anchor", required=True, help="RegMean state_dict")
    p.add_argument("--out", required=True)
    p.add_argument("--n_blocks", type=int, default=3)
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--n_val", type=int, default=1000)
    a = p.parse_args()

    cache, out = os.path.abspath(a.cache), os.path.abspath(a.out)
    anchor = torch.load(os.path.abspath(a.anchor), map_location="cpu")
    fe = FastEval(a.backbone, a.domains, n_val=a.n_val)
    base = {k: v.detach().cpu().clone() for k, v in fe.model.state_dict().items()}
    names = json.load(open(f"{cache}/lin_names.json"))
    n_layers = max(C.layer_index(n) for n in names if C.layer_index(n) is not None) + 1
    bm = BMM(cache, a.domains, base, anchor, n_layers, a.n_blocks)

    fe.load(anchor)
    ref = fe.recall10("val")
    score = lambda v: sum(v[d] / max(ref[d], 1e-9) for d in a.domains) / len(a.domains)
    blocks = range(a.n_blocks)

    def objective(tr):
        lam = {(b, g): tr.suggest_float(f"lam_{b}_{g}", 1e-3, 1e2, log=True) for b in blocks for g in GROUPS}
        alpha = {b: tr.suggest_float(f"alpha_{b}", 0.5, 2.0) for b in blocks}
        t = time.time()
        fe.load(bm.build(lam, alpha))
        s = score(fe.recall10("val"))
        print(f"trial {tr.number} val_rel={s:.4f} ({time.time() - t:.0f}s)", flush=True)
        return s

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GPSampler(seed=0))
    study.enqueue_trial({**{f"lam_{b}_{g}": 1.0 for b in blocks for g in GROUPS}, **{f"alpha_{b}": 1.0 for b in blocks}})
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=a.n_trials)
    bp = study.best_params
    best = bm.build({(b, g): bp[f"lam_{b}_{g}"] for b in blocks for g in GROUPS}, {b: bp[f"alpha_{b}"] for b in blocks})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(best, out)
    json.dump({"best_val_rel": study.best_value, "params": bp}, open(out.replace(".pt", ".json"), "w"), indent=1)
    print(f"best trial {study.best_trial.number} val_rel={study.best_value:.4f}", flush=True)


if __name__ == "__main__":
    main()
