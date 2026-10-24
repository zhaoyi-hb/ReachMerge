"""CAT-Merging (conflict-aware task merging), adapted to text-encoder recommenders.

Linear layers: each task vector is projected off the top-`ratio` conflict directions of every
other task. LayerNorm weights and biases are masked. Features come from `n_probe` unlabelled
item texts per domain. The merged model is theta_0 + scaling * sum_d tau_d.
"""
import argparse
import os

import numpy as np
import torch

from reachmerge import common as C
from baselines import probe


def collect_features(backbone, base, T, domains, keys, n_probe, tokenizer):
    feats = {}
    for d in domains:
        m = C.build_model(backbone).cuda()
        m.load_state_dict({k: base[k] + T[d][k] for k in base}, strict=True)
        if backbone == "blair_base":
            batch = probe.blair_batch(tokenizer, probe.item_texts(d, n_probe))
        else:
            batch = probe.recformer_batch(tokenizer, [d], n_probe)
        lin_in, ln_out, hs = {}, {}, []
        for name, mod in m.named_modules():
            if isinstance(mod, torch.nn.Linear):
                hs.append(mod.register_forward_hook(
                    lambda mo, inp, out, p=name + ".weight": lin_in.__setitem__(p, inp[0].detach().cpu())))
            elif isinstance(mod, torch.nn.LayerNorm):
                def h(mod2, inp, out, p=name + ".weight"):
                    w = mod2.weight.detach().cpu()
                    b = mod2.bias.detach().cpu() if mod2.bias is not None else torch.zeros_like(w)
                    ln_out[p] = (out.detach().cpu() - b) / w
                hs.append(mod.register_forward_hook(h))
        with torch.no_grad():
            probe.forward(m, backbone, batch)
        for x in hs:
            x.remove()
        del m
        torch.cuda.empty_cache()
        f = {}
        for k in keys:
            if k in lin_in:
                f[k] = lin_in[k].reshape(-1, lin_in[k].shape[-1]).float()
            elif k in ln_out:
                f[k] = ln_out[k].reshape(-1, ln_out[k].shape[-1]).float()
            else:
                f[k] = None
        feats[d] = f
    return feats


def cat_merge(T, feats, domains, keys, ratio=4, lambd=5.0, ratio_ln=4, lambd_ln=10.0, ratio_bias=0.25):
    D = domains
    K = len(D)
    tv = {d: {k: T[d][k].clone() for k in keys} for d in D}
    ref = T[D[0]]
    is_lin = lambda k: ref[k].ndim == 2 and "embedding" not in k
    is_ln = lambda k: k.endswith(".weight") and ref[k].ndim == 1 and "LayerNorm" in k
    is_bias = lambda k: k.endswith(".bias")

    basis = {}
    for kd in D:
        basis[kd] = {}
        for nm in keys:
            if not is_lin(nm) or feats[kd][nm] is None:
                continue
            Xk = feats[kd][nm]
            xtx_k = Xk.t().mm(Xk)
            G = None
            for jd in D:
                if jd == kd:
                    continue
                Xj = feats[jd][nm]
                Wj = tv[jd][nm]
                Gj = Wj.mm(xtx_k - lambd * Xj.t().mm(Xj)).mm(Wj.t())
                G = Gj if G is None else G + Gj
            G = (G + G.t()) / 2
            ev, evec = np.linalg.eigh(G.double().numpy())
            ev = torch.from_numpy(ev).float()
            evec = torch.from_numpy(evec).float()
            basis[kd][nm] = None if (ev <= 0).all() else evec[:, 0:int(ratio)]
    for jd in D:
        for kd in D:
            if kd == jd:
                continue
            for nm in keys:
                if not is_lin(nm):
                    continue
                b = basis[kd].get(nm)
                if b is None or b.numel() == 0:
                    continue
                v = tv[jd][nm]
                vo = v - b.mm(b.t()).mm(v)
                if vo.norm() > 0:
                    tv[jd][nm] = vo * (v.norm() / vo.norm())

    if ratio_ln:
        for nm in keys:
            if not is_ln(nm) or feats[D[0]][nm] is None:
                continue
            masks = {}
            for kd in D:
                score = 0
                for jd in D:
                    if kd == jd:
                        continue
                    score = score + torch.mean((feats[kd][nm] * tv[jd][nm]) ** 2, dim=0) \
                        - lambd_ln * torch.mean((feats[jd][nm] * tv[jd][nm]) ** 2, dim=0)
                idx = torch.topk(score, min(ratio_ln, score.numel()), largest=True).indices
                mk = torch.ones_like(score, dtype=torch.int)
                mk[idx] = 0
                masks[kd] = mk
            for kd in D:
                for jd in D:
                    if kd == jd:
                        continue
                    v = tv[jd][nm]
                    vm = v * masks[kd]
                    tv[jd][nm] = vm * (v.norm() / vm.norm()) if vm.norm() > 0 else vm

    if ratio_bias:
        for nm in keys:
            if not is_bias(nm):
                continue
            masks = [torch.ones_like(tv[D[0]][nm]) for _ in range(K)]
            for i in range(K):
                other = 0
                for j in range(K):
                    if i != j:
                        other = other + tv[D[i]][nm] ** 2
                flat = other.reshape(-1)
                kk = max(1, min(int(flat.numel() * (1 - ratio_bias)), flat.numel()))
                mk = other < torch.kthvalue(flat, kk).values
                for j in range(K):
                    if i != j:
                        masks[i] = masks[i] * mk
            for j in range(K):
                v = tv[D[j]][nm]
                vm = v * masks[j]
                tv[D[j]][nm] = vm * (v.norm() / vm.norm()) if vm.norm() > 0 else vm
    return tv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=list(C.BACKBONES), required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_probe", type=int, default=128)
    p.add_argument("--scaling", type=float, default=0.4)
    a = p.parse_args()

    out = os.path.abspath(a.out)
    C.enter_mergerec()
    m0 = C.build_model(a.backbone)
    base = {k: v.float().clone() for k, v in m0.state_dict().items()}
    T = C.task_vectors(base, a.ckpt_dir, a.domains)
    keys = list(base)
    feats = collect_features(a.backbone, base, T, a.domains, keys, a.n_probe, m0.tokenizer)
    tv = cat_merge(T, feats, a.domains, keys)
    C.save_state({k: base[k] + a.scaling * sum(tv[d][k] for d in a.domains) for k in keys}, m0, out)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
