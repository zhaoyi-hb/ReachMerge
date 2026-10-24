"""MAGIC (magnitude calibration) on top of TA or NAN.

WSC rescales every output row of a merged weight matrix by its magnitude relative to the
experts' updates. FSC rescales each encoder layer's output channels so that the merged
model's feature change matches the experts' average change on unlabelled probe items; the
per-channel ratio is folded into the weights. DSC applies FSC after WSC.
"""
import argparse
import os

import torch

from reachmerge import common as C
from baselines import probe
from baselines.coefficient import nan_coefficients


def wsc(tau_m, T, domains):
    K = len(domains)
    out = {}
    for k, m in tau_m.items():
        if "weight" not in k or "ln" in k or m.ndim != 2:
            out[k] = m
            continue
        A = torch.stack([T[d][k] for d in domains]).permute(1, 0, 2).contiguous().reshape(m.shape[0], K, -1)
        x = m.reshape(m.shape[0], -1)
        alpha = (A * x.unsqueeze(1)).sum(2) / ((A ** 2).sum(2) + 1e-12)
        x_in = (alpha.unsqueeze(2) * A).sum(1)
        An = A.norm(dim=2)
        lam_in = ((alpha * An) / (An + 1e-8)).pow(2).sum(1)
        lam_out = (x - x_in).norm(dim=1).pow(2) / (An.mean(1).pow(2) + 1e-8)
        r = (1.0 / torch.sqrt(lam_in + lam_out + 1e-8)).unsqueeze(1)
        out[k] = torch.where(torch.isnan(r), torch.ones_like(r), r) * m
    return out


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
    K = len(a.domains)

    if a.backbone == "blair_base":
        texts = [t for d in a.domains for t in probe.item_texts(d, 128)]
        batch = probe.blair_batch(m0.tokenizer, texts)
    else:
        batch = probe.recformer_batch(m0.tokenizer, a.domains, 16)
    layers = [f"model.encoder.layer.{i}" for i in range(12)]

    def outputs(state):
        m = C.build_model(a.backbone).cuda()
        m.load_state_dict({k: v.clone() for k, v in state.items()}, strict=True)
        mods = dict(m.named_modules())
        cap, hs = {}, []
        for ln in layers:
            hs.append(mods[ln].register_forward_hook(
                lambda mo, i, o, ln=ln: cap.__setitem__(ln, (o[0] if isinstance(o, tuple) else o).detach())))
        with torch.no_grad():
            probe.forward(m, a.backbone, batch)
        for h in hs:
            h.remove()
        del m
        torch.cuda.empty_cache()
        return cap

    pre = outputs(base)
    acc = {ln: 0 for ln in layers}
    for d in a.domains:
        s = outputs({k: base[k] + T[d][k] for k in base})
        for ln in layers:
            acc[ln] = acc[ln] + (s[ln] - pre[ln]).abs().mean(dim=(0, 1)).cpu()
    target = {ln: acc[ln] / K for ln in layers}

    def fsc(delta):
        s = outputs({k: base[k] + delta[k] for k in base})
        tau = {k: delta[k].clone() for k in base}
        for ln in layers:
            r = target[ln] / ((s[ln] - pre[ln]).abs().mean(dim=(0, 1)).cpu() + 1e-8)
            r = torch.where(torch.isnan(r) | torch.isinf(r), torch.ones_like(r), r)
            for k in tau:
                if not k.startswith(ln + ".") or tau[k].shape[0] != r.numel():
                    continue
                if tau[k].ndim == 2:
                    tau[k] = tau[k] * r.unsqueeze(1)
                elif tau[k].ndim == 1:
                    tau[k] = tau[k] * r
        return tau

    for name, lam in (("ta", [a.ta_weight] * K), ("nan", nan_coefficients(T, a.domains))):
        delta = {k: sum(l * T[d][k] for l, d in zip(lam, a.domains)) for k in base}
        tau = fsc(wsc(delta, T, a.domains))
        C.save_state({k: base[k] + tau[k] for k in base}, m0, f"{out}/magic_{name}/state_dict.pt")
        print("saved magic_" + name, flush=True)


if __name__ == "__main__":
    main()
