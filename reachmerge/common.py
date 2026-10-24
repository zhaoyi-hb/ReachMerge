"""Shared helpers: MergeRec setup, backbones, expert loading, Gram statistics."""
import os
import re
import sys
import pathlib

import torch

MERGEREC_ROOT = os.path.abspath(os.environ.get("MERGEREC_ROOT", "third_party/MergeRec"))
if MERGEREC_ROOT not in sys.path:
    sys.path.insert(0, MERGEREC_ROOT)

from rec_retrieval.configs import NegativeSampleConfig  # noqa: E402
from rec_retrieval.module.models import ModelType  # noqa: E402
from utils import _get_data_module, remove_duplicate_prefix  # noqa: E402

RECFORMER_CKPT = "checkpoints/pretrained/recformer_base_pretrained.pt"
BACKBONES = {
    "blair_base": dict(model_type="BLAIR_BASE",
                       path=os.environ.get("BLAIR_PATH", "hyp1231/blair-roberta-base"),
                       kwargs={}),
    "recformer_base": dict(model_type="RECFORMER_BASE",
                           path=os.environ.get("LONGFORMER_PATH", "allenai/longformer-base-4096"),
                           kwargs={"ckpt_path": RECFORMER_CKPT}),
}


def enter_mergerec():
    """Dataset and checkpoint paths are relative to the MergeRec root."""
    os.chdir(MERGEREC_ROOT)


def build_model(backbone):
    b = BACKBONES[backbone]
    return ModelType[b["model_type"]].value(
        model_name_or_path=b["path"], tokenizer_name_or_path=b["path"], lora_config=None,
        pooling_method="cls", model_kwargs=dict(b["kwargs"]), tokenizer_kwargs={})


def load_expert(ckpt_dir, domain):
    sd = torch.load(f"{ckpt_dir}/{domain}/state_dict.pt", map_location="cpu")
    sd.pop("item_embeddings", None)
    return remove_duplicate_prefix(sd)


def task_vectors(base, ckpt_dir, domains):
    out = {}
    for d in domains:
        sd = load_expert(ckpt_dir, d)
        out[d] = {k: sd[k].float() - base[k] for k in base if k in sd and tuple(sd[k].shape) == tuple(base[k].shape)}
        assert len(out[d]) == len(base), f"{d}: expert keys do not match the backbone"
    return out


def data_module(backbone, domain, tokenizer, batch_size=32, num_workers=0):
    dm = _get_data_module(
        model_type=ModelType[BACKBONES[backbone]["model_type"]], batch_size=batch_size,
        data_path=pathlib.Path(f"datasets/{domain}"), item_prompt=None, max_attribute_len=32,
        max_items=50, max_seq_len=512, model_tokenizer=tokenizer,
        negative_sample_config=NegativeSampleConfig(), num_workers=num_workers,
        reverse_sequence=True, sequence_prompt=None)
    dm.setup("fit")
    return dm


def linear_names(model):
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


def layer_index(name):
    m = re.search(r"layer\.(\d+)\.", name)
    return int(m.group(1)) if m else None


@torch.no_grad()
def collect_grams(model, dm, names, n_seq, n_item, seed=None, device="cuda"):
    """Token-normalised input Gram matrices of the given linear layers.

    Inputs are training sequences and item texts of one domain; padding tokens are dropped.
    """
    mods = dict(model.named_modules())
    G, cnt, mask = {}, {}, {}

    def hook(name):
        def h(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            am = mask.get("m")
            if am is not None and x.shape[0] == am.numel():
                x = x[am.reshape(-1).bool()]
            x = x.double()
            G[name] = G.get(name, 0) + x.T @ x
            cnt[name] = cnt.get(name, 0) + x.shape[0]
        return h

    hs = [mods[n].register_forward_hook(hook(n)) for n in names]
    if seed is not None:
        torch.manual_seed(seed)
    n = 0
    for b in dm.train_dataloader():
        enc = (b.sequences if hasattr(b, "sequences") else b.sequence).to(device)
        mask["m"] = enc.get("attention_mask")
        model(enc)
        n += enc["input_ids"].shape[0]
        if n >= n_seq:
            break
    n = 0
    for b in dm.item_dataloader():
        enc = b.items.to(device)
        mask["m"] = enc.get("attention_mask")
        model(enc)
        n += enc["input_ids"].shape[0]
        if n >= n_item:
            break
    for h in hs:
        h.remove()
    return {k: (v / cnt[k]).cpu() for k, v in G.items()}


def shrink(G, alpha):
    return alpha * G + (1 - alpha) * torch.diag(torch.diag(G))


def solve_layer(Ws, Gs, base_w, alpha, ridge):
    """RegMean solution of one linear layer with ridge towards the expert average.

    Falls back to the expert average if the merged update is more than three times
    longer than the longest expert update.
    """
    num = 0
    den = 0
    w_sum = 0
    longest = 0.0
    for W, G in zip(Ws, Gs):
        Gt = shrink(G, alpha)
        num = num + W @ Gt
        den = den + Gt
        w_sum = w_sum + W
        longest = max(longest, float((W - base_w).norm()))
    w_avg = w_sum / len(Ws)
    lam = ridge * float(torch.diag(den).mean())
    eye = torch.eye(den.shape[0], dtype=den.dtype)
    W = torch.linalg.solve((den + lam * eye).T, (num + lam * w_avg).T).T
    if float((W - base_w).norm()) > 3 * longest:
        return w_avg, True
    return W, False


def save_state(sd, ref_model, path):
    dt = {k: v.dtype for k, v in ref_model.state_dict().items()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({k: v.to(dt[k]) for k, v in sd.items()}, path)
