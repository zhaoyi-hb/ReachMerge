"""Unlabelled probe inputs (item texts) for the data-dependent baselines (CAT-Merging, MAGIC)."""
import json
import pathlib
import random

import torch

from reachmerge import common as C


def item_texts(domain, n, seed=42):
    meta = json.load(open(f"datasets/{domain}/meta_data.json"))
    smap = json.load(open(f"datasets/{domain}/smap.json"))
    ids = list(smap.keys())
    random.Random(seed).shuffle(ids)
    out = []
    for i in ids:
        r = meta.get(i) or {}
        t = " ".join(str(r.get(k, "")).strip() for k in ("title", "brand", "category") if r.get(k))
        if t.strip():
            out.append(t.strip())
        if len(out) >= n:
            break
    return out


def blair_batch(tokenizer, texts):
    b = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return {k: v.cuda() for k, v in b.items()}


_RF_DM = {}


def recformer_batch(tokenizer, domains, n_per, seed=42):
    from rec_retrieval.configs import NegativeSampleConfig
    from rec_retrieval.datamodule.recommender.recformer import RecDataModuleForRecformer
    parts = []
    for d in domains:
        if d not in _RF_DM:
            dm = RecDataModuleForRecformer(
                dataset_path=pathlib.Path(f"datasets/{d}"), tokenizer=tokenizer, batch_size=n_per,
                max_seq_len=512, max_attribute_len=32, max_items=50, num_workers=0,
                negative_sample=NegativeSampleConfig())
            dm.setup("test")
            _RF_DM[d] = dm
        dm = _RF_DM[d]
        ids = list(range(len(dm.item_dataset)))
        random.Random(seed).shuffle(ids)
        parts.append(dm.item_collator([dm.item_dataset[i] for i in ids[:n_per]]).items)
    if len(parts) == 1:
        out = parts[0]
    else:
        L = max(b["input_ids"].shape[1] for b in parts)
        merged = {}
        for k in parts[0].keys():
            vs = []
            for b in parts:
                v = b[k]
                pad = L - v.shape[1]
                if pad > 0:
                    v = torch.nn.functional.pad(v, (0, pad), value=(1 if k == "input_ids" else 0))
                vs.append(v)
            merged[k] = torch.cat(vs, 0)
        out = type(parts[0])(merged)
    return type(out)({k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in out.items()})


def forward(model, backbone, batch):
    """BLaIR probes go through the encoder directly; RecFormer through the full model."""
    if backbone == "blair_base":
        return model.model(**batch)
    return model(batch)
