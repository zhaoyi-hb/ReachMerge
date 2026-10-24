"""In-process Recall@10 over the full item catalogue (cosine similarity), used for model selection."""
import os
import random

import torch
import torch.nn.functional as F

from reachmerge import common as C


class FastEval:
    def __init__(self, backbone, domains, n_val=10 ** 9, seed=0, batch_size=64, device="cuda"):
        cwd = os.getcwd()
        C.enter_mergerec()
        self.domains = domains
        self.device = device
        self.bs = batch_size
        self.model = C.build_model(backbone).to(device).eval()
        self.dms, self.val_idx = [], []
        for i, d in enumerate(domains):
            dm = C.data_module(backbone, d, self.model.tokenizer, batch_size, num_workers=4)
            self.dms.append(dm)
            idx = list(range(len(dm.val_dataset)))
            random.Random(seed + i).shuffle(idx)
            self.val_idx.append(sorted(idx[:n_val]))
        os.chdir(cwd)

    def load(self, sd):
        _, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected[:3]

    @torch.no_grad()
    def _encode(self, enc):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return F.normalize(self.model(enc.to(self.device)).float(), dim=-1)

    @torch.no_grad()
    def recall10(self, split="val"):
        out = {}
        for i, d in enumerate(self.domains):
            dm = self.dms[i]
            items = torch.cat([self._encode(b.items) for b in dm.item_dataloader()])
            ds = dm.val_dataset if split == "val" else dm.test_dataset
            idx = self.val_idx[i] if split == "val" else list(range(len(ds)))
            hits = 0
            for k in range(0, len(idx), self.bs):
                b = dm.sequence_eval_collator([ds[j] for j in idx[k:k + self.bs]])
                top = (self._encode(b.sequence) @ items.T).topk(10, dim=-1).indices.cpu()
                hits += (top == b.labels.unsqueeze(1)).any(1).sum().item()
            out[d] = hits / len(idx)
        return out
