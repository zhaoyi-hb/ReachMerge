"""Build the six Goodreads genre domains in MergeRec format.

A book can belong to several genres. For each (user, genre) the last two interactions by
date_added are the validation and test targets. To keep the domains isolated, every target
of a user is removed from that user's training sequences in all other genres, so no expert
is trained on another domain's evaluation answers.

Input: the `books` and `behavior` parquet shards of the Goodreads dump on the Hugging Face
hub (liyucheng/goodreads). The first two behavior shards are used.
"""
import argparse
import collections
import glob
import itertools
import json
import os
import random
import statistics as st

import pyarrow.parquet as pq

GENRES = ["fiction", "romance", "nonfiction", "children", "poetry", "business"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="folder with books/ and behavior/ parquet shards")
    p.add_argument("--out", required=True, help="MergeRec datasets/ folder")
    p.add_argument("--core", type=int, default=5)
    p.add_argument("--max_users", type=int, default=40000)
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--n_behavior_shards", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    books = {}
    for f in sorted(glob.glob(f"{a.raw}/books/*.parquet")):
        for b in pq.ParquetFile(f).iter_batches(batch_size=50000, columns=["book_id", "title", "description", "genres"]):
            d = b.to_pydict()
            for i in range(len(d["book_id"])):
                t, desc = d["title"][i] or "", d["description"][i] or ""
                g = d["genres"][i] or []
                gs = [str(x) for x in (g if isinstance(g, (list, tuple)) else [g])]
                if (t or desc) and any(x in GENRES for x in gs):
                    books[str(d["book_id"][i])] = {"title": t[:300], "description": desc[:400], "genres": gs}

    rows = []
    for f in sorted(glob.glob(f"{a.raw}/behavior/*.parquet"))[:a.n_behavior_shards]:
        for b in pq.ParquetFile(f).iter_batches(batch_size=200000, columns=["user_id", "book_id", "date_added"]):
            d = b.to_pydict()
            for i in range(len(d["user_id"])):
                bid = str(d["book_id"][i])
                if bid in books:
                    rows.append((str(d["user_id"][i]), bid, str(d["date_added"][i] or "")))
    print("interactions:", len(rows), flush=True)

    by_genre = collections.defaultdict(list)
    for u, bid, ts in rows:
        for g in books[bid]["genres"]:
            if g in GENRES:
                by_genre[g].append((u, bid, ts))

    seqs = {}
    for g in GENRES:
        sub = by_genre[g]
        users = sorted({u for u, _, _ in sub})
        if len(users) > a.max_users:
            keep = set(random.Random(a.seed).sample(users, a.max_users))
            sub = [x for x in sub if x[0] in keep]
        for _ in range(3):
            uc = collections.Counter(u for u, _, _ in sub)
            ic = collections.Counter(b for _, b, _ in sub)
            sub = [(u, b, t) for u, b, t in sub if uc[u] >= a.core and ic[b] >= a.core]
        s = collections.defaultdict(list)
        for u, b, t in sub:
            s[u].append((t, b))
        seqs[g] = {u: [b for _, b in sorted(x)][-a.max_len:] for u, x in s.items() if len(x) >= a.core}

    targets = collections.defaultdict(set)
    for g in GENRES:
        for u, x in seqs[g].items():
            targets[u].update(x[-2:])

    stats, pairs = {}, {}
    for g in GENRES:
        train, val, test, items, removed = {}, {}, {}, set(), 0
        for u, x in seqs[g].items():
            ans = x[-2:]
            tr = [b for b in x[:-2] if not (b in targets[u] and b not in ans)]
            removed += len(x) - 2 - len(tr)
            if len(tr) < a.core - 2:
                continue
            train[u], val[u], test[u] = tr, [ans[0]], [ans[1]]
            items.update(tr)
            items.update(ans)
        items = sorted(items)
        users = sorted(train)
        smap = {b: k for k, b in enumerate(items)}
        umap = {u: k for k, u in enumerate(users)}
        out = {
            "smap": smap, "umap": umap,
            "train": {umap[u]: [smap[b] for b in train[u]] for u in users},
            "val": {umap[u]: [smap[b] for b in val[u]] for u in users},
            "test": {umap[u]: [smap[b] for b in test[u]] for u in users},
            "meta_data": {b: {"title": books[b]["title"], "description": books[b]["description"],
                              "genres": " ".join(books[b]["genres"])[:120]} for b in items},
        }
        dst = f"{a.out}/G4_{g}"
        os.makedirs(dst, exist_ok=True)
        for name, obj in out.items():
            json.dump(obj, open(f"{dst}/{name}.json", "w"))
        lens = [len(v) for v in out["train"].values()]
        stats[f"G4_{g}"] = {"items": len(smap), "users": len(umap), "mean_train_len": round(st.mean(lens), 2),
                            "removed_cross_domain_targets": removed}
        pairs[g] = {"train": {(u, b) for u in users for b in train[u] + val[u]}, "test": {(u, test[u][0]) for u in users}}
        print(g, stats[f"G4_{g}"], flush=True)
    json.dump(stats, open(f"{a.out}/goodreads_stats.json", "w"), indent=1)

    leak = []
    for g in GENRES:
        other = set().union(*[pairs[o]["train"] for o in GENRES if o != g])
        leak.append(sum(x in other for x in pairs[g]["test"]) / len(pairs[g]["test"]))
    overlap = [len(set(json.load(open(f"{a.out}/G4_{x}/smap.json"))) & set(json.load(open(f"{a.out}/G4_{y}/smap.json"))))
               / min(stats[f"G4_{x}"]["items"], stats[f"G4_{y}"]["items"]) for x, y in itertools.combinations(GENRES, 2)]
    print("test pairs seen in other domains' train/val:", [round(x, 4) for x in leak])
    print("mean item overlap between genres:", round(st.mean(overlap), 3))


if __name__ == "__main__":
    main()
