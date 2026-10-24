"""Relative R@10 (% of each domain's own expert, averaged over domains) for every result in a run."""
import argparse
import csv
import glob
import os
import statistics

DOMAINS = {
    "amazon": "Arts Beauty Instruments Office Pantry Scientific Sports Toys".split(),
    "goodreads": "G4_fiction G4_romance G4_nonfiction G4_children G4_poetry G4_business".split(),
}


def read(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["dataset"].split("/")[-1]] = float(r["test/Recall@10"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="outputs/<run>")
    p.add_argument("--platform", choices=list(DOMAINS), required=True)
    a = p.parse_args()
    res = f"outputs/{a.run}/results"
    D = DOMAINS[a.platform]
    expert = {}
    for d in D:
        expert.update(read(f"{res}/expert_{d}.csv"))
    methods = {}
    for f in sorted(glob.glob(f"{res}/*.csv")):
        name = os.path.basename(f)[:-4]
        if name.startswith("expert_"):
            continue
        m = name
        for d in D:
            if name.endswith("_" + d):
                m = name[: -len(d) - 1]
        methods.setdefault(m, {}).update(read(f))
    print(f"{'method':<24}{'Rel. (%)':>10}{'R@10':>9}")
    for m, v in methods.items():
        if not all(d in v for d in D):
            continue
        rel = statistics.mean(100 * v[d] / expert[d] for d in D)
        print(f"{m:<24}{rel:>10.2f}{statistics.mean(v[d] for d in D):>9.4f}")


if __name__ == "__main__":
    main()
