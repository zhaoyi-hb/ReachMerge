# ReachMerge

Code for the paper *Merged Models Move Too Little: Dose Calibration for Data-Isolated Cross-Domain Sequential Recommendation* (under review).

ReachMerge merges per-domain sequential recommenders without access to any other domain's logs. It
solves the closed-form RegMean problem layer by layer in depth order (RegMean++), then rescales the
merged update by a single factor `s` chosen on the source domains' validation splits.

## Layout

```
reachmerge/       RegMean, RegMean++, dose calibration, reproduction-ratio diagnostic
baselines/        coefficient merges (WA, TA, NAN, DARE), CAT-Merging, LARV, MAGIC, BMM
data/             construction of the Goodreads genre domains
scripts/          expert training, merging, evaluation, joint training
patches/          changes applied to the MergeRec code base
```

TIES and MergeRec (domain-wise / layer-wise) are run with the MergeRec implementation.

## Setup

The code builds on [MergeRec](https://github.com/DIALLab-SKKU/MergeRec) (data loaders, backbones,
TIES and the learned-coefficient baselines).

```bash
git clone https://github.com/DIALLab-SKKU/MergeRec third_party/MergeRec
cd third_party/MergeRec && git checkout 01812f5 && git apply ../../patches/mergerec_py310.patch && cd ../..
pip install -r third_party/MergeRec/requirements.txt optuna pyarrow
```

`mergerec_py310.patch` replaces Python 3.12 type syntax so that the code runs on Python 3.10 and
lets `finetune_test.py` load checkpoints without item embeddings.

For the joint-training reference, make a second copy and apply both patches; the second one
fixes how `ChainedDataset` unpacks indexed samples:

```bash
cp -r third_party/MergeRec third_party/MergeRec_joint
cd third_party/MergeRec_joint && git apply ../../patches/mergerec_joint.patch && cd ../..
```

Follow the MergeRec instructions to unpack the eight Amazon datasets and to download the
RecFormer checkpoint into `checkpoints/pretrained/`. Backbones are loaded from the Hugging Face
hub by default (`hyp1231/blair-roberta-base`, `allenai/longformer-base-4096`); set `BLAIR_PATH` /
`LONGFORMER_PATH` to use local copies.

### Goodreads

Download the `books` and `behavior` parquet shards of `liyucheng/goodreads` from the Hugging
Face hub, then

```bash
python data/build_goodreads.py --raw <goodreads_dir> --out third_party/MergeRec/datasets
```

This writes `G4_fiction`, `G4_romance`, `G4_nonfiction`, `G4_children`, `G4_poetry` and
`G4_business`. Each user's validation and test targets in one genre are removed from their
training sequences in all other genres; the script prints the fraction of test pairs found in
other domains' training data (zero).

## Running

All commands run from the repository root. `<backbone>` is `blair_base` or `recformer_base`,
`<platform>` is `amazon` or `goodreads`.

```bash
bash scripts/train_experts.sh   <backbone> <platform> <seed>   # one expert per domain
bash scripts/eval_experts.sh    <backbone> <platform> <seed>   # normalisers for relative R@10
bash scripts/run_reachmerge.sh  <backbone> <platform> <seed>   # RegMean, RegMean++, ReachMerge
bash scripts/run_baselines.sh   <backbone> <platform> <seed>   # all merging baselines
bash scripts/run_joint.sh       <backbone> <platform> <seed>   # pooled-log reference
python -m scripts.summarize --run <backbone>_<platform>_s<seed> --platform <platform>
```

The paper uses seeds 42-45 for BLaIR and 42-43 for RecFormer. Merged models and per-domain
CSV files are written to `outputs/<backbone>_<platform>_s<seed>/`.

The diagnostic of Section 4 (reproduction ratio and least-squares scale) is computed with

```bash
python -m reachmerge.gram_cache  --backbone <b> --ckpt_dir checkpoints/<run> --domains ... --cache outputs/<run>/gram_cache
python -m reachmerge.diagnostics --backbone <b> --cache outputs/<run>/gram_cache --merged outputs/<run>/regmean_pp.pt
```

### Settings

| | |
|---|---|
| Gram statistics | 2,000 training sequences and 2,000 item texts per domain, padding removed |
| RegMean | off-diagonal shrinkage 0.9, ridge 1e-2 x mean diagonal, fallback to the expert average if the update exceeds 3x the longest expert update |
| Dose grid | {1.0, 1.1, 1.2, 1.3, 1.45}, selected on the full validation split of every domain |
| Task Arithmetic / TIES / DARE | coefficient 0.2 / 0.8 / 0.2 with 10% drop |
| CAT-Merging | 128 probe items per domain, scaling 0.4 |
| BMM | 3 blocks x (scale + 4 ridge groups), 50 GP trials, 1,000 validation users per domain |

The coefficient-based baselines use the setting with the best test score on seed 42, fixed for
all seeds.

## Hardware

Every step runs on one 80 GB GPU. Training the joint-data reference takes 26-60 GPU hours per
platform and seed with RecFormer.

## License

MIT. The MergeRec code base is distributed under its own MIT license.
