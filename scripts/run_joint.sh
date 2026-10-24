#!/bin/bash
# Reference model trained on the pooled logs of all domains (not data-isolated).
# Uses a separate MergeRec copy with patches/mergerec_joint.patch applied.
# usage: bash scripts/run_joint.sh <backbone> <platform> <seed>
set -eu
source scripts/env.sh
setup $1 $2; SEED=$3
RUN=$1_$2_s$SEED; O=$REPO/outputs/$RUN; mkdir -p $O/results
DP=""; for d in $DOMS; do DP="$DP datasets/$d"; done
cd $JOINT_ROOT
RID=joint_${RUN}_$(date +%s)
WANDB_RUN_ID=$RID python finetune_joint.py --model_type $BACKBONE --model_path $MP --tokenizer_path $MP $MK \
  --data_paths $DP --batch_size 64 --negative_sample.in_batch --temperature 0.05 --warmup_steps 100 \
  --learning_rate 5e-5 --log_every_n_steps 50 --seed $SEED
CKPT=$(ls -t MergeRecJoint/$RID/checkpoints/*.ckpt | head -1)
python - "$CKPT" $O/joint.pt <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu"); sd = c.get("state_dict", c)
torch.save({k: v for k, v in sd.items() if torch.is_tensor(v) and not k.startswith("item_embeddings")}, sys.argv[2])
PY
cd $MERGEREC_ROOT
for d in $DOMS; do
  python finetune_test.py --model_type $BACKBONE --model_path $MP --tokenizer_path $MP $MK --data_path datasets/$d \
    --data_split test --batch_size 32 --finetune_checkpoint_path $O/joint.pt --metrics_path $O/results/joint_$d.csv
done
