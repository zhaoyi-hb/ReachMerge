#!/bin/bash
# Fine-tune one expert per domain with the MergeRec recipe.
# usage: bash scripts/train_experts.sh <blair_base|recformer_base> <amazon|goodreads> <seed>
set -eu
source scripts/env.sh
setup $1 $2; SEED=$3
RUN=$1_$2_s$SEED
cd $MERGEREC_ROOT
for d in $DOMS; do
  [ -f checkpoints/$RUN/$d/state_dict.pt ] && continue
  RID=${RUN}_${d}_$(date +%s)
  WANDB_RUN_ID=$RID python finetune_train.py --model_type $BACKBONE --model_path $MP --tokenizer_path $MP $MK \
    --batch_size 64 --negative_sample.in_batch --temperature 0.05 --warmup_steps 100 \
    --data_path datasets/$d --learning_rate 5e-5 --log_every_n_steps 50 --seed $SEED
  CKPT=$(ls -t MergeRecFineTune/$RID/checkpoints/*.ckpt | head -1)
  mkdir -p checkpoints/$RUN/$d
  python scripts/2_ft_postprocess/extract.py "$CKPT" checkpoints/$RUN/$d/
done
echo "experts in $MERGEREC_ROOT/checkpoints/$RUN"
