#!/bin/bash
# Test every expert on its own domain; these scores normalise the relative R@10.
# usage: bash scripts/eval_experts.sh <backbone> <platform> <seed>
set -eu
source scripts/env.sh
setup $1 $2; SEED=$3
RUN=$1_$2_s$SEED; O=$REPO/outputs/$RUN/results; mkdir -p $O
cd $MERGEREC_ROOT
for d in $DOMS; do
  python finetune_test.py --model_type $BACKBONE --model_path $MP --tokenizer_path $MP $MK --data_path datasets/$d \
    --data_split test --batch_size 32 --finetune_checkpoint_path checkpoints/$RUN/$d/state_dict.pt --metrics_path $O/expert_$d.csv
done
