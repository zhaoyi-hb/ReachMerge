#!/bin/bash
# Test a merged state_dict on every domain of a platform.
# usage: bash scripts/eval.sh <backbone> <platform> <run> <state_dict> <out_prefix> [split]
#   <run> is the expert folder under checkpoints/ (any expert of it satisfies the loader's required argument)
set -eu
source scripts/env.sh
setup $1 $2; RUN=$3; SD=$(realpath $4); OUT=$(realpath -m $5); SPLIT=${6:-test}
mkdir -p $(dirname $OUT)
F0=$(echo $DOMS | awk '{print $1}')
cd $MERGEREC_ROOT
for d in $DOMS; do
  STATE_PATH=$SD python $REPO/reachmerge/eval_state.py --model_type ${BACKBONE^^} --model_path $MP --tokenizer_path $MP $MK \
    --train_data_split test --test_data_split $SPLIT --learn_type task_wise --batch_size 32 --merge_type task_vector \
    --finetune_checkpoint_paths checkpoints/$RUN/$F0/state_dict.pt --data_paths datasets/$d --metrics_path ${OUT}_$d.csv
done
