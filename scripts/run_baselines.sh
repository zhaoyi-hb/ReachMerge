#!/bin/bash
# All baselines of the main table for one backbone / platform / seed.
# Coefficients are those reported in the paper (TA 0.2, TIES 0.8, DARE 0.2 with 10% drop, CAT scaling 0.4).
# usage: bash scripts/run_baselines.sh <backbone> <platform> <seed>
set -eu
source scripts/env.sh
setup $1 $2; SEED=$3
RUN_DIR=$1_$2_s$SEED
CK=checkpoints/$RUN_DIR
O=$REPO/outputs/$RUN_DIR; B=$O/baselines; mkdir -p $B $O/results
DP=""; CP=""; IP=""
for d in $DOMS; do DP="$DP datasets/$d"; CP="$CP $CK/$d/state_dict.pt"; IP="$IP $CK/$d/item_embedding.pt"; done

# weight averaging, Task Arithmetic, NAN, DARE, CAT-Merging, LARV, MAGIC
python -m baselines.coefficient --backbone $BACKBONE --ckpt_dir $CK --domains $DOMS --out_dir $B
python -m baselines.cat_merging --backbone $BACKBONE --ckpt_dir $CK --domains $DOMS --out $B/cat/state_dict.pt
python -m baselines.larv        --backbone $BACKBONE --ckpt_dir $CK --domains $DOMS --out_dir $B
python -m baselines.magic       --backbone $BACKBONE --ckpt_dir $CK --domains $DOMS --out_dir $B
for m in wa ta nan dare cat larv_ta larv_nan magic_ta magic_nan; do
  bash scripts/eval.sh $1 $2 $RUN_DIR $B/$m/state_dict.pt $O/results/$m
done

# BMM around the RegMean anchor (run_reachmerge.sh first)
python -m reachmerge.gram_cache --backbone $BACKBONE --ckpt_dir $CK --domains $DOMS --cache $O/gram_cache
python -m baselines.bmm --backbone $BACKBONE --domains $DOMS --cache $O/gram_cache --anchor $O/regmean.pt --out $B/bmm/state_dict.pt
bash scripts/eval.sh $1 $2 $RUN_DIR $B/bmm/state_dict.pt $O/results/bmm

cd $MERGEREC_ROOT
# TIES with the MergeRec implementation
python merge_test.py --model_type $BACKBONE --model_path $MP --tokenizer_path $MP $MK \
  --data_paths $DP --finetune_checkpoint_paths $CP --train_data_split test --test_data_split test \
  --learn_type task_wise --batch_size 32 --merge_type ties --weight_file uniform --weight_file_line 0.8 \
  --metrics_path $O/results/ties.csv
# MergeRec, domain-wise and layer-wise
for v in task_wise layer_wise; do
  python merge_train.py --model_type $BACKBONE --model_path $MP --tokenizer_path $MP $MK \
    --data_paths $DP --test_data_paths $DP --finetune_checkpoint_paths $CP \
    --item_embeddings_paths $IP --sequence_embeddings_paths $IP \
    --batch_size 16 --train_data_split item --test_data_split test --sample_method random \
    --max_steps 500 --loss_type SINGLE_PSEUDO_LABEL_KD --coefficient 1000 \
    --initial_per_weight 0.2 --merge_type task_vector --learn_type $v \
    --learning_rate 0.001 --seed $SEED --metrics_path $O/results/mergerec_$v.csv
done
