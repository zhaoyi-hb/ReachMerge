#!/bin/bash
# RegMean, RegMean++ and ReachMerge (RegMean++ followed by dose calibration).
# usage: bash scripts/run_reachmerge.sh <backbone> <platform> <seed>
set -eu
source scripts/env.sh
setup $1 $2; SEED=$3
RUN_DIR=$1_$2_s$SEED
CK=checkpoints/$RUN_DIR
O=$REPO/outputs/$RUN_DIR; mkdir -p $O

python -m reachmerge.regmean    --backbone $BACKBONE --ckpt_dir $CK --domains $DOMS --out $O/regmean.pt
python -m reachmerge.regmean_pp --backbone $BACKBONE --ckpt_dir $CK --domains $DOMS --out $O/regmean_pp.pt
python -m reachmerge.dose       --backbone $BACKBONE --domains $DOMS --merged $O/regmean_pp.pt --out_dir $O/reachmerge

for m in regmean regmean_pp; do bash scripts/eval.sh $1 $2 $RUN_DIR $O/$m.pt $O/results/$m; done
bash scripts/eval.sh $1 $2 $RUN_DIR $O/reachmerge/state_dict.pt $O/results/reachmerge
python -m scripts.summarize --run $RUN_DIR --platform $2
