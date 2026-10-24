# Paths and domain lists shared by the scripts. Source it from the repository root.
export REPO=$(pwd)
export MERGEREC_ROOT=${MERGEREC_ROOT:-$REPO/third_party/MergeRec}
export JOINT_ROOT=${JOINT_ROOT:-$REPO/third_party/MergeRec_joint}
export PYTHONPATH=$REPO:$PYTHONPATH
export BLAIR_PATH=${BLAIR_PATH:-hyp1231/blair-roberta-base}
export LONGFORMER_PATH=${LONGFORMER_PATH:-allenai/longformer-base-4096}
export WANDB_MODE=${WANDB_MODE:-offline} TOKENIZERS_PARALLELISM=false

AMAZON="Arts Beauty Instruments Office Pantry Scientific Sports Toys"
GOODREADS="G4_fiction G4_romance G4_nonfiction G4_children G4_poetry G4_business"

# setup <backbone> <platform>: sets DOMS, MP (backbone path) and MK (extra model args)
setup() {
  BACKBONE=$1
  case $2 in amazon) DOMS=$AMAZON;; goodreads) DOMS=$GOODREADS;; *) echo "unknown platform $2"; exit 1;; esac
  if [ "$BACKBONE" = blair_base ]; then MP=$BLAIR_PATH; MK=""
  else MP=$LONGFORMER_PATH; MK="--model_kwargs ckpt_path checkpoints/pretrained/recformer_base_pretrained.pt"; fi
}
