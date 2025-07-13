#!/bin/bash
ds_name="cifar10"
kl_val="0.1"
seed="1"
fstr="1"
modeltype="evidential"
conda activate prompt

CUDA_VISIBLE_DEVICES=0 python3 train_new_feb21_full.py \
    --config-file configs/prompt/${ds_name}_al.yaml \
    SEED ${seed} \
    SOLVER.KL_VAL ${kl_val} \
    SOLVER.MODELTYPE ${modeltype} \
    SOLVER.FSTR ${fstr} \
    OUTPUT_DIR "logs/"${modeltype}"_"${fstr}"_FSTR_"${ds_name}"_kl_"${kl_val}"/seed"${seed}
