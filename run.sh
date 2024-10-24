#!/bin/bash
ds_name="cifar100"
kl_val="0.0"
seed="1"
fstr="1"
modeltype="evidential"
hypa="100"
conda activate prompt

CUDA_VISIBLE_DEVICES=0 python3 train_new_feb21_full.py \
    --config-file configs/prompt/${ds_name}_al.yaml \
    SEED ${seed} \
    SOLVER.KL_VAL ${kl_val} \
    SOLVER.MODELTYPE ${modeltype} \
    SOLVER.FSTR ${fstr} \
    SOLVER.HYP_A ${hypa} \
    OUTPUT_DIR "logs/"${modeltype}"_"${fstr}"_FSTR_"${ds_name}"_kl_"${kl_val}"/seed"${seed}
