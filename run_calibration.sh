#!/bin/bash
ds_name="cifar100"
kl_val="0.0"
seed="1"
fstr="1"
modeltype="evidential"
hypa="1.5"
conda activate prompt

CUDA_VISIBLE_DEVICES=0 python calib_mar4_evid.py \
    --config-file configs/prompt/${ds_name}_al.yaml \
    SOLVER.FSTR  ${fstr} \
    SOLVER.FSVAL ${fstr} \
    SEED ${seed} \
    SOLVER.KL_VAL ${kl_val} \
    SOLVER.HYP_A ${hypa} \
    SOLVER.MODELTYPE ${modeltype} \
    OUTPUT_DIR "calibration_logs/"${modeltype}"_"${fstr}"_FSTR_"${ds_name}"_kl_"${kl_val}"/seed"${seed}