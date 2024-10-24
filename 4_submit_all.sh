#!/bin/bash
modeltype="evidential"
identifier_name="Neurips_001"
CUDA_VISIBLE_DEVICES=0
ds_name="cifar10" # The dataset name
ctr=0


for fstr in 1 2 5 10 20 50; do 
    for kl_val in 0.0 1.0 0.1 10.0 100.0 1000.0; do  #  1000.0 5000.0 10000.0 50000.0 # 0.0 1.0 0.1 10.0 100.0 1000.0
        for sd in 1 2 3 4 5; do 
            export CUDA_VISIBLE_DEVICES
            seed=$((sd))
            timestamp=$(date +---20%y-%m-%d-%T---)

            python3 train_new_feb21_full.py \
                --config-file configs/prompt/${ds_name}_al.yaml \
                SEED ${seed} \
                SOLVER.KL_VAL ${kl_val} \
                SOLVER.MODELTYPE ${modeltype} \
                SOLVER.FSTR ${fstr} \
                OUTPUT_DIR "logs_live/"${timestamp}"res_mod_"${modeltype}"_"${fstr}"_FSTR_"${identifier_name}"_"${ds_name}"_kl_"${kl_val}"/seed"${seed}

        done
    done
done