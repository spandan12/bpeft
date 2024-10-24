# %%bash
# launch final training with five random seeds for VTAB-dmlab, sun397 and eurosat. The hyperparameters are the same from our paper.


#  for fstr in 2; do
#      for seed in "1"; do
#          CUDA_VISIBLE_DEVICES=0 python train_new_feb21_full.py \
#              --config-file configs/prompt/cifar10_al.yaml \
#              SOLVER.FSTR  ${fstr} \
#              SEED ${seed} \
#              OUTPUT_DIR "Temp_Del_soon_Mar25_cifar10_First_ce/seed${seed}"
#      done
#  done

for fstr in 10; do
    for kl_val in  0.0 0.1; do
        for hypa in 1.0; do
            for seed in "1"; do
                CUDA_VISIBLE_DEVICES=0 python calib_mar4_evid.py \
                    --config-file configs/prompt/cifar100_al.yaml \
                    SOLVER.FSTR  ${fstr} \
                    SOLVER.FSVAL 2 \
                    SOLVER.KL_VAL ${kl_val} \
                    SOLVER.HYP_A ${hypa} \
                    SEED ${seed} \
                    SOLVER.MODELTYPE 'evidential' \
                    OUTPUT_DIR "Final_runs_Apr10/cifar10/${fstr}_shot_cifar_10_kl_${kl_val}_m_${hypa}_${timestamp}/seed${seed}"
            done
        done
   done
done
