# %%bash
# launch final training with five random seeds for VTAB-dmlab, sun397 and eurosat. The hyperparameters are the same from our paper.


# for fstr in 2; do
#     for seed in "1"; do
#         CUDA_VISIBLE_DEVICES=0 python train_new_feb21_full.py \
#             --config-file configs/prompt/cifar10_al.yaml \
#             SOLVER.FSTR  ${fstr} \
#             SEED ${seed} \
#             OUTPUT_DIR "Mar7_cifar10_First_ce/seed${seed}"
#     done
# done

for fstr in 1; do
    for seed in "1"; do
        CUDA_VISIBLE_DEVICES=0 python calib_mar4_evid.py \
            --config-file configs/prompt/cifar10_al.yaml \
            SOLVER.FSTR  ${fstr} \
            SOLVER.FSVAL ${fstr} \
            SEED ${seed} \
            SOLVER.MODELTYPE 'evidential' \
            OUTPUT_DIR "Apr10"${fstr}"_shot_cifar10_4_kl_1.0/seed${seed}"

    done
done