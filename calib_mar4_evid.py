#!/usr/bin/env python3
"""
major actions here: fine-tune the features and evaluate different settings
"""
import os
import pdb
import torch
import warnings

import numpy as np
import random

from time import sleep
from random import randint

import src.utils.logging as logging
from src.configs.config import get_cfg

from src.engine.evaluator import Evaluator
from src.engine.trainer_evidential import Trainer
from src.models.build_model import build_model
from src.utils.file_io import PathManager

import matplotlib.pyplot as plt


from D_ALL.data_loaders import get_loaders_cif100_deep, get_loaders_svhn_deep,  get_loaders_cifar10_deep, get_svhn_loader_as_ood
from launch import default_argument_parser, logging_train_setup
warnings.filterwarnings("ignore")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    output_folder = os.path.join(
        cfg.DATA.NAME, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}")

    # train cfg.RUN_N_TIMES times
    count = 1
    while count <= cfg.RUN_N_TIMES:
        output_path = os.path.join(output_dir, output_folder, f"run{count}")
        # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
        sleep(randint(3, 30))
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)
            cfg.OUTPUT_DIR = output_path
            break
        else:
            count += 1
    if count > cfg.RUN_N_TIMES:
        raise ValueError(
            f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")

    cfg.freeze()
    return cfg



def train(cfg, args):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # main training / eval actions here

    # fix the seed for reproducibility
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(cfg.SEED)

    openset_eval = False
    ood_dataset_name = ''
    # setup training env including loggers
    if  cfg.DATA.NAME == "cifar100":
        get_dataset_loader = get_loaders_cif100_deep
        get_openset_loader = get_svhn_loader_as_ood
        # openset_eval = True
        ood_dataset_name = 'svhn'
    elif cfg.DATA.NAME == "tiny":
        get_dataset_loader = get_loaders_tiny_deep
    elif cfg.DATA.NAME == "svhn":
        get_dataset_loader = get_loaders_svhn_deep
    elif cfg.DATA.NAME == "cifar10":
        get_dataset_loader = get_loaders_cifar10_deep
        get_openset_loader = get_loaders_cif100_deep
        # openset_eval = True
        ood_dataset_name = 'cifar100'
    else:
        print("See the dataloader")
        raise NotImplementedError

    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")


    print(cfg.OUTPUT_DIR)
    data_path = # Supply the data path 


    if  cfg.DATA.NAME == "cifar100":
        data_path = f"{supply_the_link}/4_res_mod_evidential_{cfg.SOLVER.FSTR}_FSTR_Mar15.0939_cifar100_kl_{cfg.SOLVER.KL_VAL}"
    elif cfg.DATA.NAME == 'cifar10':
        data_path = f"{supply_the_link}/res_mod_ce_{cfg.SOLVER.FSTR}_FSTR_Apr8.0640._cifar10_kl_{cfg.SOLVER.KL_VAL}"
    else:
        data_path = f"{supply_the_data_path}"
    
    data_path += f"/seed{cfg.SEED}/{cfg.DATA.NAME}/sup_vitb16_imagenet21k"
    data_path += "/lr0.1_wd0.01/run1/"
        
    cfg['MODEL']['WEIGHT_PATH'] = f'{data_path}'
    # cfg['MODEL']['WEIGHT_PATH'] += f'/prompt_ep11.pth'
    cfg['MODEL']['WEIGHT_PATH'] += f'/prompt_ep50.pth' # The model
    
    logger.info(f"Down... lr0.1_wd0.01/run1/prompt_ep{cfg.SOLVER.TOTAL_EPOCH}.pth")
    logger.info(f"Model... {cfg['MODEL']['WEIGHT_PATH']}")


    _,fs_val_loader,test_loader,_ = get_dataset_loader(cfg = cfg)
    if openset_eval == True:
        if ood_dataset_name == 'svhn':
            openset_test_loader = get_openset_loader()
        else:
            _,_,openset_test_loader,_ = get_openset_loader(cfg = cfg)
    
    # pdb.set_trace()
    logger.info("Constructing models...")
    model, cur_device = build_model(cfg)
    # pdb.set_trace()

    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)

    # model.eval()
    
    # trainer.eval_classifier_deep_ece(fs_val_loader,  id = 'fs_val')
    # print("CALIBRATION")
    # print("Test set: ")
    # # trainer.calib_val_set(test_loader, cfg=cfg, tau = 1.0)
    # print("Validation Set")
    # tau = trainer.calib_val_set(fs_val_loader, cfg=cfg, tau = 1.0)
    # print("test set with tau = ", tau)
    # # trainer.calib_val_set(test_loader, cfg=cfg, tau = tau)
    # print("CALIBRATION END")


    trainer.eval_classifier_deep_ece(fs_val_loader,  id = 'fs_val')
    test_pred_matrix = trainer.eval_classifier_deep_ece(test_loader, id = 'test')

    if openset_eval == True:
        openset_pred_matrix = trainer.eval_classifier_deep_openset(openset_test_loader, id='openset')
        
        setting_name = f'{cfg.SOLVER.FSTR}_shot_{cfg.DATA.NAME}-_KL_{cfg.SOLVER.KL_VAL}'

        # Index of vacuity in pred_matrix is 2.
        plot_ood_separation_diagram(
        in_distribution_values = test_pred_matrix[:, 2], 
        ood_values = openset_pred_matrix[:, 2], 
        title = setting_name, 
        x_axis = 'vacuity', 
        in_distribution_dataset = cfg.DATA.NAME, 
        ood_dataset = ood_dataset_name, 
        output_dir = cfg.OUTPUT_DIR
        )

        # Index of maxp in pred_matrix is 3.
        plot_ood_separation_diagram(
        in_distribution_values = test_pred_matrix[:, 3], 
        ood_values = openset_pred_matrix[:, 3], 
        title = setting_name, 
        x_axis = 'Max_probability', 
        in_distribution_dataset = cfg.DATA.NAME, 
        ood_dataset = ood_dataset_name, 
        output_dir = cfg.OUTPUT_DIR
        )

    import sys 
    sys.exit()



def plot_ood_separation_diagram(
    in_distribution_values, 
    ood_values, 
    title, 
    x_axis, 
    in_distribution_dataset, 
    ood_dataset, 
    output_dir
    ):

    fig, ax = plt.subplots()
    ax.set_xlabel(x_axis, fontsize=20)
    ax.set_ylabel('Density', fontsize=20)

    ax.hist(
        [in_distribution_values, ood_values], 
        bins=np.arange(0,1,0.1), 
        density=True, 
        label=[f"In Distribution: {in_distribution_dataset}", f"OOD: {ood_dataset}"]
    )
    ax.legend(fontsize=16)
    ax.set_title(title, fontsize=25)
    fig.savefig(f'{output_dir}/{title}_{x_axis}.png')



def main(args):
    """main function to call from workflow"""

    # set up cfg and args
    cfg = setup(args)

    # Perform training.
    train(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
