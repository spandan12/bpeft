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


from D_ALL.data_loaders import get_loaders_cif100_deep 
from D_ALL.data_loaders import get_loaders_svhn_deep
from D_ALL.data_loaders import get_loaders_cifar10_deep
from D_ALL.data_loader_tiny import get_loaders_tiny_deep
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

    # setup training env including loggers
    if  cfg.DATA.NAME == "cifar100":
        get_dataset_loader = get_loaders_cif100_deep
    elif cfg.DATA.NAME == "tiny":
        get_dataset_loader = get_loaders_tiny_deep
    elif cfg.DATA.NAME == "svhn":
        get_dataset_loader = get_loaders_svhn_deep
    elif cfg.DATA.NAME == "cifar10":
        get_dataset_loader = get_loaders_cifar10_deep
    else:
        print("See the dataloader")
        raise NotImplementedError

    al_cycle = 0
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")
    logger.info(f"UP... lr0.1_wd0.01/run1/prompt_ep{cfg.SOLVER.TOTAL_EPOCH}_AL_{al_cycle-1}.pth")
    logger.info(f"AL CYCLE... S... {al_cycle}")


    print(cfg.OUTPUT_DIR)
    data_path = "paste_the_data_path_here"

    seed = cfg.SEED

    the_path = "paste_the_path_here"
    if  cfg.DATA.NAME == "cifar100":
        data_path = f"{os.getcwd()}/{the_path}/results_feb22_FULL_DEEP_50P_100E_cifar100_kl_{cfg.SOLVER.KL_VAL}"
    else:
        data_path = f"{os.getcwd()}/{the_path}/results_feb_tiny_kl_{cfg.SOLVER.KL_VAL}"
    
    data_path += f"/seed{seed}/{cfg.DATA.NAME}/sup_vitb16_imagenet21k"
    data_path += "/lr0.1_wd0.01/run1/"
        
    cfg['MODEL']['WEIGHT_PATH'] = f'{data_path}'
    cfg['MODEL']['WEIGHT_PATH'] += f'/prompt_ep51.pth'
    # cfg['MODEL']['WEIGHT_PATH'] += f'/prompt_ep50_AL_0.pth'
    
    logger.info(f"Down... lr0.1_wd0.01/run1/prompt_ep{cfg.SOLVER.TOTAL_EPOCH}.pth")
    logger.info(f"Model... {cfg['MODEL']['WEIGHT_PATH']}")


    _,_,test_loader,_ = get_dataset_loader(cfg = cfg)
    # pdb.set_trace()
    logger.info("Constructing models...")
    model, cur_device = build_model(cfg)
    # pdb.set_trace()

    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)

    # model.eval()
    trainer.eval_classifier_deep_ece(test_loader, cfg=cfg)


def main(args):
    """main function to call from workflow"""

    # set up cfg and args
    cfg = setup(args)

    # Perform training.
    train(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
