#!/usr/bin/env python3
"""
a trainer class
"""
import random
import datetime
import time
import torch
import torch.nn as nn
import os
import pandas as pd

import pdb
import itertools
import matplotlib.pyplot as plt

import numpy as np

import torch.nn.functional as F

from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import logging
from ..utils.train_utils import AverageMeter, gpu_mem_usage

logger = logging.get_logger("visual_prompt")


class Trainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """
    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        evaluator: Evaluator,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # solver related
        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.cls_weights = [1. for _ in range(self.cfg.DATA.NUMBER_CLASSES)]

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            # checkpointables = [key for key in self.checkpointer.checkpointables 
                            #    if key not in ["head.last_layer.bias",  "head.last_layer.weight"]]
            # self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            checkpoint = torch.load(cfg.MODEL.WEIGHT_PATH)
            shallow_prompt = torch.from_numpy(checkpoint['shallow_prompt'])
            self.model.enc.transformer.prompt_embeddings.data = shallow_prompt.to(self.device)
            self.model.head.load_state_dict(checkpoint['head'])
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

    def forward_one_batch(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, outputs

    def get_input(self, data):
        return data[0].float(), data[1]

    def train_classifier(self, train_loader, val_loader, test_loader, al_cycle=0):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        self.model.eval()
        self.save_prompt(0, al_cycle = al_cycle)

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        # DONE: comment next 2 lines, then the next line
        # self.cls_weights = train_loader.dataset.get_class_weights(
            # self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # self.cls_weights = [1. for _ in range(100)]
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break
                
                X, targets = self.get_input(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                train_loss, _ = self.forward_one_batch(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch*total_data*(total_epoch-epoch-1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
             # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()

            # Enable eval mode
            self.model.eval()

            # %  epoch%5 == 0 or 
            if epoch == 0 or epoch > 48 or epoch+2 > self.cfg.SOLVER.TOTAL_EPOCH:
                self.save_prompt(epoch + 1, al_cycle = al_cycle)

                # eval at each epoch for single gpu training
                self.evaluator.update_iteration(epoch)
                self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
                if test_loader is not None:
                    self.eval_classifier(
                        test_loader, "test", epoch == total_epoch - 1)

                # check the patience
                t_name = "val_" + val_loader.dataset.name
                try:
                    curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
                except KeyError:
                    return

                if curr_acc > best_metric:
                    best_metric = curr_acc
                    best_epoch = epoch + 1
                    logger.info(
                        f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
                    patience = 0
                else:
                    patience += 1
                if patience >= self.cfg.SOLVER.PATIENCE:
                    logger.info("No improvement. Breaking out of loop.")
                    break

        # save the last checkpoints
        if self.cfg.MODEL.SAVE_CKPT:
            Checkpointer(
                self.model,
                save_dir=self.cfg.OUTPUT_DIR,
                save_to_disk=True
            ).save("last_model")

    @torch.no_grad()
    def save_prompt(self, epoch, al_cycle=0):
        # only save the prompt embed if below conditions are satisfied
        if self.cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH:
            if self.cfg.MODEL.TYPE == "vit" and "prompt" in self.cfg.MODEL.TRANSFER_TYPE:
                prompt_embds = self.model.enc.transformer.prompt_embeddings.cpu().numpy()
                out = {"shallow_prompt": prompt_embds}
                out['head'] = self.model.head.state_dict()
                if self.cfg.MODEL.PROMPT.DEEP:
                    deep_embds = self.model.enc.transformer.deep_prompt_embeddings.cpu().numpy()
                    out["deep_prompt"] = deep_embds
                torch.save(out, os.path.join(
                    self.cfg.OUTPUT_DIR, f"prompt_ep{epoch}_AL_{al_cycle}.pth"))

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")
            
    @torch.no_grad()
    def get_pred_list(self, data_loader, tiny=False):
        '''
         zip(all_predicted_labels, all_entropy, all_labels, all_index)
        '''
        all_labels,all_predicted_labels, all_entropy, all_index = [], [], [], []
        all_paths = []
        bvs_prob_diff = []
        max_prob_all = []
        correct_predictions_sum = 0
        for idx, input_data in enumerate(data_loader):
            if idx > 20: break
            if tiny:
                X, targets = self.get_input(input_data[:2])
                paths = input_data[2]
                
            else:
                X, targets = self.get_input(input_data)
                paths = [0 for _ in range(len(targets))]
            
            # pdb.set_trace()
            all_paths += [x for x in paths]


            _, outputs = self.forward_one_batch(X, targets, False)
            probabilities = F.softmax(outputs, dim = 1).to(targets.device)
            # pdb.set_trace()

            entropy = -torch.sum(probabilities * torch.log(probabilities), dim = 1)
            predicted_classes = torch.argmax(probabilities, dim = 1)

            indices = [idx*len(entropy) + a for a in range(len(entropy))]

            all_labels += targets.tolist()
            all_predicted_labels += predicted_classes.tolist()
            all_entropy += entropy.tolist()
            all_index += indices

            top2_prob, _ = probabilities.topk(2, dim = 1)
            prob_diff = top2_prob[:,0] - top2_prob[:,1]
            bvs_prob_diff += prob_diff.tolist()
            max_prob_all += top2_prob[:,0].tolist()

            # print(entropy)
            correct_predictions = (predicted_classes == targets).sum().item()/len(predicted_classes)
            correct_predictions_sum += correct_predictions
            # pdb.set_trace()

            if idx%10 == 0:
                print(idx, correct_predictions)
        
        logger.info(f"Accuracy...  {correct_predictions_sum/(idx+1)}")

        one_list = zip(all_predicted_labels, all_entropy, all_labels,all_index, all_paths, bvs_prob_diff, max_prob_all)
        one_list = list(one_list)
        # pdb.set_trace()

        logger.info(f"Num classes... {len(set(all_labels))} Number of classes predicted... {len(set(all_predicted_labels))}")

        return one_list
    
    @torch.no_grad()
    def get_pred_all(self, data_loader, tiny=False):
        '''
         zip(all_predicted_labels, all_entropy, all_labels, all_index)
        '''
        all_labels,all_predicted_labels, all_entropy, all_index = [], [], [], []
        all_paths = []
        bvs_prob_diff = []
        max_prob_all = []
        all_predictions = None
        correct_predictions_sum = 0
        for idx, input_data in enumerate(data_loader):
            if idx > 20: break
            if tiny:
                X, targets = self.get_input(input_data[:2])
                paths = input_data[2]
                
            else:
                X, targets = self.get_input(input_data)
                paths = [0 for _ in range(len(targets))]
            
            _, outputs = self.forward_one_batch(X, targets, False)
            probabilities = F.softmax(outputs, dim = 1).to(targets.device)

            if all_predictions is None:
                all_predictions = probabilities.detach().cpu().numpy()
            else:
                attach = probabilities.detach().cpu().numpy()
                all_predictions = np.concatenate((all_predictions, attach), axis = 0)
            
            predicted_classes = torch.argmax(probabilities, dim = 1)

            indices = [idx*len(predicted_classes) + a for a in range(len(predicted_classes))]

            all_labels += targets.tolist()
            all_predicted_labels += predicted_classes.tolist()
            all_index += indices

            
            # print(entropy)
            correct_predictions = (predicted_classes == targets).sum().item()/len(predicted_classes)
            correct_predictions_sum += correct_predictions
            # pdb.set_trace()

            if idx%10 == 0:
                print(idx, correct_predictions)
        
        logger.info(f"Accuracy...  {correct_predictions_sum/(idx+1)}")

        all_predicted_labels = np.array(all_predicted_labels)
        all_labels = np.array(all_labels)
        all_index = np.array(all_index)
        one_np_mat = np.concatenate((all_predicted_labels, all_labels, all_index, all_predictions), axis = 1)
        return one_np_mat
    
    def sort_one_list(self, one_list, strategy):
        if strategy == "class_and_entropy":
            one_list.sort(key = lambda x: (x[0], -x[1]))
        elif strategy == "entropy":
            one_list.sort(key = lambda x: (-x[1]))
        elif strategy == "random":
            random.shuffle(one_list)
        elif strategy == "class_and_random":
            random.shuffle(one_list)
            one_list.sort(key = lambda x: (x[0]))
        elif strategy == "bvs":
            one_list.sort(key = lambda x: (x[5]))
        elif strategy == "class_and_bvs":
            one_list.sort(key = lambda x: (x[0], x[5]))
        return one_list
    
    def get_previous_data_path(self, al_cycle = 0, cfg=None):
        if not cfg:
            print("Pass CFG")
            raise NotImplementedError
        
        
        if al_cycle == 0:
            f_name = cfg.SOLVER.INIT_FILE_NAME
            return f"./D_ALL/References/ys_dataset/{f_name}"
        
        elif al_cycle >= 1:
            
            old_sample_count = int(cfg.SOLVER.INIT_POOL + cfg.SOLVER.SEL_AL*al_cycle)
            f_name = f'{cfg.SOLVER.STRATEGY}_after_{al_cycle}_{old_sample_count}'
            logger.info(f"The old file... {f_name}")
            if cfg.DATA.NAME == "tiny":
                return f"{self.cfg.OUTPUT_DIR}/al_data/{f_name}.csv"
            else:
                return f"{self.cfg.OUTPUT_DIR}/al_data/{f_name}.npy"
        
    def save_al_data(self, total_new, cfg, al_cycle):

        
        al_cycle += 1
        
        num_samples_new = cfg.SOLVER.INIT_POOL + cfg.SOLVER.SEL_AL*al_cycle
        al_save_path = f"{self.cfg.OUTPUT_DIR}/al_data/{cfg.SOLVER.STRATEGY}_after_{al_cycle}_{num_samples_new}"
        if not os.path.exists(f"{self.cfg.OUTPUT_DIR}/al_data/"):
            os.mkdir(f"{self.cfg.OUTPUT_DIR}/al_data/")
        print("AL save Path: ", al_save_path)
        if cfg.DATA.NAME == "tiny":
            al_save_path += ".csv"
            total_new.to_csv(al_save_path)
        else:
            al_save_path += ".npy"
            np_save_arr = np.array(total_new)
            np.save(al_save_path, np_save_arr)
    
    @torch.no_grad()
    def eval_classifier_deep(self, data_loader, al_cycle = 0, cfg=None):
        """evaluate classifier"""

        if not cfg:
            print("Pass CFG")
            raise NotImplementedError



        one_list = self.get_pred_list(data_loader, tiny = cfg.DATA.NAME=="tiny")
        one_list = self.sort_one_list(one_list, cfg.SOLVER.STRATEGY)

        previous_al_data_path = self.get_previous_data_path(al_cycle=al_cycle, cfg=cfg)

        if cfg.DATA.NAME == "tiny":
            selected_data_already = pd.read_csv(previous_al_data_path)
        else:
            selected_data_already = np.load(previous_al_data_path)

        logger.info(f"Total: {len(one_list)}, Data indices... {len(selected_data_already)},{selected_data_already}")
        logger.info(f"Data indices Unique... {len(set(list(selected_data_already)))}")
        if cfg.DATA.NAME != "tiny":
            selected_data_already = set(list(selected_data_already))
            logger.info(f"Data indices Unique... {len(selected_data_already)}")

        new_indices, new_indices_class = [], []
        temp_count =0

        new_indices_path = []

        per_class_to_add = int(cfg.SOLVER.SEL_AL//cfg.DATA.NUMBER_CLASSES)
        if cfg.SOLVER.STRATEGY in ["class_and_entropy","class_and_random", "class_and_bvs"]:
            current_class = one_list[0][0]
            num_elements_added = 0
            for d in one_list:
                if d[0] == current_class and d[3] not in selected_data_already and num_elements_added < per_class_to_add:
                    new_indices.append(d[3])
                    new_indices_class.append(d[2])
                    new_indices_path.append((d[4], d[2]))
                    num_elements_added += 1
                    temp_count += 1
                elif d[0] != current_class:
                    current_class = d[0]
                    num_elements_added = 1
                    new_indices.append(d[3])
                    new_indices_class.append(d[2])
                    new_indices_path.append((d[4], d[2]))

   
                
        elif cfg.SOLVER.STRATEGY in ["entropy" ,"random", "bvs"]:
            new_indices = [a[3] for a in one_list[:cfg.SOLVER.SEL_AL]]
            new_indices_class = [a[2] for a in one_list[:cfg.SOLVER.SEL_AL]]
            new_indices_path = [(a[4], a[2]) for a in one_list[:cfg.SOLVER.SEL_AL]]
            


        

        logger.info(f"Temp Count: {temp_count}, New indices... {len(new_indices)}, {new_indices}")
        logger.info(f"New indices Class...{len(new_indices_class)}, {new_indices_class}")

        

        if cfg.DATA.NAME == "tiny":
            new_data = pd.DataFrame(new_indices_path, columns = ['ImageName', "ClassId"])
            overall_data = pd.concat([selected_data_already, new_data], axis=0, ignore_index=True)
            total_new = overall_data
        else:
            total_new = new_indices + list(selected_data_already)

        self.save_al_data(total_new, cfg=cfg, al_cycle=al_cycle)


    def plot_reliability_diagrams(self,save_loc, one_list, n_bins = 10):
        true_classes = np.array([x[2] for x in one_list])
        predicted_probs = np.array([x[6] for x in one_list])
        predicted_classes = np.array([x[0] for x in one_list])
        
        bin_limits = np.linspace(0, 1, n_bins + 1)
        accuracies = []
        confidences = []
        bin_counts = []

        for i in range(n_bins):
            bin_lower, bin_upper = bin_limits[i], bin_limits[i + 1]
            in_bin = np.logical_and(predicted_probs >= bin_lower, predicted_probs < bin_upper)
            bin_probs = predicted_probs[in_bin]
            bin_true = true_classes[in_bin]
            bin_pred = predicted_classes[in_bin]

            if len(bin_probs) > 0:
                bin_accuracy = np.mean(bin_pred == bin_true)
                bin_confidence = np.mean(bin_probs)
                accuracies.append(bin_accuracy)
                confidences.append(bin_confidence)
                bin_counts.append(len(bin_probs))

        # Plotting the accuracy-confidence curve
        plt.figure(figsize=(8, 6))
        plt.plot(confidences, accuracies, marker='o', linestyle='-', color='b', label='Accuracy vs Confidence')
        width = 0.09  # Width of bars
        plt.bar(confidences, accuracies, width, color='skyblue', alpha=0.7, label='Accuracy')
        

        # Adding a reference line
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')

        
        plt.title('Accuracy-Confidence Plot')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        # plt.show()

        plt.savefig(f"{save_loc}/acc_conf_fig.png")
        plt.clf()
        # Adding a reference line
        plt.bar(confidences, bin_counts, width, color='skyblue', alpha=1.0, label='Accuracy')

        
        plt.title(f'Number of points: {np.sum(bin_counts)}')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_loc}/num_samples.png")
        plt.clf()
        # plt.show()


        

    def calculate_ece(self, one_list, n_bins=10):
        true_classes = [x[2] for x in one_list]
        max_probs = [x[6] for x in one_list]
        predicted_classes = [x[0] for x in one_list]
        
        bins = np.linspace(0, 1, n_bins + 1)

        ece = 0.0

        for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
            # Find the indices of probabilities in this bin
            in_bin = np.where((max_probs > bin_lower) & (max_probs <= bin_upper))[0]
            if len(in_bin) > 0:
                # Calculate accuracy and average confidence for this bin
                accuracy = np.mean(np.array(predicted_classes)[in_bin] == np.array(true_classes)[in_bin])
                avg_confidence = np.mean(np.array(max_probs)[in_bin])
                # Update ECE
                ece += np.abs(avg_confidence - accuracy) * len(in_bin) / len(predicted_classes)

        return ece
  
    @torch.no_grad()
    def eval_classifier_deep_ece(self, data_loader, al_cycle = 0, cfg=None):
        """evaluate classifier"""

        if not cfg:
            print("Pass CFG")
            raise NotImplementedError



        one_list = self.get_pred_list(data_loader, tiny = cfg.DATA.NAME=="tiny")
        one_pred_mat = self.get_pred_all(data_loader, tiny = cfg.DATA.NAME=="tiny")
        

        ece = self.calculate_ece(one_list)
        self.plot_reliability_diagrams(self.cfg.OUTPUT_DIR, one_list)
        np.set_printoptions(formatter={'float': '{:0.3f}'.format})
        np_array = np.array(one_list)

        print("Save location: ", self.cfg.OUTPUT_DIR )
        file_name = self.cfg.OUTPUT_DIR + "/pred_array_seed_1.csv"
        np.savetxt(file_name, np_array, delimiter=',')

        file_name = self.cfg.OUTPUT_DIR + "/full_matrix_seed.csv" 
        np.savetxt(file_name, one_pred_mat, delimiter=',')
        print("ECE: ", ece)
        import sys 
        sys.exit()
        