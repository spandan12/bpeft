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

from sklearn.metrics import roc_auc_score
from ..engine.evidential_loss import Evidential_loss

from ..engine.calib_helpers import ModelWithTemperature
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
        
        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )
        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

        #### CHANGED HEAD
        print(self.model.head)
        # two_layer_head = False
        # if two_layer_head:
        #     self.model.head = nn.Sequential(
        #                             nn.Linear(self.model.head.last_layer.in_features, 256),  # 1st layer (in_features from previous layer)
        #                             nn.ReLU(),                           # Non-linear activation
        #                             nn.Linear(256, self.model.head.last_layer.out_features)
        #                         ).to(self.device)
        #     print("New head")
        #     print(self.model.head)

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            # checkpointables = [key for key in self.checkpointer.checkpointables 
                            #    if key not in ["head.last_layer.bias",  "head.last_layer.weight"]]
            # self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            print("LOADING MODEL...")
            checkpoint = torch.load(cfg.MODEL.WEIGHT_PATH)
            shallow_prompt = torch.from_numpy(checkpoint['shallow_prompt'])
            self.model.enc.transformer.prompt_embeddings.data = shallow_prompt.to(self.device)

            if cfg.MODEL.PROMPT.DEEP:
                deep_prompt = torch.from_numpy(checkpoint['deep_prompt'])
                self.model.enc.transformer.deep_prompt_embeddings.data = deep_prompt.to(self.device)
            # print("Saved head: ", checkpoint['head'])
            self.model.head.load_state_dict(checkpoint['head'])
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")
        

        # import sys 
        # sys.exit()
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        evid_args ={
            'unc_act':'exp', 
            'ev_unc_type':'log', 
            'use_vac_reg':True, 
            'kl_strength':float(cfg.SOLVER.KL_VAL)
            }
        logger.info(f"Evid Args: {evid_args}")
        self.loss_evidential = Evidential_loss(evid_args)



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
                if self.cfg.SOLVER.MODELTYPE == 'ce':
                    loss_all = F.cross_entropy(outputs, targets, reduction="none")
                    loss = torch.sum(loss_all) / targets.shape[0]
                elif self.cfg.SOLVER.MODELTYPE == 'evidential':
                    loss = self.loss_evidential.edl_overall_loss(outputs, targets)

            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                if self.cfg.SOLVER.MODELTYPE == 'ce':
                    loss_all = F.cross_entropy(outputs, targets, reduction="none")
                    loss = torch.sum(loss_all) / targets.shape[0]
                elif self.cfg.SOLVER.MODELTYPE == 'evidential':
                    loss = self.loss_evidential.edl_overall_loss(outputs, targets)
                
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

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        self.model.eval()
        
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
                
                X, targets = self.get_input(input_data)
                # logger.info(X.shape)
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
            if (epoch > 1 and epoch %10 == 0) or epoch+2 > self.cfg.SOLVER.TOTAL_EPOCH:
            # print("Epoch: ", epoch)
            # if epoch > -1:
                # eval at each epoch for single gpu training
                self.eval_classifier_deep_ece(test_loader)
                self.evaluator.update_iteration(epoch)
                self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
                if test_loader is not None:
                    self.eval_classifier(test_loader, "test", epoch == total_epoch - 1)

                
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
                self.save_prompt(epoch + 1)
        # save the last checkpoints
        if self.cfg.MODEL.SAVE_CKPT:
            Checkpointer(
                self.model,
                save_dir=self.cfg.OUTPUT_DIR,
                save_to_disk=True
            ).save("last_model")

    @torch.no_grad()
    def save_prompt(self, epoch):
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
                    self.cfg.OUTPUT_DIR, f"prompt_ep{epoch}.pth"))

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
    def get_pred_list(self, data_loader):
        '''
         zip(all_predicted_labels, all_entropy, all_labels, all_index)
        '''
        
        pred_matrix = np.zeros((len(data_loader.dataset), 4))

        correct_predictions_sum = 0
        till_now = 0
        for idx, input_data in enumerate(data_loader):
            # if idx > 10: break
            if self.cfg.DATA.NAME=="tiny":
                X, targets = self.get_input(input_data[:2])
                paths = input_data[2]
                
            else:
                X, targets = self.get_input(input_data)
                paths = [0 for _ in range(len(targets))]
            
            # pdb.set_trace()


            _, outputs = self.forward_one_batch(X, targets, False)
            evidence = self.loss_evidential.get_evidence(outputs)
            if self.cfg.SOLVER.MODELTYPE == 'ce':
                alpha = outputs
            elif self.cfg.SOLVER.MODELTYPE == 'evidential':
                alpha = evidence + 1

            S = torch.sum(alpha, dim=1, keepdim=True)
            probabilities = (alpha/S).to(targets.device)
            vacuity = alpha.shape[-1] / S.detach()
            vacuity = vacuity.squeeze()
            # pdb.set_trace()

            predicted_classes = torch.argmax(alpha, dim = 1).to(targets.device)
            top2_prob, _ = probabilities.topk(2, dim = 1)
            

            pred_matrix[till_now:till_now + len(targets), 0] = predicted_classes.detach().cpu().numpy()
            pred_matrix[till_now:till_now + len(targets), 1] = targets.detach().cpu().numpy()
            pred_matrix[till_now:till_now + len(targets), 2] = vacuity.detach().cpu().numpy()**1
            pred_matrix[till_now:till_now + len(targets), 3] = top2_prob[:,0].detach().cpu().numpy()

            till_now += len(targets)

            # For printing
            correct_predictions = (predicted_classes == targets).sum().item()/len(predicted_classes)
            correct_predictions_sum += correct_predictions
            if idx%10 == 0:
                print(idx, correct_predictions)
        
        logger.info(f"Accuracy...  {correct_predictions_sum/(idx+1)}")

        return pred_matrix
    
    @torch.no_grad()
    def get_pred_all(self, data_loader, id=None):
        '''
         zip(all_predicted_labels, all_entropy, all_labels, all_index)
        '''
        pred_matrix = np.zeros((len(data_loader.dataset), 4))

        all_predictions = None
        correct_predictions_sum = 0
        till_now = 0

        for idx, input_data in enumerate(data_loader):
            # if idx > 10: break
            if self.cfg.DATA.NAME=="tiny":
                X, targets = self.get_input(input_data[:2])
                paths = input_data[2]
                
            else:
                X, targets = self.get_input(input_data)
                paths = [0 for _ in range(len(targets))]
            
            if id == 'openset':
                targets = torch.zeros(len(targets), dtype=int)

            _, outputs = self.forward_one_batch(X, targets, False)
            if self.cfg.SOLVER.MODELTYPE != 'ce': 
                evidence = self.loss_evidential.get_evidence(outputs)
            else:
                evidence = outputs # CE directly save evidence
            
            if self.cfg.SOLVER.MODELTYPE == 'ce':
                alpha = outputs # The logits
            elif self.cfg.SOLVER.MODELTYPE == 'evidential':
                alpha = evidence + 1
            
            S = torch.sum(alpha, dim=1, keepdim=True)
            probabilities = (alpha/S).to(targets.device)
            vacuity = alpha.shape[-1] / S.detach()
            vacuity = vacuity.squeeze()

            if all_predictions is None:
                all_predictions = evidence.detach().cpu().numpy()
            else:
                attach = evidence.detach().cpu().numpy()
                all_predictions = np.concatenate((all_predictions, attach), axis = 0)
            
            predicted_classes = torch.argmax(alpha, dim = 1).to(targets.device)

            
            top2_prob, _ = probabilities.topk(2, dim = 1)

            pred_matrix[till_now:till_now + len(targets), 0] = predicted_classes.detach().cpu().numpy()
            pred_matrix[till_now:till_now + len(targets), 1] = targets.detach().cpu().numpy()
            pred_matrix[till_now:till_now + len(targets), 2] = vacuity.detach().cpu().numpy()**1
            pred_matrix[till_now:till_now + len(targets), 3] = top2_prob[:,0].detach().cpu().numpy()

            till_now += len(targets)

            correct_predictions = (predicted_classes == targets).sum().item()/len(predicted_classes)
            correct_predictions_sum += correct_predictions
            
            if idx%10 == 0:
                print(idx, correct_predictions)
        
        logger.info(f"Accuracy...  {correct_predictions_sum/(idx+1)}")


        one_np_mat = np.concatenate((pred_matrix, all_predictions), axis = 1)
        return one_np_mat

    def plot_reliability_diagrams(self,save_loc, pred_matrix, n_bins = 50):
        true_classes = pred_matrix[:, 1] #np.array([x[1] for x in one_list])
        # predicted_probs = 1 - pred_matrix[:,2] #np.array([x[3] for x in one_list])
        predicted_probs = pred_matrix[:,3] #np.array([x[3] for x in one_list])
        predicted_classes = pred_matrix[:, 0] #np.array([x[0] for x in one_list])
        
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


        

    def calculate_ece(self, pred_matrix, n_bins=50, confidence_metric = 'max_prob'):
        if confidence_metric == 'max_prob':
            max_probs = pred_matrix[:,3] #[x[3] for x in one_list]
        elif confidence_metric == 'vacuity':
            max_probs = (1.0 - pred_matrix[:,2])

        
        true_classes = pred_matrix[:,1] #[x[1] for x in one_list]
        predicted_classes = pred_matrix[:,0] #[x[0] for x in one_list]
        
        bins = np.linspace(0, 1, n_bins + 1)

        ece = 0.0

        for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
            # Find the indices of probabilities in this bin
            in_bin = np.where((max_probs > bin_lower) & (max_probs <= bin_upper))[0]
            if len(in_bin) > 0:
                # Calculate accuracy and average confidence for this bin
                # accuracy = np.mean(np.array(predicted_classes)[in_bin] == np.array(true_classes)[in_bin])
                accuracy = np.mean(predicted_classes[in_bin] == true_classes[in_bin])
                # avg_confidence = np.mean(np.array(max_probs)[in_bin])
                avg_confidence = np.mean(max_probs[in_bin])
                # Update ECE
                ece += np.abs(avg_confidence - accuracy) * len(in_bin) / len(predicted_classes)

        return ece
  
  
    def calib_val_set(self, val_loader, tau = 1.0):
        self.model2 = ModelWithTemperature(self.model, tau = tau)
        return self.model2.set_temperature(val_loader)


    @torch.no_grad()
    def eval_classifier_deep_openset(self, openset_loader, tau = 1.0, id = 'openset'):
        self.model.eval()
        logger.info("OPENSET EVAL FIRST")
        
        np.set_printoptions(formatter={'float': '{:0.3f}'.format})

        file_name = self.cfg.OUTPUT_DIR + f"/{id}_full_matrix_seed.csv" 
        one_pred_mat = self.get_pred_all(openset_loader, id = id)

        header_full = 'pred,gt,vacuity, maxp,'
        num_classes = self.cfg.DATA.NUMBER_CLASSES
        header_full += ','.join([f'class_{c}'for c in range(1,num_classes)]) 
        gt_classes = np.array(one_pred_mat[:, 1], dtype = int)
        pred_scores = one_pred_mat[:,4:]
        pred_scores = pred_scores / np.sum(pred_scores, axis = -1, keepdims = True)
        
        print("Saving file with evidences: ")
        np.savetxt(file_name, one_pred_mat, delimiter=',', header = header_full)
        
        return one_pred_mat
        

    @torch.no_grad()
    def eval_classifier_deep_ece(self, data_loader, tau = 1.0, id = ''):
        """evaluate classifier"""
        self.model.eval()
        logger.info("EVAL FIRST")
        self.eval_classifier(data_loader, "test", False)
        logger.info("NOW OTHERS")


        pred_matrix = self.get_pred_list(data_loader)
        header = 'pred,gt,vacuity, maxp'
        # np_array = np.array(one_list)
        # pred_matrix = np_array

        ece = self.calculate_ece(pred_matrix, confidence_metric = 'max_prob')
        logger.info(f"Max Probability based ECE: {ece}")
        vac_ece = self.calculate_ece(pred_matrix, confidence_metric = 'vacuity')
        logger.info(f"Vacuity based ECE: {vac_ece}")
        self.plot_reliability_diagrams(self.cfg.OUTPUT_DIR, pred_matrix)
        np.set_printoptions(formatter={'float': '{:0.3f}'.format})
        

        print("Save location: ", self.cfg.OUTPUT_DIR )
        file_name = self.cfg.OUTPUT_DIR + f"/{id}_pred_array_seed_1.csv"
        np.savetxt(file_name, pred_matrix, delimiter=',', header = header)

        file_name = self.cfg.OUTPUT_DIR + f"/{id}_full_matrix_seed.csv" 
        one_pred_mat = self.get_pred_all(data_loader)

        header_full = 'pred,gt,vacuity, maxp,'
        num_classes = self.cfg.DATA.NUMBER_CLASSES
        header_full += ','.join([f'class_{c}'for c in range(1,num_classes+1)]) 
        gt_classes = np.array(one_pred_mat[:, 1], dtype = int)
        pred_scores = one_pred_mat[:,4:]
        pred_scores = pred_scores / np.sum(pred_scores, axis = -1, keepdims = True)
        auc = roc_auc_score(gt_classes, pred_scores, multi_class='ovr', average='macro')
        logger.info(f"AUC: {auc}")
        print("AUC: ", auc)
        np.savetxt(file_name, one_pred_mat, delimiter=',', header = header_full)

        return one_pred_mat
        
        # import sys 
        # sys.exit()
        