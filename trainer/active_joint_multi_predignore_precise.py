import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch_scatter import scatter

from trainer import active_joint_multi_predignore
from models import get_model
from trainer.active_joint_multi_predignore import GroupMultiLabelCE_
from utils.miou import MeanIoU
from utils.loss import MyCrossEntropyLoss
r"""
Precise cross entropy instead of multi-positive CE loss (kind of oracle experiment)
"""

class ActiveTrainer(active_joint_multi_predignore.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)

    def get_criterion(self):
        ''' Define criterion '''
        self.ce_loss = MyCrossEntropyLoss(ignore_index=self.args.ignore_idx, reduction='mean', temperature=self.args.ce_temp)
        self.group_multi_loss = GroupMultiLabelCE_(args=self.args, num_class=self.num_classes, num_superpixel= self.args.nseg, temperature=self.args.group_ce_temp)

    def train_impl(self, total_itrs, val_period):
        self.net.train()
        pbar = tqdm(range(total_itrs), ncols=150)

        for iteration in pbar:
            ### Data Loading
            batch = self.train_dataset_loader.__next__()
            images = batch['images'].to(self.device, dtype=torch.float32)
            labels = batch['labels'].to(self.device, dtype=torch.long)
            targets = batch['target'].to(self.device, dtype=torch.uint8)
            superpixels = batch['spx'].to(self.device)
            spmasks = batch['spmask'].to(self.device)

            ### Forward
            self.optimizer.zero_grad()
            preds = self.net(images)

            ### Loss
            group_loss = self.group_multi_loss(preds, targets, superpixels, spmasks)
            ce_loss = self.zero_if_nan(self.ce_loss(preds, labels))
            loss = ce_loss + group_loss

            ### Update (Model, Scheduler)
            self.update(loss)

            ### Logging
            self.update_average_meter({'train-loss': loss,
                                       'pos-loss': ce_loss,
                                       'group-loss': group_loss})

            ### Logging intervals
            self.log_training(iteration, pbar, total_itrs)
            self.log_validation(iteration, val_period)