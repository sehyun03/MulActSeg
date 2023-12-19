import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch_scatter import scatter

from trainer import active_joint_multi_predignore
from trainer.active_joint_multi_predignore_lossdecomp_sequence import PlblGroupMultiLabelCELog, PlblOnehotCEMultihotChoice
from models import get_model
from utils.loss import GroupMultiLabelCE, MultiChoiceCE
r"""
Sequential training script using pseudo label trained from previous round
"""

class ActiveTrainer(active_joint_multi_predignore.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)

    def get_criterion(self):
        self.group_multi_loss = PlblGroupMultiLabelCELog(args=self.args, num_class=self.num_classes, num_superpixel= self.args.nseg, temperature=self.args.group_ce_temp)
        self.multi_pos_loss = PlblOnehotCEMultihotChoice(num_class=self.num_classes, temperature=self.args.multi_ce_temp, reduction='none')

    def train_impl(self, total_itrs, val_period):
        self.net.train()
        pbar = tqdm(range(total_itrs), ncols=180)

        for iteration in pbar:
            ### Data Loading
            batch = self.train_dataset_loader.__next__()
            images = batch['images']
            labels = batch['target']
            labels_plbl = batch['labels'].to(self.device, dtype=torch.long)
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=self.target_dtype)
            superpixels = batch['spx'].to(self.device)
            spmasks = batch['spmask'].to(self.device)

            ### Forward
            self.optimizer.zero_grad()
            preds = self.net(images)

            ### Loss
            group_loss = self.group_multi_loss(preds, labels, superpixels, spmasks, labels_plbl)
            
            ce_loss_sum, ce_loss_num, mc_loss_sum, mc_loss_num = self.multi_pos_loss(preds, labels, superpixels, spmasks, labels_plbl)
            multi_positive_loss = (ce_loss_sum + mc_loss_sum) / (ce_loss_num + mc_loss_num)

            loss = (self.args.coeff * multi_positive_loss) + group_loss

            ### Update (Model, Scheduler)
            self.update(loss)

            ### Logging
            self.update_average_meter({'train-loss': loss,
                                       'pos-loss': multi_positive_loss,
                                       'group-loss': group_loss})

            ### Logging intervals
            self.log_training(iteration, pbar, total_itrs)
            self.log_validation(iteration, val_period)    