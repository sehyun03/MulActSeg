import torch
import numpy as np
from tqdm import tqdm

from trainer import active
from utils.loss import GroupMultiLabelCE, MultiChoiceCE

class ActiveTrainer(active.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)

    def get_criterion(self):
        ''' Define criterion '''
        self.group_multi_loss = GroupMultiLabelCE(args=self.args, num_class=self.num_classes, num_superpixel= self.args.nseg, temperature=self.args.group_ce_temp)
        self.multi_pos_loss = MultiChoiceCE(num_class=self.num_classes, temperature=self.args.multi_ce_temp)

    def zero_if_nan(self, loss):
        if torch.isnan(loss):
            return 0
        else:
            return loss

    def check_loss_sanity(self, loss):
        if loss == 0:
            return False
        elif torch.isnan(loss):
            raise ValueError
        else:
            return True

    def update(self, loss):
        if self.check_loss_sanity(loss):
            loss.backward()
            self.optimizer.step()

        if self.args.scheduler == 'poly':
            self.scheduler.step()

    def update_average_meter(self, dict):
        for key, value in dict.items():
            if self.check_loss_sanity(value):
                self.am.add({key: value.detach().cpu().item()})

    def train_impl(self, total_itrs, val_period):
        self.net.train()
        pbar = tqdm(range(total_itrs), ncols=150)

        for iteration in pbar:
            ### Data Loading
            batch = self.train_dataset_loader.__next__()
            images = batch['images']
            labels = batch['labels']
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=self.target_dtype)
            superpixels = batch['spx'].to(self.device)
            spmasks = batch['spmask'].to(self.device)

            ### Forward
            self.optimizer.zero_grad()
            preds = self.net(images)

            ### Loss
            group_loss = self.group_multi_loss(preds, labels, superpixels, spmasks)
            pos_loss = self.multi_pos_loss(preds, labels, superpixels, spmasks)            
            loss = self.args.coeff * pos_loss + group_loss

            ### Update (Model, Scheduler)
            self.update(loss)

            ### Logging
            self.update_average_meter({'train-loss': loss,
                                       'pos-loss': pos_loss,
                                       'group-loss': group_loss})

            ### Logging intervals
            self.log_training(iteration, pbar, total_itrs)
            self.log_validation(iteration, val_period)