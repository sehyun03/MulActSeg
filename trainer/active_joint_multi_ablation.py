import torch
import numpy as np
from tqdm import tqdm

from trainer import active
from models import freeze_bn
from utils.loss import GroupMultiLabelCE, RandMultiChoiceCE, MaxMultiChoiceCE, RCMultiChoiceCE

class ActiveTrainer(active.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)

    def get_criterion(self):
        ''' Define criterion '''
        if self.args.loss_type == 'rand_multi_ce':
            self.multi_pos_loss = RandMultiChoiceCE(num_class=self.num_classes, temperature=self.args.multi_ce_temp)
        elif self.args.loss_type == 'max_multi_ce':
            self.multi_pos_loss = MaxMultiChoiceCE(num_class=self.num_classes, temperature=self.args.multi_ce_temp)
        elif self.args.loss_type == 'rc_multi_ce':
            self.multi_pos_loss = RCMultiChoiceCE(num_class=self.num_classes, temperature=self.args.multi_ce_temp)
        else:
            raise NotImplementedError

        self.group_multi_loss = GroupMultiLabelCE(args=self.args, num_class=self.num_classes, num_superpixel= self.args.nseg, temperature=self.args.group_ce_temp)

    def check_loss_sanity(self, loss):
        if loss == 0:
            return False
        elif torch.isnan(loss):
            return False
        else:
            return True

    def train_impl(self, total_itrs, val_period):
        self.net.train()
        if self.args.freeze_bn is True:
            freeze_bn(self.net)
        pbar = tqdm(range(total_itrs), ncols=150)

        for iteration in pbar:
            ### Loading
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

            ### Update
            if self.check_loss_sanity(loss):
                loss.backward()
                self.optimizer.step()

            ### Scheduler
            if self.args.scheduler == 'poly':
                self.scheduler.step()

            ### Logging
            if self.check_loss_sanity(loss):
                self.am.add({'train-loss': loss.detach().cpu().item()})
            if self.check_loss_sanity(pos_loss):
                self.am.add({'pos-loss': pos_loss.detach().cpu().item()})
            if self.check_loss_sanity(group_loss):
                self.am.add({'group-loss': group_loss.detach().cpu().item()})

            if iteration % self.args.log_period == (self.args.log_period - 1):
                pbar.set_description('[AL {}-round] (step{}): Loss {:.4f} Session {}'.format(
                    self.selection_iter,
                    iteration,
                    self.am.get('train-loss'),
                    self.args.session_name
                ))                
                global_step = iteration + (total_itrs * (self.selection_iter - 1))
                lr_f = self.optimizer.param_groups[-1]['lr']
                wlog_train = {'learning-rate cls': lr_f}
                wlog_train.update({k:self.am.pop(k) for k,v in self.am.get_whole_data().items()})
                self.args.wandb.log(wlog_train, step=global_step)

            ### Validation
            if iteration % val_period == (val_period - 1) and iteration > self.args.val_start:
                self.logger.info('**** EVAL ITERATION %06d ****' % (iteration))
                self.validate(trainiter=iteration)
                self.net.train()
                if self.args.freeze_bn is True:
                    freeze_bn(self.net)