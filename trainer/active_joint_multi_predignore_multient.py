import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from models import freeze_bn
from trainer import active_joint_multi_predignore
from trainer.active_joint_multi_predignore import MultiChoiceCE_, GroupMultiLabelCE_
from utils.loss import MultiChoiceEnt

class MultiChoiceEnt_(MultiChoiceEnt):
    def __init__(self, num_class, temperature=1.0, reduction='mean'):
        super().__init__(num_class, temperature, reduction)

    def forward(self, inputs, targets, superpixels, spmasks):
        ''' additional ignore label => C (= num_class + 1)
            inputs:  N x C x H x W
            targets: N x self.num_superpiexl x C
            superpixels: N x H x W
            spmasks: N x H x W
        '''

        N, C, H, W = inputs.shape
        inputs = inputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW
        loss = 0
        num_valid = 1

        for i in range(N):
            '''
            inputs[i] ### HW x C
            superpixels[i] ### HW x 1
            spmasks[i] ### HW x 1
            '''
            ### filtered inputs
            valid_mask = spmasks[i] ### HW
            if not torch.any(valid_mask):
                continue
            valid_input = inputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID
            trg_sup = targets[i] ### self.num_superpixel x C: multi-hot annotation
            trg_pixel = trg_sup[valid_superpixel.squeeze(dim=1)].detach() ### HW' x C : pixel-wise multi-hot annotation
            
            ### filter out single target
            multi_trg_mask = (1 < trg_pixel.sum(dim=1)) ### HW'
            if not torch.any(multi_trg_mask):
                continue
            valid_input = valid_input[multi_trg_mask] ### HW'' x C
            trg_pixel = trg_pixel[multi_trg_mask] ### HW'' x C
            pos_pred = valid_input * trg_pixel

            ### softmax on candidate label set
            ### -inf insertion (for softmax)
            pos_pred = pos_pred.view(-1) ### HW'' x C
            pos_pred[pos_pred == 0] = float('-inf') ### HW'' x C
            pos_pred = pos_pred.view(-1, trg_pixel.shape[1]) ### HW'' x C
            valid_output = F.softmax(pos_pred / self.temp, dim=1) ### HW'' x C

            ent_output = -torch.sum(valid_output * torch.log(valid_output + self.eps), dim=1) ### HW''
            num_valid += ent_output.shape[0]
            loss += ent_output.sum()

        if self.reduction == 'mean':
            return loss / num_valid
        elif self.reduction == 'none':
            return loss, num_valid
        else:
            NotImplementedError

class ActiveTrainer(active_joint_multi_predignore.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)

    def get_criterion(self):
        ''' Define criterion '''
        self.group_multi_loss = GroupMultiLabelCE_(args=self.args, num_class=self.num_classes, num_superpixel= self.args.nseg, temperature=self.args.group_ce_temp)
        self.multi_pos_loss = MultiChoiceCE_(num_class=self.num_classes, temperature=self.args.multi_ce_temp)
        self.multi_ent_loss = MultiChoiceEnt_(num_class=self.num_classes, temperature=self.args.multi_ce_temp)

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
            ent_loss = self.multi_ent_loss(preds, labels, superpixels, spmasks)            
            loss = self.args.coeff * pos_loss + group_loss + self.args.entcoeff * ent_loss

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
            if self.check_loss_sanity(ent_loss):
                self.am.add({'ent-loss': ent_loss.detach().cpu().item()})

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