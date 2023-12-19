import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from models import freeze_bn
from trainer import active_joint_multi_predignore
from trainer.active_joint_multi_predignore import MultiChoiceCE_, GroupMultiLabelCE_
from utils.loss import MultiChoiceEnt
from utils.scheduler import ramp_up

class TopOnePlbl(MultiChoiceEnt):
    def __init__(self, num_class, within_filtering=False, filtering_threshold=0, temperature=1.0, reduction='mean'):
        super().__init__(num_class, temperature, reduction)
        self.plbl_th = filtering_threshold
        self.within_filtering = within_filtering

    def forward(self, inputs, plbl_inputs, targets, superpixels, spmasks):
        ''' inputs:  N x C x H x W
            plbl_inputs: N x C x H x W
            targets: N x self.num_superpiexl x C+1
            superpixels: N x H x W
            spmasks: N x H x W
        '''

        N, C, H, W = inputs.shape
        inputs = inputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        outputs = F.softmax(inputs / self.temp, dim=2) ### N x HW x C
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW: binary mask indicating current selected spxs
        loss = 0
        num_valid = 1

        with torch.no_grad():
            plbl_inputs = plbl_inputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
            plbl_outputs = F.softmax(plbl_inputs / self.temp, dim=2) ### N x HW x C

        for i in range(N):
            '''
            outputs[i] ### HW x C
            plbl_outputs[i] ### HW x C
            superpixels[i] ### HW x 1
            spmasks[i] ### HW x 1
            '''
            ### filtered outputs
            valid_mask = spmasks[i] ### HW
            if not torch.any(valid_mask): ### 더 이상 뽑을게 없는 경우
                continue
            valid_output = outputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID

            trg_sup = targets[i] ### self.num_superpixel x C: multi-hot annotation
            trg_pixel = trg_sup[valid_superpixel.squeeze(dim=1)].detach() ### HW' x C : pixel-wise multi-hot annotation
            
            ### filter out empty target
            non_single_trg_mask = (1 < trg_pixel.sum(dim=1)) ### HW'
            valid_output = valid_output[non_single_trg_mask] ### HW'' x C
            trg_pixel = trg_pixel[non_single_trg_mask] ### HW'' x C
            pos_pred = valid_output * trg_pixel ### HW'' x C: output of positive set

            ### pseudo label generation
            with torch.no_grad():
                valid_plbl_output = plbl_outputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
                valid_plbl_output = valid_plbl_output[non_single_trg_mask] ### HW'' x C
                pos_plbl_pred = valid_plbl_output * trg_pixel ### HW'' x C: output of positive set
                if not self.within_filtering:
                    pos_plbl_pred = pos_plbl_pred.max(dim=1)[0].detach() ### HW'': top-1 selection
                else:
                    pos_plbl_pred = (pos_plbl_pred / pos_plbl_pred.sum(dim=1)[..., None]).max(dim=1)[0].detach()

            ### loss calculation
            filtered_pred = pos_pred[self.plbl_th < pos_plbl_pred].max(dim=1)[0]
            num_valid += filtered_pred.shape[0]
            if filtered_pred.shape[0] == 0:
                continue
            loss += -torch.log(filtered_pred + self.eps).sum()

        if self.reduction == 'mean':
            return loss / num_valid
        else:
            NotImplementedError

class ActiveTrainer(active_joint_multi_predignore.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)

    def get_criterion(self):
        ''' Define criterion '''
        self.group_multi_loss = GroupMultiLabelCE_(args=self.args, num_class=self.num_classes, num_superpixel= self.args.nseg, temperature=self.args.group_ce_temp)
        self.multi_pos_loss = MultiChoiceCE_(num_class=self.num_classes, temperature=self.args.multi_ce_temp)
        self.multi_top1_loss = TopOnePlbl(num_class=self.num_classes, within_filtering=self.args.within_filtering, filtering_threshold=self.args.plbl_th, temperature=1.0)

    def train_impl(self, total_itrs, val_period):
        args = self.args
        self.net.train()
        if args.freeze_bn is True:
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
            
            # ### Generate pseudo label
            with torch.no_grad():
                self.net.eval()
                plbl_preds = self.net(images).detach()
                self.net.train()

            ### Loss
            group_loss = self.group_multi_loss(preds, labels, superpixels, spmasks)
            pos_loss = self.multi_pos_loss(preds, labels, superpixels, spmasks)            
            top1_loss = self.multi_top1_loss(preds, plbl_preds, labels, superpixels, spmasks)            
            lamda = ramp_up(iteration / total_itrs, lamparam=args.lamparam, scale=args.lamscale, dorampup=args.dorampup)
            loss = args.coeff * pos_loss + group_loss + lamda * top1_loss

            ### Update
            if self.check_loss_sanity(loss):
                loss.backward()
                self.optimizer.step()

            ### Scheduler
            if args.scheduler == 'poly':
                self.scheduler.step()

            ### Logging
            if self.check_loss_sanity(loss):
                self.am.add({'train-loss': loss.detach().cpu().item()})
            if self.check_loss_sanity(pos_loss):
                self.am.add({'pos-loss': pos_loss.detach().cpu().item()})
            if self.check_loss_sanity(group_loss):
                self.am.add({'group-loss': group_loss.detach().cpu().item()})
            if self.check_loss_sanity(top1_loss):
                self.am.add({'top1-loss': top1_loss.detach().cpu().item()})

            if iteration % args.log_period == (args.log_period - 1):
                pbar.set_description('[AL {}-round] (step{}): Loss {:.4f} Session {}'.format(
                    self.selection_iter,
                    iteration,
                    self.am.get('train-loss'),
                    args.session_name
                ))                
                global_step = iteration + (total_itrs * (self.selection_iter - 1))
                lr_f = self.optimizer.param_groups[-1]['lr']
                wlog_train = {'learning-rate cls': lr_f}
                wlog_train.update({k:self.am.pop(k) for k,v in self.am.get_whole_data().items()})
                args.wandb.log(wlog_train, step=global_step)

            ### Validation
            if iteration % val_period == (val_period - 1) and iteration > args.val_start:
                self.logger.info('**** EVAL ITERATION %06d ****' % (iteration))
                self.validate(trainiter=iteration)
                self.net.train()
                if args.freeze_bn is True:
                    freeze_bn(self.net)