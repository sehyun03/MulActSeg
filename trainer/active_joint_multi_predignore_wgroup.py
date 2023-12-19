import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch_scatter import scatter

from trainer import active_joint_multi_predignore
from trainer.active_joint_multi_predignore import MultiChoiceCE_, GroupMultiLabelCE_
from models import freeze_bn

class WeightedGroupMultiLabelCE(GroupMultiLabelCE_):
    def __init__(self, args, num_class, num_superpixel, temperature=1.0, reduction='mean'):
        super().__init__(args, num_class, num_superpixel, temperature, reduction)
        self.ignore_only_single = args.group_only_single

    def forward(self, inputs, plbl_preds, targets, superpixels, spmasks):
        ''' inputs: N x C x H x W
            plbl_preds: N x C x H x W
            targets: N x self.num_superpixel x C+1
            superpixels: NxHxW
            spmasks: NxHxW
            
            Apply max operation over predicted probabilities for each multi-hot label within the superpixel, and apply CE loss​.
            '''
        N, C, H, W = inputs.shape
        outputs = F.softmax(inputs / self.temp, dim=1) ### N x C x H x W
        outputs = outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW
        if self.ignore_only_single:
            empty_trg_mask = (1 < targets.sum(dim=2))
        else:
            empty_trg_mask = torch.any(targets, dim=2).bool() ### N x self.num_superpixel
        loss = 0
        num_valid = 1

        with torch.no_grad():
            plbl_outputs = F.softmax(plbl_preds / self.temp, dim=1) ### N x C x H x W
            plbl_outputs = plbl_outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C

        for i in range(N):
            '''
            outputs[i] ### HW x C
            superpixels[i] ### HW x 1
            spmasks[i] ### HW x 1
            plbl_outputs[i] ### HW x C
            '''

            ### filtered outputs
            valid_mask = spmasks[i]
            if not torch.any(valid_mask):
                continue
            valid_output = outputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID
            out_sup_mxpool = scatter(valid_output, valid_superpixel, dim=0, reduce='max', dim_size=self.num_superpixel)
                ### self.num_superpixel x C : sp 영역 내 class 별 max predicted prob, invalid superpixel 은 모두 0 으로 채워짐.
            trg_sup_mxpool = targets[i] ### self.num_superpixel x C: multi-hot annotation
            out_sup_mxpool = out_sup_mxpool[empty_trg_mask[i]]
            trg_sup_mxpool = trg_sup_mxpool[empty_trg_mask[i]]
            top_one_preds = out_sup_mxpool * trg_sup_mxpool ### self.num_superpixel x C: 존재하는 multi-hot 으로 filtering
            if not torch.any(top_one_preds):
                continue
            top_one_preds_nonzero = top_one_preds[top_one_preds.nonzero(as_tuple=True)] ### 해당 value indexing

            ### filtered plbl_outputs
            with torch.no_grad():
                valid_plbl_output = plbl_outputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
                plbl_out_sup_mxpool = scatter(valid_plbl_output, valid_superpixel, dim=0, reduce='max', dim_size=self.num_superpixel)
                plbl_out_sup_mxpool = plbl_out_sup_mxpool[empty_trg_mask[i]]
                plbl_top_one_preds = plbl_out_sup_mxpool * trg_sup_mxpool ### self.num_superpixel x C: 존재하는 multi-hot 으로 filtering
                plbl_top_one_preds_nonzero = plbl_top_one_preds[top_one_preds.nonzero(as_tuple=True)] ### 해당 value indexing
                
            num_valid += top_one_preds_nonzero.shape[0] ### valid pixel 개수 측정
            loss += -(plbl_top_one_preds_nonzero.detach() * torch.log(top_one_preds_nonzero + self.eps)).sum()

        if self.reduction == 'mean':
            return loss / num_valid
        elif self.reduction == 'none':
            return loss, num_valid
        else:
            raise NotImplementedError

class ActiveTrainer(active_joint_multi_predignore.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)

    def get_criterion(self):
        ''' Define criterion '''
        self.group_multi_loss = WeightedGroupMultiLabelCE(args=self.args, num_class=self.num_classes, num_superpixel= self.args.nseg, temperature=self.args.group_ce_temp)
        self.multi_pos_loss = MultiChoiceCE_(num_class=self.num_classes, temperature=self.args.multi_ce_temp)

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

            ### Generated pseudo prediction
            with torch.no_grad():
                self.net.eval()
                plbl_preds = self.net(images).detach()
                self.net.train()

            ### Loss
            group_loss = self.group_multi_loss(preds, plbl_preds, labels, superpixels, spmasks)
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