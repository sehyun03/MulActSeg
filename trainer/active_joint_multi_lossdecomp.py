import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch_scatter import scatter

from trainer import active_joint_multi
from trainer.active_joint_multi_predignore_mclossablation2 import GroupMultiLabelCE_onlymulti
from utils.loss import MultiChoiceCE
r"""
Decomposition of previous multi-positive loss & group-multi loss
- One-hot spxs: CE loss
- Multi-hot spxs: Multi-positive, Group Multi
- without predignore
"""
class OnehotCEMultihotChoice(MultiChoiceCE):
    def __init__(self, num_class, temperature=1.0, reduction='mean'):
        super().__init__(num_class, temperature, reduction)
        assert(self.reduction == 'mean')

    def forward(self, inputs, targets, superpixels, spmasks):
        ''' inputs:  N x C x H x W
            targets: N x self.num_superpiexl x C+1
            superpixels: N x H x W
            spmasks: N x H x W
        '''

        N, C, H, W = inputs.shape
        inputs = inputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        outputs = F.softmax(inputs / self.temp, dim=2) ### N x HW x C
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW: binary mask indicating current selected spxs
        oh_loss = 0
        oh_num_valid = 1
        mh_loss = 0
        mh_num_valid = 1

        for i in range(N):
            '''
            outputs[i] ### HW x C
            superpixels[i] ### HW x 1
            spmasks[i] ### HW x 1
            '''
            r''' skip this image if valid superpixel is not included '''
            valid_mask = spmasks[i] ### HW
            if not torch.any(valid_mask): continue ### empty image

            r''' calculate pixel-wise (CE, MC) loss jointly'''
            valid_output = outputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID

            trg_sup = targets[i] ### self.num_superpixel x C: multi-hot annotation
            trg_pixel = trg_sup[valid_superpixel.squeeze(dim=1)] ### HW' x C : pixel-wise multi-hot annotation
            
            pos_pred = (valid_output * trg_pixel).sum(dim=1)

            r''' ce loss on one-hot spx '''
            onehot_trg = (1 == trg_pixel.sum(dim=1))
            if torch.any(onehot_trg):
                oh_pos_pred = pos_pred[onehot_trg]
                oh_loss += -torch.log(oh_pos_pred + self.eps).sum()
                oh_num_valid += oh_pos_pred.shape[0]

            r''' mc loss on multi-hot spx '''
            # multihot_trg = torch.logical_not(onehot_trg)
            multihot_trg = (1 < trg_pixel.sum(dim=1))
            if torch.any(multihot_trg):
                # assert(torch.all(multihot_trg == (1 < trg_pixel.sum(dim=1))))
                mh_pos_pred = pos_pred[multihot_trg]
                mh_loss += -torch.log(mh_pos_pred + self.eps).sum()
                mh_num_valid += mh_pos_pred.shape[0]

        return oh_loss / oh_num_valid, mh_loss / mh_num_valid

class ActiveTrainer(active_joint_multi.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)

    def get_criterion(self):
        ''' Define criterion '''
        self.group_multi_loss = GroupMultiLabelCE_onlymulti(args=self.args, num_class=self.num_classes, num_superpixel= self.args.nseg, temperature=self.args.group_ce_temp)
        self.multi_pos_loss = OnehotCEMultihotChoice(num_class=self.num_classes, temperature=self.args.multi_ce_temp)

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
            ce_loss, mc_loss = self.multi_pos_loss(preds, labels, superpixels, spmasks)
            loss = (self.args.coeff * ce_loss) + (self.args.coeff_mc * mc_loss) + (self.args.coeff_gm * group_loss)

            ### Update (Model, Scheduler)
            self.update(loss)

            ### Logging
            self.update_average_meter({'train-loss': loss,
                                       'ce-loss': ce_loss,
                                       'pos-loss': mc_loss,
                                       'group-loss': group_loss})

            ### Logging intervals
            self.log_training(iteration, pbar, total_itrs)
            self.log_validation(iteration, val_period)