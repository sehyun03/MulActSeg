import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch_scatter import scatter

from trainer import active_joint_multi_predignore
from models import get_model
from trainer.active_joint_multi_predignore import GroupMultiLabelCE_
from utils.loss import MultiChoiceCE
from utils.miou import MeanIoU
from utils.miou_evalignore import IoUIgnore
r"""
Exclusitve Cross entropy loss
- remove candidate label from denominator of the softmax.
- then, apply softmax multiple times into the network.
"""

class ExclusiveCE(nn.Module):
    def __init__(self, num_class, temperature=1.0, reduction='mean'):
        super().__init__()
        self.num_class = num_class
        self.reduction = reduction
        self.eps = 1e-8
        self.temp = temperature

    def forward(self, inputs, targets, superpixels, spmasks):
        ''' inputs:  N x C x H x W
            targets: N x self.num_superpiexl x C+1
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
            outputs[i] ### HW x C
            superpixels[i] ### HW x 1
            spmasks[i] ### HW x 1
            '''
            ### filtered outputs
            valid_mask = spmasks[i] ### HW
            if not torch.any(valid_mask):
                continue
            valid_output = inputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID

            trg_sup = targets[i] ### self.num_superpixel x C: multi-hot annotation
            trg_pixel = trg_sup[valid_superpixel.squeeze(dim=1)].detach() ### HW' x C : pixel-wise multi-hot annotation
            
            ### filter out empty target
            empty_trg_mask = torch.any(trg_pixel, dim=1).bool() ### HW'
            valid_output = valid_output[empty_trg_mask]
            trg_pixel = trg_pixel[empty_trg_mask]
            exp_logit = torch.exp(valid_output)
            
            denominator = (exp_logit * torch.logical_not(trg_pixel)).sum(dim=1)
            denominator = denominator[..., None].repeat(1, 20)
            denominator = (denominator + exp_logit) * trg_pixel
            exclusive_softmax =  (exp_logit * trg_pixel) / (denominator + self.eps)
            exclusive_ce = -torch.log(exclusive_softmax + self.eps) * trg_pixel
            pix_exclusive_ce = exclusive_ce.sum(dim=1)
            pix_ce = pix_exclusive_ce / trg_pixel.sum(dim=1)
            
            num_valid += pix_ce.shape[0]
            loss += pix_ce.sum()

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
        # self.multi_pos_loss = MultiChoiceCE_(num_class=self.num_classes, temperature=self.args.multi_ce_temp)
        self.multi_pos_loss = ExclusiveCE(num_class=self.num_classes, temperature=self.args.multi_ce_temp)