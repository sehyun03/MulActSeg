import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch_scatter import scatter

from trainer import active_joint_multi_predignore
from trainer.active_joint_multi_predignore import GroupMultiLabelCE_
from utils.loss import MultiChoiceCE
from utils.miou import MeanIoU
from utils.miou_evalignore import IoUIgnore
r"""
Remove multi-choice loss within multi-hot loss
- Dominant spx: pixel-wise classification loss
- Multi-hot spx + Dominant spx: group-multi loss
"""

class MultiChoiceCE_onlyDom(MultiChoiceCE):
    def __init__(self, num_class, temperature=1.0, reduction='mean'):
        super().__init__(num_class, temperature, reduction)

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
        if self.reduction == 'none':
            pixel_loss = torch.zeros_like(spmasks, dtype=torch.float)
        loss = 0
        num_valid = 1

        r''' goal: generate pseudo label for multi-hot superpixels ''' 
        is_trg_dominant = (1 == targets.sum(dim=2)) ### N x self.num_superpixel

        for i in range(N):
            '''
            outputs[i] ### HW x C
            superpixels[i] ### HW x 1
            spmasks[i] ### HW x 1
            '''
            ### filtered outputs
            valid_mask = spmasks[i] ### HW
            if not torch.any(valid_mask): ### 더 이상 뽑을게 없는 경우
                continue
            multi_mask = is_trg_dominant[i][superpixels[i].squeeze(dim=1)[spmasks[i]]].detach()
            valid_mask = spmasks[i].clone()
            valid_mask[spmasks[i]] = multi_mask
            if not torch.any(valid_mask):
                continue

            valid_output = outputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID

            trg_sup = targets[i] ### self.num_superpixel x C: multi-hot annotation
            trg_pixel = trg_sup[valid_superpixel.squeeze(dim=1)].detach() ### HW' x C : pixel-wise multi-hot annotation
            
            ### filter out empty target
            empty_trg_mask = torch.any(trg_pixel, dim=1).bool() ### HW'
            valid_output = valid_output[empty_trg_mask]
            trg_pixel = trg_pixel[empty_trg_mask]
            
            pos_pred = (valid_output * trg_pixel).sum(dim=1)
            num_valid += pos_pred.shape[0]
            if self.reduction == 'mean':
                loss += -torch.log(pos_pred + self.eps).sum()
            elif self.reduction == 'none':
                new_valid_mask = valid_mask.clone()
                new_valid_mask[valid_mask] = empty_trg_mask
                pixel_loss[i, new_valid_mask] = -torch.log(pos_pred + self.eps)

        if self.reduction == 'mean':
            return loss / num_valid
        elif self.reduction == 'none':
            return pixel_loss
        else:
            NotImplementedError

class ActiveTrainer(active_joint_multi_predignore.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)

    def get_criterion(self):
        ''' Define criterion '''
        self.group_multi_loss = GroupMultiLabelCE_(args=self.args, num_class=self.num_classes, num_superpixel= self.args.nseg, temperature=self.args.group_ce_temp)
        self.multi_pos_loss = MultiChoiceCE_onlyDom(num_class=self.num_classes, temperature=self.args.multi_ce_temp)