import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch_scatter import scatter

from trainer import active_joint_multi_predignore
from trainer.active_joint_multi_predignore import GroupMultiLabelCE_
from trainer.active_joint_multi_predignore_mclossablation import MultiChoiceCE_onlyDom
r"""
Remove multi-choice loss within multi-hot loss
- Dominant spx: pixel-wise classification loss
- Multi-hot spx: group-multi loss
"""

class GroupMultiLabelCE_onlymulti(GroupMultiLabelCE_):
    def __init__(self, args, num_class, num_superpixel, temperature=1.0, reduction='mean'):
        super().__init__(args, num_class, num_superpixel, temperature, reduction)


    def forward(self, inputs, targets, superpixels, spmasks):
        ''' inputs: NxCxHxW
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
        empty_trg_mask = torch.any(targets, dim=2).bool() ### N x self.num_superpixel
        is_trg_multi = (1 < targets.sum(dim=2)) ### N x self.num_superpixel
        loss = 0
        num_valid = 1

        for i in range(N):
            '''
            outputs[i] ### HW x C
            superpixels[i] ### HW x 1
            spmasks[i] ### HW x 1
            '''

            ### filtered outputs
            valid_mask = spmasks[i]
            if not torch.any(valid_mask):
                continue
            multi_mask = is_trg_multi[i][superpixels[i].squeeze(dim=1)[spmasks[i]]].detach()
            valid_mask = spmasks[i].clone()
            valid_mask[spmasks[i]] = multi_mask
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

            top_one_preds_nonzero = top_one_preds[top_one_preds.nonzero(as_tuple=True)] ### 해당 value indexing
            num_valid += top_one_preds_nonzero.shape[0] ### valid pixel 개수 측정

            loss += -torch.log(top_one_preds_nonzero + self.eps).sum()

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
        self.group_multi_loss = GroupMultiLabelCE_onlymulti(args=self.args, num_class=self.num_classes, num_superpixel= self.args.nseg, temperature=self.args.group_ce_temp)
        self.multi_pos_loss = MultiChoiceCE_onlyDom(num_class=self.num_classes, temperature=self.args.multi_ce_temp)