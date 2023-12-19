import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch_scatter import scatter

from trainer import active_joint_multi_predignore
from models import get_model
from trainer.active_joint_multi_predignore import GroupMultiLabelCE_, MultiChoiceCE_
from utils.miou import MeanIoU
from utils.miou_evalignore import IoUIgnore
r"""
Additionally predict undefined (ignore) class within cityscapes
"""

class MultiChoiceCEScale(MultiChoiceCE_):
    def __init__(self, num_class, temperature=1.0, reduction='mean'):
        super().__init__(num_class, temperature, reduction)
        ncls_scale_map_abs = torch.Tensor([2.995732307434082,
                                            2.944438934326172,
                                            2.890371799468994,
                                            2.8332133293151855,
                                            2.7725887298583984,
                                            2.70805025100708,
                                            2.6390573978424072,
                                            2.5649492740631104,
                                            2.4849066734313965,
                                            2.397895336151123,
                                            2.3025851249694824,
                                            2.1972246170043945,
                                            2.079441547393799,
                                            1.945910096168518,
                                            1.7917594909667969,
                                            1.6094379425048828,
                                            1.3862943649291992,
                                            1.0986123085021973,
                                            0.6931471824645996])

        self.ncls_scale_map_rel = ncls_scale_map_abs[0] / ncls_scale_map_abs
        self.ncls_scale_map_rel = self.ncls_scale_map_rel.cuda()
        # ㄴ 19

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
            nhot = trg_pixel.sum(dim=1)
            ncls_weight = self.ncls_scale_map_rel[nhot-1]

            if self.reduction == 'mean':
                loss += -(ncls_weight * torch.log(pos_pred + self.eps)).sum()
            elif self.reduction == 'none':
                new_valid_mask = valid_mask.clone()
                new_valid_mask[valid_mask] = empty_trg_mask
                pixel_loss[i, new_valid_mask] = -(ncls_weight * torch.log(pos_pred + self.eps))

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
        self.multi_pos_loss = MultiChoiceCEScale(num_class=self.num_classes, temperature=self.args.multi_ce_temp)