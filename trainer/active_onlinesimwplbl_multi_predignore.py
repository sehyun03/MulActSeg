import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_max

from trainer import active_onlineplbl_multi_predignore
from trainer.active_joint_multi_predignore import MultiChoiceCE_
from trainer.active_onlineplbl_multi_predignore import LocalProtoCE
r""" online pseudo labeling with local prototype-based pseudo labeling.
- Additional weighting on the pseudo label using model predicted probability.
"""

class LocalsimWProtoCE(LocalProtoCE):
    def __init__(self, args, num_superpixel, temperature=1.0, reduction='mean'):
        super().__init__(args, num_superpixel, temperature, reduction)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255, reduction='none')

    def generate_plbl(self, inputs_plbl, feats_plbl, targets, superpixels, spmasks):
        r"""
        Args::
            inputs_plbl: NxCxHxW
            feats_plbl: NxChannelxHxW
            targets: N x self.num_superpixel x C+1
            superpixels: NxHxW
            spmasks: NxHxW
        
        Returns::
            nn_plbl: N x HW x1
        """
        N, C, H, W = inputs_plbl.shape
        outputs = F.softmax(inputs_plbl / self.temp, dim=1) ### N x C x H x W
        outputs = outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C

        _, Ch, _, _ = feats_plbl.shape
        feats_plbl = feats_plbl.permute(0,2,3,1).reshape(N, -1, Ch) ### N x HW x Ch
        
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW

        is_trg_multi = (1 < targets.sum(dim=2)) ### N x self.num_superpixel

        r''' goal: generate pseudo label for multi-hot superpixels ''' 
        nn_plbl = torch.ones_like(superpixels).squeeze(dim=2) * 255 ### N x HW x 1
        weight = torch.zeros_like(feats_plbl[..., 0]) ### N x HW

        for i in range(N):
            '''
            outputs[i] ### HW x C
            feats_plbl[i] : HW x Ch
            superpixels[i] ### HW x 1
            targets[i] : self.num_superpiexl x C
            spmasks[i] ### HW x 1
            '''
            multi_hot_target = targets[i] ### self.num_superpixel x C

            r''' valid mask (== spmasks && multi_mask) filtering outputs '''
            ### spmasks 에 안걸러졌기 때문에 superpixels[i] 는 invalid spx id 를 포함할 수 있음.
            if not torch.any(spmasks[i]):
                continue
            multi_mask = is_trg_multi[i][superpixels[i].squeeze(dim=1)[spmasks[i]]].detach()
            valid_mask = spmasks[i].clone()
            valid_mask[spmasks[i]] = multi_mask
            if not torch.any(valid_mask):
                continue

            valid_output = outputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
            vpx_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID
            valid_feat = feats_plbl[i][valid_mask] ### HW' x Ch

            r''' get max pixel for each class within superpixel '''
            _, vdx_sup_mxpool = scatter_max(valid_output, vpx_superpixel, dim=0, dim_size=self.args.nseg)
            ### ㄴ self.num_superpixel x C: 각 (superpixel, class) pair 의 max 값을 가지는 index

            r''' filter invalid && single superpixels '''
            is_spx_valid = vdx_sup_mxpool[:,0] < valid_output.shape[0]
            ### ㄴ vpx_superpixel 에 포함되지 않은 superpixel id 에 대해서는 max index 가
            ### valid_output index 최대값 (==크기)로 잡힘. 이 값을 통해 쓸모없는 spx filtering
            vdx_vsup_mxpool = vdx_sup_mxpool[is_spx_valid]
            ### ㄴ nvalidseg x C : index of max pixel for each class (for valid spx)
            trg_vsup_mxpool = multi_hot_target[is_spx_valid]
            ### ㄴ nvalidseg x C : multi-hot label (for valid spx)

            r''' Index conversion (valid pixel -> pixel) '''
            validex_to_pixdex = valid_mask.nonzero().squeeze(dim=1)
            ### ㄴ translate valid_pixel -> pixel space
            vspxdex, vcdex = trg_vsup_mxpool.nonzero(as_tuple=True)
            ### ㄴ valid superpixel index && valid class index
            top1_vdx = vdx_vsup_mxpool[vspxdex, vcdex]
            ### ㄴ vdx_sup_mxpool 중에서 valid 한 superpixel 과 target 에서의 valid index
            # top1_pdx = validex_to_pixdex[top1_vdx]
            # ### ㄴ max index 들을 pixel space 로 변환

            r''' Inner product between prototype features & superpixel features '''
            prototypes = valid_feat[top1_vdx]
            ### ㄴ nproto x Ch
            similarity = torch.mm(prototypes, valid_feat.T)
            ### ㄴ nproto x nvalid_pixels: 각 prototype 과 모든 valid pixel feature 사이의 유사도
            
            r''' Nearest prototype selection '''
            mxproto_pxl, idx_mxproto_pxl = scatter_max(similarity, vspxdex, dim=0)
            ### ㄴ nvalidspx x nvalid_pixels: pixel 별 가장 유사한 prototype id

            r''' Assign pseudo label of according prototype
            - idx_mxproto_pxl 중에서 각 pixel 이 해당하는 superpixel superpixel 의 값을 얻기
            - 이를 위해 우선 (superpixel -> valid superpixel)로 index conversion 을 만듦
            - pixel 별 superpixel id 를 pixel 별 valid superpixel id 로 변환 (=nearest_vspdex)
            - 각 valid superpixel 의 label 로 pseudo label assign (=plbl_vdx)
            - pseudo label map 의 해당 pixel 에 valid pixel 별 pseudo label 할당 (nn_plbl)
            '''
            spdex_to_vspdex = torch.ones_like(is_spx_valid) * -1
            spdex_to_vspdex[is_spx_valid] = torch.unique(vspxdex)
            vspdex_superpixel = spdex_to_vspdex[vpx_superpixel.squeeze(dim=1)]
            ### ㄴ HW': 여기 vpx_superpixel 의 id value 는 superpixel 의 id 이다. 이를 통해 valid superpixel idex conversion
            nearest_vspdex = idx_mxproto_pxl.T[torch.arange(vspdex_superpixel.shape[0]), vspdex_superpixel]
            nearest_similarity_vspdex = mxproto_pxl.T[torch.arange(vspdex_superpixel.shape[0]), vspdex_superpixel]
            plbl_vdx = vcdex[nearest_vspdex]
            nn_plbl[i, validex_to_pixdex] = plbl_vdx
            weight[i, validex_to_pixdex] = nearest_similarity_vspdex

        nn_plbl = nn_plbl.reshape(N, H, W)
        weight = weight.reshape(N, H, W)

        return weight, nn_plbl

    def forward(self, inputs_plbl, feats_plbl, inputs, targets, superpixels, spmasks):
        r"""
        Args::
            inputs:  N x C x H x W
            nn_plbl: N x H x W
        """
        with torch.no_grad():
            weight, nn_plbl = self.generate_plbl(inputs_plbl, feats_plbl, targets, superpixels, spmasks)

        r''' CE loss between plbl and prediction '''
        if self.args.th_wplbl is None:
            loss = weight * self.cross_entropy(inputs / self.temp, nn_plbl)
        else:
            loss = (self.args.th_wplbl < weight) * self.cross_entropy(inputs / self.temp, nn_plbl)

        r''' checkloss sanity (avoid nan loss) '''
        if torch.any(loss):
            loss = torch.masked_select(loss, loss != 0).mean()
        else:
            loss = 0

        return loss

class ActiveTrainer(active_onlineplbl_multi_predignore.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)

    def get_criterion(self):
        ''' Define criterion '''
        self.multi_local_proto_loss = LocalsimWProtoCE(args=self.args, num_superpixel=self.args.nseg, temperature=self.args.group_ce_temp)
        self.multi_pos_loss = MultiChoiceCE_(num_class=self.num_classes, temperature=self.args.multi_ce_temp)