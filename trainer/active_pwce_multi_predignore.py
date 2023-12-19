import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_max
from torch_scatter import scatter_log_softmax
from torch_scatter import scatter_softmax

from trainer import active_joint_multi_predignore
from trainer.active_joint_multi_predignore import MultiChoiceCE_
from utils.scheduler import ramp_up
from utils.loss import MyCrossEntropyLoss
r""" Local prorotype similarity-based weighted cross entropy loss for partial labeled semantic segmentation training.
"""

class JointLocalProtoWeightingCE(nn.Module):
    def __init__(self, args, num_superpixel, temperature=1.0, reduction='mean'):
        super().__init__()
        self.args = args
        self.num_superpixel = num_superpixel
        self.ce_temp = args.ce_temp
        # self.simw_temp = args.simw_temp
        self.temp = temperature
        self.reduction = reduction
        self.eps = 1e-8
        assert(reduction == 'mean')
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

    def forward(self, inputs_plbl, feats_plbl, inputs, targets, superpixels, spmasks):
        r"""
        Args::
            inputs_plbl: NxCxHxW
            feats_plbl: NxChannelxHxW
            inputs: NxCxHxW
            targets: N x self.num_superpixel x C+1
            superpixels: NxHxW
            spmasks: NxHxW
        
        Returns::
            loss: 
        """
        # TODO) torch.no_grad 작업
        # with torch.no_grad():

        N, C, H, W = inputs_plbl.shape
        outputs_plbl = F.softmax(inputs_plbl / self.ce_temp, dim=1) ### N x C x H x W
        outputs_plbl = outputs_plbl.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C

        _, Ch, _, _ = feats_plbl.shape
        feats_plbl = feats_plbl.permute(0,2,3,1).reshape(N, -1, Ch) ### N x HW x Ch
        
        outputs = F.softmax(inputs / self.ce_temp, dim=1) ### N x HW x C
        outputs = outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C

        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW

        is_trg_multihot = (1 < targets.sum(dim=2)) ### N x self.num_superpixel

        loss = 0
        num_valid = 0

        for i in range(N):
            '''
            outputs_plbl[i] ### HW x C
            feats_plbl[i] : HW x Ch
            superpixels[i] ### HW x 1
            targets[i] : self.num_superpiexl x C
            spmasks[i] ### HW x 1
            '''
            
            with torch.no_grad():
                r''' Superpixel mask filtering (for efficiency)
                - spmasks[i]: selected superpixel regions
                - ㄴ If no selected spx within image, skip this image
                - valid mask (== spmasks && multi_mask) filtering outputs_plbl 
                - ㄴspmasks 에 안걸러졌기 때문에 superpixels[i] 는 invalid spx id 를 포함할 수 있음.
                '''
                if not torch.any(spmasks[i]): continue
                validall_superpixel = superpixels[i][spmasks[i]]
                validall_trg_pixel = targets[i][validall_superpixel.squeeze(dim=1)]

                multi_mask = is_trg_multihot[i][validall_superpixel.squeeze(dim=1)].detach()
                valid_mask = spmasks[i].clone()
                valid_mask[spmasks[i]] = multi_mask
                if not torch.any(valid_mask): continue

                valid_output = outputs_plbl[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
                vpx_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID
                valid_feat = feats_plbl[i][valid_mask] ### HW' x Ch

                r''' get max (=prototype) pixel for each class within superpixel '''
                _, maxvidx_per_spx_cls = scatter_max(valid_output, vpx_superpixel, dim=0, dim_size=self.args.nseg)
                ### ㄴ self.num_superpixel x C: 각 (superpixel, class) pair 의 max 값을 가지는 index (valid)

                r''' filter invalid superpixels '''
                is_spx_valid = maxvidx_per_spx_cls[:,0] < valid_output.shape[0]
                ### ㄴ self.num_superpixel: vpx_superpixel 에 포함되지 않은 superpixel id 에 대해서는 valid_output index 최대값 (max index)로 잡힘. 이 값을 통해 쓸모없는 spx filtering
                maxvidx_per_vspx_cls = maxvidx_per_spx_cls[is_spx_valid]
                ### ㄴ nvalidseg x C
                multihot_vspx = targets[i][is_spx_valid]
                ### ㄴ nvalidseg x C

                r''' Weight generation
                - Get prototype index
                - Caculated smilarity score and obtain weights
                '''
                proto_vspx, proto_clsdex = multihot_vspx.nonzero(as_tuple=True)
                ### ㄴ (nproto, nproto): List of all (valid spx, valid C) pair, nvalidspx x c == nproto
                prototype_videx = maxvidx_per_vspx_cls[proto_vspx, proto_clsdex]
                ### ㄴ maxvidx_per_spx_cls 중에서 valid 한 superpixel 과 target 에서의 valid index

                prototypes = valid_feat[prototype_videx]
                ### ㄴ nproto x Ch
                similarity = torch.mm(prototypes, valid_feat.T)
                ### ㄴ nproto x nvalid_pixels: prototype-wise similarity to every valid pixel
                proto_vpixel_simialrity = scatter_softmax(similarity / self.args.simw_temp, proto_vspx, dim=0)
                ### ㄴ nproto x nvalid_pixels: prototype-wise similarity to each pixel w/ softmax normalization within spx

                r''' Weight assignment
                - validall_trg_pixel 의 중 multi-hot label 들에게 'proto_vpixel_simialrity' 를 weight 값으로 assign.
                - proto_vpixel_simialrity 에 대한 indexing 을 위해 같은 size 의 index matrix 정의
                    -- index matrix 계산을 위해 proto_vspx index space 변환
                    -- validall_trg_pixel 에 assign 하기 위해 multi index 를 validall index 로 변환
                '''
                vspdex_to_spdex = is_spx_valid.nonzero(as_tuple=True)[0]
                proto_spx = vspdex_to_spdex[proto_vspx] ### to calcualte equal operation in all index
                multispx = validall_superpixel[multi_mask].squeeze(dim=1)

                is_similarity_valid = torch.eq(proto_spx[..., None], multispx[None, ...])
                valid_similarity = torch.masked_select(proto_vpixel_simialrity.T.reshape(-1), is_similarity_valid.T.reshape(-1))

                multidex_to_valall_idx = multi_mask.nonzero().squeeze(dim=1)
                multi_trg_pixel = validall_trg_pixel[multi_mask]
                update_multi_idx, update_cdx = multi_trg_pixel.nonzero(as_tuple=True)
                update_valall_idx = multidex_to_valall_idx[update_multi_idx]
                validall_trg_pixel = validall_trg_pixel.float()
                validall_trg_pixel[update_valall_idx, update_cdx] = valid_similarity
            
            r''' Cross entropy loss with according weighting '''
            valid_prediction_output = outputs[i][spmasks[i]]
            valid_pred_ce = -torch.log(valid_prediction_output + self.eps)
            weighted_ce = valid_pred_ce * validall_trg_pixel

            loss += weighted_ce.sum()
            num_valid += weighted_ce.shape[0]

        if loss == 0:
            return 0
        
        if torch.isnan(loss):
            loss = 0

        return loss / num_valid

class ActiveTrainer(active_joint_multi_predignore.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)
        self.simw_temp_orig = args.simw_temp

    def get_criterion(self):
        ''' Define criterion '''
        self.joint_local_proto_weight_loss = JointLocalProtoWeightingCE(args=self.args, num_superpixel=self.args.nseg, temperature=self.args.group_ce_temp)

    def train_impl(self, total_itrs, val_period):
        self.net.train()
        pbar = tqdm(range(total_itrs), ncols=150)
        args = self.args

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

            ### Another forward for pseudo labeling
            with torch.no_grad():
                self.net.eval()
                self.net.set_return_feat()
                feats_plbl, preds_plbl = self.net.feat_forward(images)
                self.net.unset_return_feat()

            ### Loss
            if args.simw_temp_schedule:
                if iteration < 20000:
                    self.args.simw_temp = 1000.0
                else:
                    self.args.simw_temp = self.simw_temp_orig

            joint_ce_loss = self.joint_local_proto_weight_loss(preds_plbl, feats_plbl, preds, labels, superpixels, spmasks)
            # lamda = ramp_up(iteration / total_itrs, lamparam=args.lamparam, scale=args.lamscale, dorampup=args.dorampup)
            loss = joint_ce_loss

            ### Update (Model, Scheduler)
            self.update(loss)

            ### Logging
            self.update_average_meter({'train-loss': loss})

            ### Logging intervals
            self.log_training(iteration, pbar, total_itrs)
            self.log_validation(iteration, val_period)