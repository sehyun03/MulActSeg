import torch
import numpy as np
import os
from tqdm import tqdm, trange
import torch.nn.functional as F
from torch_scatter import scatter, scatter_max
from PIL import Image
from skimage.segmentation import mark_boundaries

from dataloader import get_dataset
from dataloader.utils import DataProvider
from trainer.eval_within_multihot import ActiveTrainer
from models import get_model, freeze_bn
from utils.miou import MeanIoU

class ActiveTrainer(ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        self.selection_iter = selection_iter
        super().__init__(args, logger, selection_iter)

    def inference(self, loader, prefix=''):
        args = self.args
        iou_helper = MeanIoU(self.num_classes+1, args.ignore_idx)
        iou_helper._before_epoch()
        N = loader.__len__()

        ### model forward
        self.net.eval()
        self.net.set_return_feat()
        with torch.no_grad():
            for iteration in trange(N):
                batch = loader.__next__()
                images = batch['images'].to(self.device, dtype=torch.float32)
                labels = batch['labels'].to(self.device, dtype=torch.long)

                feats, outputs = self.net.feat_forward(images)

                r''' NN based pseudo label acquisition '''
                superpixels = batch['spx'].to(self.device)
                spmasks = batch['spmask'].to(self.device)
                targets = batch['target'].to(self.device)
                nn_pseudo_label = self.pseudo_label_generation(labels, feats, outputs, targets, spmasks, superpixels)
                ### ㄴ N x H x W

                output_dict = {
                    'outputs': nn_pseudo_label,
                    'targets': labels
                }
                iou_helper._after_step(output_dict)

        iou_table = []
        ious = iou_helper._after_epoch()
        miou = np.mean(ious)
        iou_table.append(f'{miou:.2f}')
        
        ### Append per-class ious
        for class_iou in ious:
            iou_table.append(f'{class_iou:.2f}')
        iou_table_str = ','.join(iou_table)

        del iou_table
        del output_dict
        print("\n[AL {}-round]: {}\n{}".format(self.selection_iter, prefix, iou_table_str), flush=True)

        return miou, iou_table_str


    def pseudo_label_generation(self, labels, feats, inputs, targets, spmasks, superpixels):
        r'''
        Args::
            feats: N x Channel x H x W
            inputs: N x C x H x W
            targets: N x self.num_superpiexl x C
            spmasks: N x H x W
            superpixels: N x H x W
            superpixel_smalls: N x H x W
            
            Apply max operation over predicted probabilities for each multi-hot label within the superpixel,
            and highlight selected top-1 pixel with its corresponding labels
            
        return::
            pseudo_label (torch.Tensor): pseudo label map to be evaluated
                                         N x H x W
            '''

        N, C, H, W = inputs.shape
        outputs = F.softmax(inputs, dim=1) ### N x C x H x W
        outputs = outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        N, Ch, H, W = feats.shape
        feats = feats.permute(0,2,3,1).reshape(N, -1, Ch) ### N x HW x Ch
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW
        nn_plbl = torch.ones_like(labels) * 255 ### N x H x W
        nn_plbl = nn_plbl.reshape(N, -1)

        for i in range(N):
            '''
            outputs[i] : HW x C
            feats[i] : HW x Ch
            superpixels[i] : HW x 1
            superpixel_smalls[i] : HW x 1
            targets[i] : self.num_superpiexl x C
            spmasks[i] : HW
            '''
            multi_hot_target = targets[i] ### self.num_superpixel x C

            r''' filtered outputs '''
            valid_mask = spmasks[i] ### HW
            if not torch.any(valid_mask): ### valid pixel 이 하나도 없으면 loss 계산 X
                continue #TODO
            valid_output = outputs[i][valid_mask] ### HW' x C
            valid_feat = feats[i][valid_mask] ### HW' x Ch
            vpx_superpixel = superpixels[i][valid_mask] ### HW' x 1

            r''' get max pixel for each class within superpixel '''
            _, vdx_sup_mxpool = scatter_max(valid_output, vpx_superpixel, dim=0, dim_size=self.args.nseg)
            ### ㄴ self.num_superpixel x C: 각 (superpixel, class) pair 의 max 값을 가지는 index
           
            r''' filter invalid superpixels '''
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
            _, idx_mxproto_pxl = scatter_max(similarity, vspxdex, dim=0)
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
            ### ㄴ 여기 vpx_superpixel 의 id value 는 superpixel 의 id 이다.
            nearest_vspdex = idx_mxproto_pxl.T[torch.arange(vspdex_superpixel.shape[0]), vspdex_superpixel]
            plbl_vdx = vcdex[nearest_vspdex]
            nn_plbl[i, validex_to_pixdex] = plbl_vdx

        nn_plbl = nn_plbl.reshape(N, H, W)
        
        return nn_plbl