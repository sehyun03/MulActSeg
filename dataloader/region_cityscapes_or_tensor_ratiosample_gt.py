import json
import os, sys
import dataloader.ext_transforms as et
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
from time import time

from .constant import train_id_to_color, id_to_train_id
# for synthia
import imageio
imageio.plugins.freeimage.download()
from . import region_cityscapes_or_tensor
r"""
tiny filtering with gt class-wise superpixel ratios
- remove small ratio class from multi-hot label using ground truth class-wise sizes
"""

class RegionCityscapesOr(region_cityscapes_or_tensor.RegionCityscapesOr):

    def __init__(self, args, root, datalist, split='train', transform=None, return_spx=False,
                 region_dict="dataloader/init_data/cityscapes/train.dict", mask_region=True, dominant_labeling=False, loading='binary', load_smaller_spx=False):
        super().__init__(args, root, datalist, split, transform, return_spx, region_dict, mask_region, dominant_labeling, loading, load_smaller_spx)

        ''' Load gt class-wise superpixel sizes '''
        gtcls_sp_size_path = '{}/superpixel_seed/cityscapes/seeds_{}/train/gtFine_multi_tensor/sp_gt_size.npy'.format(self.root, args.nseg)
        self.gtcls_sp_size = torch.from_numpy(np.load(gtcls_sp_size_path)) # N x nseg x C

        ''' Remove tiny multi-hot labels from "multi_hot_cls" '''
        self.ratiofilter_tiny_cls_wgt()

    def ratiofilter_tiny_cls_wgt(self):
        r"""
        - remove small cls label from multi-hot labels (using ratio)
        - - removing size criteria: args.multihot_filter_ratio
        """
        args = self.args
        eps = 1e-12

        n_data, nseg, ncls = self.gtcls_sp_size.shape

        r''' filtering by size criteria '''
        ### 각 size 의 ratio value 얻기 (0 으로 갈아끼우고 -> sum -> 나누기)
        gtcls_sp_ratio = self.gtcls_sp_size.clone()
        invalid_mask = (gtcls_sp_ratio == -1)
        gtcls_sp_ratio = torch.masked_fill(gtcls_sp_ratio, invalid_mask, value = 0)
        gtcls_sp_ratio = gtcls_sp_ratio / (gtcls_sp_ratio.sum(dim=2)[..., None] + eps)

        ### Size ratio 에 따라서 sampling 수행
        gtcls_sp_ratio_ = gtcls_sp_ratio.view(-1, gtcls_sp_ratio.shape[-1])
        max_nmulti_cls = self.multi_hot_cls.sum(dim=2).max().item()
        seleted_indices = torch.multinomial(gtcls_sp_ratio_ + eps, num_samples=max_nmulti_cls, replacement=False)
        num_data, num_seg, num_cls = self.multi_hot_cls.shape
        db_multi_hot_cls = self.multi_hot_cls.clone()
        self.multi_hot_cls = self.multi_hot_cls.view(-1, num_cls)


        ### 만약 뽑힌 superpixel 의 ratio 가 threshold 이하일 경우에만 sampling 수행
        cumulative_sp_ratio = torch.zeros_like(gtcls_sp_ratio_[:, 0])
        label_assign_mask = torch.ones_like(cumulative_sp_ratio).bool()
        for count in range(max_nmulti_cls):
            cumulative_sp_ratio += gtcls_sp_ratio_[torch.arange(gtcls_sp_ratio_.shape[0]), seleted_indices[:, count]]
            label_assign_mask[cumulative_sp_ratio == 0] = False

            ### label assignment
            self.multi_hot_cls[torch.arange(gtcls_sp_ratio_.shape[0]), seleted_indices[:, count]] = label_assign_mask.byte()
            label_assign_mask[(1 - args.multihot_filter_ratio) < cumulative_sp_ratio] = False

        self.multi_hot_cls = self.multi_hot_cls.view(num_data, num_seg, num_cls)