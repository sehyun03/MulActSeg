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

        ### ratio 비율을 기준으로 gt filtering 진행
        small_ratio_mask = (gtcls_sp_ratio < args.multihot_filter_ratio) # N x nseg x C
        self.multi_hot_cls = torch.masked_fill(self.multi_hot_cls, small_ratio_mask, value = 0)