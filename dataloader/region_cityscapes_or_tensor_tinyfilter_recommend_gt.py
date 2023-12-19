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
tiny filtering with gt class-wise superpixel sizes
- remove small pixels from multi-hot label using ground truth class-wise sizes
"""

class RegionCityscapesOr(region_cityscapes_or_tensor.RegionCityscapesOr):

    def __init__(self, args, root, datalist, split='train', transform=None, return_spx=False,
                 region_dict="dataloader/init_data/cityscapes/train.dict", mask_region=True, dominant_labeling=False, loading='binary', load_smaller_spx=False):
        super().__init__(args, root, datalist, split, transform, return_spx, region_dict, mask_region, dominant_labeling, loading, load_smaller_spx)

        ''' Load gt class-wise superpixel sizes '''
        gtcls_sp_size_path = '{}/superpixel_seed/cityscapes/seeds_{}/train/gtFine_multi_tensor/sp_gt_size.npy'.format(self.root, args.nseg)
        self.gtcls_sp_size = torch.from_numpy(np.load(gtcls_sp_size_path)) # N x nseg x C

        ''' Remove tiny multi-hot labels from "multi_hot_cls" '''
        self.filter_tiny_cls_wgt()

    def filter_tiny_cls_wgt(self):
        r"""
        - remove small cls label from multi-hot labels
        - - removing size criteria: args.multihot_filter_size        
        """
        args = self.args

        n_data, nseg, ncls = self.gtcls_sp_size.shape

        r''' filtering by size criteria
        - small_size_mask: is (spx, class) smaller than criteria
        - dominant_mask: is spx chosen to be dominant label
        '''
        small_size_mask = (self.gtcls_sp_size < args.multihot_filter_size) # N x nseg x C
        large_size_mask = torch.logical_not(small_size_mask) # N x nseg x C
        dominant_mask = large_size_mask.sum(dim=2) < 2 # N x nseg
        replace_target_mask = torch.logical_and(small_size_mask, dominant_mask[..., None]) # N x nseg x C
        self.multi_hot_cls = torch.masked_fill(self.multi_hot_cls, replace_target_mask, value = 0) # N x nseg x C

        r''' keep top-1 class within every superpixels '''
        max_cdx = torch.argmax(self.gtcls_sp_size, dim=2) # N x nseg
        max_cdx = max_cdx.view(-1) # N * nseg
        self.multi_hot_cls = self.multi_hot_cls.view(-1, ncls)
        self.multi_hot_cls[torch.arange(n_data * nseg), max_cdx] = 1
        self.multi_hot_cls = self.multi_hot_cls.view(n_data, nseg, ncls)