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
Only highlight top-1 class as a basline experiment
"""

class RegionCityscapesOr(region_cityscapes_or_tensor.RegionCityscapesOr):

    def __init__(self, args, root, datalist, split='train', transform=None, return_spx=False,
                 region_dict="dataloader/init_data/cityscapes/train.dict", mask_region=True, dominant_labeling=False, loading='binary', load_smaller_spx=False):
        super().__init__(args, root, datalist, split, transform, return_spx, region_dict, mask_region, dominant_labeling, loading, load_smaller_spx)

        ''' Load gt class-wise superpixel sizes '''
        gtcls_sp_size_path = '{}/superpixel_seed/cityscapes/seeds_{}/train/gtFine_multi_tensor/sp_gt_size.npy'.format(self.root, args.nseg)
        self.gtcls_sp_size = torch.from_numpy(np.load(gtcls_sp_size_path)) # N x nseg x C

        self.topone_baseline()

    def topone_baseline(self):
        args = self.args
        n_data, nseg, ncls = self.gtcls_sp_size.shape
        gtcls_sp_size = self.gtcls_sp_size.view(-1, ncls)
        topone_indices = gtcls_sp_size.argmax(dim=1)
        self.multi_hot_cls.fill_(0)
        self.multi_hot_cls = self.multi_hot_cls.view(-1, ncls)
        self.multi_hot_cls[torch.arange(n_data * nseg), topone_indices] = 1
        self.multi_hot_cls= self.multi_hot_cls.view(n_data, nseg, ncls)