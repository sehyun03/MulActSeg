import json
import os
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
from  dataloader import region_cityscapes

class RegionCityscapes(region_cityscapes.RegionCityscapes):

    def __init__(self,
                 args,
                 root,
                 datalist,
                 split='train',
                 transform=None,
                 return_spx=False,
                 region_dict="dataloader/init_data/cityscapes/train.dict",
                 mask_region=True,
                 dominant_labeling=False):
        super().__init__(args, root, datalist, split, transform, return_spx, region_dict, mask_region, dominant_labeling)
        assert(self.dominant_labeling)

    def __getitem__(self, index):
        img_fname, lbl_fname, spx_fname = self.im_idx[index]
        '''Load image, label, and superpixel'''
        image = Image.open(img_fname).convert('RGB')
        superpixel = self.open_spx(spx_fname)
        target = Image.open(lbl_fname)

        r''' convert ignore label into new class '''
        target = torch.from_numpy(np.array(target))
        target = torch.masked_fill(target, target == 255, 19) ### original label as 19th class
        target = Image.fromarray(target.numpy())

        image, lbls = self.transform(image, [target, superpixel])
        target, superpixel = lbls
        # ㄴ  19: discovered ignore labels, 255: unselected superpixels

        ''' GT masking (mimic region-based annotation) '''
        if self.mask_region is True:
            h, w = target.shape
            target = target.reshape(-1)
            superpixel = superpixel.reshape(-1)
            if spx_fname in self.suppix:
                preserving_labels = self.suppix[spx_fname]
            else:
                preserving_labels = []
            mask = np.isin(superpixel, preserving_labels)
            target = np.where(mask, target, 255) ### filter selected superpixel
            target = target.reshape(h, w)
            superpixel = superpixel.reshape(h, w)

        if self.return_spx is False:
            sample = {'images': image, 'labels': target, 'fnames': self.im_idx[index]}
        else:
            sample = {'images': image, 'labels': target, 'spx': superpixel, 'fnames': self.im_idx[index]}
        return sample