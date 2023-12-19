import json
import os
import dataloader.ext_transforms as et
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch

from .constant import train_id_to_color, id_to_train_id
# for synthia
import imageio
imageio.plugins.freeimage.download()
from .region_cityscapes import RegionCityscapes

class RegionCityscapesDominantAll(RegionCityscapes):

    def __init__(self, args, root, datalist, split='train', transform=None, return_spx=False,
                 region_dict="dataloader/init_data/cityscapes/train.dict", mask_region=True, dominant_labeling=False,
                 or_labeling=False,
                 generate_ignore=False):
        super().__init__(args, root, datalist, split, transform, return_spx, region_dict, mask_region, dominant_labeling)
        self.generate_ignore = generate_ignore

    def __getitem__(self, index):
        img_fname, lbl_fname, spx_fname = self.im_idx[index]
        '''Load image, label, and superpixel'''
        image = Image.open(img_fname).convert('RGB')
        target = Image.open(lbl_fname)
        superpixel = self.open_spx(spx_fname)
        image, lbls = self.transform(image, [target, superpixel])
        target, superpixel = lbls
        target = self.encode_target(target)

        '''GT masking (mimic region-based annotation)'''
        if self.mask_region is True:
            h, w = target.shape
            target = target.reshape(-1)
            ignore_mask  = torch.from_numpy(target==255)
            superpixel = superpixel.reshape(-1)
            preserving_labels = self.suppix[spx_fname]

            ''' dominant label assignment '''
            for p in preserving_labels:
                if self.generate_ignore:
                    sp_mask = (superpixel == p)
                else:
                    sp_mask = torch.logical_and((superpixel == p), torch.logical_not(ignore_mask))
                u, c = np.unique(target[sp_mask], return_counts=True)
                if c.size != 0:
                    target[sp_mask] = u[c.argmax()]
            if self.generate_ignore:
                pass
            else:
                target[ignore_mask] = 255
            target = target.reshape(h, w)
            superpixel = superpixel.reshape(h, w)
    
        if self.return_spx is False:
            sample = {'images': image, 'labels': target, 'fnames': self.im_idx[index]}
        else:
            sample = {'images': image, 'labels': target, 'spx': superpixel, 'fnames': self.im_idx[index]}
        return sample