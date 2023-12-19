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

class RegionCityscapesAll(RegionCityscapes):

    def __init__(self, args, root, datalist, split='train', transform=None, region_dict="dataloader/init_data/cityscapes/train.dict"):
        super().__init__(args, root, datalist, split, transform, False, region_dict, True, False)

    def __getitem__(self, index):
        img_fname, lbl_fname, spx_fname = self.im_idx[index]
        '''Load image, label, and superpixel'''
        image = Image.open(img_fname).convert('RGB')
        target = Image.open(lbl_fname)
        superpixel = self.open_spx(spx_fname)
        image, lbls = self.transform(image, [target, superpixel])
        target, superpixel = lbls
        target = self.encode_target(target)

        ''' superpixel dictionary generation '''
        superpixel_info = {}

        '''GT masking (mimic region-based annotation)'''
        target = target.reshape(-1)
        superpixel = superpixel.reshape(-1)
        preserving_labels = self.suppix[spx_fname]

        ''' label assignment '''
        for p in preserving_labels:
            sp_mask = (superpixel == p)
            u, c = np.unique(target[sp_mask], return_counts=True)
            isignore = 255 in u
            allignore = np.all((u != 255))
            npx = sp_mask.sum()
            if not allignore:
                u_valid = u[u != 255]
                c_valid = c[u != 255]
                c_order = c_valid.argsort()[::-1]
                cls = u_valid[c_order].tolist()
                cpx = c_valid[c_order].tolist()
            else:
                cls = []
                cpx = []
            superpixel_info[p] = {'cls': cls, 'cpx': cpx, 'npx': npx, 'isignore': isignore, 'allignore': allignore}

        sample = {'superpixel_info': superpixel_info, 'fname': self.im_idx[index]}

        return sample