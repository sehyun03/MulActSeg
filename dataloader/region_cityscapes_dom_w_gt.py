import json
import os
import dataloader.ext_transforms as et
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
from time import time
from . import region_cityscapes

from .constant import train_id_to_color, id_to_train_id
# for synthia
import imageio
imageio.plugins.freeimage.download()

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
        super().__init__(args,
                        root,
                        datalist,
                        split,
                        transform,
                        return_spx,
                        region_dict,
                        mask_region,
                        dominant_labeling)
        assert(self.dominant_labeling)
        if 'predignore' in args.init_checkpoint:
            self.pred_ignore = True
        else:
            self.pred_ignore = False

    def __getitem__(self, index):
        img_fname, lbl_fname, spx_fname = self.im_idx[index]
        '''Load image, label, and superpixel'''
        image = Image.open(img_fname).convert('RGB')
        target = Image.open(lbl_fname)
        target = torch.from_numpy(np.array(target))
        if self.pred_ignore:
            target = torch.masked_fill(target, target == 255, 19) ### original label as 19th class
        target = Image.fromarray(target.numpy())

        superpixel = self.open_spx(spx_fname)

        id = lbl_fname.split('/')[-1].split('.')[0]
        target_path = '{}/gtFine/train/{}/{}_gtFine_labelIds.png'.format(self.root, id.split('_')[0], id)
        r''' 본래 255 였던 영역을 19 번째 클래스로 치환해준다 (undefined label 예측을 위함)
        - 해당 loader 에서 255 index 는 'unselected region' 용으로 활용된다. '''
        target_precise = Image.open(target_path)
        target_precise = torch.from_numpy(self.encode_target(target_precise).astype('uint8'))
        if self.pred_ignore:
            target_precise = torch.masked_fill(target_precise, target_precise == 255, 19) ### original label as 19th class
        label = Image.fromarray(target_precise.numpy())

        image, lbls = self.transform(image, [target, label, superpixel])
        target, label, superpixel = lbls

        ''' GT masking (mimic region-based annotation) '''
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

        sp_mask = torch.from_numpy(mask.reshape(h,w))
        # sp_mask = torch.isin(superpixel, preserving_labels)

        sample = {'images': image, 'target': target, 'labels': label, 'spx': superpixel, 'spmask': sp_mask, 'fnames': self.im_idx[index]}

        return sample