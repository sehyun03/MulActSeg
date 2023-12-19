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
from . import region_cityscapes
r""" Pseudo label region cityscapes
- Same as precise label loader (but without label encoding)
"""
class RegionCityscapes(region_cityscapes.RegionCityscapes):
    def __init__(self, args, root, datalist, split='train', transform=None, return_spx=False,
                 region_dict="dataloader/init_data/cityscapes/train.dict", mask_region=True, dominant_labeling=False):
        super().__init__(args, root, datalist, split, transform, return_spx, region_dict, mask_region, dominant_labeling)
        
        r''' Get pseudo label directory from resume_checkpoint '''
        round = args.resume_checkpoint[-6:-4]
        assert(int(round) == args.init_iteration)
        ckpt_root = '/'.join(args.resume_checkpoint.split('/')[:-1])
        if args.plbl_type is not None:
            self.plbl_root = '{}/plbl_gen_{}/round_{}'.format(ckpt_root, args.plbl_type, round)
        else:
            self.plbl_root = '{}/plbl_gen/round_{}'.format(ckpt_root, round)
        assert(os.path.exists(self.plbl_root))

    def __getitem__(self, index):
        img_fname, _, spx_fname = self.im_idx[index]
        
        '''Load image amd label'''
        image = Image.open(img_fname).convert('RGB')

        img_id = img_fname.split('/')[-1].split('_leftImg8bit')[0]
        target_path = "{}/{}.png".format(self.plbl_root, img_id)
        target_plbl = Image.open(target_path)

        image, lbls = self.transform(image, [target_plbl])
        target_plbl = lbls[0]

        sample = {'images': image, 'labels': target_plbl, 'fnames': self.im_idx[index]}

        return sample