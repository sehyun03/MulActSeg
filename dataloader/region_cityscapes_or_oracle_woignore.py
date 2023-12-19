from PIL import Image
import numpy as np
import torch
import os

import imageio
imageio.plugins.freeimage.download()
from .region_cityscapes_or_tensor import RegionCityscapesOr

class RegionCityscapesOr(RegionCityscapesOr):

    def __init__(self, args, root, datalist, split='train', transform=None, return_spx=False,
                 region_dict="dataloader/init_data/cityscapes/train.dict", mask_region=True, dominant_labeling=False, loading='binary', load_smaller_spx=False):
        super().__init__(args, root, datalist, split, transform, return_spx, region_dict, mask_region, dominant_labeling, loading, load_smaller_spx)
        assert(self.mask_region)
        assert(not self.load_smaller_spx)

    def __getitem__(self, index):
        img_fname, lbl_fname, spx_fname = self.im_idx[index] ### warnning: index => superpixel-wise 로 정의됨

        ''' Load image, label, and superpixel '''
        image = Image.open(img_fname).convert('RGB')
        superpixel = self.open_spx(spx_fname)

        ''' Return image and superpixel for pooling dataset '''
        if self.split == 'active-ulabel':
            return self.__getpoolitem__(image, superpixel)

        id = lbl_fname.split('/')[-1].split('.')[0]
        target_path = '{}/gtFine/train/{}/{}_gtFine_labelIds.png'.format(self.root, id.split('_')[0], id)
        target_precise = Image.open(target_path)
        target_precise = torch.from_numpy(self.encode_target(target_precise).astype('uint8'))
        target_precise = Image.fromarray(target_precise.numpy())

        ''' Augment both images, superpixel map '''
        image, lbls = self.transform(image, [target_precise, superpixel])
        target_precise, superpixel = lbls

        ''' Get actively sampled superpixel ids '''
        preserving_labels = self.suppix[spx_fname] if spx_fname in self.suppix else []

        ''' Filter unselected superpixels & ignored regions '''
        sp_mask = torch.isin(superpixel, torch.tensor(preserving_labels))
        target_precise = torch.masked_fill(target_precise, torch.logical_not(sp_mask), 255)

        sample = {'images': image,
                  'labels': target_precise,
                  'spx': superpixel,
                  'spmask': sp_mask,
                  'fnames': self.im_idx[index]}

        return sample