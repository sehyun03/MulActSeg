from sys import implementation
from PIL import Image
import numpy as np
import torch

import imageio
imageio.plugins.freeimage.download()
from .region_cityscapes_or_tensor import RegionCityscapesOr
import dataloader.ext_transforms as et

class RegionCityscapesOr(RegionCityscapesOr):

    def __init__(self, args, root, datalist, split='train', transform=None, return_spx=False,
                 region_dict="dataloader/init_data/cityscapes/train.dict", mask_region=True, dominant_labeling=False, loading='binary', load_smaller_spx=False):
        super().__init__(args, root, datalist, split, transform, return_spx, region_dict, mask_region, dominant_labeling, loading, load_smaller_spx)
        
        self.weak_transform = et.ExtCompose([
            et.ExtResize((1024, 2048)),
            et.ExtToTensor(dtype_list=['int', 'int','int']),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        assert(self.mask_region)
        img_fname, lbl_fname, spx_fname = self.im_idx[index] ### warnning: index => superpixel-wise 로 정의됨
        ''' Load image, label, and superpixel '''
        image = Image.open(img_fname).convert('RGB')
        superpixel = self.open_spx(spx_fname)

        id = lbl_fname.split('/')[-1].split('.')[0]
        trg_index = self.id_to_index[id]
        target = self.multi_hot_cls[trg_index] ### [nseg x (num_classes + 1)]
        target_path = '{}/gtFine/train/{}/{}_gtFine_labelIds.png'.format(self.root, id.split('_')[0], id)
        target_precise = Image.open(target_path)

        ''' Get actively sampled superpixel ids '''
        if spx_fname in self.suppix:
            preserving_labels = self.suppix[spx_fname]
        else:
            preserving_labels = []

        ''' Augment both images, superpixel map '''
        if self.load_smaller_spx:
            superpixel_small = self.open_spx(spx_fname.replace("seeds_{}".format(self.args.nseg),"seeds_{}".format(self.args.small_nseg)))
            image_weak, lbls_weak = self.weak_transform(image.copy(), [target_precise.copy(), superpixel.copy(), superpixel_small.copy()])
            target_precise_weak, spx_weak, spx_small_weak = lbls_weak
            image, lbls = self.transform(image, [target_precise, superpixel, superpixel_small])
            target_precise, superpixel, superpixel_small = lbls
        else:
            raise NotImplementedError


        ''' Filter unselected superpixels & ignored regions '''
        h, w = superpixel.shape
        target_precise = self.encode_target(target_precise)
        sp_mask = torch.from_numpy(np.isin(superpixel.reshape(-1), preserving_labels))
        sp_mask[target_precise.reshape(-1) == 255] = False
        sp_mask = sp_mask.reshape(h, w) ### boolean mask indicating selected superpixels

        # spx_weak
        h, w = spx_weak.shape
        target_precise_weak = self.encode_target(target_precise_weak)
        sp_mask_weak = torch.from_numpy(np.isin(spx_weak.reshape(-1), preserving_labels))
        sp_mask_weak[target_precise_weak.reshape(-1) == 255] = False
        sp_mask_weak = sp_mask_weak.reshape(h, w) ### boolean mask indicating selected superpixels

        if self.load_smaller_spx:
            sample = {'images': image, 'image_weak': image_weak, 'labels': target, 'spx': superpixel, 'spx_weak': spx_weak, 'spmask': sp_mask, 'spmask_weak': sp_mask_weak, 'spx_small': superpixel_small, 'spx_small_weak': spx_small_weak,  'fnames': self.im_idx[index]}
        else:
            sample = {'images': image, 'image_weak': image_weak, 'labels': target, 'spx': superpixel, 'spx_weak': spx_weak, 'spmask': sp_mask, 'fnames': self.im_idx[index]}

        return sample