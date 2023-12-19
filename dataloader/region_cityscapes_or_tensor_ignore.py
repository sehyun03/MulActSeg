from PIL import Image
import numpy as np
import torch

import imageio
imageio.plugins.freeimage.download()
from .region_cityscapes_or_tensor import RegionCityscapesOr

class RegionCityscapesOr(RegionCityscapesOr):

    def __init__(self, args, root, datalist, split='train', transform=None, return_spx=False,
                 region_dict="dataloader/init_data/cityscapes/train.dict", mask_region=True, dominant_labeling=False, loading='binary', load_smaller_spx=False):
        super().__init__(args, root, datalist, split, transform, return_spx, region_dict, mask_region, dominant_labeling, loading, load_smaller_spx)

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
        target_precise = Image.open(target_path) ### Image
        target_precise = Image.fromarray(self.encode_target(target_precise).astype('uint8')) ### Numpy

        ''' Get actively sampled superpixel ids '''
        if spx_fname in self.suppix:
            preserving_labels = self.suppix[spx_fname]
        else:
            preserving_labels = []

        ''' Return image and superpixel for pooling dataset '''
        if self.split == 'active-ulabel':
            return self.__getpoolitem__(image, superpixel)

        ''' Augment both images, superpixel map '''
        if self.load_smaller_spx:
            superpixel_small = self.open_spx(spx_fname.replace("seeds_{}".format(self.args.nseg),"seeds_{}".format(self.args.small_nseg)))
            image, lbls = self.transform(image, [target_precise, superpixel, superpixel_small])
            target_precise, superpixel, superpixel_small = lbls
        else:
            image, lbls = self.transform(image, [target_precise, superpixel])
            target_precise, superpixel = lbls

        h, w = superpixel.shape

        ''' Filter unselected superpixels & ignored regions '''
        sp_mask = torch.from_numpy(np.isin(superpixel.reshape(-1), preserving_labels))
        sp_mask[target_precise.reshape(-1) == 255] = False
        sp_mask = sp_mask.reshape(h, w) ### boolean mask indicating selected superpixels

        if self.load_smaller_spx:
            sample = {'images': image, 'labels': target, 'spx': superpixel, 'spmask': sp_mask, 'spx_small': superpixel_small, 'fnames': self.im_idx[index]}
        else:
            sample = {'images': image, 'labels': target, 'spx': superpixel, 'spmask': sp_mask, 'fnames': self.im_idx[index]}

        return sample