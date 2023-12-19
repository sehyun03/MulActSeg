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

        r''' Get pseudo label directory from resume_checkpoint '''
        round = args.resume_checkpoint[-6:-4]
        ckpt_root = '/'.join(args.resume_checkpoint.split('/')[:-1])
        if args.plbl_type is not None:
            self.plbl_root = '{}/plbl_gen_{}/round_{}'.format(ckpt_root, args.plbl_type, round)
        else:
            self.plbl_root = '{}/plbl_gen/round_{}'.format(ckpt_root, round)
        assert(os.path.exists(self.plbl_root))

    def __getitem__(self, index):
        img_fname, lbl_fname, spx_fname = self.im_idx[index] ### warnning: index => superpixel-wise 로 정의됨

        ''' Load image, label, and superpixel '''
        image = Image.open(img_fname).convert('RGB')
        superpixel = self.open_spx(spx_fname)

        id = lbl_fname.split('/')[-1].split('.')[0]
        trg_index = self.id_to_index[id]
        target = self.multi_hot_cls[trg_index] ### [nseg x (num_classes + 1)]

        ''' Return image and superpixel for pooling dataset '''
        if self.split == 'active-ulabel':
            return self.__getpoolitem__(image, superpixel, target)

        img_id = img_fname.split('/')[-1].split('_leftImg8bit')[0]
        target_path = "{}/{}.png".format(self.plbl_root, img_id)
        target_plbl = Image.open(target_path)

        ''' Augment both images, superpixel map '''
        image, lbls = self.transform(image, [target_plbl, superpixel])
        target_plbl, superpixel = lbls

        ''' Get actively sampled superpixel ids '''
        preserving_labels = self.suppix[spx_fname] if spx_fname in self.suppix else []

        ''' Filter unselected superpixels & ignored regions '''
        sp_mask = torch.isin(superpixel, torch.tensor(preserving_labels))

        sample = {'images': image,
                  'labels': target_plbl,
                  'target': target,
                  'spx': superpixel,
                  'spmask': sp_mask,
                  'fnames': self.im_idx[index]}

        return sample