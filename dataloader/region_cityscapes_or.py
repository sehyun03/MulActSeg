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
from .region_cityscapes import RegionCityscapes

class RegionCityscapesOr(RegionCityscapes):

    def __init__(self, args, root, datalist, split='train', transform=None, return_spx=False,
                 region_dict="dataloader/init_data/cityscapes/train.dict", mask_region=True, dominant_labeling=False, loading='binary', load_smaller_spx=False):
        super().__init__(args, root, datalist, split, transform, return_spx, region_dict, mask_region, dominant_labeling)
        self.loading = loading
        self.load_smaller_spx = load_smaller_spx

        assert(not ((self.args.ignore_size != 0) and (self.args.mark_topk != -1)))

        if self.loading == 'tensor':
            self.lbl_id = [i[0].split('/')[-1].split('_leftImg8bit')[0] for i in self.im_idx]

    def get_data_list(self, datalist, json_dict):
        # im_idx contains the list of each image paths
        self.im_idx = []
        self.suppix = {}
        if datalist is not None:
            with open(datalist, 'r') as f:
                valid_list = f.read().splitlines()
            
            if self.args.ignore_size != 0:
                for i in range(len(valid_list)):
                    valid_list[i] = valid_list[i].replace('gtFine_or', 'gtFine_or_ignore_{}'.format(self.args.ignore_size))
            elif self.args.mark_topk != -1:
                for i in range(len(valid_list)):
                    valid_list[i] = valid_list[i].replace('gtFine_or', 'gtFine_or_mark_topk_{}'.format(self.args.mark_topk))
            else:
                pass

            valid_list  = [i.split('\t') for i in valid_list]

            for img_fname, lbl_fname, spx_fname in valid_list:
                img_fullname = os.path.join(self.root, img_fname)
                lbl_fullname = os.path.join(self.root, lbl_fname)
                spx_fullname = os.path.join(self.root, spx_fname)
                self.im_idx.append([img_fullname, lbl_fullname, spx_fullname]) ### list of list of three paths
                self.suppix[spx_fullname] = json_dict[spx_fname] ### list of superpixel id        

    def __getitem__(self, index):
        assert(self.mask_region)
        img_fname, lbl_fname, spx_fname = self.im_idx[index]

        '''Load image, label, and superpixel'''
        image = Image.open(img_fname).convert('RGB')
        w_orig, h_orig = image.size
        superpixel = self.open_spx(spx_fname)

        if spx_fname in self.suppix:
            preserving_labels = self.suppix[spx_fname]
        else:
            preserving_labels = []

        if self.loading == 'binary':
            ''' byte tyoe loading '''
            target = np.load(lbl_fname)[..., None]
            target = np.unpackbits(target, axis=3)[..., :5].reshape(h_orig, w_orig, 20) ### bit-wise encoded, one-hot label, H x W x 4
            target = torch.from_numpy(target)
        elif self.loading == 'dictionary':
            pass
        else:
            ''' uint loading '''
            target = torch.from_numpy(np.load(lbl_fname.replace('gtFine_or', 'gtFine_or_orig')))

        if self.load_smaller_spx:
            superpixel_small = self.open_spx(spx_fname.replace("seeds_{}".format(self.args.nseg),"seeds_{}".format(self.args.small_nseg)))
            image, lbls = self.transform(image, [target.permute(2,0,1), superpixel, superpixel_small])
            target, superpixel, superpixel_small = lbls
        else:
            image, lbls = self.transform(image, [target.permute(2,0,1), superpixel])
            target, superpixel = lbls

        h, w = superpixel.shape
        target = target.permute(1,2,0).reshape(-1, 20) ### flatten for masking
        superpixel = superpixel.reshape(-1)
        sp_mask = torch.from_numpy(np.isin(superpixel, preserving_labels))
        target[torch.logical_not(sp_mask), :-1] = 0 ### unselected superpixel -> erase label
        target[torch.logical_not(sp_mask), -1] = 1 ###  unselected superpixel -> mark ignore label

        target = target.reshape(h, w, 20).permute(2,0,1)
        superpixel = superpixel.reshape(h, w)
        
        if self.load_smaller_spx:
            sample = {'images': image, 'labels': target, 'spx': superpixel, 'spx_small': superpixel_small, 'fnames': self.im_idx[index]}
        else:
            sample = {'images': image, 'labels': target, 'spx': superpixel, 'fnames': self.im_idx[index]}

        return sample