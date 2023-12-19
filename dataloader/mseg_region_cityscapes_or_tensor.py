import json
import os
import dataloader.ext_transforms as et
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
from time import time
from collections import defaultdict

from .constant import train_id_to_color, id_to_train_id
from . import mseg_region_cityscapes
# for synthia
import imageio
imageio.plugins.freeimage.download()

class RegionCityscapesOr(mseg_region_cityscapes.RegionCityscapes):

    def __init__(self,
                 args,
                 root,
                 datalist,
                 split='train',
                 transform=None,
                 return_spx=False,
                 region_dict="dataloader/init_data/cityscapes/train.dict",
                 mask_region=True,
                 dominant_labeling=False,
                 loading='binary',
                 load_smaller_spx=False):
        super().__init__(args, root, datalist, split, transform, return_spx, region_dict, mask_region, dominant_labeling)
        self.loading = loading
        self.load_smaller_spx = load_smaller_spx

        '''Load precomputed multi-hot annotation for multiple segmention'''
        self.mseg_mh_cls = {} ### eg) {128: np.array (N x 128 x C+1), 512: np.array(N x 512 x C+1), ...}
        self.get_nseg_mh_cls(args)

        ''' generate instance id (label name) -> index dict (to index self.multi_hot_cls)'''
        self.id_to_index = {} ### eg) {'lbl':0, 'lbl':1, 'lbl':2, ...}
        self.get_id_ti_index(args)

    def get_nseg_mh_cls(self, args):
        for iter, nseg in enumerate(args.nseg_list):
            multi_hot_cls_path = '{}/superpixel_seed/cityscapes/seeds_{}/train/gtFine_multi_tensor/multi_hot_cls.npy'.format(self.root, nseg)
            self.mseg_mh_cls[nseg] = np.load(multi_hot_cls_path) ### N x nseg' x C+1

    def get_id_ti_index(self, args):
        with open(self.args.trg_datalist, 'r') as f:
            datalist = f.read().splitlines()
        lbl_fname_list = [data.split('\t')[1] for data in datalist]
        for index, lbl_fname in enumerate(lbl_fname_list):
            id = lbl_fname.split('/')[-1].split('.')[0]
            self.id_to_index[id] = index

    def __getitem__(self, index):
        r"""
        self.im_idx (list): each element is tuple consist of
                                img_path (string)
                                nseg (int) -> (lbl_path (string), spx_path (string))
            ex) [(image_path, {128: (lbl_path_128, spx_path_128), 512, 1024:}), ...] 
        self.suppix (dict): spx_path (string) -> spx_id (int list)
        """

        ''' Load img, superpixel and augment them '''
        img_fname, lbl_spx_dict = self.im_idx[index]
        image = Image.open(img_fname).convert('RGB')
        superpixel_all = []
        lbl_path = None
        for nseg, lbl_spx in sorted(lbl_spx_dict.items(), key=lambda key:int(key[0])):
            lbl_path, spx_path = lbl_spx
            superpixel_all.append(self.open_spx(spx_path))
        superpixel_all = torch.from_numpy(np.stack(superpixel_all))
        image, lbls = self.transform(image, [superpixel_all])
        superpixel_all = lbls[0]

        ''' Index target '''
        id = lbl_path.split('/')[-1].split('.')[0]
        trg_index = self.id_to_index[id]
        target = [torch.from_numpy(self.mseg_mh_cls[i][trg_index]) for i in sorted(list(lbl_spx_dict.keys()))]
        ### ㄴ [128 x C+1, 512 x C+1, 2048 x C+1, 8192 x C+1]

        ''' GT masking (mimic region-based annotation) '''
        ### self.mseg_mh_cls: {128: np.array (N x 128 x C+1), 512: np.array(N x 512 x C+1), ...}
        ### self.id_to_index: {'lbl':0, 'lbl':1, 'lbl':2, ...}
        spmasks = []
        for iter, (nseg, lbl_spx) in enumerate(sorted(lbl_spx_dict.items(), key=lambda key:int(key[0]))):
            _, spx_fname = lbl_spx
            preserving_labels = self.suppix[spx_fname] if spx_fname in self.suppix else []
            mask = torch.isin(superpixel_all[iter], torch.Tensor(preserving_labels))
            spmasks.append(mask)

        ''' Get nseg list for current image '''
        curr_nseg_list = torch.zeros((len(self.args.nseg_list),)).bool() ### current nseg_list indicator
        for idx, nseg in enumerate(self.args.nseg_list):
            curr_nseg_list[idx] = nseg in sorted(list(lbl_spx_dict.keys()))

        sample = {'images': image, 'fnames': self.im_idx[index][0],
                  'mseg_labels': target, 'mseg_spx': superpixel_all, 'mseg_spmask': torch.stack(spmasks),
                  'nseg_list': curr_nseg_list, }

        return sample