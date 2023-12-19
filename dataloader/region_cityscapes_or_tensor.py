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
        
        ### load precomputed multi-hot annotation
        # 'gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png'
        if args.trim_multihot_boundary:
            multi_tensor_name = "gtFine_multi_tensor_trim_{}x{}".format(args.trim_kernel_size, args.trim_kernel_size)
        else:
            multi_tensor_name = "gtFine_multi_tensor"
        multi_hot_cls_path = '{}/superpixel_seed/cityscapes/{}_{}/train/{}/multi_hot_cls.npy'.format(self.root, args.spx_method, args.nseg, multi_tensor_name)
        sp_size_path = '{}/superpixel_seed/cityscapes/{}_{}/train/{}/sp_size.npy'.format(self.root, args.spx_method, args.nseg, multi_tensor_name)

        self.multi_hot_cls = torch.from_numpy(np.load(multi_hot_cls_path)) ### (N, Nseg, Ncls)
        self.isselected = torch.zeros(self.multi_hot_cls.shape[:-1]) ### (N, Nseg)

        ''' generate instance id (label name) to index dict to index self.multi_hot_cls'''
        self.id_to_index = {}
        with open(self.args.trg_datalist, 'r') as f:
            datalist = f.read().splitlines()
        lbl_fname_list = [data.split('\t')[1] for data in datalist]
        for index, lbl_fname in enumerate(lbl_fname_list):
            id = lbl_fname.split('/')[-1].split('.')[0]
            self.id_to_index[id] = index

    def __getpoolitem__(self, image, superpixel, target):
        image, lbls = self.transform(image, [superpixel])
        superpixel = lbls[0]
        sample = {'images': image, 'spx': superpixel, 'labels': target}

        return sample

    def __getitem__(self, index):
        assert(self.mask_region)
        img_fname, lbl_fname, spx_fname = self.im_idx[index] ### warnning: index => superpixel-wise 로 정의됨
        ''' Load image, label, and superpixel '''
        image = Image.open(img_fname).convert('RGB')
        superpixel = self.open_spx(spx_fname)

        id = lbl_fname.split('/')[-1].split('.')[0]
        trg_index = self.id_to_index[id]
        target = self.multi_hot_cls[trg_index] ### [nseg x (num_classes + 1)]

        ''' Get actively sampled superpixel ids '''
        if spx_fname in self.suppix:
            preserving_labels = self.suppix[spx_fname]
        else:
            preserving_labels = []

        ''' Return image and superpixel for pooling dataset '''
        if self.split == 'active-ulabel':
            return self.__getpoolitem__(image, superpixel, target)

        ''' Augment both images, superpixel map '''
        if self.load_smaller_spx:
            assert(self.args.spx_method == 'seeds')
            superpixel_small = self.open_spx(spx_fname.replace("seeds_{}".format(self.args.nseg),"seeds_{}".format(self.args.small_nseg)))
            image, lbls = self.transform(image, [superpixel, superpixel_small])
            superpixel, superpixel_small = lbls
        else:
            image, lbls = self.transform(image, [superpixel])
            superpixel = lbls[0]

        h, w = superpixel.shape

        ''' Filter unselected superpixels '''
        sp_mask = torch.from_numpy(np.isin(superpixel.reshape(-1), preserving_labels))
        sp_mask = sp_mask.reshape(h, w) ### boolean mask indicating selected superpixels

        if self.load_smaller_spx:
            sample = {'images': image, 'labels': target, 'spx': superpixel, 'spmask': sp_mask, 'spx_small': superpixel_small, 'fnames': self.im_idx[index]}
        else:
            sample = {'images': image, 'labels': target, 'spx': superpixel, 'spmask': sp_mask, 'fnames': self.im_idx[index]}

        return sample