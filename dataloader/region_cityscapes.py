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

class RegionCityscapes(data.Dataset):

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

        if not hasattr(args, "prob_dominant"):
            args.prob_dominant = False
                    
        self.args = args
        self.root = os.path.expanduser(root)
        if split not in ['train', 'test', 'val', 'active-label', 'active-ulabel', 'custom-set']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val" or split="active-label" or split="active-ulabel" or split="custom-set"')
        if transform is not None:
            self.transform = transform
        else:
            raise NotImplementedError

        json_dict = self._load_json(region_dict) ### superpixel id (arange(img-wise max numseg))

        self.split = split
        self.return_spx = return_spx
        self.mask_region = mask_region
        self.dominant_labeling = dominant_labeling
        self.get_data_list(datalist, json_dict)

    def get_data_list(self, datalist, json_dict):
        # im_idx contains the list of each image paths
        self.im_idx = []
        self.suppix = {}
        if datalist is not None:
            with open(datalist, 'r') as f:
                valid_list = f.read().splitlines()
            ### known ignore
            if self.args.known_ignore:
                pass
            else:
                for i in range(len(valid_list)):
                    valid_list[i] = valid_list[i].replace('gtFine_dominant', 'gtFine_dominant_ignore')

            ### dominant sampling
            if self.args.prob_dominant:
                for i in range(len(valid_list)):
                    valid_list[i] = valid_list[i].replace('gtFine_dominant', 'gtFine_dominant_ignore_sample')
            else:
                pass

            valid_list  = [i.split('\t') for i in valid_list]
            for index, (img_fname, lbl_fname, spx_fname) in enumerate(valid_list):
                img_fullname = os.path.join(self.root, img_fname)
                lbl_fullname = os.path.join(self.root, lbl_fname)
                spx_fullname = os.path.join(self.root, spx_fname)
                self.im_idx.append([img_fullname, lbl_fullname, spx_fullname]) ### list of list of three paths
                self.suppix[spx_fullname] = json_dict[spx_fname] ### list of superpixel id

    @classmethod
    def encode_target(cls, target):
        ''' apply ignore label to the mask '''
        return id_to_train_id[np.array(target)] ### index: id, value: train_id

    @classmethod
    def decode_target(cls, target):
        if isinstance(target, torch.Tensor):
            target_ = target.clone()
        else:
            target_ = target.copy()
        target_[target == 255] = 19
        
        return train_id_to_color[target_]

    @classmethod
    def open_spx(self, spx_fname):
        ''' open both png and pkl '''
        ext = spx_fname.split('.')[-1]
        if ext in ['png','jpg']:
            superpixel = Image.open(spx_fname)
        elif ext == 'pkl':
            superpixel = Image.fromarray(np.load(spx_fname, allow_pickle=True)['labels']).convert('I')
        return superpixel

    def __getitem__(self, index):
        img_fname, lbl_fname, spx_fname = self.im_idx[index]
        '''Load image, label, and superpixel'''
        image = Image.open(img_fname).convert('RGB')
        target = Image.open(lbl_fname)
        superpixel = self.open_spx(spx_fname)

        image, lbls = self.transform(image, [target, superpixel])
        target, superpixel = lbls
        target = target if self.dominant_labeling else self.encode_target(target)

        ''' GT masking (mimic region-based annotation) '''
        if self.mask_region is True:
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

        if self.return_spx is False:
            sample = {'images': image, 'labels': target, 'fnames': self.im_idx[index]}
        else:
            sample = {'images': image, 'labels': target, 'spx': superpixel, 'fnames': self.im_idx[index]}
        return sample

    def __len__(self):
        return len(self.im_idx)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        
        if isinstance(list(data.items())[0][1], int):
            assert(False), "bug: suppix_id is not continuous"

        if isinstance(list(data.items())[0][1][1], list):
            edited_data = {}
            for key, (size, nonidxs) in data.items():
                edited_list = [i for i in range(size) if i not in nonidxs]
                edited_data[key] = edited_list
            return edited_data
        elif isinstance(list(data.items())[0][1][1], int):
            return data
        else:
            raise NotImplementedError