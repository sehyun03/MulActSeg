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
        self.args = args
        self.root = os.path.expanduser(root)
        if split not in ['train', 'test', 'val', 'active-label', 'active-ulabel', 'custom-set']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val" or split="active-label" or split="active-ulabel" or split="custom-set"')
        if transform is not None:
            self.transform = transform
        else:
            raise NotImplementedError

        json_dict = {}
        for nseg in args.nseg_list:
            region_dict = 'dataloader/init_data/cityscapes/train_seed{}.dict'.format(nseg)
            json_dict_ = self._load_json(region_dict) ### superpixel id (arange(img-wise max numseg))
            json_dict = {**json_dict, **json_dict_} ### spx_file -> list of spx ids

        self.split = split
        self.return_spx = return_spx
        self.mask_region = mask_region
        self.dominant_labeling = dominant_labeling

        self.im_idx = []
        self.suppix = defaultdict(list)
        if datalist is not None:
            self.get_merged_data_list(datalist, json_dict)
    
        self.myassert()

    def myassert(self):
        assert(not self.args.known_ignore), "do not support known ignore"
        assert(self.mask_region)

    def get_merged_data_list(self, datalist, json_dict):
        r"""
        Args::
            datalist (string): path to current nseg path list file (txt)
        Returns::
        Content::
            Updates self.im_idx, self.suppix
                self.im_idx (list): each element is tuple consist of
                                        img_path (string)
                                        nseg (int) -> (lbl_path (string), spx_path (string))
                    ex) [(image_path, {128: (lbl_path_128, spx_path_128), 512, 1024:}), ...] 
                self.suppix (dict): spx_path (string) -> spx_id (int list)
        """
        args = self.args
        current_nseg = args.nseg
        nseg_path_list = {}

        ### get datalist for nseg_list
        for nseg in args.nseg_list:
            datalist = datalist.replace('{}'.format(current_nseg), '{}'.format(nseg))
            with open(datalist, 'r') as f:
                path_list = f.read().splitlines()

            for i in range(len(path_list)):
                path_list[i] = path_list[i].replace('gtFine_dominant', 'gtFine_dominant_ignore')

            nseg_path_list[nseg] = path_list
            current_nseg = nseg

        for idx in range(len(nseg_path_list[args.nseg_list[0]])):
            current_img = None
            lbl_dict = {}
            for nseg in args.nseg_list:
                img_fname, lbl_fname, spx_fname = nseg_path_list[nseg][idx].split('\t')
                img_fullname = os.path.join(self.root, img_fname)
                lbl_fullname = os.path.join(self.root, lbl_fname)
                spx_fullname = os.path.join(self.root, spx_fname)

                lbl_dict[nseg] = (lbl_fullname, spx_fullname)
                if current_img is not None: assert(current_img == img_fullname)
                current_img = img_fullname
                self.suppix[spx_fullname] = json_dict[spx_fname] # list of superpixel id

            self.im_idx.append((current_img, lbl_dict))

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
            superpixel = np.load(spx_fname, allow_pickle=True)['labels'].astype('int32')
        return superpixel

    def __getitem__(self, index):
        r"""
        self.im_idx (list): each element is tuple consist of
                                img_path (string)
                                nseg (int) -> (lbl_path (string), spx_path (string))
            ex) [(image_path, {128: (lbl_path_128, spx_path_128), 512, 1024:}), ...] 
        self.suppix (dict): spx_path (string) -> spx_id (int list)
        """
        img_fname, lbl_spx_dict = self.im_idx[index]
        image = Image.open(img_fname).convert('RGB')
        target_all = []
        superpixel_all = []
        for nseg, lbl_spx in sorted(lbl_spx_dict.items(), key=lambda key:int(key[0])):
            target_all.append(np.array(Image.open(lbl_spx[0])))
            superpixel_all.append(self.open_spx(lbl_spx[1]))
        target_all = torch.from_numpy(np.stack(target_all))
        superpixel_all = torch.from_numpy(np.stack(superpixel_all))
        image, lbls = self.transform(image, [target_all, superpixel_all])
        target_all, superpixel_all = lbls
        target_all = target_all if self.dominant_labeling else self.encode_target(target_all)

        ''' GT masking (mimic region-based annotation) '''
        ### target: nseg x H x W
        ### superpixel: nseg x H x W
        spmasks = []
        for iter, (nseg, lbl_spx) in enumerate(sorted(lbl_spx_dict.items(), key=lambda key:int(key[0]))):
            _, spx_fname = lbl_spx
            preserving_labels = self.suppix[spx_fname] if spx_fname in self.suppix else []
            mask = torch.isin(superpixel_all[iter], torch.Tensor(preserving_labels))
            if iter == 0:
                target_all[0] = torch.where(mask, target_all[0], 255)
            else:
                target_all[0] = torch.where(mask, target_all[iter], target_all[0])
            spmasks.append(mask)

        curr_nseg_list = torch.zeros((len(self.args.nseg_list),)).bool() ### current nseg_list indicator
        for idx, nseg in enumerate(self.args.nseg_list):
            curr_nseg_list[idx] = nseg in sorted(list(lbl_spx_dict.keys()))

        if self.return_spx is False:
            sample = {'images': image, 'labels': target_all[0], 'fnames': self.im_idx[index][0]}
        else:
            sample = {'images': image, 'labels': target_all[0], 'fnames': self.im_idx[index][0],
                      'mseg_spx': superpixel_all, 'mseg_spmask': torch.stack(spmasks),
                      'nseg_list': curr_nseg_list}
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

'''
key: [1,2,3,4,5,6,7, ...]
key: [32,[]]

'''