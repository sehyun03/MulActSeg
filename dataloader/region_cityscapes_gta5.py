import json
import os
import dataloader.ext_transforms as et
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch

from .constant import train_id_to_color, id_to_train_id
# for synthia
import imageio
imageio.plugins.freeimage.download()


class RegionCityscapesGTA5(data.Dataset):
    train_transform = et.ExtCompose([
        et.ExtRandomScale((0.5, 1.5)),
        et.ExtResize((1024, 2048)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        et.ExtResize((1024, 2048)),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, root, datalist, split='train', transform=None, return_spx=False,
                 region_dict="dataloader/init_data/cityscapes/train.dict", mask_region=True):
        self.root = os.path.expanduser(root)
        if split not in ['train', 'test', 'val', 'active-label', 'active-ulabel', 'custom-set']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val" or split="active-label" or split="active-ulabel" or split="custom-set"')
        if transform is not None:
            self.transform = transform
        else:  # Use default transform
            if split in ["train", "active-label"]:
                self.transform = self.train_transform
            elif split in ["val", "test", "active-ulabel", "custom-set"]:
                self.transform = self.val_transform

        json_dict = self._load_json(region_dict)

        self.split = split
        self.return_spx = return_spx
        self.mask_region = mask_region
        # im_idx contains the list of each image paths
        self.im_idx = []
        self.suppix = {}
        if datalist is not None:
            valid_list = np.loadtxt(datalist, dtype='str')
            for img_fname, lbl_fname, spx_fname in valid_list:
                img_fullname = os.path.join(self.root, img_fname)
                lbl_fullname = os.path.join(self.root, lbl_fname)
                spx_fullname = os.path.join(self.root, spx_fname)
                self.im_idx.append([img_fullname, lbl_fullname, spx_fullname])
                self.suppix[spx_fullname] = json_dict[spx_fname]

    @classmethod
    def encode_target(cls, target):
        return id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        return train_id_to_color[target]

    def __getitem__(self, index):
        img_fname, lbl_fname, spx_fname = self.im_idx[index]
        # Load image, label, and superpixel
        image = Image.open(img_fname).convert('RGB')
        target = Image.open(lbl_fname)
        superpixel = Image.open(spx_fname)
        image, lbls = self.transform(image, [target, superpixel])
        target, superpixel = lbls
        target = self.encode_target(target)
        # GT masking (mimic region-based annotation)
        if self.mask_region is True:
            h, w = target.shape
            target = target.reshape(-1)
            superpixel = superpixel.reshape(-1)
            if spx_fname in self.suppix:
                preserving_labels = self.suppix[spx_fname]
            else:
                preserving_labels = []
            mask = np.isin(superpixel, preserving_labels)
            target = np.where(mask, target, 255)
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
        return data

class RegionCityscapes(data.Dataset):

    def __init__(self, root, datalist, split='train', transform=None, return_spx=False,
                 region_dict="dataloader/init_data/cityscapes/train.dict", mask_region=True, dominant_labeling=False,
                 or_labeling=False):
        self.root = os.path.expanduser(root)
        if split not in ['train', 'test', 'val', 'active-label', 'active-ulabel', 'custom-set']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val" or split="active-label" or split="active-ulabel" or split="custom-set"')
        if transform is not None:
            self.transform = transform
        else:
            raise NotImplementedError

        json_dict = self._load_json(region_dict) ### superpixel id (arange(num_superpixel))

        self.split = split
        self.return_spx = return_spx
        self.mask_region = mask_region
        self.dominant_labeling = dominant_labeling
        self.or_labeling = or_labeling

        # im_idx contains the list of each image paths
        self.im_idx = []
        self.suppix = {}
        if datalist is not None:
            valid_list = np.loadtxt(datalist, dtype='str')
            for img_fname, lbl_fname, spx_fname in valid_list:
                img_fullname = os.path.join(self.root, img_fname)
                lbl_fullname = os.path.join(self.root, lbl_fname)
                spx_fullname = os.path.join(self.root, spx_fname)
                self.im_idx.append([img_fullname, lbl_fullname, spx_fullname]) ### list of 3 path
                self.suppix[spx_fullname] = json_dict[spx_fname] ### list of superpixel id

    @classmethod
    def encode_target(cls, target):
        ''' apply ignore label to the mask '''
        return id_to_train_id[np.array(target)] ### index: id, value: train_id

    @classmethod
    def decode_target(cls, target):
        target_ = target.clone()
        target_[target == 255] = 19
        return train_id_to_color[target_]

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
        return data

class RegionCityscapesDominantAll(RegionCityscapes):

    def __init__(self, root, datalist, split='train', transform=None, return_spx=False,
                 region_dict="dataloader/init_data/cityscapes/train.dict", mask_region=True, dominant_labeling=False,
                 or_labeling=False):
        super().__init__(root, datalist, split, transform, return_spx, region_dict, mask_region, dominant_labeling)

    def __getitem__(self, index):
        img_fname, lbl_fname, spx_fname = self.im_idx[index]
        '''Load image, label, and superpixel'''
        image = Image.open(img_fname).convert('RGB')
        target = Image.open(lbl_fname)
        superpixel = self.open_spx(spx_fname)
        image, lbls = self.transform(image, [target, superpixel])
        target, superpixel = lbls
        target = self.encode_target(target)

        '''GT masking (mimic region-based annotation)'''
        if self.mask_region is True:
            h, w = target.shape
            target = target.reshape(-1)
            ignore_mask  = torch.from_numpy(target==255)
            superpixel = superpixel.reshape(-1)
            preserving_labels = self.suppix[spx_fname]

            ''' dominant label assignment '''
            for p in preserving_labels:
                sp_mask = torch.logical_and((superpixel == p), torch.logical_not(ignore_mask))
                u, c = np.unique(target[sp_mask], return_counts=True)
                if c.size != 0:
                    target[sp_mask] = u[c.argmax()]
            target[ignore_mask] = 255
            target = target.reshape(h, w)
            superpixel = superpixel.reshape(h, w)
    
        if self.return_spx is False:
            sample = {'images': image, 'labels': target, 'fnames': self.im_idx[index]}
        else:
            sample = {'images': image, 'labels': target, 'spx': superpixel, 'fnames': self.im_idx[index]}
        return sample

# class RegionCityscapesDominantAll(RegionCityscapes):

#     def __getitem__(self, index):
#         img_fname, lbl_fname, spx_fname = self.im_idx[index]
#         '''Load image, label, and superpixel'''
#         image = Image.open(img_fname).convert('RGB')
#         target = Image.open(lbl_fname)
#         superpixel = self.open_spx(spx_fname)
#         image, lbls = self.transform(image, [target, superpixel])
#         target, superpixel = lbls
#         target = self.encode_target(target)

#         '''GT masking (mimic region-based annotation)'''
#         if self.mask_region is True:
#             h, w = target.shape
#             target = target.reshape(-1)
#             target = torch.from_numpy(target)
#             superpixel = superpixel.reshape(-1)

#             ignore_mask  = (target==255)
#             target[ignore_mask] = 20 ### to make it to one-hot
#             onehot_sp = torch.nn.functional.one_hot(superpixel, num_classes=2048).int()
#             onehot_trg = torch.nn.functional.one_hot(target, num_classes=21).int()
#             sp_trg_count = torch.einsum('ni,nj->ij', onehot_trg, onehot_sp)
#             sp_do_lbl = sp_trg_count[:20].argmax(dim=0).byte()
#             target_do = torch.sum((onehot_sp.byte() * sp_do_lbl[None,:]), dtype=torch.uint8, dim=-1)
#             # target_do_final = torch.where(torch.logical_not(ignore_mask), target_do.view(-1), 255) ### filter selected superpixel
#             target_do[ignore_mask] = 255
#             target = target_do.reshape(h, w).int()
#             superpixel = superpixel.reshape(h, w)
#         if self.return_spx is False:
#             sample = {'images': image, 'labels': target, 'fnames': self.im_idx[index]}
#         else:
#             sample = {'images': image, 'labels': target, 'spx': superpixel, 'fnames': self.im_idx[index]}
#         return sample