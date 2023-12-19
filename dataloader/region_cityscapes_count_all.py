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
from .region_cityscapes import RegionCityscapes

class RegionCityscapesCountAll(RegionCityscapes):

    def __init__(self, root, datalist, num_seg, split='train', transform=None, return_spx=False,
                 region_dict="dataloader/init_data/cityscapes/train.dict", mask_region=True, dominant_labeling=False,
                 or_labeling=False):
        super().__init__(root, datalist, split, transform, return_spx, region_dict, mask_region, dominant_labeling)
        self.num_seg = num_seg

    def __getitem__(self, index):
        img_fname, lbl_fname, spx_fname = self.im_idx[index]
        '''Load image, label, and superpixel'''
        target = Image.open(lbl_fname)
        superpixel = self.open_spx(spx_fname)
        target = self.encode_target(target)
        num_class_bin = np.zeros((self.num_seg,))


        '''GT masking (mimic region-based annotation)'''
        h, w = target.shape
        target = target.reshape(-1)
        ignore_mask  = (target==255)
        superpixel = np.array(superpixel).reshape(-1)
        preserving_labels = self.suppix[spx_fname]

        ''' assign superpixel sizes '''
        sup_size_bin = np.unique(superpixel, return_counts=True)[1]

        ''' assign num class '''
        for sdx, p in enumerate(preserving_labels):
            sp_mask = np.logical_and((superpixel == p), np.logical_not(ignore_mask))
            u, c = np.unique(target[sp_mask], return_counts=True)
            if c.size != 0:
                num_class_bin[sdx] = u.size
            else:
                num_class_bin[sdx] = 0
    
        sample = {'sup_size_bin': sup_size_bin, 'num_class_bin': num_class_bin, 'fnames': self.im_idx[index]}

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