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

class RegionCityscapesOrAll(RegionCityscapes):

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
                 or_labeling=False,
                 ignore_size=0,
                 mark_topk=2):
        super().__init__(args, root, datalist, split, transform, return_spx, region_dict, mask_region, dominant_labeling)
        self.ignore_size = ignore_size
        self.mark_topk = mark_topk

        if self.ignore_size == 0:
            self.ignore_tiny = False
        else:
            self.ignore_tiny = True

        if self.mark_topk == -1:
            self.ignore_under_k = False
        else:
            self.ignore_under_k = True

        assert(not (self.ignore_tiny and self.ignore_under_k))

    def __getitem__(self, index):
        assert(self.mask_region)
        assert(self.return_spx)

        img_fname, lbl_fname, spx_fname = self.im_idx[index]

        '''Load image, label, and superpixel'''
        image = Image.open(img_fname).convert('RGB')
        target = Image.open(lbl_fname)
        superpixel = self.open_spx(spx_fname)
        image, lbls = self.transform(image, [target, superpixel])
        target, superpixel = lbls
        target = self.encode_target(target)

        h, w = target.shape
        target = target.reshape(-1)
        ignore_mask  = torch.from_numpy(target==255)
        superpixel = superpixel.reshape(-1)
        preserving_labels = self.suppix[spx_fname]

        ''' or label assignment '''
        target[ignore_mask] = 19 ### ignore label
        onehot_trg = torch.nn.functional.one_hot(torch.from_numpy(target), num_classes=20).byte() # one-hot target
        for p in preserving_labels:
            sp_mask = torch.logical_and((superpixel == p), torch.logical_not(ignore_mask))
            if self.ignore_tiny and torch.any(sp_mask).item():
                cls_idx, count = np.unique(target[sp_mask], return_counts=True)
                count_mask = (self.ignore_size < count)
                cls_filtered = cls_idx[count_mask]
                if cls_filtered.size == 0:
                    value = torch.nn.functional.one_hot(torch.tensor(cls_idx[count.argmax()]), num_classes=20).byte()
                else:
                    value = torch.any(torch.nn.functional.one_hot(torch.from_numpy(cls_filtered), num_classes=20).byte(), dim=0)
                onehot_trg[sp_mask] = value
            elif self.ignore_under_k and torch.any(sp_mask).item():
                cls_idx, count = np.unique(target[sp_mask], return_counts=True)
                topk_idx = np.argsort(-count)[:self.mark_topk]
                cls_filtered = cls_idx[topk_idx]
                if cls_filtered.size == 0:
                    value = torch.nn.functional.one_hot(torch.tensor(cls_idx[count.argmax()]), num_classes=20).byte()
                else:
                    value = torch.any(torch.nn.functional.one_hot(torch.from_numpy(cls_filtered), num_classes=20).byte(), dim=0)
                onehot_trg[sp_mask] = value
            else:
                onehot_trg[sp_mask] = torch.any(onehot_trg[sp_mask], dim=0)


        onehot_trg = onehot_trg.reshape(h, w, -1)
        superpixel = superpixel.reshape(h, w)

        sample = {'images': image, 'labels': onehot_trg, 'spx': superpixel, 'fnames': self.im_idx[index]}

        return sample