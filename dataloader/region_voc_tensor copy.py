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
from .region_voc import RegionVOC

class RegionVOCTensor(RegionVOC):

    def __init__(self, args, root, datalist, split='train', transform=None, region_dict="dataloader/init_data/voc/train.dict"):
        super().__init__(args, root, datalist, split, transform, False, region_dict, True, False)

    def __getitem__(self, index):
        img_fname, lbl_fname, spx_fname = self.im_idx[index]
        '''Load image, label, and superpixel'''
        image = Image.open(img_fname).convert('RGB')
        target = Image.open(lbl_fname)
        superpixel = self.open_spx(spx_fname)
        image, lbls = self.transform(image, [target, superpixel])
        target, superpixel = lbls
        target = self.encode_target(target)

        ''' superpixel tensor generation '''
        superpixel_cls = torch.zeros((self.args.nseg, self.args.num_classes + 1), dtype=torch.uint8)
        superpixel_size = torch.ones((self.args.nseg, ), dtype=torch.int) * -1

        '''GT masking (mimic region-based annotation)'''
        target = target.reshape(-1)
        superpixel = superpixel.reshape(-1)
        preserving_labels = self.suppix[spx_fname]

        ''' Multi-hot label assignment '''
        for p in preserving_labels:
            sp_mask = (superpixel == p)
            u, c = np.unique(target[sp_mask], return_counts=True) ### superpixel 내부에 class 구성 파악
            isignore = 255 in u
            if isignore and len(u) == 1:
                allignore = True
            else:
                allignore = False
            npx = sp_mask.sum()
            if not allignore:
                u_valid = u[u != 255]
                c_valid = c[u != 255]
                c_order = c_valid.argsort()[::-1]
                cls = u_valid[c_order].tolist()
                cpx = c_valid[c_order].tolist()
            else:
                cls = []
                cpx = []

            if isignore:
                cls.append(-1) ### last dimension of superpixel_cls is assigned to ignore label
            else:
                pass

            superpixel_cls[p, cls] = 1
            superpixel_size[p] = npx

        sample = {'superpixel_info': (superpixel_cls, superpixel_size), 'fname': self.im_idx[index]}

        return sample