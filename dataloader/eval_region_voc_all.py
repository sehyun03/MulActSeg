from PIL import Image
import numpy as np
import torch
import os

import imageio
imageio.plugins.freeimage.download()
from . import region_voc_or_tensor

class RegionVOCOr(region_voc_or_tensor.RegionVOCOr):

    def __init__(self, args, root, datalist, split='train', transform=None, return_spx=False,
                 region_dict="dataloader/init_data/voc/train_seed32.dict", mask_region=True, dominant_labeling=False, loading='binary', load_smaller_spx=False):
        super().__init__(args, root, datalist, split, transform, return_spx, region_dict, mask_region, dominant_labeling, loading, load_smaller_spx)
        assert(self.mask_region)
        assert(not self.load_smaller_spx)

        r''' remove_dominant for analysis and visualization
        - only include dominant when saving the pseudo labels
        '''
        if 'eval_save' in args.method:
            self.remove_dominant = False
        else:
            self.remove_dominant = True

    def __getitem__(self, index):
        img_fname, lbl_fname, spx_fname = self.im_idx[index] ### warnning: index => superpixel-wise 로 정의됨

        ''' Load image, label, and superpixel '''
        image = Image.open(img_fname).convert('RGB')
        superpixel = self.open_spx(spx_fname)

        id = lbl_fname.split('/')[-1].split('.')[0]
        # print(id)

        if self.dominant_labeling:
            target_path = os.path.join(self.root, 'superpixels/pascal_voc_seg/seeds_32/train/gtFine_dominant/{}.png'.format(id))
        else:
            target_path = os.path.join(self.root, 'VOC2012/SegmentationClass/{}.png'.format(id))
        # target_path = ''
        # target_path = '{}/gtFine/train/{}/{}_gtFine_labelIds.png'.format(self.root, id.split('_')[0], id)
        r''' 본래 255 였던 영역을 19 번째 클래스로 치환해준다 (undefined label 예측을 위함)
        - 해당 loader 에서 255 index 는 'unselected region' 용으로 활용된다. '''
        target_precise = Image.open(target_path)
        img_size = target_precise.size
        target_precise = torch.from_numpy(self.encode_target(target_precise).astype('uint8'))
        target_precise = torch.masked_fill(target_precise, target_precise == 255, 21) ### original label as 21th class
        target_precise = Image.fromarray(target_precise.numpy())

        ''' Resize both images, superpixel map '''
        image, lbls = self.transform(image, [target_precise, superpixel])
        target_precise, superpixel = lbls

        ''' Get actively sampled superpixel ids '''
        preserving_labels = torch.tensor(self.suppix[spx_fname] if spx_fname in self.suppix else [])
        
        trg_index = self.id_to_index[id]
        target = self.multi_hot_cls[trg_index] ### [nseg x (num_classes + 1)]

        r''' remove dominant label within preserving_labels '''
        if self.remove_dominant:
            preserving_spx_ncls = target[preserving_labels].sum(dim=1) ### [nselected]
            is_multi = torch.logical_not(preserving_spx_ncls == 1) ### [nselected]
            preserving_labels = preserving_labels[is_multi]

        ''' Filter unselected superpixels & ignored regions '''
        valid_preserving_labels = preserving_labels[target[preserving_labels].sum(dim=1) != 0]
        sp_mask = torch.isin(superpixel, valid_preserving_labels)
        # target_precise = torch.masked_fill(target_precise, torch.logical_not(sp_mask), 255)

        sample = {'images': image,
                  'labels': target_precise,
                  'target': target,
                  'spx': superpixel,
                  'spmask': sp_mask,
                  'imsizes': img_size,
                  'fnames': self.im_idx[index]}

        return sample