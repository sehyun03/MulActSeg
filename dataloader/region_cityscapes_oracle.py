import json
import os, sys
from xml.dom.minidom import Document
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
from  dataloader import region_cityscapes
"""
- Oracle dominant region dataloader
"""

class RegionCityscapes(region_cityscapes.RegionCityscapes):

    def __init__(self, args, root, datalist, split='train', transform=None, return_spx=False,
                 region_dict="dataloader/init_data/cityscapes/train.dict", mask_region=True, dominant_labeling=False, loading='binary', load_smaller_spx=False):
        datalist = args.trg_datalist ### provide all target data list == provide full supervision
        region_dict = args.region_dict

        super().__init__(args, root, datalist, split, transform, return_spx, region_dict, mask_region, dominant_labeling)