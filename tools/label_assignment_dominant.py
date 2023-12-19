import os
import sys
import torch
import random
import logging
import argparse
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import functional as F
import importlib

sys.path.append(os.path.abspath('.'))
import dataloader.ext_transforms as et
from dataloader.utils import DataProvider
# from dataloader.region_cityscapes_dominant_all import RegionCityscapesDominantAll
# leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png	gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png	superpixel_seed/cityscapes/seeds_2048/train/label/aachen_000000_000019.pkl

def get_parser():
    # Training configurations
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--trg_data_dir', default='./data/Cityscapes')
    parser.add_argument('--nseg', type=int, default=2048, help='# superpixel component for slic')
    parser.add_argument('--generate_ignore', action='store_true', default=False)
    parser.add_argument('--num_worker', type=int, default=8, help='number of classes in dataset')
    parser.add_argument('--loader', type=str, default='region_cityscapes_dominant_all', help='Multi-hot labeling loader seleciton (dataloader/*)')
    parser.add_argument('--nvis_color', type=int, default=3000)
    parser.add_argument('--spx_method', default='seed')

    return parser

def get_lbl_fname(data_dir, img_fname):
    fname = img_fname[0][0]
    data_id = '_'.join(fname.split('/')[-1].split('_')[:3])
    os.makedirs(name=data_dir, exist_ok=True)

    lbl_fname = '{}/{}.png'.format(data_dir, data_id)

    return lbl_fname

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args.trg_datalist = 'dataloader/init_data/cityscapes/train_{}{}.txt'.format(args.spx_method, args.nseg)
    args.region_dict = 'dataloader/init_data/cityscapes/train_{}{}.dict'.format(args.spx_method, args.nseg)
    args.do_data_dir = './data/Cityscapes/superpixel_seed/cityscapes/{}_{}/train/gtFine_dominant'.format(args.spx_method, args.nseg)
    if args.generate_ignore:
        args.do_data_dir = '{}_ignore'.format(args.do_data_dir)
        args.known_ignore = False
    else:
        args.known_ignore = True

    if 'sample' in args.loader:
        args.do_data_dir = '{}_sample'.format(args.do_data_dir)

    print(args)

    identity_transform = et.ExtCompose([
        et.ExtToTensor(dtype_list = ['int', 'int'])
    ])

    ### load superpixel & max-frequent pooled target
    region_cityscapes = importlib.import_module("dataloader.{}".format(args.loader.lower()))
    region_dataset = region_cityscapes.RegionCityscapesDominantAll(args, args.trg_data_dir, args.trg_datalist, split='active-ulabel', transform=identity_transform,
                                      region_dict=args.region_dict, return_spx=True, dominant_labeling=True, generate_ignore=args.generate_ignore)

    region_loader = DataProvider(dataset=region_dataset, batch_size=1,
                                shuffle=False, num_workers=args.num_worker, pin_memory=True, drop_last=False)
    
    ''' save dominant labeled target '''
    N = region_loader.__len__()
    for iter in tqdm(range(N)):
        data = region_loader.__next__()

        imgs = data['images']
        dominant_label = data['labels'] ### encoded dominant label
        superpixel = data['spx']
        img_fname = data['fnames'] ### this is 
        lbl_fname = get_lbl_fname(args.do_data_dir, img_fname)

        ### saving
        pil_label = F.to_pil_image(pic = dominant_label.int())
        pil_label.save(lbl_fname)

        if iter < args.nvis_color:
            colored_dominant_label = region_dataset.decode_target(dominant_label)
            color_lbl_fname = lbl_fname.replace("gtFine", "gtColor")
            color_lbl_dir = '/'.join(color_lbl_fname.split('/')[:-1])
            os.makedirs(color_lbl_dir, exist_ok=True)
            pil_color_label = Image.fromarray(colored_dominant_label[0].astype('uint8'), mode='RGB')
            pil_color_label.save(color_lbl_fname)