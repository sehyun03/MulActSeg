import os
import sys
sys.path.append(os.path.abspath('.'))
import argparse
import numpy as np
from tqdm import tqdm

import dataloader.ext_transforms as et
from dataloader.region_cityscapes_tensor import RegionCityscapesTensor
from dataloader.utils import DataProvider

def get_parser():
    # Training configurations
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--nseg', type=int, default=2048, help='# superpixel component for slic')
    parser.add_argument('--save_data_dir', help='superpixel directory root')
    parser.add_argument('--num_worker', type=int, default=8, help='number of classes in dataset')
    parser.add_argument('--ignore_size', type=int, default=0, help='(or_lbeling) ignore class region smaller than this')
    parser.add_argument('--mark_topk', type=int, default=-1, help='(or_lbeling) ignore classes with the region size under than kth order')
    parser.add_argument('--num_classes', type=int, default=19, help='number of classes in dataset')
    parser.add_argument('--trim_kernel_size', type=int, default=3)
    parser.add_argument('--trim_multihot_boundary', action='store_true', default=False)
    parser.add_argument('--prob_dominant', action='store_true', default=False)
    parser.add_argument('--spx_method', default='seed')

    parser.add_argument('--trg_data_dir', default='./data/Cityscapes')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args.trg_datalist = 'dataloader/init_data/cityscapes/train_{}{}.txt'.format(args.spx_method, args.nseg)
    args.region_dict = 'dataloader/init_data/cityscapes/train_{}{}.dict'.format(args.spx_method, args.nseg)
    args.known_ignore = False
    print(args)

    identity_transform = et.ExtCompose([et.ExtToTensor(dtype_list=['int','int'])])

    ### load superpixel & max-frequent pooled target
    region_dataset = RegionCityscapesTensor(args,
                                         args.trg_data_dir,
                                         args.trg_datalist,
                                         split='active-ulabel',
                                         transform=identity_transform,
                                         region_dict=args.region_dict)

    region_loader = DataProvider(dataset=region_dataset, batch_size=1,
                                shuffle=False, num_workers=args.num_worker, pin_memory=True, drop_last=False)
    
    ''' save dominant labeled target '''
    N = region_loader.__len__()
    multi_hot_cls = np.zeros((len(region_dataset), args.nseg, args.num_classes + 1)).astype('uint8')
    multi_hot_size = np.zeros((len(region_dataset), args.nseg)).astype('int')

    for iter in tqdm(range(N)):
        data = region_dataset.__getitem__(iter)
        superpixel_cls, superpixel_size  = data['superpixel_info']
        multi_hot_cls[iter] = superpixel_cls
        multi_hot_size[iter] = superpixel_size

    os.makedirs(args.save_data_dir, exist_ok=True)
    cls_name = "{}/multi_hot_cls.npy".format(args.save_data_dir)
    size_name = "{}/sp_size.npy".format(args.save_data_dir)

    ### saving
    np.save(cls_name, multi_hot_cls)
    np.save(size_name, multi_hot_size)