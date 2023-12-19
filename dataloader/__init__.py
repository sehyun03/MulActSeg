import importlib
from dataloader.dataset import CityscapesGTA5, VOC
from dataloader.region_active_dataset import RegionActiveDataset
# from dataloader.region_voc_or import RegionVOCOr
import dataloader.ext_transforms as et
from dataloader.transform import get_train_transform
from dataloader.transform_voc import get_train_transform_voc

def get_dataset(args, name, data_root, datalist, total_itrs=None, imageset='train'):
    """Obtain a specified dataset class.
    Args:
        name (str): the name of datasets, now only support cityscapes.
        data_root (str): the root directory of data.
        datalist (str): the name of initialized datalist for all mode.
        total_itrs (int): the number of total training iterations.
        imageset (str): "train", "val", "active-label", "active-ulabel" 4 different sets.

    """
    assert(imageset in ["val", "eval"])
    assert(name in ["cityscapes","voc"])

    # if imageset == "val":
    #     transform = et.ExtCompose([
    #         et.ExtResize((1024, 2048)),
    #         et.ExtToTensor(),
    #         et.ExtNormalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225]),
    #     ])
    # elif imageset == "eval":
    #     transform = et.ExtCompose([
    #         et.ExtResize((1024, 2048)),
    #         et.ExtToTensor(),
    #         et.ExtNormalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225]),
    #     ])
    # else:
    #     raise NotImplementedError
    if name == "cityscapes":
        if imageset == "val":
            transform = et.ExtCompose([
                et.ExtResize((1024, 2048)),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        elif imageset == "eval":
            transform = et.ExtCompose([
                et.ExtResize((1024, 2048)),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            raise NotImplementedError
        dataset = CityscapesGTA5(data_root, datalist, imageset, transform=transform)

    elif name == "voc":
        if imageset == "val":
            transform = et.ExtCompose([
                et.ExtResize(513),
                et.ExtCenterCrop(513),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        elif imageset == "eval":
            transform = et.ExtCompose([
                et.ExtResize(513),
                et.ExtCenterCrop(513),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            raise NotImplementedError
        dataset = VOC(data_root, datalist, imageset, transform=transform, dominant_labeling=args.dominant_labeling)

    return dataset

def get_slide_dataset(name, data_root, datalist, total_itrs=None, imageset='train'):
    assert(imageset == "eval")
    assert(name in ["cityscapes","voc"])

    # transform = et.ExtCompose([
    #     et.ExtResize((1024, 2048)),
    #     et.ExtToTensor(),
    #     et.ExtNormalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225]),
    # ])

    # dataset = CityscapesGTA5(data_root, datalist, imageset, transform=transform)
    if name == "cityscapes":
        transform = et.ExtCompose([
            et.ExtResize((1024, 2048)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        dataset = CityscapesGTA5(data_root, datalist, imageset, transform=transform)
    elif name == "voc":
        transform = et.ExtCompose([
            et.ExtResize(513),
            et.ExtCenterCrop(513),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        dataset = VOC(data_root, datalist, imageset, transform=transform)

    return dataset

def get_active_dataset(args, train_transform=None):
    ''' Active segmentation dataset 
        Main difference is train, val transformation
    '''
    if args.src_dataset == 'cityscapes':
        ### train transform
        lbl_transform = get_train_transform(args, train_transform)

        ### validation transform (for pool dataset)
        val_transform = et.ExtCompose([
            et.ExtResize((1024, 2048)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        val_do_transform = et.ExtCompose([
            et.ExtResize((1024, 2048)),
            et.ExtToTensor(dtype_list=['int', 'int']),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        if args.active_mode == 'scan':
            raise NotImplementedError
        elif args.active_mode == 'region':
            if args.or_labeling:
                region_cityscapes_or = importlib.import_module("dataloader.{}".format(args.loader.lower()))
                trg_label_dataset = region_cityscapes_or.RegionCityscapesOr(args, args.trg_data_dir, None, split='active-label', transform=lbl_transform, dominant_labeling=args.dominant_labeling, loading=args.loading, load_smaller_spx=args.load_smaller_spx)
                trg_pool_dataset = region_cityscapes_or.RegionCityscapesOr(args, args.trg_data_dir, args.trg_datalist, region_dict=args.region_dict, split='active-ulabel', transform=val_transform, return_spx=True)
            else:
                region_cityscapes = importlib.import_module("dataloader.{}".format(args.loader.lower()))
                trg_label_dataset = region_cityscapes.RegionCityscapes(args, args.trg_data_dir, None, split='active-label', transform=lbl_transform, dominant_labeling=args.dominant_labeling)
                trg_pool_dataset = region_cityscapes.RegionCityscapes(args, args.trg_data_dir, args.trg_datalist, region_dict=args.region_dict, split='active-ulabel', transform=val_do_transform, return_spx=True, dominant_labeling=args.dominant_labeling)

            region_active_dataset = 'mseg_' if 'mseg' in args.loader.lower() else ''
            region_active = importlib.import_module("dataloader.{}region_active_dataset".format(region_active_dataset))
            dataset = region_active.RegionActiveDataset(args, trg_pool_dataset, trg_label_dataset)
    
    elif args.src_dataset == 'voc':
        ### train transform
        lbl_transform = get_train_transform_voc(args, train_transform)

        ### validation transform (for pool dataset)
        val_transform = et.ExtCompose([
            et.ExtResize(513),
            et.ExtCenterCrop(513),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        val_do_transform = et.ExtCompose([
            et.ExtResize(513),
            et.ExtCenterCrop(513),
            et.ExtToTensor(dtype_list=['int', 'int']),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        if args.active_mode == 'scan':
            raise NotImplementedError
        elif args.active_mode == 'region':
            if args.or_labeling:
                region_voc_or = importlib.import_module("dataloader.{}".format(args.loader.lower()))
                trg_label_dataset = region_voc_or.RegionVOCOr(args, args.trg_data_dir, None, split='active-label', transform=lbl_transform, dominant_labeling=args.dominant_labeling, loading=args.loading, load_smaller_spx=args.load_smaller_spx)
                trg_pool_dataset = region_voc_or.RegionVOCOr(args, args.trg_data_dir, args.trg_datalist, region_dict=args.region_dict, split='active-ulabel', transform=val_transform, return_spx=True)
            else:
                region_voc = importlib.import_module("dataloader.{}".format(args.loader.lower()))
                trg_label_dataset = region_voc.RegionVOC(args, args.trg_data_dir, None, split='active-label', transform=lbl_transform, dominant_labeling=args.dominant_labeling)
                trg_pool_dataset = region_voc.RegionVOC(args, args.trg_data_dir, args.trg_datalist, region_dict=args.region_dict, split='active-ulabel', transform=val_do_transform, return_spx=True, dominant_labeling=args.dominant_labeling)

            region_active_dataset = 'mseg_' if 'mseg' in args.loader.lower() else ''
            region_active = importlib.import_module("dataloader.{}region_active_dataset".format(region_active_dataset))
            dataset = region_active.RegionActiveDataset(args, trg_pool_dataset, trg_label_dataset)
    return dataset