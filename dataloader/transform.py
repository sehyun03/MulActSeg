import dataloader.ext_transforms as et
from torchvision import transforms as T
import torch

def get_train_transform(args, transform):
    if transform is None:
        lbl_transform = None
    elif transform == 'orig_notrg':
        lbl_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(512, 1024)), ### no need to pad
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(dtype_list=['int']), ### no need to augment target
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    elif transform == 'orig_notrg_load_small':
        lbl_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(512, 1024)), ### no need to pad
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(dtype_list=['int', 'int']), ### no need to augment target
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    elif transform == 'orig_ignore_notrg':
        lbl_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(512, 1024)), ### no need to pad
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(dtype_list=['int', 'int']), ### no need to augment target
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    elif transform == 'orig_ignore_notrg_load_small':
        lbl_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(512, 1024)), ### no need to pad
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(dtype_list=['int', 'int', 'int']), ### no need to augment target
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    elif transform == 'rescale':
        lbl_transform = et.ExtCompose([
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(512, 1024)), ### no need to pad
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(dtype_list=['int', 'int']),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    elif transform == 'rescale_769_nospx':
        lbl_transform = et.ExtCompose([
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(768, 768), pad_values=[args.ignore_idx], padding=(124, 116, 104), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(dtype_list=['int']),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    elif transform == 'rescale_769':
        lbl_transform = et.ExtCompose([
            et.ExtRandomScale((0.5, 2.0)),
            # et.ExtScale(0.5), ### minimum size test for debugging 512 x 1024
            et.ExtRandomCrop(size=(768, 768), pad_values=[args.ignore_idx, args.nseg], padding=(124, 116, 104), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(dtype_list=['int', 'int']),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    elif transform == 'rescale_769_multi':
        transform_list = [
                        et.ExtRandomScale((0.5, 2.0)),
                        # et.ExtScale(0.5), ### minimum size test for debugging 512 x 1024
                        ]

        if args.load_smaller_spx:
            transform_list.extend([
                            et.ExtRandomCrop(size=(768, 768), pad_values=[args.ignore_idx, args.nseg, args.small_nseg], padding=(124, 116, 104), pad_if_needed=True),
                            et.ExtRandomHorizontalFlip(),
                            et.ExtToTensor(dtype_list=['uint8', 'int', 'int']),
                            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            ])            
        else:
            transform_list.extend([
                            et.ExtRandomCrop(size=(768, 768), pad_values=[args.ignore_idx, args.nseg], padding=(124, 116, 104), pad_if_needed=True),
                            et.ExtRandomHorizontalFlip(),
                            et.ExtToTensor(dtype_list=['uint8', 'int']),
                            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            ])
        lbl_transform = et.ExtCompose(transform_list)
    elif transform == 'rescale_769_multi_notrg': ### only augment superpixel map
        transform_list = [
                        et.ExtRandomScale((0.5, 2.0)),
                        # et.ExtScale(0.5), ### minimum size test for debugging 512 x 1024
                        ]

        if args.load_smaller_spx:
            transform_list.extend([
                            et.ExtRandomCrop(size=(768, 768), pad_values=[args.nseg, args.small_nseg], padding=(124, 116, 104), pad_if_needed=True),
                            et.ExtRandomHorizontalFlip(),
                            et.ExtToTensor(dtype_list=['int', 'int']),
                            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            ])
        else:
            transform_list.extend([
                            et.ExtRandomCrop(size=(768, 768), pad_values=[args.nseg], padding=(124, 116, 104), pad_if_needed=True),
                            et.ExtRandomHorizontalFlip(),
                            et.ExtToTensor(dtype_list=['int']),
                            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

            ])
        lbl_transform = et.ExtCompose(transform_list)
    elif transform == 'rescale_769_multi_notrg_ignore':
        transform_list = [
                        et.ExtRandomScale((0.5, 2.0)),
                        # et.ExtScale(0.5), ### minimum size test for debugging 512 x 1024
                        ]

        if args.load_smaller_spx:
            transform_list.extend([
                            et.ExtRandomCrop(size=(768, 768), pad_values=[0, args.nseg, args.small_nseg], padding=(124, 116, 104), pad_if_needed=True),
                            et.ExtRandomHorizontalFlip(),
                            et.ExtToTensor(dtype_list=['int', 'int', 'int']),
                            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            ])
        else:
            transform_list.extend([
                            et.ExtRandomCrop(size=(768, 768), pad_values=[0, args.nseg], padding=(124, 116, 104), pad_if_needed=True),
                            et.ExtRandomHorizontalFlip(),
                            et.ExtToTensor(dtype_list=['int', 'int']),
                            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

            ])
        lbl_transform = et.ExtCompose(transform_list)
    elif transform == 'rescale_769_multi_notrg_ignore_strongv1':
        transform_list = [
                        et.ExtRandomScale((0.5, 2.0)),
                        ]
        assert(args.load_smaller_spx)
        transform_list.extend([
                        et.ExtRandomCrop(size=(768, 768), pad_values=[0, args.nseg, args.small_nseg], padding=(124, 116, 104), pad_if_needed=True),
                        et.ExtRandomHorizontalFlip(),
                        et.ExtColorJitter(0.4, 0.4, 0.4, 0.1, p=0.2),
                        et.ExtRandomGrayscale(p=0.2),
                        et.ExtToTensor(dtype_list=['int', 'int', 'int']),
                        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        ])
        lbl_transform = et.ExtCompose(transform_list)
    elif transform == 'eval_spx':
        lbl_transform = et.ExtCompose([
            et.ExtResize((1024, 2048)),
            et.ExtToTensor(dtype_list=['int', 'int']), ### no need to augment target
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    elif transform == 'eval_dom_gt_spx':
        lbl_transform = et.ExtCompose([
            et.ExtResize((1024, 2048)),
            et.ExtToTensor(dtype_list=['int', 'int', 'int']), ### no need to augment target
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise NotImplementedError

    return lbl_transform