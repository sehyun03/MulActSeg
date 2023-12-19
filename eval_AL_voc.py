#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic
import os
import sys
from datetime import datetime
import wandb
import torch

# custom
from dataloader import get_active_dataset
from utils.common_voc import initialization, get_parser, preprocess, arg_assert
from utils.mylog import finalization, init_logging
import importlib

def main(args):
    ''' initialization '''
    logger = initialization(args)
    t_start = datetime.now()
    val_result = {}
    init_logging(args)

    '''Active Learning dataset'''
    active_set = get_active_dataset(args, train_transform=args.train_transform)
    Trainer = importlib.import_module("trainer.{}".format(args.method.lower()))

    ### Active Learning iteration
    print('Start active learning iteration from {}'.format(args.init_iteration))
    selection_iter = 0
    trainer = Trainer.ActiveTrainer(args, logger, selection_iter) ### caution: reinitialize to ImageNet pretrained model
    active_set.selection_iter = selection_iter

    ''' Resume from previous model and selection '''
    ### resume actively sampled data before 'selection_iter' rounds.
    active_set.load_datalist(args.datalist_path)
    trainer.load_checkpoint(args.init_checkpoint, load_optimizer=args.load_optim)

    trainer.eval(active_set, selection_iter = (args.init_iteration - 1))

    # try:
    #     trainer.eval(active_set, selection_iter = (args.init_iteration - 1))
    # except:
    #     trainer.eval(selection_iter = (args.init_iteration - 1))

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print(' '.join(sys.argv))
    preprocess(args)
    arg_assert(args)
    print(args)
    if args.set_num_threads != -1:
        torch.set_num_threads(args.set_num_threads)

    '''Wandb log'''
    if args.dontlog:
        print("skip logging...")
        os.environ['WANDB_SILENT'] = 'true'
        os.environ['WANDB_MODE'] = 'dryrun'
    else:
        os.environ['WANDB_SILENT'] = 'false'
        os.environ['WANDB_MODE'] = 'run'

    '''Wandb sweep argument'''
    wandb.init(name="{}".format(args.model_save_dir.split('/')[-1], ),
               project='query-designed-active-segmentation', tags=[str(i) for i in args.wandb_tags], group=args.wandb_group,
               settings=wandb.Settings(start_method="fork"))
    wandb.config.update(args)
    args.wandb = wandb

    main(args)