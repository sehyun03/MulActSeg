#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic
import os
import sys
from datetime import datetime
import torch
import wandb
import numpy as np


# custom
from dataloader import get_active_dataset
from utils.common import initialization, get_parser, preprocess, arg_assert
from utils.mylog import finalization, init_logging
import importlib
def main(args):
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    
    ''' initialization '''
    logger = initialization(args)
    t_start = datetime.now()
    val_result = {}
    init_logging(args)

    '''Active Learning dataset'''
    active_set = get_active_dataset(args, train_transform=args.train_transform)
    Initial_selector = importlib.import_module("active_selection.{}".format(args.initial_active_method))
    initial_selector = Initial_selector.RegionSelector(args)
    Active_selector = importlib.import_module("active_selection.{}".format(args.active_method))
    active_selector = Active_selector.RegionSelector(args)
    Trainer = importlib.import_module("trainer.{}".format(args.method.lower()))

    ### Active Learning iteration
    print('Start active learning iteration from {}'.format(args.init_iteration))
    for selection_iter in range(args.init_iteration, args.max_iterations + 1): # 1 ~ args.max_iteration
        trainer = Trainer.ActiveTrainer(args, logger, selection_iter) ### caution: reinitialize to ImageNet pretrained model
        active_set.selection_iter = selection_iter

        r''' Resume from previous model and selection '''
        ### optionally resume actively sampled data before 'selection_iter' rounds.
        if (args.datalist_path is not None) and (selection_iter == args.init_iteration):
            active_set.load_datalist(args.datalist_path)
        
        ### Model loading for active selection (there are four scenarios)
        if selection_iter == 1 and selection_iter == args.init_iteration: ### init pretrained model loading
            # import pdb; pdb.set_trace()
            trainer.load_checkpoint(args.init_checkpoint, load_optimizer=args.load_optim)
        elif selection_iter != 1 and selection_iter != args.init_iteration: ### load best model from previous round
            prevckpt_fname = os.path.join(args.model_save_dir, f'checkpoint{selection_iter-1:02d}.tar')
            trainer.load_checkpoint(prevckpt_fname, load_optimizer=args.load_optim)
        elif selection_iter != 1 and selection_iter == args.init_iteration: ### use loaded resume_checkpoint
            assert(args.resume_checkpoint is not None) 
            trainer.load_checkpoint(args.resume_checkpoint, load_optimizer=args.load_optim)
        else: ### init_iteration < 1 is not supported
            raise NotImplementedError

        ### sanity check evluation (and logging)
        if (not args.skip_first_eval) and (selection_iter == args.init_iteration):
            iou_table_str = trainer.eval(selection_iter = (args.init_iteration - 1))

        ''' 1. Active sampling '''
        print("[AL {}-round]: Active sampling starts".format(selection_iter))
        if selection_iter == 1: ### random sampling for initial round
            initial_selector.select_next_batch(trainer, active_set, args.active_selection_size)
        else:
            active_selector.select_next_batch(trainer, active_set, args.active_selection_size)
        active_set.dump_datalist()

        ''' 2. Model update (training) and selection (saving)'''
        print("[AL {}-round]: Model training starts".format(selection_iter))
        if args.start_over: ### initialize model with imagenet pretrained model
            trainer.load_checkpoint(args.init_checkpoint, load_optimizer=args.load_optim)
        
        trainer.train(active_set) ### finetune & save best val checkpoint

        ''' 3. Load best checkpoint + Evaluation '''
        fname = os.path.join(args.model_save_dir, f'checkpoint{selection_iter:02d}.tar')
        trainer.load_checkpoint(fname) ### To use best val model for active sampling (instead of current model)
        val_return = trainer.eval(selection_iter = selection_iter)
        val_result[selection_iter] = val_return
        print("[AL {}-round]: best miou/iou:\n{}\n\n".format(selection_iter, val_return))
        logger.info(f"AL {selection_iter}: Get best validation result")
        torch.cuda.empty_cache()

    ### finalization
    finalization(t_start, val_result, logger, args)

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
