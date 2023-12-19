import torch
import numpy as np
import os
from tqdm import tqdm

from dataloader import get_dataset
from trainer.base import BaseTrainer
from models import get_model, freeze_bn

class ActiveTrainer(BaseTrainer):
    def __init__(self, args, logger, selection_iter):
        self.selection_iter = selection_iter
        super().__init__(args, logger)

        if self.args.or_labeling:
            self.target_dtype = torch.uint8
        else:
            self.target_dtype = torch.long
        
        self.target_dtype = torch.long if 'oracle' in self.args.loader else self.target_dtype

        ''' validation/evaluation dataloaders '''
        val_dataset = get_dataset(args, name=self.args.val_dataset, data_root=self.args.val_data_dir,
                                  datalist=self.args.val_datalist, imageset='val')
        eval_dataset = get_dataset(args, name=self.args.val_dataset, data_root=self.args.val_data_dir,
                                  datalist=self.args.val_datalist, imageset='eval')
        self.val_dataset_loader = self.get_valloader(val_dataset)
        self.eval_dataset_loader = self.get_valloader(eval_dataset)

    def get_optim(self, my_lr):
        if self.args.adaptive_train_lr:
            my_lr = self.args.train_lr * self.selection_iter

        super().get_optim(my_lr = my_lr)

    def train(self, active_set, fname=None):
        ''' prepare datasets / dataloaders / checkpoint filename '''
        train_dataset = active_set.get_trainset() ### get labeled set

        ### trainer track best val checkpoint
        if fname is None: ### Orig implementation
            self.checkpoint_file = \
                os.path.join(self.model_save_dir, f'checkpoint{active_set.selection_iter:02d}.tar')
        else: ### For stage2 training with script
            self.checkpoint_file = fname

        '''finetune & save checkpoint (best val)'''
        self.train_dataset_loader = self.get_trainloader(train_dataset)
        total_iterations = int(self.args.finetune_itrs)
        val_period = int(self.args.val_period)
        self.train_impl(total_iterations, val_period)

    def log_validation(self, iteration, val_period):
        if iteration % val_period == (val_period - 1) and iteration > self.args.val_start:
            self.logger.info('**** EVAL ITERATION %06d ****' % (iteration))
            self.validate(trainiter=iteration)
            self.net.train()

    def log_training(self, iteration, pbar, total_itrs):
        if iteration % self.args.log_period == (self.args.log_period - 1):
            pbar.set_description('[AL {}-round] (step{}): Loss {:.4f} Session {}'.format(
                self.selection_iter,
                iteration,
                self.am.get('train-loss'),
                self.args.session_name
            ))                
            global_step = iteration + (total_itrs * (self.selection_iter - 1))
            lr_f = self.optimizer.param_groups[-1]['lr']
            wlog_train = {'learning-rate cls': lr_f}
            wlog_train.update({k:self.am.pop(k) for k,v in self.am.get_whole_data().items()})
            self.args.wandb.log(wlog_train, step=global_step)

    def train_impl(self, total_itrs, val_period):
        self.net.train()
        pbar = tqdm(range(total_itrs), ncols=150)

        for iteration in pbar:
            ### Data Loading
            batch = self.train_dataset_loader.__next__()
            images = batch['images']
            labels = batch['labels']
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=self.target_dtype)

            ### Forward
            self.optimizer.zero_grad()
            preds = self.net(images)
            if isinstance(preds, tuple):  # for multi-level
                preds = preds[1]

            ### Loss
            loss = self.loss_fun(preds, labels)

            if not torch.isnan(loss):
                loss.backward()
                self.optimizer.step()
            if self.args.scheduler == 'poly':
                self.scheduler.step()

            if not torch.isnan(loss):
                self.am.add({'train-loss': loss.detach().cpu().item()})
                
            self.log_training(iteration, pbar, total_itrs)
            self.log_validation(iteration, val_period)