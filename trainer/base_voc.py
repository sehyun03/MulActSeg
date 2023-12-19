# torch
from sys import prefix
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import wandb
from tqdm import trange

# model, dataset, utils
from dataloader.utils import DataProvider
from models import get_model
from utils.miou import MeanIoU
from utils.miou_evalignore import IoUIgnore
from utils.scheduler import PolyLR
from utils.loss import FocalLoss, MyCrossEntropyLoss
from utils.loss import MultiChoiceCE, GroupMultiLabelCE, JointMultiLoss, HierGroupMultiLabelCE, JointHierarchyLoss
from utils.loss import RCCE_asym, JointRcceAsym
from utils.common_voc import AverageMeter

class BaseTrainer(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.model_save_dir = args.model_save_dir
        self.best_iou = 0
        self.device = torch.device('cuda:0')
        self.local_rank = 0
        self.am = AverageMeter()

        ''' Define Model '''
        self.num_classes = args.num_classes
        self.net = self.get_al_model()
        self.net.to(self.device)
        
        ''' Define Optimizer '''
        self.get_optim(my_lr = args.train_lr)

        ''' Define Scheduler '''
        if hasattr(self.args, "finetune_itrs"):
            total_itrs = self.args.finetune_itrs
        else:
            total_itrs = self.args.total_itrs
            
        if self.args.scheduler == 'poly':
            self.scheduler = PolyLR(self.optimizer, total_itrs, power=args.power, min_lr=args.min_lr)
        elif self.args.scheduler == 'none':
            pass
        else:
            raise NotImplementedError

        ''' Define criterion '''
        self.get_criterion()

        print("Trainer initialized", flush=True)


    def get_al_model(self):
        args = self.args
        net = get_model(model=args.model, num_classes=self.num_classes,
                        output_stride=args.output_stride, separable_conv=args.separable_conv)
        return net

    def get_optim(self, my_lr):
        if self.args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(params=[
                {'params': self.net.backbone.parameters(), 'lr': my_lr},
                {'params': self.net.classifier.parameters(), 'lr': self.args.cls_lr_scale * my_lr},
            ], lr=my_lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(params=[
                {'params': self.net.backbone.parameters(), 'lr': my_lr},
                {'params': self.net.classifier.parameters(), 'lr': self.args.cls_lr_scale * my_lr},
            ], lr=my_lr, momentum=0.9, weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError

    def get_criterion(self):
        ''' Define criterion '''

        if self.args.loss_type == 'focal_loss':
            self.loss_fun = FocalLoss(ignore_index=self.args.ignore_idx, size_average=True)
        elif self.args.loss_type == 'cross_entropy':
            self.loss_fun = MyCrossEntropyLoss(ignore_index=self.args.ignore_idx, reduction='mean', temperature=self.args.ce_temp)
        elif self.args.loss_type == 'multi_choice_ce':
            self.loss_fun = MultiChoiceCE(num_class=self.num_classes, temperature=self.args.multi_ce_temp)
        elif self.args.loss_type == 'group_multi_label_ce':
            self.loss_fun = GroupMultiLabelCE(num_class=self.num_classes, num_superpixel= self.args.nseg, temperature=self.args.group_ce_temp)
        elif self.args.loss_type == 'hierarchy_group_multi_label_ce':
            self.loss_fun = HierGroupMultiLabelCE(num_class=self.num_classes, num_superpixel= self.args.nseg, only_single=self.args.group_only_single, gumbel_scale=self.args.gumbel_scale, temperature=self.args.group_ce_temp)
        elif self.args.loss_type == 'joint_multi_loss':
            group_multi_label_ce = GroupMultiLabelCE(num_class=self.num_classes, num_superpixel= self.args.nseg, temperature=self.args.group_ce_temp)
            multi_pos_choice_ce = MultiChoiceCE(num_class=self.num_classes, temperature=self.args.multi_ce_temp)
            self.loss_fun = JointMultiLoss(group_multi_label_ce, multi_pos_choice_ce)
        elif self.args.loss_type == 'joint_multi_loss_weight':
            group_multi_label_ce = GroupMultiLabelCE(num_class=self.num_classes, num_superpixel= self.args.nseg, temperature=self.args.group_ce_temp, reduction='none')
            multi_pos_choice_ce = MultiChoiceCE(num_class=self.num_classes, temperature=self.args.multi_ce_temp, reduction='none')
            self.loss_fun = JointMultiLoss(group_multi_label_ce, multi_pos_choice_ce, reduction='none')
        elif self.args.loss_type == 'joint_hierarchy_multi_loss':
            hierarchy_multi_label_ce = HierGroupMultiLabelCE(num_class=self.num_classes, num_superpixel= self.args.nseg, only_single=self.args.group_only_single, gumbel_scale=self.args.gumbel_scale, temperature=self.args.group_ce_temp)
            multi_pos_choice_ce = MultiChoiceCE(num_class=self.num_classes, temperature=self.args.multi_ce_temp)
            self.loss_fun = JointHierarchyLoss(hierarchy_multi_label_ce, multi_pos_choice_ce)
        elif self.args.loss_type == 'joint_hierarchy_multi_loss_weight':
            hierarchy_multi_label_ce = HierGroupMultiLabelCE(num_class=self.num_classes, num_superpixel= self.args.nseg, only_single=self.args.group_only_single, gumbel_scale=self.args.gumbel_scale, temperature=self.args.group_ce_temp, reduction='none')
            multi_pos_choice_ce = MultiChoiceCE(num_class=self.num_classes, temperature=self.args.multi_ce_temp, reduction='none')
            self.loss_fun = JointHierarchyLoss(hierarchy_multi_label_ce, multi_pos_choice_ce, reduction='none')
        elif self.args.loss_type == 'rc_asym_ce':
            self.loss_fun = RCCE_asym(num_class=self.num_classes, temperature=self.args.multi_ce_temp)
        elif self.args.loss_type == 'joint_multi_rc_asym':
            group_multi_label_ce = GroupMultiLabelCE(num_class=self.num_classes, num_superpixel= self.args.nseg, temperature=self.args.group_ce_temp)
            rc_asym_ce = RCCE_asym(num_class=self.num_classes, temperature=self.args.multi_ce_temp)
            self.loss_fun = JointRcceAsym(group_multi_label_ce, rc_asym_ce)
        else:
            raise NotImplementedError

    def get_trainloader(self, dataset):
        data_provider = DataProvider(dataset=dataset,
                                     batch_size=self.args.train_batch_size,
                                     shuffle=True,
                                     num_workers=self.args.num_workers,
                                     pin_memory=True, ### debugging
                                    #  pin_memory=False, ### debugging
                                     drop_last=True)
        return data_provider

    def get_valloader(self, dataset):
        data_provider = DataProvider(dataset=dataset, batch_size=self.args.val_batch_size,
                                     shuffle=False, num_workers=self.args.val_num_workers,
                                     pin_memory=True, drop_last=False)
        return data_provider

    def train(self):
        raise NotImplementedError

    def train_impl(self, total_itrs, val_period):
        raise NotImplementedError

    def inference(self, loader, prefix=''):
        iou_helper = MeanIoU(self.num_classes, self.args.ignore_idx)
        iou_helper._before_epoch()
        N = loader.__len__()

        ### model forward
        self.net.eval()
        with torch.no_grad():
            for iteration in trange(N):
                batch = loader.__next__()
                images = batch['images'].to(self.device, dtype=torch.float32)
                labels = batch['labels'].to(self.device, dtype=torch.long)

                outputs = self.net(images)
                preds = outputs.detach().max(dim=1)[1]
                # preds = outputs.detach().max(dim=1)[1]

                output_dict = {
                    'outputs': preds,
                    'targets': labels
                }
                iou_helper._after_step(output_dict)

        iou_table = []
        ious = iou_helper._after_epoch()
        miou = np.mean(ious)
        iou_table.append(f'{miou:.2f}')
        
        ### Append per-class ious
        for class_iou in ious:
            iou_table.append(f'{class_iou:.2f}')
        iou_table_str = ','.join(iou_table)

        del iou_table
        del output_dict
        print("\n[AL {}-round]: {}\n{}".format(self.selection_iter, prefix, iou_table_str), flush=True)

        return miou, iou_table_str

    def inference_predignore(self, loader, prefix=''):
        r"""  inference_pred_ignore
        - additionally calculate iou for undefined (ignore) class
        - - extra_class is expected to be calculated as num_classes + 1 th class
        """
        args = self.args
        iou_helper = MeanIoU(self.num_classes, args.ignore_idx)
        ignore_iou_helper = IoUIgnore(num_classes=self.num_classes, ignore_label=args.ignore_idx)
        ### ㄴ 이미지 내에 255 인 영역과 19 label index 사이의 iou 측정
        iou_helper._before_epoch()
        N = loader.__len__()

        ### model forward
        self.net.eval()
        with torch.no_grad():
            for iteration in trange(N):
                batch = loader.__next__()
                images = batch['images'].to(self.device, dtype=torch.float32)
                labels = batch['labels'].to(self.device, dtype=torch.long)

                outputs = self.net(images)
                preds = outputs.detach()

                iou_helper._after_step({'outputs': preds[:, :-1, :, :].max(dim=1)[1], 'targets': labels})
                ### ㄴ for conventional iou computation ignore label is not considered
                ### 이렇게 해야만 undefined 가 아닌 자리에 undefined 라고 예측하는 불상사를 막을 수 있다.
                ignore_iou_helper._after_step({'outputs': preds.max(dim=1)[1], 'targets': labels})

        iou_table = []
        ious = iou_helper._after_epoch()
        miou = np.mean(ious)
        iou_table.append(f'{miou:.2f}')
        
        ### Append per-class ious
        ignore_iou = ignore_iou_helper._after_epoch()
        for class_iou in ious:
            iou_table.append(f'{class_iou:.2f}')
        iou_table.append(f'{ignore_iou:.2f}')
        iou_table_str = ','.join(iou_table)

        del iou_table
        print("\n[AL {}-round]: {}\n{}".format(self.selection_iter, prefix, iou_table_str), flush=True)

        return miou, iou_table_str

    def validate(self, trainiter=None, prefix=''):
        miou, iou_table_str  = self.inference(loader=self.val_dataset_loader, prefix='validation')

        ''' file logging '''
        self.logger.info('[Validation Result]')
        self.logger.info('%s' % (iou_table_str))
        
        ''' Save best miou checkpoint & result '''
        if self.best_iou < miou:
            self.best_iou = miou
            self.save_checkpoint()
        self.logger.info('Current val miou is %.3f %%, while the best val miou is %.3f %%' % (miou, self.best_iou))

        ''' Wamdb logging
        - ex) val_period: 10 --> current step: 9 --> logging step = 10
        - Current global step will also be used in next train logging (if period matches)
        '''
        total_itrs = int(self.args.finetune_itrs)
        global_step = trainiter + (total_itrs * (self.selection_iter - 1))
        wlog_test = {'{}val-miou'.format(prefix): miou, '{}val-best-miou'.format(prefix): self.best_iou, '{}selection_iter'.format(prefix): self.selection_iter}
        self.args.wandb.log(wlog_test, step=global_step + 1)

        return iou_table_str

    def eval(self, selection_iter):
        args = self.args
        miou, iou_table_str  = self.inference(loader=self.eval_dataset_loader, prefix='evaluation')

        ''' file logging '''
        self.logger.info('[Evaluation Result]')
        self.logger.info('%s' % (iou_table_str))
        self.logger.info('Current eval miou is %.3f %%' % (miou))

        ''' Wamdb logging
        - After every rounds
        - Current step: global_step -1 --> loging step = global_step
        - Current global step will also be used in next train logging (if period matches)
        '''
        global_step = int(args.finetune_itrs) * (selection_iter)
        wlog_test = {'eval-miou': miou, 'selection_iter': selection_iter}
        args.wandb.log(wlog_test, step=global_step)

        ''' Additional summary logging
        - round_v_miou: round-wise miou table (row: run, column: round) => 그냥 string 으로 변경
        - r{}_class_v_miou: class-wise iou table (row: run, column: class name경
            -- round 개수만큼 존재
        '''
        ### round_v_miou update
        current_miou = args.wandb_iou_table.loc[0]['round_v_miou']
        args.wandb_iou_table.loc[0]['round_v_miou'] = "{}{:.2f},".format(current_miou, miou)

        ### class_v_iou
        args.wandb_iou_table.loc[0]["round-{}".format(selection_iter)] = iou_table_str

        # args.wandb.run.summary["round_v_miou_v3"] = args.wandb_iou_table
        args.wandb.run.summary["round_v_miou"] = args.wandb_iou_table

        return iou_table_str

    def save_checkpoint(self):
        checkpoint = {
                        'model_state_dict': self.net.state_dict(),
                        'opt_state_dict': self.optimizer.state_dict()
                     }
        torch.save(checkpoint, self.checkpoint_file)

    def load_checkpoint(self, fname, load_optimizer=False):
        print('Load checkpoint', flush=True)
        map_location = self.device
        checkpoint = torch.load(fname, map_location=map_location)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer is True:
            self.optimizer.load_state_dict(checkpoint['opt_state_dict'])