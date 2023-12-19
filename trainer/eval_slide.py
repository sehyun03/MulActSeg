import torch
import numpy as np
import os
from tqdm import tqdm, trange

from dataloader import get_dataset
from dataloader.utils import DataProvider
from trainer.base import BaseTrainer
from models import get_model, freeze_bn
from utils.miou_evalignore import IoUIgnore
from utils.miou import MeanIoU
from utils.sliding_evaluator import SlidingEval
r"""
Naive evaluation with sliding window mode
"""

class ActiveTrainer(BaseTrainer):
    def __init__(self, args, logger, selection_iter):
        self.selection_iter = selection_iter
        super().__init__(args, logger)

    def get_al_model(self):
        args = self.args
        checkpoint = torch.load(args.init_checkpoint, map_location=self.device)
        num_classes = checkpoint['model_state_dict']['classifier.proxy'].shape[0]
        net = get_model(model=args.model, num_classes=num_classes,
                        output_stride=args.output_stride, separable_conv=args.separable_conv)
        return net

    def eval(self, active_set, selection_iter):
        args = self.args
        ''' validation/evaluation dataloaders '''
        eval_dataset = get_dataset(args=args, name=self.args.val_dataset, data_root=self.args.val_data_dir,
                                  datalist=self.args.val_datalist, imageset='eval')
        self.eval_dataset_loader = self.get_valloader(eval_dataset)
        miou, iou_table_str  = self.inference(loader=self.eval_dataset_loader, prefix='evaluation')

        ''' file logging '''
        self.logger.info('[Evaluation Result]')
        self.logger.info('%s' % (iou_table_str))
        self.logger.info('Current eval miou is %.3f %%' % (miou))

        return iou_table_str

    def inference(self, loader, prefix=''):
        iou_helper = MeanIoU(self.num_classes, self.args.ignore_idx)
        iou_helper._before_epoch()
        N = loader.__len__()

        ### model forward
        self.net.eval()
        self.evaluator = SlidingEval(model=self.net,
                                     crop_size=800,
                                     stride_rate=2/3,
                                     device="cuda:0",
                                     val_id=1)
        with torch.no_grad():
            for iteration in trange(N):
                batch = loader.__next__()
                images = batch['images'].to(self.device, dtype=torch.float32)
                labels = batch['labels'].to(self.device, dtype=torch.long)

                # outputs = self.net(images)
                # preds = outputs.detach().max(dim=1)[1]
                ''' TODO: data 형식 맞추기 '''
                preds = self.evaluator(img=images)
                preds = torch.from_numpy(preds).cuda(non_blocking=True).argmax(dim=0)[None,...]

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