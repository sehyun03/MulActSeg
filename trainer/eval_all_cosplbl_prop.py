import torch
import numpy as np
import os
from tqdm import tqdm, trange
import torch.nn.functional as F
from torch_scatter import scatter, scatter_max
from PIL import Image
from skimage.segmentation import mark_boundaries
from skimage.morphology import binary_dilation

from dataloader import get_dataset
from dataloader.utils import DataProvider
from trainer.eval_save_cosplbl_prop import ActiveTrainer
from models import get_model, freeze_bn
from utils.miou import MeanIoU
r'''
- Cosine pseudo label with label propagation
'''

class ActiveTrainer(ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        self.selection_iter = selection_iter
        super().__init__(args, logger, selection_iter)

    def inference(self, loader, prefix=''):
        args = self.args
        iou_helper = MeanIoU(self.num_classes+1, args.ignore_idx)
        iou_helper._before_epoch()
        N = loader.__len__()

        ### model forward
        self.net.eval()
        self.net.set_return_feat()
        with torch.no_grad():
            for iteration in trange(N):
                batch = loader.__next__()
                images = batch['images'].to(self.device, dtype=torch.float32)
                labels = batch['labels'].to(self.device, dtype=torch.long)

                feats, outputs = self.net.feat_forward(images)

                r''' NN based pseudo label acquisition '''
                superpixels = batch['spx'].to(self.device)
                spmasks = batch['spmask'].to(self.device)
                targets = batch['target'].to(self.device)
                nn_pseudo_label = self.pseudo_label_generation(labels, feats, outputs, targets, spmasks, superpixels)
                ### ㄴ N x H x W

                output_dict = {
                    'outputs': nn_pseudo_label,
                    'targets': labels
                }
                # iou_helper._after_step(output_dict)
                iou_helper._after_step_within_predregion(output_dict)

        iou_table = []
        precision_table = []
        recall_table = []
        ious, precisions, recalls = iou_helper._after_epoch_ipr()

        miou = np.mean(ious)
        iou_table.append(f'{miou:.2f}')
        for class_iou in ious:
            iou_table.append(f'{class_iou:.2f}')
        iou_table_str = ','.join(iou_table)

        mprecision = np.mean(precisions)
        precision_table.append(f'{mprecision:.2f}')
        for class_precision in precisions:
            precision_table.append(f'{class_precision:.2f}')
        precision_table_str = ','.join(precision_table)

        mrecall = np.mean(recalls)
        recall_table.append(f'{mrecall:.2f}')
        for class_recall in recalls:
            recall_table.append(f'{class_recall:.2f}')
        recall_table_str = ','.join(recall_table)

        del iou_table
        del precision_table
        del recall_table
        del output_dict
        print("\n[AL {}-round] IoU: {}\n{}".format(self.selection_iter, prefix, iou_table_str), flush=True)
        print("\n[AL {}-round] Precision: {}\n{}".format(self.selection_iter, prefix, precision_table_str), flush=True)
        print("\n[AL {}-round] Recall: {}\n{}".format(self.selection_iter, prefix, recall_table_str), flush=True)

        return miou, iou_table_str