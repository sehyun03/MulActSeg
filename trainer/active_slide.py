import torch
import numpy as np
import os
from tqdm import tqdm

from dataloader import get_slide_dataset
from trainer.active import ActiveTrainer
from models import get_model, freeze_bn

class ActiveTrainer(ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)

        ''' validation/evaluation dataloaders '''
        eval_dataset = get_slide_dataset(name=self.args.val_dataset, data_root=self.args.val_data_dir,
                                  datalist=self.args.val_datalist, imageset='eval')
        self.eval_dataset_loader = self.get_valloader(eval_dataset)

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