import torch
import numpy as np
import os
from tqdm import tqdm

from dataloader import get_dataset
from trainer import active
from models import get_model, freeze_bn
from utils.miou import MeanIoU
from utils.miou_evalignore import IoUIgnore

class ActiveTrainer(active.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)
        self.target_dtype = torch.long

    def get_al_model(self):
        args = self.args
        num_classes = self.num_classes + 1 # additional class for ignore
        net = get_model(model=args.model, num_classes=num_classes,
                        output_stride=args.output_stride, separable_conv=args.separable_conv)
        return net

    def load_checkpoint(self, fname, load_optimizer=False):
        print('Load checkpoint', flush=True)
        map_location = self.device
        checkpoint = torch.load(fname, map_location=map_location)
        
        '''
        - remove final cls weight from load dict for imagenet pretrained model
        - because the # cls have been changed while cls weight are ramdomly initialized
        - this is done for every rounds (because of the start_over options)
        '''
        if 'imagenet_pretrained' in fname:
            del checkpoint['model_state_dict']['classifier.final.weight']

            try:
                del checkpoint['model_state_dict']['classifier.final.bias']
            except:
                pass

            r''' For compatibility with cosine similarity classifier '''
            try:
                del checkpoint['model_state_dict']['classifier.proxy']
            except:
                pass


        ''' Nothing to do with best val loading '''
        self.net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if load_optimizer is True:
            self.optimizer.load_state_dict(checkpoint['opt_state_dict'])

    def inference(self, loader, prefix=''):
        r"""  inference_pred_ignore
        - additionally calculate iou for undefined (ignore) class
        - - extra_class is expected to be calculated as num_classes + 1 th class
        """
        args = self.args
        iou_helper = MeanIoU(self.num_classes, args.ignore_idx)
        ignore_iou_helper = IoUIgnore(num_classes=self.num_classes, ignore_label=args.ignore_idx)
        iou_helper._before_epoch()
        N = loader.__len__()

        ### model forward
        self.net.eval()
        with torch.no_grad():
            for iteration in range(N):
                batch = loader.__next__()
                images = batch['images'].to(self.device, dtype=torch.float32)
                labels = batch['labels'].to(self.device, dtype=torch.long)

                outputs = self.net(images)
                preds = outputs.detach()

                iou_helper._after_step({'outputs': preds[:, :-1, :, :].max(dim=1)[1], 'targets': labels})
                ### ㄴ for conventional iou computation ignore label is not considered
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