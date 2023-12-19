import torch
import numpy as np
import os
from dataloader import get_dataset
from models import get_model
from trainer.base import BaseTrainer
from utils.miou import MeanIoU
from utils.miou_evalignore import IoUIgnore
from tqdm import trange
from PIL import Image


class ActiveTrainer(BaseTrainer):
    def __init__(self, args, logger, selection_iter):
        self.selection_iter = selection_iter
        super().__init__(args, logger)

    def get_al_model(self):
        args = self.args
        num_classes = self.num_classes + 1 # additional class for ignore
        net = get_model(model=args.model, num_classes=num_classes,
                        output_stride=args.output_stride, separable_conv=args.separable_conv)
        return net

    def eval(self, active_set, selection_iter):
        args = self.args
        eval_dataset = get_dataset(args, name=self.args.val_dataset, data_root=self.args.val_data_dir,
                                  datalist=self.args.val_datalist, imageset='eval')
        self.eval_dataset_loader = self.get_valloader(eval_dataset)


        miou, iou_table_str  = self.inference(loader=self.eval_dataset_loader, prefix='evaluation')

        ''' file logging '''
        self.logger.info('[Evaluation Result]')
        self.logger.info('%s' % (iou_table_str))
        self.logger.info('Current eval miou is %.3f %%' % (miou))

        return iou_table_str
    
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

        decode_target = loader.dataset.decode_target

        round = self.args.init_checkpoint.split('/')[-1][-6:-4]
        save_dir_gt = 'vis/neurips23_supp_qual/gt'
        save_dir = 'vis/neurips23_supp_qual/round_{}'.format(round)
        os.makedirs(save_dir_gt, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        ### model forward
        self.net.eval()
        with torch.no_grad():
            for iteration in trange(N):
                batch = loader.__next__()
                images = batch['images'].to(self.device, dtype=torch.float32)
                labels = batch['labels'].to(self.device, dtype=torch.long)

                outputs = self.net(images)
                preds = outputs.detach()
                pred_within = preds[:, :-1, :, :].max(dim=1)[1]

                fname = batch['fnames'][0][1]
                lbl_id = fname.split('/')[-1].split('.')[0]

                ### save GT
                if args.save_vis:
                    vis_labels = torch.masked_fill(labels[0], labels[0]==255, 19).cpu()
                    vis_labels = decode_target(vis_labels).astype('uint8')
                    Image.fromarray(vis_labels).save("{}/{}.png".format(save_dir_gt, lbl_id))

                ### save plbl
                vis_pred_within = decode_target(pred_within.cpu()).astype('uint8')[0]
                Image.fromarray(vis_pred_within).save("{}/{}.png".format(save_dir, lbl_id))

                iou_helper._after_step({'outputs': pred_within, 'targets': labels})
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