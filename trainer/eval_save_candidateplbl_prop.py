import torch
import numpy as np
import os
from tqdm import tqdm, trange
from PIL import Image

from trainer import eval_within_multihot
from utils.miou import MeanIoU
r""" Generate (save) pseudo label using top-1 prediction within candiate set
- additionally, for unselected superpixel we assign pseudo label by naive thresholding
"""

class ActiveTrainer(eval_within_multihot.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        self.selection_iter = selection_iter
        super().__init__(args, logger, selection_iter)
        assert(args.val_batch_size == 1)

    def inference(self, loader, prefix=''):
        args = self.args
        iou_helper = MeanIoU(self.num_classes+1, args.ignore_idx)
        iou_helper._before_epoch()
        N = loader.__len__()
        
        checkpoint_dir = '/'.join(self.args.init_checkpoint.split('/')[:-1])
        if args.plbl_type is None:
            args.plbl_type = 'wcand'
        save_dir = "{}/plbl_gen_{}/round_{}".format(checkpoint_dir, args.plbl_type, self.args.init_checkpoint.split('/')[-1][-6:-4])
        os.makedirs(name=save_dir, exist_ok=True)

        ### model forward
        self.net.eval()
        self.net.set_return_feat()
        with torch.no_grad():
            for iteration in trange(N):
                batch = loader.__next__()
                images = batch['images'].to(self.device, dtype=torch.float32)
                labels = batch['labels'].to(self.device, dtype=torch.long)

                feats, outputs = self.net.feat_forward(images)

                r''' top-1 within candidate based pseudo label acquisition '''
                superpixels = batch['spx'].to(self.device)
                spmasks = batch['spmask'].to(self.device)
                targets = batch['target'].to(self.device)
                wcand_pseudo_label = self.top_pseudo_label_generation(labels, outputs, targets, spmasks, superpixels)
                ### ㄴ N x H x W

                r''' top-1 probability filtered pseudo label '''
                pred_prob = torch.softmax(outputs / args.ce_temp, dim=1)
                pred_top1_prob, pred_cls = torch.max(pred_prob, dim=1)
                plbl_th_mask = pred_top1_prob > args.plbl_th
                plbl_th_mask = torch.logical_and(plbl_th_mask, torch.logical_not(spmasks))
                orig_shape = wcand_pseudo_label.shape
                ### plbl assignment
                wcand_pseudo_label = wcand_pseudo_label.view(-1)
                plbl_th_mask = plbl_th_mask.view(-1)
                wcand_pseudo_label[plbl_th_mask] = pred_cls.view(-1)[plbl_th_mask]
                wcand_pseudo_label = wcand_pseudo_label.reshape(orig_shape)

                output_dict = {
                    'outputs': wcand_pseudo_label,
                    'targets': labels
                }
                iou_helper._after_step(output_dict)

                r''' Save pseudo labels '''
                fname = batch['fnames'][0][1]
                lbl_id = fname.split('/')[-1].split('.')[0]
                plbl_save = wcand_pseudo_label[0].cpu().numpy().astype('uint8')
                Image.fromarray(plbl_save).save("{}/{}.png".format(save_dir, lbl_id))


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