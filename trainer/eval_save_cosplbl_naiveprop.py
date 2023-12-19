import torch
import numpy as np
import os
from tqdm import tqdm, trange
from PIL import Image
from skimage.segmentation import mark_boundaries

from trainer import eval_save_cosplbl
from utils.miou import MeanIoU
r""" Generate (save) pseudo label using proposed prototype-based expansion
- additionally, for unselected superpixel we assign pseudo label by naive thresholding
"""

class ActiveTrainer(eval_save_cosplbl.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        self.selection_iter = selection_iter
        super().__init__(args, logger, selection_iter)
        assert(args.val_batch_size == 1)

    def inference(self, loader, prefix=''):
        args = self.args
        iou_helper = MeanIoU(self.num_classes+1, args.ignore_idx)
        iou_helper._before_epoch()
        N = loader.__len__()
        
        decode_target = loader.dataset.decode_target

        round = self.args.init_checkpoint.split('/')[-1][-6:-4]
        checkpoint_dir = '/'.join(self.args.init_checkpoint.split('/')[:-1])
        if args.plbl_type is not None:
            save_dir = '{}/plbl_gen_{}/round_{}'.format(checkpoint_dir, args.plbl_type, round)
        else:
            save_dir = '{}/plbl_gen/round_{}'.format(checkpoint_dir, round)
        save_vid_dir = '{}_vis'.format(save_dir)
        os.makedirs(name=save_dir, exist_ok=True)
        if args.save_vis:
            os.makedirs(name=save_vid_dir, exist_ok=True)

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
                nn_pseudo_label = self.pseudo_label_generation(labels, feats, outputs, targets, spmasks, superpixels)
                ### ㄴ N x H x W

                r''' top-1 probability filtered pseudo label '''
                pred_prob = torch.softmax(outputs / args.ce_temp, dim=1)
                pred_top1_prob, pred_cls = torch.max(pred_prob, dim=1)
                plbl_th_mask = pred_top1_prob > args.plbl_th
                plbl_th_mask = torch.logical_and(plbl_th_mask, torch.logical_not(spmasks))
                orig_shape = nn_pseudo_label.shape
                ### plbl assignment
                nn_pseudo_label = nn_pseudo_label.view(-1)
                plbl_th_mask = plbl_th_mask.view(-1)
                nn_pseudo_label[plbl_th_mask] = pred_cls.view(-1)[plbl_th_mask]
                nn_pseudo_label = nn_pseudo_label.reshape(orig_shape)

                output_dict = {
                    'outputs': nn_pseudo_label,
                    'targets': labels
                }
                iou_helper._after_step(output_dict)

                r''' Save pseudo labels '''
                fname = batch['fnames'][0][1]
                lbl_id = fname.split('/')[-1].split('.')[0]
                plbl_save = nn_pseudo_label[0].cpu().numpy().astype('uint8')
                Image.fromarray(plbl_save).save("{}/{}.png".format(save_dir, lbl_id))

                r''' Visualize pseudo labels '''
                if args.save_vis:
                    fname = batch['fnames'][0][1]
                    lbl_id = fname.split('/')[-1].split('.')[0]

                    vis_superpixel = superpixels[0].cpu().numpy()
                    vis_nn_plbl = torch.masked_fill(nn_pseudo_label[0], nn_pseudo_label[0]==255, 20).cpu()
                    vis_nn_plbl = decode_target(vis_nn_plbl).astype('uint8')
                    vis_nn_plbl = mark_boundaries(vis_nn_plbl, vis_superpixel) * 255
                    vis_nn_plbl = vis_nn_plbl.astype('uint8')
                    Image.fromarray(vis_nn_plbl).save("{}/{}.png".format(save_vid_dir, lbl_id))


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