import torch
import numpy as np
import os
from tqdm import tqdm, trange
import torch.nn.functional as F
from torch_scatter import scatter, scatter_max
from PIL import Image
from skimage.segmentation import mark_boundaries

from dataloader import get_dataset
from dataloader.utils import DataProvider
from trainer.eval_cosplbl_within_multihot import ActiveTrainer
from models import get_model, freeze_bn
from utils.miou_evalignore import IoUIgnore
from utils.miou import MeanIoU


class ActiveTrainer(ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        self.selection_iter = selection_iter
        super().__init__(args, logger, selection_iter)

    def denormalize(self, img):
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        return (img * std[..., None, None]) + mean[..., None, None]

    def inference(self, loader, prefix=''):
        iou_helper = MeanIoU(self.num_classes, self.args.ignore_idx)
        iou_helper._before_epoch()
        N = loader.__len__()
        checkpoint_id = self.args.init_checkpoint.split('/')[1].split('_nlbl')[0]
        save_dir = "vis/spx_tvis/{}/{}_round_{}".format(self.args.nseg, checkpoint_id, self.args.init_checkpoint.split('/')[-1][-6:-4])
        os.makedirs(name=save_dir, exist_ok=True)

        decode_target = loader.dataset.decode_target

        ### model forward
        self.net.eval()
        self.net.set_return_feat()
        with torch.no_grad():
            for iteration in trange(N):
                batch = loader.__next__()
                images = batch['images'].to(self.device, dtype=torch.float32)
                labels = batch['labels'].to(self.device, dtype=torch.long)
                superpixels = batch['spx'].to(self.device)
                spmasks = batch['spmask'].to(self.device)
                targets = batch['target'].to(self.device)
                fnames = batch['fnames']

                feats, outputs = self.net.feat_forward(images)
                preds = outputs.detach().max(dim=1)[1]

                r''' Top-1 selected point acquisition '''
                top_one = self.get_top_one(labels, outputs, targets, spmasks, superpixels)
                ### ㄴ N x H x W

                r''' NN pseudo label acquisition '''
                nn_pseudo_label = self.pseudo_label_generation(labels, feats, outputs, targets, spmasks, superpixels)

                r''' Visualization
                - (0) target precise + (with or without) spx boundaries
                - (1) Top-1 plbl + spx boundaries
                - (2) NN pseudo label + spx boundaries
                - (3) Image + spx boundaries + top-1 points
                - (4) Top-1 poins + spx boundaries
                '''
                vis_superpixel = superpixels[0].cpu().numpy()
                img_id = fnames[0][0].split('/')[-1].split('_left')[0]

                r''' (0) target precise + (with or without) spx boundaries '''
                vis_label = torch.masked_fill(labels[0], labels[0]==255, 20).cpu()
                vis_label = decode_target(vis_label).astype('uint8')
                vis_label = mark_boundaries(vis_label, vis_superpixel) * 255
                vis_label = vis_label.astype('uint8')
                Image.fromarray(vis_label).save("{}/{}_gt_bdry.png".format(save_dir, img_id))

                r''' (1) Top-1 plbl + (with or without) spx boundaries '''
                vis_pred_plbl = torch.masked_fill(preds[0], labels[0]==255, 20).cpu()
                vis_pred_plbl = decode_target(vis_pred_plbl).astype('uint8')
                vis_pred_plbl = mark_boundaries(vis_pred_plbl, vis_superpixel) * 255
                vis_pred_plbl = vis_pred_plbl.astype('uint8')
                Image.fromarray(vis_pred_plbl).save("{}/{}_pred_plbl_bdry.png".format(save_dir, img_id))

                r''' (2) NN plbl + (with or without) spx boundaries '''
                vis_nn_plbl = torch.masked_fill(nn_pseudo_label[0], labels[0]==255, 20).cpu()
                vis_nn_plbl = decode_target(vis_nn_plbl).astype('uint8')
                vis_nn_plbl = mark_boundaries(vis_nn_plbl, vis_superpixel) * 255
                vis_nn_plbl = vis_nn_plbl.astype('uint8')
                Image.fromarray(vis_nn_plbl).save("{}/{}_nn_plbl_bdry.png".format(save_dir, img_id))

                r''' (3) Image + spx boundaries + top-1 points '''
                vis_img = self.denormalize(images[0].cpu()).permute(1,2,0).numpy()
                vis_img = mark_boundaries(vis_img, vis_superpixel)
                vis_img = (vis_img * 255).astype('uint8')
                top_one_ = torch.masked_fill(top_one[0], top_one[0]==255, 20).cpu()
                top_one_vis = decode_target(top_one_).astype('uint8')

                vis_img = torch.from_numpy(vis_img)
                top_one_vis = torch.from_numpy(top_one_vis)
                vis_img = torch.where((top_one_ == 20)[..., None], vis_img, top_one_vis)
                vis_img = vis_img.numpy()

                Image.fromarray(vis_img).save("{}/{}_img_bdry.png".format(save_dir, img_id))

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

    def get_top_one(self, labels, inputs, targets, spmasks, superpixels):
        r'''
        Args::
            inputs: N x C x H x W
            targets: N x self.num_superpiexl x C+1
            spmasks: N x H x W
            superpixels: N x H x W
            superpixel_smalls: N x H x W
            
            Apply max operation over predicted probabilities for each multi-hot label within the superpixel,
            and highlight selected top-1 pixel with its corresponding labels
            
        return::
            top_one_vis (torch.Tensor): top-1 highlighted target map
                                        N x H x W
            '''

        N, C, H, W = inputs.shape
        outputs = F.softmax(inputs, dim=1) ### N x C x H x W
        outputs = outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW
        top_one_vis = torch.ones_like(labels) * 255 ### N x H x W
        top_one_vis = top_one_vis.reshape(N, -1)

        for i in range(N):
            '''
            outputs[i] : HW x C
            superpixels[i] : HW x 1
            superpixel_smalls[i] : HW x 1
            targets[i] : self.num_superpiexl x C+1
            spmasks[i] : HW
            '''

            r''' filtered outputs '''
            valid_mask = spmasks[i] ### HW
            if not torch.any(valid_mask): ### valid pixel 이 하나도 없으면 loss 계산 X
                continue #TODO
            valid_output = outputs[i][valid_mask] ### HW' x C
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1

            r''' get max pixel for each class within superpixel '''
            _, idx_sup_mxpool = scatter_max(valid_output, valid_superpixel, dim=0, dim_size=self.args.nseg)
            ### ㄴ self.num_superpixel x C
           
            trg_sup_mxpool = targets[i] ### self.num_superpixel x C
            
            valid_idx_sup_mxpool = idx_sup_mxpool[idx_sup_mxpool[:,0] < valid_output.shape[0]]
            ### ㄴ nvalidseg x C : index of max pixel for each class
            valid_trg_sup_mxpool = trg_sup_mxpool[idx_sup_mxpool[:,0] < valid_output.shape[0]]
            ### ㄴ nvalidseg x C : multi-hot label

            r''' Index conversion '''
            index_map = valid_mask.nonzero().squeeze(dim=1)
            top1_vdx = valid_idx_sup_mxpool[valid_trg_sup_mxpool.nonzero(as_tuple=True)]
            top1_pdx = index_map[top1_vdx]
            top1_cdx = valid_trg_sup_mxpool.nonzero()[:,1]
            top_one_vis[i, top1_pdx] = top1_cdx

        top_one_vis = top_one_vis.reshape(N, H, W)
        
        return top_one_vis