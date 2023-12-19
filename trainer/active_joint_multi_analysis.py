import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from trainer import active
from models import freeze_bn
from utils.loss import MultiChoiceCE, GroupMultiLabelCE
from torch_scatter import scatter, scatter_max
from utils.miou import MeanIoU
import dataloader.ext_transforms as et

class ActiveTrainer(active.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)

    def train_impl(self, total_itrs, val_period):
        raise NotImplementedError

    def get_criterion(self):
        ''' Define criterion '''
        self.group_multi_loss = GroupMultiLabelCE(args=self.args, num_class=self.num_classes, num_superpixel= self.args.nseg, temperature=self.args.group_ce_temp)
        self.multi_pos_loss = MultiChoiceCE(num_class=self.num_classes, temperature=self.args.multi_ce_temp)

    def eval(self, active_set, selection_iter):
        ### orig evaluation
        # super().eval(selection_iter)

        ### top-1 selection accuracy evaluation
        train_dataset = active_set.get_trainset() ### get labeled set
        identity_transform = et.ExtCompose([
            et.ExtToTensor(dtype_list=['int','int']) ### target, spx
        ])        
        self.train_dataset_loader = self.get_trainloader(train_dataset)
        self.net.eval()

        ncorr_cls = torch.zeros(19)
        ncorr_total = 0
        n_cls = torch.zeros(19)
        n_total = 0

        with torch.no_grad():
            N = self.train_dataset_loader.__len__()
            loader = self.train_dataset_loader
                
            for iteration in tqdm(range(N)):
                batch = loader.__next__()

                images = batch['images']
                labels = batch['labels']
                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)
                superpixels = batch['spx'].to(self.device)
                spmasks = batch['spmask'].to(self.device)
                precise_targets = batch['target'].to(self.device)

                outputs = self.net(images)

                N, C, H, W = outputs.shape
                outputs = F.softmax(outputs, dim=1) ### N x C x H x W
                outputs = outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
                superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
                spmasks = spmasks.reshape(N, -1) ### N x HW
                empty_trg_mask = torch.any(labels[..., :-1], dim=2).bool() ### N x self.num_superpixel
                gt_targets = precise_targets.reshape(N, -1)

                for i in range(N):
                    '''
                    outputs[i] ### HW x C
                    superpixels[i] ### HW x 1
                    spmasks[i] ### HW x 1
                    gt_targets[i] ### HW x 1
                    '''

                    ### filtered outputs
                    valid_mask = spmasks[i]
                    if not torch.any(valid_mask):
                        continue
                    valid_output = outputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
                    valid_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID
                    valid_gt_target = gt_targets[i][valid_mask]

                    _, idx_sup_mxpool = scatter_max(valid_output, valid_superpixel, dim=0, dim_size=self.args.nseg) ### self.num_superpixel x C
                    valid_idx_sup_mxpool = idx_sup_mxpool[idx_sup_mxpool[:,0] < valid_output.shape[0]] ### nvalidseg x C : index of max pixel for each class
                    trg_sup_mxpool = labels[i, :, :-1] ### self.num_superpixel x C
                    valid_trg_sup_mxpool = trg_sup_mxpool[idx_sup_mxpool[:,0] < valid_output.shape[0]] ### nvalidseg x C : multi-hot label

                    gt_cls = valid_gt_target[valid_idx_sup_mxpool] ### nvalidseg x C (without dummy rows)
                    idxpred = valid_trg_sup_mxpool.nonzero(as_tuple=True)
                    gt = gt_cls[idxpred]
                    pred = idxpred[1]
                    
                    ncorr_total += (gt == pred).sum().item()
                    n_total += pred.shape[0]
                    for i in range(pred.shape[0]):
                        ncorr_cls[gt[i]] += (gt == pred)[i].item()
                        n_cls[gt[i]] += 1

            acc_total = ncorr_total / n_total
            acc_cls = ncorr_cls / n_cls
            acc_table_str = ",".join([str(acc) for acc in acc_cls.tolist()])
            print("[AL {}-round]: evaluation\n{},{}".format(selection_iter, acc_total, acc_table_str), flush=True)