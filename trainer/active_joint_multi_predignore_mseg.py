import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch_scatter import scatter, scatter_max, scatter_mul, scatter_sum

from trainer import active_joint_multi_predignore
from models import freeze_bn
from utils.loss import GroupMultiLabelCE, MultiChoiceCE

class MultisegGroupMultiLabelCE(GroupMultiLabelCE):
    def __init__(self, args, num_class, num_superpixel, temperature=1.0, reduction='mean'):
        super().__init__(args, num_class, num_superpixel, temperature=1.0, reduction='mean')
        self.nseg_list = np.array(sorted([i for i in self.args.nseg_list]))

    def forward(self, inputs, targets, superpixels, spmasks, nseg_lbl):
        ''' inputs: NxCxHxW
            targets: list of list ex) [[128 x C+1, 512 x C+1, 2048 x C+1, 8192 x C+1], [], ...]
            superpixels: list of nseg_list x H x W
            spmasks: list of nseg_list x H x W
            
            Apply max operation over predicted probabilities for each multi-hot label within the superpixel, and apply CE loss​.
            '''

        N, C, H, W = inputs.shape
        outputs = F.softmax(inputs / self.temp, dim=1) ### N x C+1 x H x W
        outputs = outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C+1

        mseg_nseg = nseg_lbl.sum(dim=1).tolist()
        superpixels = [i.reshape(nseg, -1) for nseg, i in zip(mseg_nseg, superpixels)] # list of nseg_list x HW
        spmasks = [i.reshape(nseg, -1) for nseg, i in zip(mseg_nseg, spmasks)] # list of nseg_list x HW
        loss = 0
        num_valid = 1

        for b_iter in range(N):
            '''
            outputs[i] ### HW x C+1
            superpixels[i] ### nseg_list x HW x 1
            spmasks[i] ### nseg_list x HW x 1
            '''
            for sdx, nseg in enumerate(self.nseg_list[nseg_lbl[b_iter]]):
                valid_mask = spmasks[b_iter][sdx]
                if not torch.any(valid_mask):
                    continue
                valid_output = outputs[b_iter][valid_mask]
                valid_superpixel = superpixels[b_iter][sdx][valid_mask]
                out_sup_mxpool = scatter(valid_output, valid_superpixel, dim=0, reduce='max', dim_size=nseg)
                ### ㄴ nseg x C+1 : sp 영역 내 class 별 max predicted prob, invalid superpixel 은 모두 0 으로 채워짐.
                trg_sup_mxpool = targets[b_iter][sdx] ### nseg x C+1

                top_one_preds = out_sup_mxpool * trg_sup_mxpool
                top_one_preds_nonzero = top_one_preds[top_one_preds.nonzero(as_tuple=True)] ### 해당 value indexing
                num_valid += top_one_preds_nonzero.shape[0] ### valid pixel 개수 측정

                loss += -torch.log(top_one_preds_nonzero + self.eps).sum()

        if self.reduction == 'mean':
            return loss / num_valid
        elif self.reduction == 'none':
            return loss, num_valid
        else:
            raise NotImplementedError

class MultisegMultiChoiceCE(MultiChoiceCE):
    def __init__(self, args, num_class, temperature=1.0, reduction='mean'):
        super().__init__(num_class, temperature, reduction)
        self.args = args
        self.nseg_list = np.array(sorted([i for i in self.args.nseg_list]))

    def forward(self, inputs, targets, superpixels, spmasks, nseg_lbl):
        ''' inputs: NxCxHxW
            targets: list of list ex) [[128 x C+1, 512 x C+1, 2048 x C+1, 8192 x C+1], [], ...]
            superpixels: list of nseg_list x H x W
            spmasks: list of nseg_list x H x W
            '''

        N, C, H, W = inputs.shape
        inputs = inputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C+1
        outputs = F.softmax(inputs / self.temp, dim=2) ### N x HW x C+1

        mseg_nseg = nseg_lbl.sum(dim=1).tolist()
        superpixels = [i.reshape(nseg, -1) for nseg, i in zip(mseg_nseg, superpixels)] # list of nseg_list x HW
        spmasks = [i.reshape(nseg, -1) for nseg, i in zip(mseg_nseg, spmasks)] # list of nseg_list x HW
        loss = 0
        num_valid = 1

        for b_iter in range(N):
            '''
            outputs[i] ### HW x C+1
            superpixels[i] ### nseg_list x HW x 1
            spmasks[i] ### nseg_list x HW x 1
            '''
            for sdx, nseg in enumerate(self.nseg_list[nseg_lbl[b_iter]]):
                valid_mask = spmasks[b_iter][sdx]
                if not torch.any(valid_mask):
                    continue
                valid_output = outputs[b_iter][valid_mask]
                valid_superpixel = superpixels[b_iter][sdx][valid_mask]

                trg_sup = targets[b_iter][sdx] ### self.num_superpixel x C+1: multi-hot annotation
                trg_pixel = trg_sup[valid_superpixel].detach() ### HW' x C : pixel-wise multi-hot annotation
                
                pos_pred = (valid_output * trg_pixel).sum(dim=1)
                num_valid += pos_pred.shape[0]
                loss += -torch.log(pos_pred + self.eps).sum()

        if self.reduction == 'mean':
            return loss / num_valid
        elif self.reduction == 'none':
            return loss, num_valid
        else:
            NotImplementedError

class ActiveTrainer(active_joint_multi_predignore.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)

    def get_criterion(self):
        ''' Define criterion '''
        self.group_multi_loss = MultisegGroupMultiLabelCE(args=self.args, num_class=self.num_classes, num_superpixel= self.args.nseg, temperature=self.args.group_ce_temp)
        self.multi_pos_loss = MultisegMultiChoiceCE(args=self.args, num_class=self.num_classes, temperature=self.args.multi_ce_temp)

    def train_impl(self, total_itrs, val_period):
        self.net.train()
        if self.args.freeze_bn is True:
            freeze_bn(self.net)
        pbar = tqdm(range(total_itrs), ncols=150)

        for iteration in pbar:
            ### Loading
            batch = self.train_dataset_loader.__next__()
            images = batch['images'].to(self.device)
            labels = batch['mseg_labels'] ### list of list
            images = images.to(self.device, dtype=torch.float32)
            labels = [[j.to(self.device, dtype=self.target_dtype) for j in i] for i in labels]
            superpixels = [i.to(self.device) for i in batch['mseg_spx']]
            spmasks = [i.to(self.device) for i in batch['mseg_spmask']]
            nseg_lbl = batch['nseg_list']

            ### Forward
            self.optimizer.zero_grad()
            preds = self.net(images)

            ### Loss
            group_loss = self.group_multi_loss(preds, labels, superpixels, spmasks, nseg_lbl)
            pos_loss = self.multi_pos_loss(preds, labels, superpixels, spmasks, nseg_lbl)
            loss = self.args.coeff * pos_loss + group_loss

            ### Update
            if self.check_loss_sanity(loss):
                loss.backward()
                self.optimizer.step()

            ### Scheduler
            if self.args.scheduler == 'poly':
                self.scheduler.step()

            ### Logging
            if self.check_loss_sanity(loss):
                self.am.add({'train-loss': loss.detach().cpu().item()})
            if self.check_loss_sanity(pos_loss):
                self.am.add({'pos-loss': pos_loss.detach().cpu().item()})
            if self.check_loss_sanity(group_loss):
                self.am.add({'group-loss': group_loss.detach().cpu().item()})

            if iteration % self.args.log_period == (self.args.log_period - 1):
                pbar.set_description('[AL {}-round] (step{}): Loss {:.4f} Session {}'.format(
                    self.selection_iter,
                    iteration,
                    self.am.get('train-loss'),
                    self.args.session_name
                ))                
                global_step = iteration + (total_itrs * (self.selection_iter - 1))
                lr_f = self.optimizer.param_groups[-1]['lr']
                wlog_train = {'learning-rate cls': lr_f}
                wlog_train.update({k:self.am.pop(k) for k,v in self.am.get_whole_data().items()})
                self.args.wandb.log(wlog_train, step=global_step)

            ### Validation
            if iteration % val_period == (val_period - 1) and iteration > self.args.val_start:
                self.logger.info('**** EVAL ITERATION %06d ****' % (iteration))
                self.validate(trainiter=iteration)
                self.net.train()
                if self.args.freeze_bn is True:
                    freeze_bn(self.net)