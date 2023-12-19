import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch_scatter import scatter

from trainer import active_joint_multi
from models import get_model
from utils.loss import GroupMultiLabelCE, MultiChoiceCE
from utils.miou import MeanIoU
from utils.miou_evalignore import IoUIgnore
r"""
Additionally predict undefined (ignore) class within cityscapes
"""

class MultiChoiceCE_(MultiChoiceCE):
    def __init__(self, num_class, temperature=1.0, reduction='mean'):
        super().__init__(num_class, temperature, reduction)

    def forward(self, inputs, targets, superpixels, spmasks):
        ''' inputs:  N x C x H x W
            targets: N x self.num_superpiexl x C+1
            superpixels: N x H x W
            spmasks: N x H x W
        '''

        N, C, H, W = inputs.shape
        inputs = inputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        outputs = F.softmax(inputs / self.temp, dim=2) ### N x HW x C
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW: binary mask indicating current selected spxs
        if self.reduction == 'none':
            pixel_loss = torch.zeros_like(spmasks, dtype=torch.float)
        loss = 0
        num_valid = 1

        for i in range(N):
            '''
            outputs[i] ### HW x C
            superpixels[i] ### HW x 1
            spmasks[i] ### HW x 1
            '''
            ### filtered outputs
            valid_mask = spmasks[i] ### HW
            if not torch.any(valid_mask): ### 더 이상 뽑을게 없는 경우
                continue
            valid_output = outputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID

            trg_sup = targets[i] ### self.num_superpixel x C: multi-hot annotation
            trg_pixel = trg_sup[valid_superpixel.squeeze(dim=1)].detach() ### HW' x C : pixel-wise multi-hot annotation
            
            ### filter out empty target
            empty_trg_mask = torch.any(trg_pixel, dim=1).bool() ### HW'
            valid_output = valid_output[empty_trg_mask]
            trg_pixel = trg_pixel[empty_trg_mask]
            
            pos_pred = (valid_output * trg_pixel).sum(dim=1)
            num_valid += pos_pred.shape[0]
            if self.reduction == 'mean':
                loss += -torch.log(pos_pred + self.eps).sum()
            elif self.reduction == 'none':
                new_valid_mask = valid_mask.clone()
                new_valid_mask[valid_mask] = empty_trg_mask
                pixel_loss[i, new_valid_mask] = -torch.log(pos_pred + self.eps)

        if self.reduction == 'mean':
            return loss / num_valid
        elif self.reduction == 'none':
            return pixel_loss
        else:
            NotImplementedError
class GroupMultiLabelCE_(GroupMultiLabelCE):
    def __init__(self, args, num_class, num_superpixel, temperature=1.0, reduction='mean'):
        super().__init__(args, num_class, num_superpixel, temperature, reduction)

    def forward(self, inputs, targets, superpixels, spmasks):
        ''' inputs: NxCxHxW
            targets: N x self.num_superpixel x C+1
            superpixels: NxHxW
            spmasks: NxHxW
            
            Apply max operation over predicted probabilities for each multi-hot label within the superpixel, and apply CE loss​.
            '''
        N, C, H, W = inputs.shape
        outputs = F.softmax(inputs / self.temp, dim=1) ### N x C x H x W
        outputs = outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW
        empty_trg_mask = torch.any(targets, dim=2).bool() ### N x self.num_superpixel
        loss = 0
        num_valid = 1

        for i in range(N):
            '''
            outputs[i] ### HW x C
            superpixels[i] ### HW x 1
            spmasks[i] ### HW x 1
            '''

            ### filtered outputs
            valid_mask = spmasks[i]
            if not torch.any(valid_mask):
                continue
            valid_output = outputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID

            out_sup_mxpool = scatter(valid_output, valid_superpixel, dim=0, reduce='max', dim_size=self.num_superpixel)
                ### self.num_superpixel x C : sp 영역 내 class 별 max predicted prob, invalid superpixel 은 모두 0 으로 채워짐.
            trg_sup_mxpool = targets[i] ### self.num_superpixel x C: multi-hot annotation
            
            out_sup_mxpool = out_sup_mxpool[empty_trg_mask[i]]
            trg_sup_mxpool = trg_sup_mxpool[empty_trg_mask[i]]

            top_one_preds = out_sup_mxpool * trg_sup_mxpool ### self.num_superpixel x C: 존재하는 multi-hot 으로 filtering

            top_one_preds_nonzero = top_one_preds[top_one_preds.nonzero(as_tuple=True)] ### 해당 value indexing
            num_valid += top_one_preds_nonzero.shape[0] ### valid pixel 개수 측정

            loss += -torch.log(top_one_preds_nonzero + self.eps).sum()

        if self.reduction == 'mean':
            return loss / num_valid
        elif self.reduction == 'none':
            return loss, num_valid
        else:
            raise NotImplementedError

class ActiveTrainer(active_joint_multi.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)

    def get_criterion(self):
        ''' Define criterion '''
        self.group_multi_loss = GroupMultiLabelCE_(args=self.args, num_class=self.num_classes, num_superpixel= self.args.nseg, temperature=self.args.group_ce_temp)
        self.multi_pos_loss = MultiChoiceCE_(num_class=self.num_classes, temperature=self.args.multi_ce_temp)

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