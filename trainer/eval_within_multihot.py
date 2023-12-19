import torch
import numpy as np
import os
from tqdm import tqdm, trange

from dataloader import get_dataset
from dataloader.utils import DataProvider
from trainer.base import BaseTrainer
from models import get_model, freeze_bn
from utils.miou_evalignore import IoUIgnore
from utils.miou import MeanIoU


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
        eval_dataset = active_set.trg_label_dataset
        r'''
        - unselected/dominant label 모두 ignore_class (255) 로 처리
        - multi-hot label 내부의 ignore_class 들은 모두 extra_label (=19) 부여
        '''
        eval_dataset.im_idx = sorted(eval_dataset.im_idx)
        self.eval_dataset_loader = DataProvider(dataset=eval_dataset,
                                                batch_size=args.val_batch_size,
                                                shuffle=False,
                                                num_workers=8,
                                                pin_memory=True,
                                                drop_last=False)
        
        miou, iou_table_str  = self.inference(loader=self.eval_dataset_loader, prefix='evaluation')

        ''' file logging '''
        self.logger.info('[Evaluation Result]')
        self.logger.info('%s' % (iou_table_str))
        self.logger.info('Current eval miou is %.3f %%' % (miou))

        return iou_table_str

    def inference(self, loader, prefix=''):
        iou_helper = MeanIoU(self.num_classes + 1, self.args.ignore_idx)
        iou_helper._before_epoch()
        N = loader.__len__()

        ### model forward
        self.net.eval()
        with torch.no_grad():
            for iteration in trange(N):
                batch = loader.__next__()
                images = batch['images'].to(self.device, dtype=torch.float32)
                labels = batch['labels'].to(self.device, dtype=torch.long)
                superpixels = batch['spx'].to(self.device)
                spmasks = batch['spmask'].to(self.device)
                targets = batch['target'].to(self.device)

                outputs = self.net(images).detach()

                r''' Generate pseudo label within candidate label set '''
                pseudo_label = self.top_pseudo_label_generation(labels, outputs, targets, spmasks, superpixels)

                output_dict = {
                    'outputs': pseudo_label,
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
    
    def top_pseudo_label_generation(self, labels, inputs, targets, spmasks, superpixels):
        r'''
        Args::
            inputs: N x C x H x W
            targets: N x self.num_superpiexl x C
            spmasks: N x H x W
            superpixels: N x H x W
            superpixel_smalls: N x H x W
            
            Apply max operation over predicted probabilities for each multi-hot label within the superpixel,
            and highlight selected top-1 pixel with its corresponding labels
            
        return::
            pseudo_label (torch.Tensor): pseudo label map to be evaluated
                                         N x H x W
            '''

        N, C, H, W = inputs.shape
        outputs = inputs
        outputs = outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW
        nn_plbl = torch.ones_like(labels) * 255 ### N x H x W
        nn_plbl = nn_plbl.reshape(N, -1)

        for i in range(N):
            '''
            outputs[i] : HW x C
            superpixels[i] : HW x 1
            superpixel_smalls[i] : HW x 1
            targets[i] : self.num_superpiexl x C
            spmasks[i] : HW
            '''
            multi_hot_target = targets[i] ### self.num_superpixel x C

            r''' filtered outputs '''
            valid_mask = spmasks[i] ### HW
            if not torch.any(valid_mask): ### valid pixel 이 하나도 없으면 loss 계산 X
                continue #TODO
            valid_output = outputs[i][valid_mask] ### HW' x C
            vpx_superpixel = superpixels[i][valid_mask] ### HW' x 1

            trg_pixel = multi_hot_target[vpx_superpixel.squeeze(dim=1)].detach() ### HW' x C : pixel-wise multi-hot annotation

            output_within_candidate = (valid_output * trg_pixel)
            plbl_win_candidtae = output_within_candidate.max(dim=1)[1]

            r''' Index conversion (valid pixel -> pixel) '''
            validex_to_pixdex = valid_mask.nonzero().squeeze(dim=1)
            nn_plbl[i, validex_to_pixdex] = plbl_win_candidtae

        nn_plbl = nn_plbl.reshape(N, H, W)
        
        return nn_plbl