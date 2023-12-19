from email.headerregistry import Group
from tokenize import group
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_scatter import scatter, scatter_max, scatter_mul, scatter_sum
from torch.distributions.gumbel import Gumbel


class MyCrossEntropyLoss(nn.CrossEntropyLoss):
    r"""
    Cross entropy with temperature term
    """
    def __init__(self, ignore_index, reduction='mean', temperature=1.0):
        super().__init__(ignore_index=ignore_index, reduction=reduction)
        self.temperature = temperature

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input / self.temperature

        return super().forward(input, target)
        
class JointHierarchyLoss(nn.Module):
    def __init__(self, hierarchy_multi_label_ce, multi_pos_loss, reduction='mean'):
        super().__init__()
        self.hierarchy_multi_label_ce = hierarchy_multi_label_ce
        self.multi_pos_loss = multi_pos_loss
        self.reduction = reduction

    def forward(self, inputs, targets, superpixels, superpixel_smalls):
        if self.reduction == 'mean':
            loss_group = self.hierarchy_multi_label_ce(inputs, targets, superpixels, superpixel_smalls)
            loss_pos = self.multi_pos_loss(inputs, targets)

            return loss_group, loss_pos
        elif self.reduction == 'none':
            loss_group, num_valid = self.hierarchy_multi_label_ce(inputs, targets, superpixels, superpixel_smalls)
            loss_pos = self.multi_pos_loss(inputs, targets) ### (N, )

            return loss_group, loss_pos, num_valid
        else:
            raise NotImplementedError            

class JointMultiLoss(nn.Module):
    def __init__(self, group_multi_loss, multi_pos_loss, reduction='mean'):
        super().__init__()
        self.group_multi_loss = group_multi_loss
        self.multi_pos_loss = multi_pos_loss
        self.reduction = reduction
    
    def forward(self, inputs, targets, superpixels, spmasks):
        if self.reduction == 'mean':
            loss_group = self.group_multi_loss(inputs, targets, superpixels, spmasks)
            loss_pos = self.multi_pos_loss(inputs, targets, superpixels, spmasks)

            return loss_group, loss_pos
        elif self.reduction == 'none':
            loss_group, num_valid = self.group_multi_loss(inputs, targets, superpixels, spmasks)
            loss_pos = self.multi_pos_loss(inputs, targets, spmasks)

            return loss_group, loss_pos, num_valid
        else:
            raise NotImplementedError

class JointRcceAsym(nn.Module):
    def __init__(self, group_multi_loss, rc_asym_loss, reduction='mean'):
        super().__init__()
        self.group_multi_loss = group_multi_loss
        self.rc_asym_loss = rc_asym_loss
        self.reduction = reduction
    
    def forward(self, inputs, inputs2, targets, superpixels):
        if self.reduction == 'mean':
            loss_group = self.group_multi_loss(inputs, targets, superpixels)
            loss_pos = self.rc_asym_loss(inputs, inputs2, targets)

            return loss_group, loss_pos
        else:
            raise NotImplementedError

class GroupMultiLabelCE(nn.Module):
    def __init__(self, args, num_class, num_superpixel, temperature=1.0, reduction='mean'):
        super().__init__()
        self.args = args
        self.num_class = num_class
        self.num_superpixel = num_superpixel
        self.eps = 1e-8
        self.temp = temperature
        self.reduction = reduction

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
        empty_trg_mask = torch.any(targets[..., :-1], dim=2).bool() ### N x self.num_superpixel
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
            trg_sup_mxpool = targets[i][..., :-1] ### self.num_superpixel x C: multi-hot annotation
            
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

class HierGroupMultiLabelCE(GroupMultiLabelCE):
    def __init__(self, args, num_class, num_superpixel, only_single, gumbel_scale, reduction='mean', temperature=1.0):
        super().__init__(args, num_class, num_superpixel, temperature=1.0)
        self.num_small_superpixel = args.small_nseg
        self.only_single = only_single
        self.gumbel_scale = gumbel_scale
        if gumbel_scale != -1:
            self.gumbel = Gumbel(loc=torch.tensor(0.0).to(torch.device('cuda:0')), scale=torch.tensor(gumbel_scale).to(torch.device('cuda:0')))
        self.reduction = reduction

    def forward(self, inputs, targets, spmasks, superpixels, superpixel_smalls):
        ''' inputs: N x C x H x W
            targets: N x self.num_superpiexl x C+1
            spmasks: N x H x W
            superpixels: N x H x W
            superpixel_smalls: N x H x W
            
            Apply max operation over predicted probabilities for each multi-hot label within the superpixel, and apply CE loss​.
            '''
        N, C, H, W = inputs.shape
        outputs = F.softmax(inputs / self.temp, dim=1) ### N x C x H x W
        outputs = outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        superpixel_smalls = superpixel_smalls.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW
        empty_trg_mask = torch.any(targets[..., :-1], dim=2).bool() ### N x self.num_superpixel
        loss = 0
        num_valid = 1

        if self.gumbel_scale != -1:
            inputs = inputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C

        for i in range(N):
            '''
            outputs[i] : HW x C
            superpixels[i] : HW x 1
            superpixel_smalls[i] : HW x 1
            targets[i] : self.num_superpiexl x C+1
            spmasks[i] : HW
            '''

            ### filtered outputs
            valid_mask = spmasks[i] ### HW
            if not torch.any(valid_mask): ### valid pixel 이 하나도 없으면 loss 계산 X
                continue
            valid_output = outputs[i][valid_mask] ### HW' x C
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1
            valid_superpixel_small = superpixel_smalls[i][valid_mask] ### HW' x 1

            ### get max pixel for each class within superpixel
            if self.gumbel_scale != -1:
                ### use logit for gumbel max sampling
                valid_input = inputs[i][valid_mask] ### HW' x C
                _, idx_sup_mxpool = scatter_max(valid_input + self.gumbel.sample(valid_output.shape), valid_superpixel, dim=0, dim_size=self.num_superpixel) ### self.num_superpixel x C
            else:
                _, idx_sup_mxpool = scatter_max(valid_output, valid_superpixel, dim=0, dim_size=self.num_superpixel) ### self.num_superpixel x C
           
            trg_sup_mxpool = targets[i, :, :-1] ### self.num_superpixel x C
            
            ### For non-existing superpixels: value == 0.0, index == (max_index + 1)
            ### So we can filter out empty idx_sup_mxpool
            valid_idx_sup_mxpool = idx_sup_mxpool[idx_sup_mxpool[:,0] < valid_output.shape[0]] ### nvalidseg x C : index of max pixel for each class
            valid_trg_sup_mxpool = trg_sup_mxpool[idx_sup_mxpool[:,0] < valid_output.shape[0]] ### nvalidseg x C : multi-hot label

            ### TODO filter empty target
            if self.only_single:
                single_class_mask = (1 < valid_trg_sup_mxpool.sum(dim=1))
                valid_idx_sup_mxpool = valid_idx_sup_mxpool[single_class_mask]
                valid_trg_sup_mxpool = valid_trg_sup_mxpool[single_class_mask]

            ### get according (superpixel, class index) pair using max pixel calculated above
            selected_small_superpixels = valid_superpixel_small.squeeze(dim=1)[valid_idx_sup_mxpool] ### nvalidseg x C (without dummy rows)
            valid_trg_sup_mxpool_idx = valid_trg_sup_mxpool.nonzero(as_tuple=True)
            selected_small_superpixels = selected_small_superpixels[valid_trg_sup_mxpool_idx] ### nvalidseg_small : only selected classes within multi-hot labels
            selected_small_classes = valid_trg_sup_mxpool_idx[1]

            ### aggreate prediction for selected_small_superpixels
            size_placeholder = torch.ones_like(valid_superpixel) ### HW' x 1
            valid_likelihood = -torch.log(valid_output + self.eps) ### negative log softmax
            out_small_sup_sumpool = scatter(valid_likelihood, valid_superpixel_small, dim=0, reduce='sum', dim_size=self.num_small_superpixel) ### (self.num_small_superpixel x C): sum-likelikhood from non-ignore superpixels
            size_sup_sumpool = scatter(size_placeholder.int(), valid_superpixel_small, dim=0, reduce='sum', dim_size=self.num_small_superpixel) ### (self.num_small_superpixel x C): get each size of the non-ignore superpixels
            value = out_small_sup_sumpool[selected_small_superpixels, selected_small_classes] ### extract selected (superpixel_id, class) pair
            size = size_sup_sumpool.squeeze()[selected_small_superpixels]

            num_valid += size.sum()
            loss += value.sum()

        if self.reduction == 'mean':
            return loss / num_valid
        elif self.reduction == 'none':
            return loss, num_valid
        else:
            raise NotImplementedError

class WeightAsyncHierGroupMultiLabelCE(HierGroupMultiLabelCE):
    def __init__(self, args, num_class, num_superpixel, only_single, gumbel_scale, reduction='mean', temperature=1.0, weight_reduce='max'):
        super().__init__(args, num_class, num_superpixel, only_single, gumbel_scale, reduction, temperature)
        self.weight_reduce = weight_reduce

    def forward(self, inputs, inputs_weak, targets, spmasks, spmasks_weak, superpixels, superpixels_weak, superpixel_smalls, spx_smalls_weak):
        ''' inputs: N x C x H x W
            targets: N x self.num_superpiexl x C+1
            spmasks: N x H x W
            superpixels: N x H x W

            inputs_weak: N x H_o x W_o
            spmasks_weak: N x H_o x W_o
            superpixels_weak: N x H_o x W_o
            spx_smalls_weak: N x H_o x W_o
            Apply max operation over predicted probabilities for each multi-hot label within the superpixel, and apply CE loss​.
            '''

        ### weak augmented data
        N, C, H_o, W_o = inputs_weak.shape
        outputs_weak = F.softmax(inputs_weak / self.temp, dim=1) ### N x C x H_o x W_o
        outputs_weak = outputs_weak.permute(0,2,3,1).reshape(N, -1, C) ### N x H_oW_o x C
        superpixels_weak = superpixels_weak.reshape(N, -1, 1) ### N x H_oW_o x 1
        spx_smalls_weak = spx_smalls_weak.reshape(N, -1, 1) ### N x H_oW_o x 1
        spmasks_weak = spmasks_weak.reshape(N, -1) ### N x H_oW_o

        ### strong augmented data
        _, _, H, W = inputs.shape
        outputs = F.softmax(inputs / self.temp, dim=1) ### N x C x H x W
        outputs = outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        superpixel_smalls = superpixel_smalls.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW        

        ###
        loss = 0
        num_valid = 1

        for i in range(N):
            '''
            outputs_weak[i] : H_oW_o x C
            superpixels_weak[i] : H_oW_o x 1
            spx_smalls_weak[i] : H_oW_o x 1
            targets[i] : self.num_superpiexl x C+1
            spmasks_weak[i] : H_oW_o
            '''

            ''' get small (superpixel_id, class) tuple by class-wise max operation '''
            ### filtered outputs for trainin
            valid_mask = spmasks[i] ### HW
            if not torch.any(valid_mask): ### valid pixel 이 하나도 없으면 loss 계산 X
                continue
            valid_output = outputs[i][valid_mask] ### HW' x C
            valid_spx_small = superpixel_smalls[i][valid_mask] ### HW' x 1

            ### filtered outputs with weak augmentation
            valid_mask_weak = spmasks_weak[i] ### H_oW_o
            valid_output_weak = outputs_weak[i][valid_mask_weak] ### H_oW_o' x C
            valid_superpixel_weak = superpixels_weak[i][valid_mask_weak] ### H_oW_o' x 1
            valid_spx_small_weak = spx_smalls_weak[i][valid_mask_weak] ### H_oW_o' x 1

            ### get max pixel for each class within superpixel
            _, idx_sup_mxpool_weak = scatter_max(valid_output_weak, valid_superpixel_weak, dim=0, dim_size=self.num_superpixel) ### self.num_superpixel x C
            trg_sup_mxpool_weak = targets[i, :, :-1] ### self.num_superpixel x C
            
            ### For non-existing superpixels: value == 0.0, index == (max_index + 1)
            ### So we can filter out empty idx_sup_mxpool using index value of first class (row share the same out-of-bound value)
            valid_idx_sup_mxpool_weak = idx_sup_mxpool_weak[idx_sup_mxpool_weak[:,0] < valid_output_weak.shape[0]] ### nvalid_superpixel x C : index of max pixel for each class
            valid_trg_sup_mxpool_weak = trg_sup_mxpool_weak[idx_sup_mxpool_weak[:,0] < valid_output_weak.shape[0]] ### nvalid_superpixel x C : multi-hot label

            ### get according (superpixel, class index) pair using max pixel calculated above
            ### index one dimentional tensor with 2-D tensor => tensor with according values
            selected_small_superpixels = valid_spx_small_weak.squeeze(dim=1)[valid_idx_sup_mxpool_weak] ### nvalid_superpixel x C: selected small superpixel ids
            valid_trg_sup_mxpool_idx = valid_trg_sup_mxpool_weak.nonzero(as_tuple=True) ### tuple with size equals to valid class (=1) with in 'nvalid_superpixel x C' target
            selected_small_superpixels = selected_small_superpixels[valid_trg_sup_mxpool_idx] ### valid small superpixel_ids (list tensor)
            selected_small_classes = valid_trg_sup_mxpool_idx[1] ### class indices with the size same as 'selected_small_superpixels' (list tensor)

            ''' apply loss for (superpixel_id, class) small superpixel on the strong augmented outputs '''
            ### aggreate prediction from strong augmented output using selected_small_superpixels
            valid_likelihood = -torch.log(valid_output + self.eps) ### negative log softmax
            out_small_sup_sumpool = scatter(valid_likelihood, valid_spx_small, dim=0, reduce='sum', dim_size=self.num_small_superpixel) ### (self.num_small_superpixel x C): sum-likelikhood from non-ignore superpixels
            value = out_small_sup_sumpool[selected_small_superpixels, selected_small_classes] ### extract selected (superpixel_id, class) pair

            ### calculate top-1 predicted probability for each small superpixel for weighting
            sp_top1_pred = scatter(valid_output_weak, valid_spx_small_weak, dim=0, reduce=self.weight_reduce, dim_size=self.num_small_superpixel) ### (self.num_small_superpixel x C): mean predicted probability
            weight = sp_top1_pred[selected_small_superpixels, selected_small_classes] ### extract selected (superpixel_id, class) pair
            value = value * weight.detach()

            ### size calculation normalization
            size_placeholder = torch.ones_like(valid_spx_small) ### HW' x 1
            size_sup_sumpool = scatter(size_placeholder.int(), valid_spx_small, dim=0, reduce='sum', dim_size=self.num_small_superpixel) ### (self.num_small_superpixel x C): get each size of the non-ignore superpixels
            size = size_sup_sumpool.squeeze(dim=1)[selected_small_superpixels]
            ### remove size of excluded small superpixels in the strong augmented version
            size = size[value.nonzero()]

            num_valid += size.sum()
            loss += value.sum()

        if self.reduction == 'mean':
            return loss / num_valid
        elif self.reduction == 'none':
            return loss, num_valid
        else:
            raise NotImplementedError

class AsyncHierGroupMultiLabelCE(HierGroupMultiLabelCE):
    def __init__(self, args, num_class, num_superpixel, only_single, gumbel_scale, reduction='mean', temperature=1.0):
        super().__init__(args, num_class, num_superpixel, only_single, gumbel_scale, reduction, temperature)

    def forward(self, inputs, inputs_weak, targets, spmasks, spmasks_weak, superpixels, superpixels_weak, superpixel_smalls, spx_smalls_weak):
        ''' inputs: N x C x H x W
            targets: N x self.num_superpiexl x C+1
            spmasks: N x H x W
            superpixels: N x H x W

            inputs_weak: N x H_o x W_o
            spmasks_weak: N x H_o x W_o
            superpixels_weak: N x H_o x W_o
            spx_smalls_weak: N x H_o x W_o
            Apply max operation over predicted probabilities for each multi-hot label within the superpixel, and apply CE loss​.
            '''

        ### weak augmented data
        N, C, H_o, W_o = inputs_weak.shape
        outputs_weak = F.softmax(inputs_weak / self.temp, dim=1) ### N x C x H_o x W_o
        outputs_weak = outputs_weak.permute(0,2,3,1).reshape(N, -1, C) ### N x H_oW_o x C
        superpixels_weak = superpixels_weak.reshape(N, -1, 1) ### N x H_oW_o x 1
        spx_smalls_weak = spx_smalls_weak.reshape(N, -1, 1) ### N x H_oW_o x 1
        spmasks_weak = spmasks_weak.reshape(N, -1) ### N x H_oW_o

        ### strong augmented data
        _, _, H, W = inputs.shape
        outputs = F.softmax(inputs / self.temp, dim=1) ### N x C x H x W
        outputs = outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        superpixel_smalls = superpixel_smalls.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW        

        ###
        loss = 0
        num_valid = 1

        for i in range(N):
            '''
            outputs_weak[i] : H_oW_o x C
            superpixels_weak[i] : H_oW_o x 1
            spx_smalls_weak[i] : H_oW_o x 1
            targets[i] : self.num_superpiexl x C+1
            spmasks_weak[i] : H_oW_o
            '''

            ''' get small (superpixel_id, class) tuple by class-wise max operation '''
            ### filtered outputs for trainin
            valid_mask = spmasks[i] ### HW
            if not torch.any(valid_mask): ### valid pixel 이 하나도 없으면 loss 계산 X
                continue
            valid_output = outputs[i][valid_mask] ### HW' x C
            valid_spx_small = superpixel_smalls[i][valid_mask] ### HW' x 1

            ### filtered outputs with weak augmentation
            valid_mask_weak = spmasks_weak[i] ### H_oW_o
            valid_output_weak = outputs_weak[i][valid_mask_weak] ### H_oW_o' x C
            valid_superpixel_weak = superpixels_weak[i][valid_mask_weak] ### H_oW_o' x 1
            valid_spx_small_weak = spx_smalls_weak[i][valid_mask_weak] ### H_oW_o' x 1

            ### get max pixel for each class within superpixel
            _, idx_sup_mxpool_weak = scatter_max(valid_output_weak, valid_superpixel_weak, dim=0, dim_size=self.num_superpixel) ### self.num_superpixel x C
            trg_sup_mxpool_weak = targets[i, :, :-1] ### self.num_superpixel x C
            
            ### For non-existing superpixels: value == 0.0, index == (max_index + 1)
            ### So we can filter out empty idx_sup_mxpool using index value of first class (row share the same out-of-bound value)
            valid_idx_sup_mxpool_weak = idx_sup_mxpool_weak[idx_sup_mxpool_weak[:,0] < valid_output_weak.shape[0]] ### nvalid_superpixel x C : index of max pixel for each class
            valid_trg_sup_mxpool_weak = trg_sup_mxpool_weak[idx_sup_mxpool_weak[:,0] < valid_output_weak.shape[0]] ### nvalid_superpixel x C : multi-hot label

            ### get according (superpixel, class index) pair using max pixel calculated above
            ### index one dimentional tensor with 2-D tensor => tensor with according values
            selected_small_superpixels = valid_spx_small_weak.squeeze(dim=1)[valid_idx_sup_mxpool_weak] ### nvalid_superpixel x C: selected small superpixel ids
            valid_trg_sup_mxpool_idx = valid_trg_sup_mxpool_weak.nonzero(as_tuple=True) ### tuple with size equals to valid class (=1) with in 'nvalid_superpixel x C' target
            selected_small_superpixels = selected_small_superpixels[valid_trg_sup_mxpool_idx] ### valid small superpixel_ids (list tensor)
            selected_small_classes = valid_trg_sup_mxpool_idx[1] ### class indices with the size same as 'selected_small_superpixels' (list tensor)

            ''' apply loss for (superpixel_id, class) small superpixel on the strong augmented outputs '''
            ### aggreate prediction from strong augmented output using selected_small_superpixels
            valid_likelihood = -torch.log(valid_output + self.eps) ### negative log softmax
            out_small_sup_sumpool = scatter(valid_likelihood, valid_spx_small, dim=0, reduce='sum', dim_size=self.num_small_superpixel) ### (self.num_small_superpixel x C): sum-likelikhood from non-ignore superpixels
            value = out_small_sup_sumpool[selected_small_superpixels, selected_small_classes] ### extract selected (superpixel_id, class) pair

            ### size calculation normalization
            size_placeholder = torch.ones_like(valid_spx_small) ### HW' x 1
            size_sup_sumpool = scatter(size_placeholder.int(), valid_spx_small, dim=0, reduce='sum', dim_size=self.num_small_superpixel) ### (self.num_small_superpixel x C): get each size of the non-ignore superpixels
            size = size_sup_sumpool.squeeze(dim=1)[selected_small_superpixels]
            ### remove size of excluded small superpixels in the strong augmented version
            size = size[value.nonzero()]

            num_valid += size.sum()
            loss += value.sum()

        if self.reduction == 'mean':
            return loss / num_valid
        elif self.reduction == 'none':
            return loss, num_valid
        else:
            raise NotImplementedError

class AugHierGroupMultiLabelCE(HierGroupMultiLabelCE):
    def __init__(self, args, num_class, num_superpixel, only_single, gumbel_scale, reduction='mean', temperature=1.0):
        super().__init__(args, num_class, num_superpixel, only_single, gumbel_scale, reduction, temperature)

    def forward(self, inputs, targets, spmasks, superpixels, superpixel_smalls):
        ''' inputs: N x C x H x W
            targets: N x self.num_superpiexl x C+1
            spmasks: N x H x W
            superpixels: N x H x W
            superpixel_smalls: N x H x W
            
            Apply max operation over predicted probabilities for each multi-hot label within the superpixel, and apply CE loss​.
            '''
        N, C, H, W = inputs.shape
        outputs = F.softmax(inputs / self.temp, dim=1) ### N x C x H x W
        outputs = outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        superpixel_smalls = superpixel_smalls.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW
        empty_trg_mask = torch.any(targets[..., :-1], dim=2).bool() ### N x self.num_superpixel
        loss = 0
        num_valid = 1

        ### save boundary superpixels to future removal
        boundary_values = torch.cat([superpixels[:, :, 0], superpixels[:, 0, :], superpixels[:, -1, :], superpixels[:, :, -1]], dim=1)
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1

        if self.gumbel_scale != -1:
            inputs = inputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C

        for i in range(N):
            '''
            outputs[i] : HW x C
            superpixels[i] : HW x 1
            superpixel_smalls[i] : HW x 1
            targets[i] : self.num_superpiexl x C+1
            spmasks[i] : HW
            '''

            ### filtered outputs
            valid_mask = spmasks[i] ### HW
            if not torch.any(valid_mask):
                continue
            valid_output = outputs[i][valid_mask] ### HW' x C
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1
            valid_superpixel_small = superpixel_smalls[i][valid_mask] ### HW' x 1

            ### get max pixel for each class within superpixel
            if self.gumbel_scale != -1:
                ### use logit for gumbel max sampling
                valid_input = inputs[i][valid_mask] ### HW' x C
                _, idx_sup_mxpool = scatter_max(valid_input + self.gumbel.sample(valid_output.shape), valid_superpixel, dim=0, dim_size=self.num_superpixel) ### self.num_superpixel x C
            else:
                _, idx_sup_mxpool = scatter_max(valid_output, valid_superpixel, dim=0, dim_size=self.num_superpixel) ### self.num_superpixel x C
           
            trg_sup_mxpool = targets[i, :, :-1] ### self.num_superpixel x C

            ### remove bounary superpixel labels
            boundary_sp = torch.unique(boundary_values[i])
            boundary_sp = boundary_sp[boundary_sp != self.num_superpixel]
            trg_sup_mxpool[boundary_sp.tolist()] = 0

            ### For non-existing superpixels: value == 0.0, index == (max_index + 1)
            ### So we can filter out empty idx_sup_mxpool
            valid_idx_sup_mxpool = idx_sup_mxpool[idx_sup_mxpool[:,0] < valid_output.shape[0]] ### nvalidseg x C : index of max pixel for each class
            valid_trg_sup_mxpool = trg_sup_mxpool[idx_sup_mxpool[:,0] < valid_output.shape[0]] ### nvalidseg x C : multi-hot label

            ### TODO filter empty target
            if self.only_single:
                single_class_mask = (1 < valid_trg_sup_mxpool.sum(dim=1))
                valid_idx_sup_mxpool = valid_idx_sup_mxpool[single_class_mask]
                valid_trg_sup_mxpool = valid_trg_sup_mxpool[single_class_mask]

            ### get according (superpixel, class index) pair using max pixel calculated above
            selected_small_superpixels = valid_superpixel_small.squeeze(dim=1)[valid_idx_sup_mxpool] ### nvalidseg x C (without dummy rows)
            valid_trg_sup_mxpool_idx = valid_trg_sup_mxpool.nonzero(as_tuple=True)
            selected_small_superpixels = selected_small_superpixels[valid_trg_sup_mxpool_idx] ### nvalidseg_small : only selected classes within multi-hot labels
            selected_small_classes = valid_trg_sup_mxpool_idx[1]

            ### aggreate prediction for selected_small_superpixels
            size_placeholder = torch.ones_like(valid_superpixel) ### HW' x 1
            valid_likelihood = -torch.log(valid_output + self.eps) ### negative log softmax
            out_small_sup_sumpool = scatter(valid_likelihood, valid_superpixel_small, dim=0, reduce='sum', dim_size=self.num_small_superpixel) ### (self.num_small_superpixel x C): sum-likelikhood from non-ignore superpixels
            size_sup_sumpool = scatter(size_placeholder.int(), valid_superpixel_small, dim=0, reduce='sum', dim_size=self.num_small_superpixel) ### (self.num_small_superpixel x C): get each size of the non-ignore superpixels
            value = out_small_sup_sumpool[selected_small_superpixels, selected_small_classes] ### extract selected (superpixel_id, class) pair
            size = size_sup_sumpool.squeeze()[selected_small_superpixels]

            num_valid += size.sum()
            loss += value.sum()

        if self.reduction == 'mean':
            return loss / num_valid
        elif self.reduction == 'none':
            return loss, num_valid
        else:
            raise NotImplementedError

class MultiChoiceCE(nn.Module):
    def __init__(self, num_class, temperature=1.0, reduction='mean'):
        super().__init__()
        self.num_class = num_class
        self.reduction = reduction
        self.eps = 1e-8
        self.temp = temperature

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
        spmasks = spmasks.reshape(N, -1) ### N x HW
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
            if not torch.any(valid_mask):
                continue
            valid_output = outputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID

            trg_sup = targets[i][..., :-1] ### self.num_superpixel x C: multi-hot annotation
            trg_pixel = trg_sup[valid_superpixel.squeeze(dim=1)].detach() ### HW' x C : pixel-wise multi-hot annotation
            
            ### filter out empty target
            empty_trg_mask = torch.any(trg_pixel, dim=1).bool() ### HW'
            valid_output = valid_output[empty_trg_mask]
            trg_pixel = trg_pixel[empty_trg_mask]
            
            pos_pred = (valid_output * trg_pixel).sum(dim=1)
            num_valid += pos_pred.shape[0]
            loss += -torch.log(pos_pred + self.eps).sum()

        if self.reduction == 'mean':
            return loss / num_valid
        elif self.reduction == 'none':
            return loss, num_valid
        else:
            NotImplementedError

class MultiChoiceEnt(nn.Module):
    def __init__(self, num_class, temperature=1.0, reduction='mean'):
        super().__init__()
        self.num_class = num_class
        self.reduction = reduction
        self.eps = 1e-8
        self.temp = temperature

    def forward(self, inputs, targets, superpixels, spmasks):
        ''' inputs:  N x C x H x W
            targets: N x self.num_superpiexl x C+1
            superpixels: N x H x W
            spmasks: N x H x W
        '''

        N, C, H, W = inputs.shape
        inputs = inputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW
        loss = 0
        num_valid = 1

        for i in range(N):
            '''
            inputs[i] ### HW x C
            superpixels[i] ### HW x 1
            spmasks[i] ### HW x 1
            '''
            ### filtered inputs
            valid_mask = spmasks[i] ### HW
            if not torch.any(valid_mask):
                continue
            valid_intput = inputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID
            trg_sup = targets[i][..., :-1] ### self.num_superpixel x C: multi-hot annotation
            trg_pixel = trg_sup[valid_superpixel.squeeze(dim=1)].detach() ### HW' x C : pixel-wise multi-hot annotation
            
            ### filter out single target
            multi_trg_mask = (1 < trg_pixel.sum(dim=1)) ### HW'
            if not torch.any(multi_trg_mask):
                continue
            valid_input = valid_input[multi_trg_mask] ### HW'' x C
            trg_pixel = trg_pixel[multi_trg_mask] ### HW'' x C
            pos_pred = valid_input * trg_pixel

            ### softmax on candidate label set
            ### -inf insertion (for softmax)
            pos_pred = pos_pred.view(-1) ### HW'' x C
            pos_pred[pos_pred == 0] = float('-inf') ### HW'' x C
            pos_pred = pos_pred.view(-1, trg_pixel.shape[1]) ### HW'' x C
            valid_output = F.softmax(pos_pred / self.temp, dim=1) ### HW'' x C

            ent_output = -torch.sum(valid_output * torch.log(valid_output + self.eps), dim=1) ### HW''
            num_valid += ent_output.shape[0]
            loss += ent_output.sum()

        if self.reduction == 'mean':
            return loss / num_valid
        elif self.reduction == 'none':
            return loss, num_valid
        else:
            NotImplementedError

class RCMultiChoiceCE(MultiChoiceCE):
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
        spmasks = spmasks.reshape(N, -1) ### N x HW
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
            if not torch.any(valid_mask):
                continue
            valid_output = outputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID

            trg_sup = targets[i][..., :-1] ### self.num_superpixel x C: multi-hot annotation
            trg_pixel = trg_sup[valid_superpixel.squeeze(dim=1)].detach() ### HW' x C : pixel-wise multi-hot annotation
            
            ### filter out empty target
            empty_trg_mask = torch.any(trg_pixel, dim=1).bool() ### HW'
            valid_output = valid_output[empty_trg_mask]
            trg_pixel = trg_pixel[empty_trg_mask]
            
            pos_pred = (valid_output * trg_pixel) ### HW' x C: predicted probability for positive group

            with torch.no_grad():
                pos_pred_ = pos_pred.detach()
                pos_weight = pos_pred_ / pos_pred_.sum(dim=1)[..., None] ### (HW' x C): normalized pred as weight

            loss += (pos_weight * -torch.log(pos_pred + self.eps)).sum()
            num_valid += pos_pred.shape[0]

        if self.reduction == 'mean':
            return loss / num_valid
        elif self.reduction == 'none':
            return loss, num_valid
        else:
            NotImplementedError

class RCCE(nn.Module):
    def __init__(self, num_class, temperature=1.0, reduction='mean'):
        super().__init__()
        self.num_class = num_class
        self.reduction = reduction
        self.eps = 1e-8
        self.temp = temperature

    def forward(self, inputs, targets):
        ''' inputs: NxCxHxW
            targets: NxC+1xHxW '''

        inputs = inputs.permute(0,2,3,1).reshape(-1, self.num_class) ### NHW x C
        targets = targets.permute(0,2,3,1).reshape(-1, self.num_class + 1) ### additional ignore class: NHW x C+1

        ### 1. ignore masking
        inputs = inputs[torch.logical_not(targets[:,-1])] ### N' x C
        targets = targets[torch.logical_not(targets[:,-1])][:, :-1] ### N' X C

        ### 2. softmax 
        outputs = F.softmax(inputs / self.temp, dim=1)

        ### 3. index masking 
        pos_preds = (outputs * targets) ### (N' x C): prediction of positive classes
        with torch.no_grad():
            pos_preds_ = pos_preds.detach()
            pos_weight = pos_preds_ / pos_preds_.sum(dim=1)[..., None] ### (N' x C): normalized pred as weight

        loss = (pos_weight * pos_preds).sum(dim=1) ### weighted sum of candidate classes

        ### 4. negative logarithm & mean
        loss = -torch.log(loss + self.eps)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            NotImplementedError

class RCCE_asym(nn.Module):
    def __init__(self, num_class, temperature=1.0, temperature_w=1.0, reduction='mean'):
        super().__init__()
        self.num_class = num_class
        self.reduction = reduction
        self.eps = 1e-8
        self.temp = temperature
        self.temp2 = temperature_w

    def forward(self, inputs, inputs2, targets):
        ''' inputs: NxCxHxW
            inputs2: NxCxHxW
            targets: NxC+1xHxW '''

        inputs = inputs.permute(0,2,3,1).reshape(-1, self.num_class) ### NHW x C
        inputs2 = inputs2.permute(0,2,3,1).reshape(-1, self.num_class) ### NHW x C
        targets = targets.permute(0,2,3,1).reshape(-1, self.num_class + 1) ### additional ignore class: NHW x C+1

        ### 1. ignore masking
        inputs = inputs[torch.logical_not(targets[:,-1])] ### N' x C
        inputs2 = inputs2[torch.logical_not(targets[:,-1])] ### N' x C
        targets = targets[torch.logical_not(targets[:,-1])][:, :-1] ### N' X C

        ### 2. softmax 
        outputs = F.softmax(inputs / self.temp, dim=1)

        ### 3. index masking 
        pos_preds = (outputs * targets) ### (N' x C): prediction of positive classes
        with torch.no_grad():
            outputs2 = F.softmax(inputs2 / self.temp2, dim=1)
            pos_preds_ = (outputs2 * targets).detach()
            pos_weight = pos_preds_ / pos_preds_.sum(dim=1)[..., None] ### (N' x C): normalized pred as weight

        loss = (pos_weight * pos_preds).sum(dim=1) ### weighted sum of candidate classes

        ### 4. negative logarithm & mean
        loss = -torch.log(loss + self.eps)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            NotImplementedError

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()
