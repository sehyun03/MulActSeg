import torch
import numpy as np
import os
from tqdm import tqdm, trange
import torch.nn.functional as F
from torch_scatter import scatter, scatter_max
from PIL import Image
from skimage.segmentation import mark_boundaries
from skimage.morphology import binary_dilation

from dataloader import get_dataset
from dataloader.utils import DataProvider
from trainer.eval_within_multihot import ActiveTrainer
from models import get_model, freeze_bn
from utils.miou import MeanIoU
r'''
- Cosine pseudo label with label propagation
'''

class ActiveTrainer(ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        self.selection_iter = selection_iter
        super().__init__(args, logger, selection_iter)
        assert(args.val_batch_size == 1)
        self.kernel = np.ones((3,3),np.uint8)

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

                r''' NN based pseudo label acquisition '''
                superpixels = batch['spx'].to(self.device)
                spmasks = batch['spmask'].to(self.device)
                targets = batch['target'].to(self.device)
                nn_pseudo_label = self.pseudo_label_generation(labels, feats, outputs, targets, spmasks, superpixels)
                ### ㄴ N x H x W

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
                if args.save_vis and iteration < 10:
                    fname = batch['fnames'][0][1]
                    lbl_id = fname.split('/')[-1].split('.')[0]

                    vis_superpixel = superpixels[0].cpu().numpy()
                    vis_nn_plbl = torch.masked_fill(nn_pseudo_label[0], nn_pseudo_label[0]==255, 20).cpu()
                    vis_nn_plbl = decode_target(vis_nn_plbl).astype('uint8')
                    vis_nn_plbl = mark_boundaries(vis_nn_plbl, vis_superpixel) * 255
                    vis_nn_plbl = vis_nn_plbl.astype('uint8')
                    Image.fromarray(vis_nn_plbl).save("{}/{}.png".format(save_vid_dir, lbl_id))

        iou_table = []
        precision_table = []
        recall_table = []
        ious, precisions, recalls = iou_helper._after_epoch_ipr()

        miou = np.mean(ious)
        iou_table.append(f'{miou:.2f}')
        for class_iou in ious:
            iou_table.append(f'{class_iou:.2f}')
        iou_table_str = ','.join(iou_table)

        mprecision = np.mean(precisions)
        precision_table.append(f'{mprecision:.2f}')
        for class_precision in precisions:
            precision_table.append(f'{class_precision:.2f}')
        precision_table_str = ','.join(precision_table)

        mrecall = np.mean(recalls)
        recall_table.append(f'{mrecall:.2f}')
        for class_recall in recalls:
            recall_table.append(f'{class_recall:.2f}')
        recall_table_str = ','.join(recall_table)

        del iou_table
        del precision_table
        del recall_table
        del output_dict
        print("\n[AL {}-round] IoU: {}\n{}".format(self.selection_iter, prefix, iou_table_str), flush=True)
        print("\n[AL {}-round] Precision: {}\n{}".format(self.selection_iter, prefix, precision_table_str), flush=True)
        print("\n[AL {}-round] Recall: {}\n{}".format(self.selection_iter, prefix, recall_table_str), flush=True)

        return miou, iou_table_str

    def pseudo_label_generation(self, labels, feats, inputs, targets, spmasks, superpixels):
        r'''
        Args::
            feats: N x Channel x H x W
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
        outputs = F.softmax(inputs, dim=1) ### N x C x H x W
        outputs = outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        preds = outputs.argmax(dim=2)
        N, Ch, H, W = feats.shape
        feats = feats.permute(0,2,3,1).reshape(N, -1, Ch) ### N x HW x Ch
        superpixels_orig = superpixels.cpu().numpy()
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW
        is_trg_multihot = (1 < targets.sum(dim=2)) ### N x self.num_superpixel

        nn_plbl = torch.ones_like(labels) * 255 ### N x H x W
        nn_plbl = nn_plbl.reshape(N, -1)

        for i in range(N):
            '''
            outputs[i] : HW x C
            feats[i] : HW x Ch
            superpixels[i] : HW x 1
            superpixel_smalls[i] : HW x 1
            targets[i] : self.num_superpiexl x C
            spmasks[i] : HW
            '''

            r''' filtered outputs '''
            if not torch.any(spmasks[i]): continue
            validall_superpixel = superpixels[i][spmasks[i]]
            # validall_trg_pixel = targets[i][validall_superpixel.squeeze(dim=1)]

            multi_mask = is_trg_multihot[i][validall_superpixel.squeeze(dim=1)].detach()
            valid_mask = spmasks[i].clone()
            valid_mask[spmasks[i]] = multi_mask
            if not torch.any(valid_mask): continue

            valid_output = outputs[i][valid_mask] ### HW' x C
            valid_feat = feats[i][valid_mask] ### HW' x Ch
            vpx_superpixel = superpixels[i][valid_mask] ### HW' x 1
            multi_hot_target = targets[i] ### self.num_superpixel x C

            r''' get max pixel for each class within superpixel '''
            _, vdx_sup_mxpool = scatter_max(valid_output, vpx_superpixel, dim=0, dim_size=self.args.nseg)
            ### ㄴ self.num_superpixel x C: 각 (superpixel, class) pair 의 max 값을 가지는 index
           
            r''' filter invalid superpixels '''
            is_spx_valid = vdx_sup_mxpool[:,0] < valid_output.shape[0]
            ### ㄴ vpx_superpixel 에 포함되지 않은 superpixel id 에 대해서는 max index 가
            ### valid_output index 최대값 (==크기)로 잡힘. 이 값을 통해 쓸모없는 spx filtering            
            vdx_vsup_mxpool = vdx_sup_mxpool[is_spx_valid]
            ### ㄴ nvalidseg x C : index of max pixel for each class (for valid spx)
            multihot_vspx = multi_hot_target[is_spx_valid]
            ### ㄴ nvalidseg x C : multi-hot label (for valid spx)

            r''' Index conversion (valid pixel -> pixel) '''
            validex_to_pixdex = valid_mask.nonzero().squeeze(dim=1)
            ### ㄴ translate valid_pixel -> pixel space
            proto_vspx, proto_clsdex = multihot_vspx.nonzero(as_tuple=True)
            ### ㄴ valid superpixel index && valid class index
            top1_vdx = vdx_vsup_mxpool[proto_vspx, proto_clsdex]
            ### ㄴ vdx_sup_mxpool 중에서 valid 한 superpixel 과 target 에서의 valid index
            # top1_pdx = validex_to_pixdex[top1_vdx]
            # ### ㄴ max index 들을 pixel space 로 변환

            r''' Inner product between prototype features & superpixel features '''
            prototypes = valid_feat[top1_vdx]
            ### ㄴ nproto x Ch
            similarity = torch.mm(prototypes, valid_feat.T)
            ### ㄴ nproto x nvalid_pixels: 각 prototype 과 모든 valid pixel feature 사이의 유사도
            
            vspdex_to_spdex = is_spx_valid.nonzero(as_tuple=True)[0]
            # proto_spx = vspdex_to_spdex[proto_vspx] ### to calcualte equal operation in all index
            # multispx = validall_superpixel[multi_mask].squeeze(dim=1)

            # is_similarity_valid = torch.eq(proto_spx[..., None], multispx[None, ...])

            r''' Nearest prototype selection '''
            mxproto_sim, idx_mxproto_pxl = scatter_max(similarity, proto_vspx, dim=0)
            ### ㄴ nvalidspx x nvalid_pixels: pixel 별 가장 유사한 prototype 과의 similarity
            ### ㄴ nvalidspx x nvalid_pixels: pixel 별 가장 유사한 prototype id

            r''' Assign pseudo label of according prototype
            - idx_mxproto_pxl 중에서 각 pixel 이 해당하는 superpixel superpixel 의 값을 얻기
            - 이를 위해 우선 (superpixel -> valid superpixel)로 index conversion 을 만듦
            - pixel 별 superpixel id 를 pixel 별 valid superpixel id 로 변환 (=nearest_vspdex)
            - 각 valid superpixel 의 label 로 pseudo label assign (=plbl_vdx)
            - pseudo label map 의 해당 pixel 에 valid pixel 별 pseudo label 할당 (nn_plbl)
            '''
            spdex_to_vspdex = torch.ones_like(is_spx_valid) * -1
            vspx_ids, proto_counts = torch.unique(proto_vspx, return_counts=True)
            spdex_to_vspdex[is_spx_valid] = vspx_ids
            vspdex_superpixel = spdex_to_vspdex[vpx_superpixel.squeeze(dim=1)]
            ### ㄴ 여기 vpx_superpixel 의 id value 는 superpixel 의 id 이다.
            nn_local_cls = idx_mxproto_pxl.T[torch.arange(vspdex_superpixel.shape[0]), vspdex_superpixel]
            nn_local_similarity = mxproto_sim.T[torch.arange(vspdex_superpixel.shape[0]), vspdex_superpixel]

            r''' Prototype similarity value & neighborhood spx id acquisition'''
            trg_vsup_median_sim = torch.zeros_like(multihot_vspx).float()
            spx_neighbor_ids = {}
            offset = 0
            for vspx in range(vspx_ids.shape[0]):
                r''' Get similarity threshold value for each prototype
                - Get index value of max similarity value for each superpixel
                - For each prototype within superpixel, calculate median simialrity threshold
                '''
                indices = torch.masked_select(nn_local_cls, (vspdex_superpixel == vspx))
                similarity = torch.masked_select(nn_local_similarity, (vspdex_superpixel == vspx))
                for proto_ids in range(proto_counts[vspx]):
                    proto_ids_ = proto_ids + offset
                    if self.args.cosprop_threshold_method == 'median':
                        similarity_threshold = torch.median(similarity[indices==proto_ids_])
                    elif self.args.cosprop_threshold_method == 'min':
                        similarity_threshold = torch.min(similarity[indices==proto_ids_])
                    else:
                        raise NotImplementedError
                    trg_vsup_median_sim[vspx, proto_clsdex[proto_ids_]] = similarity_threshold
                offset += proto_counts[vspx]

                r''' Get ids of surrounding superpixels
                - Get binary mask of current superpixel id --> Dilation -> Id collection
                '''
                spx_id = vspdex_to_spdex[vspx].item()
                spx_id_binmap = (superpixels_orig[i] == spx_id)
                spx_id_binmap_dilate = binary_dilation(spx_id_binmap, self.kernel)
                spx_map_tensor = torch.from_numpy(superpixels_orig[i])
                dilated_mask = torch.from_numpy(spx_id_binmap_dilate)
                selected = torch.masked_select(spx_map_tensor, dilated_mask)
                spx_neighbor_ids[spx_id] = torch.unique(selected).cuda()

                r''' TODO: Get ids of surrounding superpixels (larger superpixel)
                - Get id from larger superpixel -> select maximum index
                '''

            r''' TODO: 인접한 selected superpixel 예외 처리 '''

            r''' Similarity calculation & pseudo label assignment ''' 
            # spx_neighbor_ids = {i:j for i,j in spx_neighbor_ids.items()}
            for vspx in range(vspx_ids.shape[0]):
                r''' Get similarity betwen prototype and sourrounding regions
                - prototypes within superpixel indexing
                - sourrouding feature filtering
                - similarity calculation
                '''
                spx_id = vspdex_to_spdex[vspx].item()
                curr_spx_prototypes = prototypes[proto_vspx == vspx]
                surr_spx_mask = torch.isin(superpixels[i], spx_neighbor_ids[spx_id]).squeeze(dim=1)
                surr_feature = feats[i][surr_spx_mask] ### HW' x Ch
                curr_spx_similarity = torch.mm(curr_spx_prototypes, surr_feature.T)

                r''' Pseduo label generation from similarity and assign them into plbl map
                - prototype argmax
                - prototype index -> pseudo label index
                - Thresholding with prototype-wise threshold
                - (Skip) Exclude within superpixel indices from filtering
                - nn_plbl saving: surr_spx_mask index -> pixel index 
                '''
                prototype_cls = proto_clsdex[proto_vspx == vspx]
                plbl_prototype_id = curr_spx_similarity.argmax(dim=0)
                plbl_unfiltered = prototype_cls[plbl_prototype_id]
                similarity_threshold = trg_vsup_median_sim[vspx, prototype_cls] ### TODO: bug!
                is_plbl_valid = torch.any((similarity_threshold[..., None] < curr_spx_similarity), dim=0)
                # is_plbl_valid = torch.ones_like(is_plbl_valid).bool() ### TODO: Debug  for similarity

                surrounding_index_to_pixel_index = surr_spx_mask.nonzero(as_tuple=True)[0]
                filtered_pixel_index = surrounding_index_to_pixel_index[is_plbl_valid]
                plbl_filtered = plbl_unfiltered[is_plbl_valid]

                is_plbl_consistent = (preds[i, filtered_pixel_index] == plbl_filtered)
                nn_plbl[i, filtered_pixel_index[is_plbl_consistent]] = plbl_filtered[is_plbl_consistent]

                # TODO: if not args.within_spx_filtering:

            plbl_vdx = proto_clsdex[nn_local_cls]
            nn_plbl[i, validex_to_pixdex] = plbl_vdx

        nn_plbl = nn_plbl.reshape(N, H, W)
        
        return nn_plbl