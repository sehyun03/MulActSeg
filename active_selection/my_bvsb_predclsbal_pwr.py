import torch
from tqdm import tqdm
from torch_scatter import scatter

from active_selection.utils import get_al_loader
from active_selection import my_bvsb
"""
my_bvsb_predclsbal_pwr.py
: Best versus second Best (BvsB) selector + Pixel-wise class balancing
  Proposed sampling of [Hwang, NeurIPS 2023]

- Label distribution estimation with pixel-wise top-1 prediction
- New class weighting funciton
- Pixel-wise class weighting
"""


class RegionSelector(my_bvsb.RegionSelector):

    def __init__(self, args):
        super().__init__(args)

    def calculate_scores(self, trainer, pool_set):
        model = trainer.net
        model.eval()

        loader, idx = get_al_loader(trainer, pool_set, self.batch_size, self.num_workers)
        scores = []
        uncertainties = []
        top1nclasses = []
        tqdm_loader = tqdm(loader, total=len(loader))
        cumulated_pred_prob = torch.zeros((self.num_class)).to(trainer.device)
        nimg = 0
        with torch.no_grad():
            ### first for loop for predictive label distribution estimation
            for i_iter_test, batch in enumerate(tqdm_loader):
                images = batch['images']
                suppixs = batch['spx'] # (B, H, W) ### Note that there's no filtering for selected spxs
                images = images.to(trainer.device, dtype=torch.float32)
                suppixs = suppixs.to(trainer.device, dtype=torch.long)
                preds = model(images)  # (B, Class, H, W)
                preds_prob = torch.softmax(preds / self.args.ce_temp, dim=1)
                cumulated_pred_prob += torch.mean(preds_prob, dim=(0,2,3)) #(Class, )
                nimg += images.shape[0]
            
            cumulated_pred_prob = cumulated_pred_prob / len(tqdm_loader) #(Class,)
            cls_weight = (self.args.cls_weight_coeff * cumulated_pred_prob + 1)**(-2) # (Class,)
            
            tqdm_loader = tqdm(loader, total=len(loader))
            for i_iter_test, batch in enumerate(tqdm_loader):
                images = batch['images']
                suppixs = batch['spx'] # (B, H, W) ### Note that there's no filtering for selected spxs
                images = images.to(trainer.device, dtype=torch.float32)
                suppixs = suppixs.to(trainer.device, dtype=torch.long)
                preds = model(images)  # (B, Class, H, W)
                preds_prob = torch.softmax(preds, dim=1)  # (B, Class, H, W)
                bvsb, top1 = self.softmax_bvsb(preds) # (B, H, W)
                B, H, W = top1.shape
                pixel_wise_clsweight = cls_weight[top1.reshape(-1)].view(B, H, W)

                ### region-wise weighted uncertainty averaging
                weighted_bvsb = bvsb * pixel_wise_clsweight
                weighted_bvsb = weighted_bvsb.view(B, -1) # (B, HW)
                suppixs = suppixs.view(B, -1) # (B, HW)
                region_bvsb = scatter(weighted_bvsb, suppixs, dim=1, reduce='mean', dim_size=self.num_superpixels) ### (B, self.num_superpixels): value for non-existing spx id == 0

                top1 = top1.view(B, -1) # (B, HW)
                top1_oh = torch.nn.functional.one_hot(top1, num_classes=(self.num_class)) # (B, HW, C+1)
                region_ntop1 = scatter(top1_oh, suppixs, dim=1, reduce='sum', dim_size=self.num_superpixels) # (B, self.num_superpixels, C+1)
                
                uncertainties.append(region_bvsb.cpu())
                top1nclasses.append(region_ntop1.cpu())

            uncertainties = torch.cat(uncertainties, dim=0) ### (N, self.num_superpixels)
            top1nclasses = torch.cat(top1nclasses, dim=0) # (N, self.num_superpixels, C+1)

            ### normalize uncertainty
            uncertainties = uncertainties.view(-1) ### (N x self.num_superpixels, ) *valid: 1e-8 <= x < 1 *non-valid: 0

            ### Filter out ignore dominant regions
            top1nclasses = top1nclasses.view(-1, (self.num_class)) # (N x self.num_superpixels, C+1): 모든 region 별 top-1 예측

            ### Final score
            scores_tensor = uncertainties.view(-1, self.num_superpixels) # (N, self.num_suerpixel)

        scores = self.gen_score_list_from_tensor(pool_set, scores_tensor)

        return scores