import torch
from tqdm import tqdm
from torch_scatter import scatter

from active_selection.utils import get_al_loader
from active_selection import my_bvsb
"""
my_bvsb_banignore.py
: Best versus second Best (BvsB) selector with banning ignore

- BvSB value of each pixel within a region is averaged using torch_scatter
- Ban (i.e., set score as 0) regions whose dominant class is estimated as 'undefined class'
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
        with torch.no_grad():
            for i_iter_test, batch in enumerate(tqdm_loader):
                images = batch['images']
                suppixs = batch['spx'] # (B, H, W) ### Note that there's no filtering for selected spxs
                images = images.to(trainer.device, dtype=torch.float32)
                suppixs = suppixs.to(trainer.device, dtype=torch.long)
                preds = model(images)  # (B, Class, H, W)
                assert('predignore' in self.args.method)
                bvsb, top1 = self.softmax_bvsb(preds) # (B, H, W)

                ### region-wise uncertainty averaging
                B, _, H, W = images.shape
                bvsb = bvsb.view(B, -1) # (B, HW)
                suppixs = suppixs.view(B, -1) # (B, HW)
                top1 = top1.view(B, -1) # (B, HW)
                region_bvsb = scatter(bvsb, suppixs, dim=1, reduce='mean', dim_size=self.num_superpixels) ### (B, self.num_superpixels): value for non-existing spx id == 0
                top1_oh = torch.nn.functional.one_hot(top1, num_classes=(self.num_class + 1)) # (B, HW, C + 1)
                region_ntop1 = scatter(top1_oh, suppixs, dim=1, reduce='sum', dim_size=self.num_superpixels) # (B, self.num_superpixels, C + 1)
                
                uncertainties.append(region_bvsb.cpu())
                top1nclasses.append(region_ntop1.cpu())

            uncertainties = torch.cat(uncertainties, dim=0) ### (N, self.num_superpixels)
            top1nclasses = torch.cat(top1nclasses, dim=0) # (N, self.num_superpixels, C + 1)

            ### normalize uncertainty
            uncertainties = uncertainties.view(-1) ### (N x self.num_superpixels, ) *valid: 1e-8 <= x < 1 *non-valid: 0
            uncertainties = uncertainties - uncertainties[uncertainties != 0].min() ### calculate min excluding ignore class
            uncertainties = uncertainties / uncertainties.max() ### valid: 0 ~ 1 value, invalid: negative

            ### Filter out ignore dominant regions
            top1nclasses = top1nclasses.view(-1, (self.num_class + 1)) # (N x self.num_superpixels, C + 1): top-1 prediction of all regions
            isignoredominant = (top1nclasses.argmax(dim=1) == (top1nclasses.shape[1] - 1))
            uncertainties[isignoredominant] = 0

            uncertainties = uncertainties.view(-1, self.num_superpixels) # (N, self.num_suerpixel)
            scores_tensor = uncertainties # (N, self.num_superpixels)

        scores = self.gen_score_list_from_tensor(pool_set, scores_tensor)

        return scores