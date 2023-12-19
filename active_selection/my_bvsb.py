import torch
from tqdm import tqdm
from torch_scatter import scatter

from active_selection.utils import get_al_loader
from active_selection import base
"""
my_bvsb.py
: Best versus second Best (BvsB) selector

- BvSB value of each pixel within a region is averaged using torch_scatter
"""

class RegionSelector(base.RegionSelector):
    def __init__(self, args):
        super().__init__(args)
        self.temperature = args.ce_temp

    def softmax_bvsb(self, preds):
        prob = torch.nn.functional.softmax(preds / self.temperature, dim=1)
        top2_val, top2_idx = torch.topk(prob, 2, dim=1)
        bvsb = top2_val[:, 1] / top2_val[:, 0]
        ### ㄴ If top-2 prediction is similar --> bvsb is large (close to 1) --> High uncertainty --> high score (0 ~ 1)
        bvsb += 1e-8
        top1_idx = top2_idx[:, 0]

        return bvsb, top1_idx

    def gen_score_list_from_tensor(self, pool_set, scores_tensor):
        r""" Generate score list following d2ada implementation

        Args::
            pool_set (Dataset): pool dataset that includes image id list, superpixel id list
            scores_tensor (Torch.Tensor): (N, self.num_superpixels) score tensor
        
        Returns::
            scores (List): List of tuple = (score, joined_path, suppix_id)
        """
        scores = []
        keys = pool_set.im_idx
        sp_dict = pool_set.suppix
        for kdx, key in enumerate(keys):
            path = ','.join(key)
            spxids = sp_dict[key[2]]
            spxscores = scores_tensor[kdx][spxids]
            scores.extend([(s, path, i) for s,i in zip(spxscores.tolist(), spxids)])
    
        return scores

    def calculate_scores(self, trainer, pool_set):
        model = trainer.net
        model.eval()

        loader, idx = get_al_loader(trainer, pool_set, self.batch_size, self.num_workers)
        scores = []
        uncertainties = []
        tqdm_loader = tqdm(loader, total=len(loader))
        with torch.no_grad():
            for i_iter_test, batch in enumerate(tqdm_loader):
                images = batch['images']
                suppixs = batch['spx'] # (B, H, W) ### Note that there's no filtering for selected spxs
                images = images.to(trainer.device, dtype=torch.float32)
                suppixs = suppixs.to(trainer.device, dtype=torch.long)
                preds = model(images)  # (B, Class, H, W)
                if 'predignore' in self.args.method:
                    preds = preds[:, :-1, :, :]
                bvsb, top1 = self.softmax_bvsb(preds) # (B, H, W)

                ### region-wise uncertainty averaging
                B, _, H, W = images.shape
                bvsb = bvsb.view(B, -1) # (B, HW)
                suppixs = suppixs.view(B, -1) # (B, HW)
                region_bvsb = scatter(bvsb, suppixs, dim=1, reduce='mean', dim_size=self.num_superpixels) ### (B, self.num_superpixels): value for non-existing spx id == 0
                uncertainties.append(region_bvsb.cpu())

            uncertainties = torch.cat(uncertainties, dim=0) ### (N, self.num_superpixels)

            ### normalize uncertainty
            uncertainties = uncertainties.view(-1) ### (N x self.num_superpixels, ) *valid: 1e-8 <= x < 1 *non-valid: 0
            uncertainties = uncertainties - uncertainties[uncertainties != 0].min() ### calculate min excluding ignore class
            uncertainties = uncertainties / uncertainties.max() ### valid: 0 ~ 1 value, invalid: negative
            uncertainties = uncertainties.view(-1, self.num_superpixels) # (N, self.num_suerpixel)
            scores_tensor = uncertainties # (N, self.num_superpixels)

        scores = self.gen_score_list_from_tensor(pool_set, scores_tensor)

        return scores