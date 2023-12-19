import random
from tqdm import tqdm

from active_selection import base

"""
my_random.py
: Random selector that assign random score to every superpixel
"""

class RegionSelector(base.RegionSelector):
    def __init__(self, args):
        super().__init__(args)

    def calculate_scores(self, trainer, pool_set):
        '''Give each superpixel a random score'''
        scores = []
        for key in tqdm(pool_set.im_idx):
            rgb_fname, gt_fname, spx_fname = key
            for suppix_id in pool_set.suppix[spx_fname]:
                score = random.random()
                file_path = ",".join(key)
                item = (score, file_path, suppix_id)
                scores.append(item)

        return scores

    def select_next_batch(self, trainer, active_set, selection_count):
        scores = self.calculate_scores(trainer, active_set.trg_pool_dataset)

        ''' Sorting and sampling '''
        selected_samples = sorted(scores, reverse=True)
        active_set.expand_training_set(selected_samples, selection_count, self.active_method)