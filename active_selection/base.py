import os
import json

"""
base.py
: base class for region selection

- select_next_batch --> calculate_scores --> active_set.expand_training_set()
- calculate_scores: inference all poolset and then record their scores.
    - scores (list): list of score (float), file_path (string), suppix_id (int)
"""

class RegionSelector(object):

    def __init__(self, args):
        self.args = args
        self.batch_size = args.val_batch_size
        self.num_workers = args.val_num_workers
        self.num_superpixels = args.nseg
        self.active_method = args.active_method
        self.num_class = args.num_classes
        self.eps = 1e-8

    def calculate_scores(self, trainer, pool_set):
        return NotImplementedError

    def select_next_batch(self, trainer, active_set, selection_count):
        scores = self.calculate_scores(trainer, active_set.trg_pool_dataset)

        ''' Save calculated scores '''
        if self.args.save_scores:
            fname = os.path.join(trainer.model_save_dir, "AL_record", "region_val_{}.json".format(trainer.selection_iter))
            with open(fname, "w") as f:
                json.dump(scores, f)

        ''' Sorting and sampling '''
        selected_samples = sorted(scores, reverse=True)
        active_set.expand_training_set(selected_samples, selection_count, self.active_method)