import torch
import numpy as np
from tqdm import tqdm

from trainer import active_joint_hier_multi_async
from models import freeze_bn
from utils.loss import WeightAsyncHierGroupMultiLabelCE, MultiChoiceCE

class ActiveTrainer(active_joint_hier_multi_async.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)

    def get_criterion(self):
        ''' Define criterion '''
        self.hier_group_multi_loss = WeightAsyncHierGroupMultiLabelCE(num_class = self.num_classes,
                                                                      num_superpixel = self.args.nseg,
                                                                      only_single = self.args.group_only_single,
                                                                      gumbel_scale = self.args.gumbel_scale,
                                                                      temperature = self.args.group_ce_temp,
                                                                      weight_reduce = self.args.weight_reduce)
        self.multi_pos_loss = MultiChoiceCE(num_class=self.num_classes, temperature=self.args.multi_ce_temp)