import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_max

from trainer import active_onlineplbl_multi_predignore
from trainer.active_onlineplbl_multi_predignore import LocalProtoCE
from trainer.active_joint_multi_predignore_mclossablation import MultiChoiceCE_onlyDom
from utils.scheduler import ramp_up
r""" online pseudo labeling with local prototype-based pseudo labeling
- only apply multi-positive CE loss on dominant labels
"""

class ActiveTrainer(active_onlineplbl_multi_predignore.ActiveTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)

    def get_criterion(self):
        ''' Define criterion '''
        self.multi_local_proto_loss = LocalProtoCE(args=self.args, num_superpixel=self.args.nseg, temperature=self.args.group_ce_temp)
        self.multi_pos_loss = MultiChoiceCE_onlyDom(num_class=self.num_classes, temperature=self.args.multi_ce_temp)