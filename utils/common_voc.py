import os
import sys
import torch
import random
import logging
import argparse
import numpy as np
from time import time

class TimeLogger(object):
    def __init__(self):
        self.last_line_time = 0

    def start(self):
        self.last_line_time = time()

    def check(self, operationname):
        print("[{}]: {}".format(operationname, time() - self.last_line_time))
        self.last_line_time = time()
 
class AverageMeter(object):
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict, denominator=None):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            self.__data[k][0] += v
            if denominator is None:
                self.__data[k][1] += 1
            else:
                self.__data[k][1] += denominator

    def get(self, *keys):
        if len(keys) == 1:
            try:
                return self.__data[keys[0]][0] / self.__data[keys[0]][1]
            except:
                return 0
        else:
            v_list = [self.get(k) for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v
    
    def get_whole_data(self):
        return self.__data

def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def initialize_logging(model_save_dir):
    # mkdir
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(os.path.join(model_save_dir, "AL_record"), exist_ok=True)
    log_fname = os.path.join(model_save_dir, 'log_train.txt')
    LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    DATE_FORMAT = '%Y%m%d %H:%M:%S'
    logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
    logger = logging.getLogger("Trainer")
    logger.info(f"{'-'*20} New Experiment {'-'*20}")
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)
    return logger

def find_topk(a, k, axis=-1, largest=True, sorted=True):
    if axis is None:
        axis_size = a.size
    else:
        axis_size = a.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(a)
    if largest:
        index_array = np.argpartition(a, axis_size-k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k)-1, axis=axis)
    else:
        index_array = np.argpartition(a, k-1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)
    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis)
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis)
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices

def initialization(args):
    # set random seed
    seed_everything(args.seed)

    # Initialize Logging
    logger = initialize_logging(args.model_save_dir)
    logger.info(' '.join(sys.argv))
    logger.info(args)
    return logger

def gen_save_name(args):
    ''' add argument naming '''
    args.model_save_dir = '{}_{}_sp{}_nlbl{}k_iter{}k_method-{}-_coeff{}_ign{}_lr{}_'.format(
                                                        args.model_save_dir,
                                                        args.active_method,
                                                        args.nseg,
                                                        float(args.active_selection_size) / 1000,
                                                        float(args.finetune_itrs) / 1000,
                                                        args.method,
                                                        args.coeff,
                                                        args.known_ignore,
                                                        args.train_lr)

def avoid_duplication(args):
    ''' avoid duplicated naming '''
    if os.path.exists(args.model_save_dir) and 'naive' not in args.model_save_dir:
        tail = str(args.model_save_dir)[-1]
        if tail.isnumeric():
            args.model_save_dir= '{}{}'.format(str(args.model_save_dir)[:-1], int(tail) + 1)
        else:
            args.model_save_dir = "{}_1".format(args.model_save_dir)
        avoid_duplication(args)

def preprocess(args):
    r"""
    - For multiple nseg compatibility
    - 여러 사이즈의 nseg 를 동시에 활용할 때는 가장 큰 nseg 를 기준으로 삼음
    - - Transform 에서 padding 이 들어갈 때, 장 큰 nseg id value 로 padding 을 해줘야 하기 때문
    - - args.nseg_list 는 오름차순이라고 가정
    """
    if args.nseg_list is not None:
        args.nseg = args.nseg_list[-1]

    args.session_id = args.model_save_dir.split('/')[-1]
    args.session_name = '{}_{}'.format(args.method, args.model_save_dir.split('/')[-1])
    
    if not args.stage2:
        gen_save_name(args)
        avoid_duplication(args) ### avoid duplicated naming

    if str(args.nseg) not in args.trg_datalist:
        args.trg_datalist = "dataloader/init_data/voc/train_seed{}.txt".format(args.nseg)

    if str(args.nseg) not in args.region_dict:
        args.region_dict = 'dataloader/init_data/voc/train_seed{}.dict'.format(args.nseg)

    if args.dominant_labeling:
        if 'dominant' not in args.trg_datalist:
            args.trg_datalist = '{}_dominant.txt'.format(args.trg_datalist.split('.')[0])

    if args.or_labeling:
        if 'or' not in args.trg_datalist:
            args.trg_datalist = '{}_or.txt'.format(args.trg_datalist.split('.')[0])

    ### compatibility to previous implementation (이전 옵션으로 돌려도 동일한 실험이 되도록)
    if args.known_ignore:
        assert('ignore' in args.loader)

def arg_assert(args):
    assert args.init_checkpoint is not None

    assert(str(args.nseg) in args.trg_datalist)
    assert(str(args.nseg) in args.region_dict)

    if args.dominant_labeling:
        assert('dominant' in args.trg_datalist)
        assert("_or_" not in args.loader.lower())

    if args.or_labeling:
        assert('or' in args.trg_datalist)
        # assert(args.loss_type == 'multi_choice_ce')

    if (args.datalist_path is not None) or (args.resume_checkpoint is not None):
        ### loading from same save_dir
        if not args.stage2:
            assert(args.datalist_path.split('/')[-2] == args.resume_checkpoint.split('/')[-2])

    ### deprecated list
    assert(args.ignore_size == 0)
    assert(args.mark_topk == -1)

def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)
    torch.manual_seed(worker_id)

def get_parser(mode='sup_active'):
    parser = argparse.ArgumentParser(description='')

    r" Deeplab (architecture) Options"
    parser.add_argument("-m", "--model", type=str, default='deeplabv3plus_resnet50',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50', 'deeplabv3plus_resnet50deepstem', 'deeplabv3plus_xception', 
                                 'deeplabv3plusc1_resnet50',
                                 'deeplabv3pluswn_resnet50', 'deeplabv3pluswn_resnet50deepstem', 'deeplabv3pluswn_resnet101deepstem',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101', 'deeplabv3plus_resnet101deepstem',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet',
                                 'deeplabv2_resnet101', 'deeplabv2_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--freeze_bn", dest='freeze_bn', action='store_true',
                        help='Freeze BatchNorm Layer while training (defulat: True)')
    # parser.set_defaults(freeze_bn=True)

    r" Method configuration"
    parser.add_argument('--method', type=str, default='active_voc', help='trainer selection (trainer/*)')
    parser.add_argument('--loader', type=str, default='region_voc', help='Multi-hot labeling loader seleciton (dataloader/*)')
    parser.add_argument("--active_method", default='random')
    parser.add_argument("--initial_active_method", default='my_random')
    parser.add_argument("--active_mode", default='region', choices=['scan', 'region'],
                        help="Region-based or scan-based AL method")
    parser.add_argument("--ce_temp", type=float, default=1.0, help="temperature for CE loss")
    parser.add_argument("--multi_ce_temp", type=float, default=1.0, help="temperature for multi-label CE loss")
    parser.add_argument("--group_ce_temp", type=float, default=1.0, help="temperature for group-label CE loss")
    parser.add_argument("--simw_temp", type=float, default=0.1, help="temperature for group-label CE loss")
    parser.add_argument("--delta", type=float, default=0.7, help="multi-label pseudo labeling thrreshold")
    parser.add_argument("--lamda", type=float, default=1.0, help="multi-label pseudo labeling threshold")
    parser.add_argument("--margin", type=float, default=0.7, help="multi-label pseudo labeling threshold")
    parser.add_argument("--coeff", type=float, default=1.0, help="loss coeff for ce loss (previously for all of the multi-positive loss)")
    parser.add_argument("--coeff_mc", type=float, default=1.0, help="loss coeff for multi-positive ce loss")
    parser.add_argument("--coeff_gm", type=float, default=1.0, help="loss coeff for group-multi loss")
    parser.add_argument("--entcoeff", type=float, default=1.0, help="ent loss coeff")
    parser.add_argument("--tocoeff", type=float, default=1.0, help="top one loss coeff")
    parser.add_argument("--plbl_th", type=float, default=0.0, help="pseudo label threshold")
    parser.add_argument('--within_filtering', action='store_true', default=False)
    parser.add_argument("--lamparam", type=float, default=0.1, help="ramp up param")
    parser.add_argument("--lamscale", type=float, default=1.0, help="ramp up scale")
    parser.add_argument('--dorampup', action='store_true', default=False)
    parser.add_argument("--gumbel_scale", type=float, default=-1, help="loss coeff between multi-positive and group-multi loss")
    parser.add_argument("--multihot_filter_size", type=int, default=0)
    parser.add_argument("--multihot_filter_ratio", type=float, default=0.0)
    parser.add_argument("--th_wplbl", type=float, default=None)
    parser.add_argument('--weight_wo_proto', action='store_true', default=False)
    parser.add_argument('--simw_temp_schedule', action='store_true', default=False)
    parser.add_argument("--angle_margin", type=float, default=0.1)
    parser.add_argument("--cos_margin", type=float, default=0.05)
    parser.add_argument('--arcface_mc', action='store_true', default=False)

    r" Dataset"
    parser.add_argument('--src_dataset', default='voc', choices=['cityscapes', 'GTA5', 'SYNTHIA', 'voc'],
                        help='source domain training dataset')
    parser.add_argument('--src_data_dir', default='./data/VOCdevkit')

    parser.add_argument('--trg_dataset', default='voc', help='target domain dataset')
    parser.add_argument('--trg_data_dir', default='./data/VOCdevkit')
    parser.add_argument('--trg_datalist', default='dataloader/init_data/voc/train_seed32.txt',
                        help='target domain training list')
    parser.add_argument('--region_dict', default='dataloader/init_data/voc/train_seed32.dict',
                        help='superpixel id (just range same as # superpixel per image')

    parser.add_argument('--val_dataset', default='voc', help='validation dataset')
    parser.add_argument('--val_data_dir', default='./data/VOCdevkit')
    parser.add_argument('--val_datalist', default='dataloader/init_data/voc/val.txt', help='validation list')
    r" Dataset: augmentation"
    parser.add_argument('--train_transform', default=None)
    parser.add_argument('--prob_dominant', action='store_true', default=False)

    r" Experiment protocol"
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--known_ignore", action='store_true', default=False)
    parser.add_argument("--start_over", action='store_true', default=False)
    parser.add_argument('--init_checkpoint', type=str, default='checkpoint/resnet50_imagenet_pretrained.tar',
                        help='Initial checkpoint to start with')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='resume checkpoint')
    parser.add_argument('--datalist_path', type=str, default=None, help='Load datalist files (to continue the experiment).')
    parser.add_argument('--max_iterations', type=int, default=5,
                        help='Number of active learning iterations (default: 5)')
    parser.add_argument('--active_selection_size', type=int, default=100000,
                        help='active selection size, # of superpixel (default: 29)')
    parser.add_argument('--init_iteration', type=int, default=1,
                        help='Initial active learning round (default: 1)')
    parser.add_argument("--cls_weight_coeff", type=float, default=1.0)
    parser.add_argument('--dominant_labeling', action='store_true', default=False)
    parser.add_argument('--or_labeling', action='store_true', default=False)
    parser.add_argument("--loss_type", type=str, default='cross_entropy')
                                                                        # choices=['focal_loss', 
                                                                        #            'cross_entropy', 
                                                                        #            'multi_choice_ce', 
                                                                        #            'topone_choice_ce', 
                                                                        #            'selective_topone_choice_ce',
                                                                        #            'group_multi_label_ce',
                                                                        #            'joint_multi_loss',
                                                                        #            'joint_multi_loss_weight',
                                                                        #            'hierarchy_group_multi_label_ce',
                                                                        #            'joint_hierarchy_multi_loss',
                                                                        #            'joint_hierarchy_multi_loss_weight',
                                                                        #            'rc_asym_ce',
                                                                        #            'joint_multi_rc_asym',
                                                                        #            'topone_ent'], help="loss type (default: False)")
    parser.add_argument('--fair_counting', action='store_true', default=False)
    parser.add_argument('--save_vis', action='store_true', default=False)

    r" Experiment details"
    parser.add_argument('--num_classes', type=int, default=21, help='number of classes in dataset')
    parser.add_argument("--num_workers", type=int, default=4, help="epoch number (default: 100k)")
    parser.add_argument('--train_batch_size', type=int, default=12, help='batch size for training (default: 1)')
    parser.add_argument("--weight_decay", type=float, default=1e-5, help='weight decay (default: 5e-4)')
    parser.add_argument("--total_itrs", type=int, default=30000, help="epoch number (default: 100k)")
    parser.add_argument("--train_lr", type=float, default=0.007, help="learning rate (default: 2.5e-4)")
    parser.add_argument("--cls_lr_scale", type=float, default=10.0, help="classifier learning rate scailing (default: 10.0)")
    parser.add_argument("--optimizer", default='adamw', choices=['adamw', 'sgd'])
    parser.add_argument('--adaptive_train_lr', action='store_true', default=False)
    parser.add_argument("--scheduler", default='poly', choices=['none', 'poly'])
    parser.add_argument("--min_lr", type=float, default=1e-6, help="minimum learning rate for poly decay scheduler (default: 1e-6")
    parser.add_argument("--power", type=float, default=0.9, help="power of poly scheduler (default: 0.9")
    parser.add_argument('--load_optim', action='store_true', default=False)
    parser.add_argument('--ignore_idx', type=int, default=255, help='ignore index')
    parser.add_argument('--val_batch_size', type=int, default=12, help='batch size for validation (default: 4)')
    parser.add_argument('--val_num_workers', type=int, default=4, help='batch size for validation (default: 4)')
    parser.add_argument('--nseg', type=int, default=32, choices=[32768, 8192, 4096, 2048, 1024, 512, 256, 150, 128, 64, 32, 16, 600], help='# superpixel component for slic')
    parser.add_argument('--nseg_list', nargs='+', default=None, type=int, help='# superpixel list (when using multiple superpixel sizes, *ascending order)')
    # parser.add_argument('--wandb_tags', nargs='+', default=None)
    parser.add_argument('--plbl_type', type=str, default=None, help='multi-hot pseudo label type within: None, naive, wcand')
    parser.add_argument('--cosprop_threshold_method', type=str, default='median')


    parser.add_argument('--finetune_itrs', type=int, default=30000, help='finetune iterations (default: 120k)')
    parser.add_argument('--loading', default='binary', choices=['binary', 'naive', 'tensor']
                        , help="Deprecated!")
    parser.add_argument('--ignore_size', type=int, default=0, help='(or_lbeling) ignore class region smaller than this')
    parser.add_argument('--mark_topk', type=int, default=-1, help='(or_lbeling) ignore classes with the region size under than kth order')
    parser.add_argument("--set_num_threads", type=int, default=20, help="the number of threads")
    parser.add_argument('--stage2', action='store_true', default=False)
    parser.add_argument('--skip_plbl_generation', action='store_true', default=False)
    parser.add_argument('--naive_plbl_generation', action='store_true', default=False)
    parser.add_argument('--single_sp_plbl', action='store_true', default=False)
    parser.add_argument('--load_smaller_spx', action='store_true', default=False)
    parser.add_argument('--group_only_single', action='store_true', default=False, help="remove only single spx from group multi loss")
    parser.add_argument('--nocropsp', action='store_true', default=False)
    parser.add_argument('--weight_reduce', default='max')
    parser.add_argument('--small_nseg', type=int, default=32, help='# superpixel component for smaller superpixel')
    parser.add_argument('--weighted_uncertainty', action='store_true', default=False)
    parser.add_argument("--hitent_param", type=float, default=0.005)
    parser.add_argument('--trim_kernel_size', type=int, default=3)
    parser.add_argument('--trim_multihot_boundary', action='store_true', default=False)


    r" logging"
    parser.add_argument('-p', '--model_save_dir', default='./checkpoint/default')
    parser.add_argument('--save_feat_dir', type=str, default='log/default', help='Region feature directory.')
    parser.add_argument('--skip_first_eval', action='store_true', default=False)
    parser.add_argument('--wandb_tags', nargs='+', default=None)
    parser.add_argument('--wandb_group', default=None)
    parser.add_argument('--val_start', type=int, default=0)
    parser.add_argument("--val_period", type=int, default=5000, help="validation frequency (default: 1000)")
    parser.add_argument('--log_period', type=int, default=1000)
    parser.add_argument('--save_scores', action='store_true', default=False)
    parser.add_argument('--dontlog', action='store_true', default=False, help='control wandb logging (Not logging)')

    return parser