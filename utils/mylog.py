
from datetime import datetime
import logging
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import wandb

def timediff(t_start, t_end):
    t_diff = relativedelta(t_end, t_start)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

def log_final(t_start, val_result, logger, args):
    t_end = datetime.now()
    logger.info(f"{'%'*20} Experiment Report {'%'*20}")
    logging.info(f"0. AL Methods: {args.active_method}")
    logging.info(f"1. Takes: {timediff(t_start, t_end)}")
    logging.info(f"2. Log dir: {args.model_save_dir} (with selection json & model checkpoint)")
    logging.info("3. Validation mIoU (Be sure to submit to google form)")
    for selection_iter in range(args.init_iteration, args.max_iterations + 1):
        logging.info(f"AL {selection_iter}: {val_result[selection_iter]}")
    logger.info(f"{'%'*20} Experiment End {'%'*20}")

def init_logging(args):
    r"""
    wandb table object for per-round-miou, class-wise-iou logging
    """
    dummy_data = {"round_v_miou": [""]}
    for i in range(args.max_iterations+1):
        dummy_data["round-{}".format(i)] = [""]
    args.wandb_iou_table = pd.DataFrame(data=dummy_data)

def finalization(t_start, val_result, logger, args):
    ### log into local file
    log_final(t_start, val_result, logger, args)
    
    ### log into wandb
    # val_result: {'1': 'x.x, ..., x.x', ...}
    # round_wise_miou = [float(j.split(',')[0]) for i,j in val_result.items()]
    # d = {"round-{}".format(iter):[float(j_) for j_ in j.split(',')] for iter, j in val_result.items()}
    # df = pd.DataFrame(data=d)
    # args.wandb.log({"miou_vs_round": df})