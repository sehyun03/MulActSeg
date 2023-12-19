import os
import os.path
import hashlib
import errno
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def collate_fn(inputs):
    # train_keys = ['images', 'labels', 'fnames', 'spx']
    train_keys = list(inputs[0].keys())
    output_batch = {}
    for key in train_keys:
        if key in ['images', 'image_weak', 'spx', 'spx_weak', 'spmask', 'spmask_weak', 'labels', 'spx_small', 'spx_small_weak', 'target', 'nseg_list']:
            if type(inputs[0][key]).__module__ == np.__name__:
                output_batch[key] = torch.stack([torch.from_numpy(one_batch[key]) for one_batch in inputs])
            else:
                output_batch[key] = torch.stack([one_batch[key] for one_batch in inputs])
        elif key in ['image_list', 'fnames', 'imsizes'] or 'mseg_' in key:
            output_batch[key] = [one_batch[key] for one_batch in inputs]
        else:
            raise NotImplementedError

    return output_batch


class DataProvider():
    def __init__(self, dataset, batch_size, num_workers, drop_last, shuffle,
                 pin_memory):
        # dataset
        self.dataset = dataset
        self.iteration = 0
        self.epoch = 0

        # dataloader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.dataloader = \
            DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=collate_fn,
                       shuffle=self.shuffle, num_workers=self.num_workers, drop_last=self.drop_last,
                       pin_memory=self.pin_memory)
        self.dataiter = iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def __next__(self):
        try:
            batch = self.dataiter.next()
            self.iteration += 1
            return batch

        except StopIteration:
            self.epoch += 1
            self.dataiter = iter(self.dataloader)
            batch = self.dataiter.next()
            self.iteration += 1
            return batch


def gen_bar_updater(pbar):
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def check_integrity(fpath, md5=None):
    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True