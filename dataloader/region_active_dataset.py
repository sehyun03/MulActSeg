import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import ConcatDataset
from tqdm import tqdm

class RegionActiveDataset:
    def __init__(self, args, trg_pool_dataset, trg_label_dataset):
        # Active Learning intitial selection
        self.args = args
        self.selection_iter = 0
        self.trg_pool_dataset = trg_pool_dataset
        self.trg_label_dataset = trg_label_dataset

    def expand_training_set(self, sample_region, selection_count, selection_method):
        """
        sample_region: sorted, scored regions (list)
            [
                (score, scan_file_path, suppix_id),
                ...
            ]
        selection_count: # of superpixel to select (int)
        selection_method: name for logging (string)
        """
        # max_selection_count = int(selection_count * 1024 * 2048)  # (number of images --> number of pixels)
        max_selection_count = selection_count ### edit code: # of images --> number of superpixels
        selected_count = 0
        selected_sup_count = 0
        ''' Active Selection '''
        for idx, x in enumerate(tqdm(sample_region)): # sorted tuples: (score, file_path *join of three paths, suppix_id)
            _, scan_file_path, suppix_id = x
            scan_file_path = scan_file_path.split(",")
            spx_file_path = scan_file_path[2]
            
            '''Add into label dataset'''
            if scan_file_path not in self.trg_label_dataset.im_idx:
                self.trg_label_dataset.im_idx.append(scan_file_path) # im_idx labeled al set: join of three paths
                self.trg_label_dataset.suppix[spx_file_path] = [suppix_id] # add sp id in to suppix variable
            else:
                self.trg_label_dataset.suppix[spx_file_path].append(suppix_id)
            
            '''Remove it from unlabeled dataset'''
            self.trg_pool_dataset.suppix[spx_file_path].remove(suppix_id) ### remove superpixel id from list of superpixel within image
            if len(self.trg_pool_dataset.suppix[spx_file_path]) == 0: ### when superpixel is empty remove image path
                self.trg_pool_dataset.suppix.pop(spx_file_path)
                self.trg_pool_dataset.im_idx.remove(scan_file_path)

            '''Update isselected matrix'''
            if hasattr(self.trg_pool_dataset, 'isselected'):
                id = spx_file_path.split('/')[-1].split('.')[0]
                trg_index = self.trg_label_dataset.id_to_index[id]
                self.trg_pool_dataset.isselected[trg_index, suppix_id] = 1 ### (N, Nseg)

            '''jump out the loop when exceeding max_selection_count'''
            if self.args.fair_counting and self.args.or_labeling:
                id = spx_file_path.split('/')[-1].split('.')[0]
                trg_index = self.trg_label_dataset.id_to_index[id]
                num_cls = self.trg_label_dataset.multi_hot_cls[trg_index, suppix_id].sum()
                selected_count += num_cls
                selected_sup_count += 1
            else:
                selected_sup_count += 1
                selected_count += 1

            if selected_count > max_selection_count:
                fname = f'{selection_method}_selection_{self.selection_iter:02d}.pkl' 
                selection_path = os.path.join(self.args.model_save_dir, fname)
                ### save sorted, scored region list as pkl
                with open(selection_path, "wb") as f:
                    pickle.dump(sample_region[:idx+1], f)
                print(selected_sup_count)
                break
            
        r""" wandb logging """
        global_step = int(self.args.finetune_itrs) * (self.selection_iter - 1)
        wlog_selection = {"num_selected_spx": selected_sup_count,
                          "num_cls_spx": max_selection_count / selected_sup_count,
                          "sampling_iter": self.selection_iter} ### for wandb step sync actual selection_iter is logged in sampling_iter
        self.args.wandb.log(wlog_selection, step=global_step) ### expected step size: self.trg_pool_dataset.selection_iter

    def dump_datalist(self):
        datalist_path = os.path.join(self.args.model_save_dir, f'datalist_{self.selection_iter:02d}.pkl')
        with open(datalist_path, "wb") as f:
            store_data = {
                'trg_label_im_idx': self.trg_label_dataset.im_idx,
                'trg_pool_im_idx': self.trg_pool_dataset.im_idx,
                'trg_label_suppix': self.trg_label_dataset.suppix,
                'trg_pool_suppix': self.trg_pool_dataset.suppix,
            }
            pickle.dump(store_data, f)

    def load_datalist(self, datalist_path=None):
        print('Load datalist', flush=True)
        # Synchronize Training Path
        if datalist_path is None:
            datalist_path = os.path.join(self.args.model_save_dir, f'datalist_{self.selection_iter:02d}.pkl')
        with open(datalist_path, "rb") as f:
            pickle_data = pickle.load(f)
        self.trg_label_dataset.im_idx = pickle_data['trg_label_im_idx']
        self.trg_pool_dataset.im_idx = pickle_data['trg_pool_im_idx']
        self.trg_label_dataset.suppix = pickle_data['trg_label_suppix']
        self.trg_pool_dataset.suppix = pickle_data['trg_pool_suppix']

    def get_trainset(self):
        return self.trg_label_dataset