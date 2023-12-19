import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from collections import defaultdict

from . import region_active_dataset
Img_File_Template = 'leftImg8bit/train/{}/{}_leftImg8bit.png'
Lbl_File_Template = 'superpixel_seed/cityscapes/seeds_{}/train/gtFine_dominant_ignore/{}.png'
Spx_File_Template = 'superpixel_seed/cityscapes/seeds_{}/train/label/{}.pkl'


class RegionActiveDataset(region_active_dataset.RegionActiveDataset):
    def __init__(self, args, trg_pool_dataset, trg_label_dataset):
        super().__init__(args, trg_pool_dataset, trg_label_dataset)
        self.root = self.trg_pool_dataset.root

    def expand_training_set(self, sample_region, selection_count, selection_method):
        """
        sample_region: sorted, scored regions (list)
            [
                (score, scan_file_id, suppix_id),
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
        for idx, x in enumerate(tqdm(sample_region)): # sorted tuples: (score, scan_file_id, suppix_id)
            _, nseg_file_id, suppix_id = x
            nseg, file_id = nseg_file_id.split('/')
            nseg = int(nseg)
            img_file_path = os.path.join(self.root, Img_File_Template.format(file_id.split('_')[0], file_id))
            lbl_file_path = os.path.join(self.root, Lbl_File_Template.format(nseg, file_id))
            spx_file_path = os.path.join(self.root, Spx_File_Template.format(nseg, file_id))
            
            '''Add into label dataset'''
            img_file_path_list = [i[0] for i in self.trg_label_dataset.im_idx] ### image name list for indexing
            if img_file_path not in img_file_path_list:
                ### generate new list element ex: '(img, {'128': (lbl, spx)})'
                lbl_dict = {nseg: (lbl_file_path, spx_file_path)}
                content = (img_file_path, lbl_dict)
                self.trg_label_dataset.im_idx.append(content) # im_idx labeled al set: join of three paths
            else:
                ### add new key to lbl_dict if empty
                current_idx = img_file_path_list.index(img_file_path)
                self.trg_label_dataset.im_idx[current_idx][1].setdefault(nseg, (lbl_file_path, spx_file_path))
            self.trg_label_dataset.suppix[spx_file_path].append(suppix_id) ### if empty, list is generated
            
            '''Remove it from unlabeled dataset'''
            self.trg_pool_dataset.suppix[spx_file_path].remove(suppix_id) ### remove superpixel id from list of superpixel within image
            if len(self.trg_pool_dataset.suppix[spx_file_path]) == 0: 
                ### remove candidate list from suppix dict
                self.trg_pool_dataset.suppix.pop(spx_file_path)
                ### remove nseg key from im_idx lbl_dict
                current_idx = img_file_path_list.index(img_file_path)
                self.trg_pool_dataset.im_idx[current_idx][1].pop(nseg)
                ### if lbl_dict is empty, remove the im_idx element 
                if len(self.trg_pool_dataset.im_idx[current_idx][1]) == 0:
                    del self.trg_pool_dataset.im_idx[current_idx]

            '''jump out the loop when exceeding max_selection_count'''
            if self.args.fair_counting and self.args.or_labeling:
                trg_index = self.trg_label_dataset.id_to_index[file_id]
                num_cls = self.trg_label_dataset.mseg_mh_cls[nseg][trg_index, suppix_id].sum()
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