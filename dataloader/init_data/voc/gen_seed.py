import numpy as np
import json
from tqdm import tqdm


''' seed superpixel dict update '''
nseg_list = [16]

for nseg in nseg_list:
    
    with open('/root/data1/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt','r') as f:
        seed = f.read().splitlines()    
    # seed_spx = [i.split('\t')[-1] for i in seed]
    seed_spx = seed
    seed_dict = {}
    
    for i in tqdm(seed_spx):
        # import pdb; pdb.set_trace()
        # valid_idxes = np.load('/home/sehyun/data/Cityscapes/{}'.format(i), allow_pickle=True)['valid_idxes']
        valid_idxes = np.load('/root/data1/VOCdevkit/superpixels/pascal_voc_seg/seeds_16/train/label/{}.pkl'.format(i), allow_pickle=True)['valid_idxes']
        max_value = valid_idxes.max()
        seed_dict[i] = (max_value.item()+1, [i for i in range(max_value+1) if i not in valid_idxes])

    with open('train_seed{}.dict'.format(nseg),'w') as f:
        json.dump(seed_dict, f)

