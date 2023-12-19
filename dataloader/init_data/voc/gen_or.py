
prefix = '/root/code/query_design_for_active_seg/data/VOCdevkit/superpixels/pascal_voc_seg/seeds_16/train/gtFine_or'

with open('train_seed16.txt','r') as f:
    seed = f.read().splitlines()
import pdb; pdb.set_trace()
target_path = [i.split('\t')[0] for i in seed]
target_id = [i.split('/')[-1].split('_gtFine')[0] for i in target_path]
replaced_path = ['{}/{}.npy'.format(prefix, i.split('/')[-1]) for i in target_id]
seed_modified = ['{}\t{}\t{}'.format(i.split('\t')[0], j, i.split('\t')[2]) for i,j in zip(seed, replaced_path)]

with open('train_seed16_or.txt', 'w') as f:
    for i in seed_modified:
        f.write('{}\n'.format(i))

# ''' seed 128/512 or target list generaiton '''

# with open('train_seed32_or.txt', 'r') as f:
#     seed = f.read().splitlines()
    
# seed_512 = [i.replace('seeds_2048', 'seeds_512') for i in seed]
# seed_128 = [i.replace('seeds_2048', 'seeds_128') for i in seed]

# with open('train_seed512_or.txt', 'w') as f:
#     for i in seed_512:
#         f.write('{}\n'.format(i))
        
# with open('train_seed128_or.txt', 'w') as f:
#     for i in seed_128:
#         f.write('{}\n'.format(i))