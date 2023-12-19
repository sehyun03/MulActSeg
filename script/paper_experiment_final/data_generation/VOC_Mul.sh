# nseg 150 equals to 32x32 size superpixel
python tools/label_assignment_tensor_voc.py \
--nseg 150 \
--num_worker 12 \
--save_data_dir /home/sehyun/data/VOCdevkit/superpixels/pascal_voc_seg/seeds_32/train/gtFine_multi_tensor_trim_5x5 \
--trim_multihot_boundary \
--trim_kernel_size 5