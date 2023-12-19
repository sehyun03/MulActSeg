### =======
### Stage 1
### =======
python train_AL_voc.py -p checkpoint/voc_mul_res50 \
--model deeplabv3pluswn_resnet50deepstem \
--init_checkpoint checkpoint/res50wndeepstem_imagenet_pretrained.tar \
--method active_joint_multi_lossdecomp \
--active_method my_bvsb_predclsbal_pwr \
--cls_weight_coeff 12.0 \
--or_labeling \
--fair_counting \
--loss_type joint_multi_loss \
--nseg 150 \
--scheduler poly \
--separable_conv \
--train_lr 0.00001 \
--start_over \
--num_workers 12 \
--finetune_itrs 30000 \
--val_period 2500 \
--val_start 0 \
--max_iterations 5 \
--train_transform rescale_513_multi_notrg \
--loader region_voc_or_tensor \
--wandb_tags 10k,base,cos \
--active_selection_size 10000 \
--multi_ce_temp 0.1 \
--group_ce_temp 0.1 \
--ce_temp 0.1 \
--coeff 16.0 \
--coeff_mc 8.0 \
--coeff_gm 1.0 \
--trim_kernel_size 5 \
--trim_multihot_boundary \
--cls_weight_coeff 12.0 \
--init_iteration 1

### =======
### Stage 2
### =======
checkpoint_path=checkpoint/voc_mul_res50_my_bvsb_predclsbal_pwr_sp150_nlbl10.0k_iter30.0k_method-active_joint_multi_lossdecomp-_coeff16.0_ignFalse_lr1e-05_
# round=1
round=1
lr=0.00001
python eval_AL_voc.py -p "$checkpoint_path" \
--stage2 \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--init_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--method eval_save_cosplbl_prop_includeonehot_voc_ms \
--or_labeling \
--train_transform eval_spx_identity_ms \
--loader eval_region_voc_all_ms \
--trim_multihot_boundary \
--trim_kernel_size 5 \
--nseg 150 \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--val_batch_size 1 \
--wandb_tags eval \
--num_workers 8 \
--dontlog

python train_stage2_AL_voc.py -p "$checkpoint_path" \
--stage2 \
--init_iteration "$round" \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".pkl \
--init_checkpoint checkpoint/res50wndeepstem_imagenet_pretrained.tar \
--finetune_itrs 30000 \
--val_period 2500 \
--val_start 0 \
--active_selection_size 10000 \
--loader region_voc_plbl \
--train_transform rescale_513_notrg \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--optimizer adamw \
--train_lr "$lr" \
--ce_temp 0.1 \
--cls_lr_scale 10.0 \
--scheduler poly \
--train_batch_size 4 \
--num_workers 10 \
--val_batch_size 4 \
--nseg 150 \
--dominant_labeling \
--method active \
--plbl_type "ms" \
--wandb_tags 10k base cos

# round=2
round=2
lr=0.00001
python eval_AL_voc.py -p "$checkpoint_path" \
--stage2 \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--init_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--method eval_save_cosplbl_prop_includeonehot_voc_ms \
--or_labeling \
--train_transform eval_spx_identity_ms \
--loader eval_region_voc_all_ms \
--trim_multihot_boundary \
--trim_kernel_size 5 \
--nseg 150 \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--val_batch_size 1 \
--wandb_tags eval \
--num_workers 8 \
--dontlog

python train_stage2_AL_voc.py -p "$checkpoint_path" \
--stage2 \
--init_iteration "$round" \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".pkl \
--init_checkpoint checkpoint/res50wndeepstem_imagenet_pretrained.tar \
--finetune_itrs 30000 \
--val_period 2500 \
--val_start 0 \
--active_selection_size 10000 \
--loader region_voc_plbl \
--train_transform rescale_513_notrg \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--optimizer adamw \
--train_lr "$lr" \
--ce_temp 0.1 \
--cls_lr_scale 10.0 \
--scheduler poly \
--train_batch_size 4 \
--num_workers 10 \
--val_batch_size 4 \
--nseg 150 \
--dominant_labeling \
--method active \
--plbl_type "ms" \
--wandb_tags 10k base cos

# round=3
round=3
lr=0.00001
python eval_AL_voc.py -p "$checkpoint_path" \
--stage2 \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--init_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--method eval_save_cosplbl_prop_includeonehot_voc_ms \
--or_labeling \
--train_transform eval_spx_identity_ms \
--loader eval_region_voc_all_ms \
--trim_multihot_boundary \
--trim_kernel_size 5 \
--nseg 150 \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--val_batch_size 1 \
--wandb_tags eval \
--num_workers 8 \
--dontlog

python train_stage2_AL_voc.py -p "$checkpoint_path" \
--stage2 \
--init_iteration "$round" \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".pkl \
--init_checkpoint checkpoint/res50wndeepstem_imagenet_pretrained.tar \
--finetune_itrs 30000 \
--val_period 2500 \
--val_start 0 \
--active_selection_size 10000 \
--loader region_voc_plbl \
--train_transform rescale_513_notrg \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--optimizer adamw \
--train_lr "$lr" \
--ce_temp 0.1 \
--cls_lr_scale 10.0 \
--scheduler poly \
--train_batch_size 4 \
--num_workers 10 \
--val_batch_size 4 \
--nseg 150 \
--dominant_labeling \
--method active \
--plbl_type "ms" \
--wandb_tags 10k base cos

# round=4
round=4
lr=0.00001
python eval_AL_voc.py -p "$checkpoint_path" \
--stage2 \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--init_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--method eval_save_cosplbl_prop_includeonehot_voc_ms \
--or_labeling \
--train_transform eval_spx_identity_ms \
--loader eval_region_voc_all_ms \
--trim_multihot_boundary \
--trim_kernel_size 5 \
--nseg 150 \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--val_batch_size 1 \
--wandb_tags eval \
--num_workers 8 \
--dontlog

python train_stage2_AL_voc.py -p "$checkpoint_path" \
--stage2 \
--init_iteration "$round" \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".pkl \
--init_checkpoint checkpoint/res50wndeepstem_imagenet_pretrained.tar \
--finetune_itrs 30000 \
--val_period 2500 \
--val_start 0 \
--active_selection_size 10000 \
--loader region_voc_plbl \
--train_transform rescale_513_notrg \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--optimizer adamw \
--train_lr "$lr" \
--ce_temp 0.1 \
--cls_lr_scale 10.0 \
--scheduler poly \
--train_batch_size 4 \
--num_workers 10 \
--val_batch_size 4 \
--nseg 150 \
--dominant_labeling \
--method active \
--plbl_type "ms" \
--wandb_tags 10k base cos

# round=5
round=5
lr=0.00001
python eval_AL_voc.py -p "$checkpoint_path" \
--stage2 \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--init_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--method eval_save_cosplbl_prop_includeonehot_voc_ms \
--or_labeling \
--train_transform eval_spx_identity_ms \
--loader eval_region_voc_all_ms \
--trim_multihot_boundary \
--trim_kernel_size 5 \
--nseg 150 \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--val_batch_size 1 \
--wandb_tags eval \
--num_workers 8 \
--dontlog

python train_stage2_AL_voc.py -p "$checkpoint_path" \
--stage2 \
--init_iteration "$round" \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".pkl \
--init_checkpoint checkpoint/res50wndeepstem_imagenet_pretrained.tar \
--finetune_itrs 30000 \
--val_period 2500 \
--val_start 0 \
--active_selection_size 10000 \
--loader region_voc_plbl \
--train_transform rescale_513_notrg \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--optimizer adamw \
--train_lr "$lr" \
--ce_temp 0.1 \
--cls_lr_scale 10.0 \
--scheduler poly \
--train_batch_size 4 \
--num_workers 10 \
--val_batch_size 4 \
--nseg 150 \
--dominant_labeling \
--method active \
--plbl_type "ms" \
--wandb_tags 10k base cos