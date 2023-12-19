### =======
### Stage 1
### =======
python train_AL.py -p checkpoint/city_mul_res50 \
--model deeplabv3pluswn_resnet50deepstem \
--init_checkpoint checkpoint/city_res50deepstem_imagenet_pretrained.tar \
--method active_joint_multi_predignore_lossdecomp \
--active_method my_bvsb_predclsbal_pwr_banignore \
--cls_weight_coeff 6.0 \
--or_labeling \
--fair_counting \
--loss_type joint_multi_loss \
--nseg 2048 \
--scheduler poly \
--train_lr 0.00002 \
--start_over \
--num_workers 12 \
--finetune_itrs 80000 \
--val_period 5000 \
--val_start 0 \
--separable_conv \
--max_iterations 5 \
--train_transform rescale_769_multi_notrg \
--loader region_cityscapes_or_tensor \
--active_selection_size 100000 \
--wandb_tags 50k base cos \
--multi_ce_temp 0.1 \
--group_ce_temp 0.1 \
--ce_temp 0.1 \
--coeff 16.0 \
--coeff_mc 8.0 \
--coeff_gm 1.0 \
--init_iteration 2 \
--trim_kernel_size 5 \
--trim_multihot_boundary \
--init_iteration 1

### =======
### Stage 2
### =======
checkpoint_path=checkpoint/city_mul_res50_my_bvsb_predclsbal_pwr_banignore_sp2048_nlbl100.0k_iter80.0k_method-active_joint_multi_predignore_lossdecomp-_coeff16.0_ignFalse_lr2e-05_
#round=1
round=1
python eval_AL.py -p "$checkpoint_path" \
--stage2 \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--init_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--method eval_save_cosplbl_prop_includeonehot \
--or_labeling \
--train_transform eval_spx \
--loader eval_region_cityscapes_all \
--trim_multihot_boundary \
--trim_kernel_size 5 \
--nseg 2048 \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--val_batch_size 1 \
--wandb_tags eval \
--num_workers 8 \
--dontlog

python train_stage2_AL.py -p "$checkpoint_path" \
--stage2 \
--init_iteration "$round" \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".pkl \
--init_checkpoint checkpoint/city_res50deepstem_imagenet_pretrained.tar \
--finetune_itrs 80000 \
--val_period 5000 \
--val_start 0 \
--active_selection_size 50000 \
--train_transform rescale_769_nospx \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--optimizer adamw \
--train_lr 0.00004 \
--ce_temp 0.1 \
--cls_lr_scale 10.0 \
--scheduler poly \
--train_batch_size 4 \
--num_workers 10 \
--val_batch_size 4 \
--nseg 2048 \
--dominant_labeling \
--method active_predignore \
--loader region_cityscapes_plbl \
--wandb_tags 50k plbl cos

#round=2
round=2
python eval_AL.py -p "$checkpoint_path" \
--stage2 \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--init_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--method eval_save_cosplbl_prop_includeonehot \
--or_labeling \
--train_transform eval_spx \
--loader eval_region_cityscapes_all \
--trim_multihot_boundary \
--trim_kernel_size 5 \
--nseg 2048 \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--val_batch_size 1 \
--wandb_tags eval \
--num_workers 8 \
--dontlog

python train_stage2_AL.py -p "$checkpoint_path" \
--stage2 \
--init_iteration "$round" \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".pkl \
--init_checkpoint checkpoint/city_res50deepstem_imagenet_pretrained.tar \
--finetune_itrs 80000 \
--val_period 5000 \
--val_start 0 \
--active_selection_size 50000 \
--train_transform rescale_769_nospx \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--optimizer adamw \
--train_lr 0.00004 \
--ce_temp 0.1 \
--cls_lr_scale 10.0 \
--scheduler poly \
--train_batch_size 4 \
--num_workers 10 \
--val_batch_size 4 \
--nseg 2048 \
--dominant_labeling \
--method active_predignore \
--loader region_cityscapes_plbl \
--wandb_tags 50k plbl cos

#round=3
round=3
python eval_AL.py -p "$checkpoint_path" \
--stage2 \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--init_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--method eval_save_cosplbl_prop_includeonehot \
--or_labeling \
--train_transform eval_spx \
--loader eval_region_cityscapes_all \
--trim_multihot_boundary \
--trim_kernel_size 5 \
--nseg 2048 \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--val_batch_size 1 \
--wandb_tags eval \
--num_workers 8 \
--dontlog

python train_stage2_AL.py -p "$checkpoint_path" \
--stage2 \
--init_iteration "$round" \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".pkl \
--init_checkpoint checkpoint/city_res50deepstem_imagenet_pretrained.tar \
--finetune_itrs 80000 \
--val_period 5000 \
--val_start 0 \
--active_selection_size 50000 \
--train_transform rescale_769_nospx \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--optimizer adamw \
--train_lr 0.00004 \
--ce_temp 0.1 \
--cls_lr_scale 10.0 \
--scheduler poly \
--train_batch_size 4 \
--num_workers 10 \
--val_batch_size 4 \
--nseg 2048 \
--dominant_labeling \
--method active_predignore \
--loader region_cityscapes_plbl \
--wandb_tags 50k plbl cos

#round=4
round=4
python eval_AL.py -p "$checkpoint_path" \
--stage2 \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--init_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--method eval_save_cosplbl_prop_includeonehot \
--or_labeling \
--train_transform eval_spx \
--loader eval_region_cityscapes_all \
--trim_multihot_boundary \
--trim_kernel_size 5 \
--nseg 2048 \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--val_batch_size 1 \
--wandb_tags eval \
--num_workers 8 \
--dontlog

python train_stage2_AL.py -p "$checkpoint_path" \
--stage2 \
--init_iteration "$round" \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".pkl \
--init_checkpoint checkpoint/city_res50deepstem_imagenet_pretrained.tar \
--finetune_itrs 80000 \
--val_period 5000 \
--val_start 0 \
--active_selection_size 50000 \
--train_transform rescale_769_nospx \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--optimizer adamw \
--train_lr 0.00004 \
--ce_temp 0.1 \
--cls_lr_scale 10.0 \
--scheduler poly \
--train_batch_size 4 \
--num_workers 10 \
--val_batch_size 4 \
--nseg 2048 \
--dominant_labeling \
--method active_predignore \
--loader region_cityscapes_plbl \
--wandb_tags 50k plbl cos

#round=5
round=5
python eval_AL.py -p "$checkpoint_path" \
--stage2 \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--init_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--method eval_save_cosplbl_prop_includeonehot \
--or_labeling \
--train_transform eval_spx \
--loader eval_region_cityscapes_all \
--trim_multihot_boundary \
--trim_kernel_size 5 \
--nseg 2048 \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--val_batch_size 1 \
--wandb_tags eval \
--num_workers 8 \
--dontlog

python train_stage2_AL.py -p "$checkpoint_path" \
--stage2 \
--init_iteration "$round" \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".pkl \
--init_checkpoint checkpoint/city_res50deepstem_imagenet_pretrained.tar \
--finetune_itrs 80000 \
--val_period 5000 \
--val_start 0 \
--active_selection_size 50000 \
--train_transform rescale_769_nospx \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--optimizer adamw \
--train_lr 0.00004 \
--ce_temp 0.1 \
--cls_lr_scale 10.0 \
--scheduler poly \
--train_batch_size 4 \
--num_workers 10 \
--val_batch_size 4 \
--nseg 2048 \
--dominant_labeling \
--method active_predignore \
--loader region_cityscapes_plbl \
--wandb_tags 50k plbl cos