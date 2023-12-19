#!/bin/sh
## JOB 이름
#SBATCH -J stage2 # Job name

## LOG 적을 위치
#SBATCH -o sbatch_log/pytorch-1gpu.%j.out # Name of stdout output file (%j expands to %jobId)

## GPU 종류
#SBATCH -p 3090 # queue name or partiton name titanxp/titanrtx/2080ti/3090

## 최대 사용 시간
#SBATCH -t 3-00:00:00 # Run time (hh:mm:ss) - 1.5 hours

## 노드 지정하지않기
# #SBATCH --nodes=1
## 제외할 node (사용할거면 # 한 개만)
# #SBATCH --exclude=n28

## 사용할 node (사용할거면 # 한 개만)
# #SBTACH --nodelist=n29

#### Select GPU
## gpu 개수
#SBATCH --gres=gpu:1 # number of gpus you want to use
#SBTACH --ntasks=1
##SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10

cd $SLURM_SUBMIT_DIR
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

## path Erase because of the crash
#module load postech

echo "Start"
export WANDB_SPAWN_METHOD=fork

date
nvidia-smi
round=1
checkpoint_path=checkpoint/deepstem101_rand_mul_bdry5_my_random_sp2048_nlbl100.0k_iter80.0k_method-active_joint_multi_predignore_lossdecomp-_coeff16.0_ignFalse_lr2e-05_
### 'eval_AL.py': eval_AL stage 2, pseudo label generation
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
--model deeplabv3pluswn_resnet101deepstem \
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
--init_checkpoint checkpoint/city_res101deepstem_imagenet_pretrained.tar \
--finetune_itrs 80000 \
--val_period 5000 \
--val_start 0 \
--active_selection_size 50000 \
--train_transform rescale_769_nospx \
--model deeplabv3pluswn_resnet101deepstem \
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