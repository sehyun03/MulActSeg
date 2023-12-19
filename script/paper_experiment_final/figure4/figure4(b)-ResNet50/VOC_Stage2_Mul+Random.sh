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
save_path=checkpoint/ms_deepstem50_random_dbdry_stage2_lr00002_nopredignore_voc
checkpoint_path=checkpoint/deepstem50_dbtim_voc_mul_lr_0.00001_trim5_my_random_sp150_nlbl10.0k_iter30.0k_method-active_joint_multi_lossdecomp-_coeff16.0_ignFalse_lr1e-05__1
lr=0.00001

### start ###
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

python train_stage2_AL_voc.py -p "$save_path" \
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