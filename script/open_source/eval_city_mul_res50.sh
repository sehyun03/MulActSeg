python eval_AL.py -p checkpoint/eval \
--init_checkpoint checkpoint/stage2_checkpoint01.tar \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--stage2 \
--method eval_naive \
--wandb_tags none \
--loader region_cityscapes_all \
--train_transform eval_spx \
--nseg 2048 \
--val_batch_size 1 \
--dontlog

python eval_AL.py -p checkpoint/eval \
--init_checkpoint checkpoint/stage2_checkpoint02.tar \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--stage2 \
--method eval_naive \
--wandb_tags none \
--loader region_cityscapes_all \
--train_transform eval_spx \
--nseg 2048 \
--val_batch_size 1 \
--dontlog

python eval_AL.py -p checkpoint/eval \
--init_checkpoint checkpoint/stage2_checkpoint03.tar \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--stage2 \
--method eval_naive \
--wandb_tags none \
--loader region_cityscapes_all \
--train_transform eval_spx \
--nseg 2048 \
--val_batch_size 1 \
--dontlog

python eval_AL.py -p checkpoint/eval \
--init_checkpoint checkpoint/stage2_checkpoint04.tar \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--stage2 \
--method eval_naive \
--wandb_tags none \
--loader region_cityscapes_all \
--train_transform eval_spx \
--nseg 2048 \
--val_batch_size 1 \
--dontlog

python eval_AL.py -p checkpoint/eval \
--init_checkpoint checkpoint/stage2_checkpoint05.tar \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--stage2 \
--method eval_naive \
--wandb_tags none \
--loader region_cityscapes_all \
--train_transform eval_spx \
--nseg 2048 \
--val_batch_size 1 \
--dontlog