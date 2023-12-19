# Active Learning for Semantic Segmentation with Multi-class Label Query

### [Project page]()
This repository is the official implementation of "[Active Learning for Semantic Segmentation with Multi-class Label Query. Hwang et al., Neurips 2023](https://arxiv.org/abs)".

> Active Learning for Semantic Segmentation with Multi-class Label Query    
> [Sehyun Hwang](https://sehyun03.github.io/), [Sohyun Lee](https://sohyun-l.github.io/), [Hoyoung Kim](https://cskhy16.github.io/), Minhyeon Oh, [Jungseul Ok](https://sites.google.com/view/jungseulok), [Suha Kwak](https://suhakwak.github.io/)     
> POSTECH   
> NeurIPS 2023

## Overview
This paper proposes a new active learning method for semantic segmentation.
The core of our method lies in a new annotation query design.
It samples informative local image regions (_e.g._,, superpixels), and for each of such regions, asks an oracle for a multi-hot vector indicating all classes existing in the region. 
This multi-class labeling strategy is substantially more efficient than existing ones like segmentation, polygon, and even dominant class labeling in terms of annotation time per click. 
However, it introduces the class ambiguity issue in training since it assigns partial labels (_i.e._, a set of candidate classes) to individual pixels.
We thus propose a new algorithm for learning semantic segmentation while disambiguating the partial labels in two stages.
In the first stage, it trains a segmentation model directly with the partial labels through two new loss functions motivated by partial label learning and multiple instance learning. 
In the second stage, it disambiguates the partial labels by generating pixel-wise pseudo labels, which are used for supervised learning of the model.
Equipped with a new acquisition function dedicated to the multi-class labeling, our method outperformed previous work on Cityscapes and PASCAL VOC 2012 while spending less annotation cost.

<!-- ### What is multi-class labeling -->

<!-- ### Training altorithm -->

<!-- ### Experimental results -->

<!-- ## Citation -->
<!-- If you find our code or paper useful, please consider citing our paper: -->


## Requirements

To install requirements:

```setup
conda env create -f actsegmul.yml
```

## Dataset & pretrained model
Prepare dataset:

- Download superpixel region and multi-class label in [data](https://e.pcloud.link/publink/show?code=XZoAgsZXlmcsf7tA2pUpQiGTwnp4zadenUX), and put 'data' under `<GitRoot>`.
- Download [Cityscapes](https://www.cityscapes-dataset.com) and put 'gtFine' and 'leftImg8bit' in '`<GitRoot>`/data/Cityscapes' (check the following directory tree).
- Download [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and put 'VOC2012' in '`<GitRoot>`/data/VOCdevkit' (check the following directory tree).

Prepare trained model:
- Download initial weight and pretrained model in [checkpoint](https://e.pcloud.link/publink/show?code=XZvAgsZ5h0IrVsf9vLkTlU2fKTzOXp2s5Ak), and put 'checkpoint' under `<GitRoot>`.

The directory needs to be:
```
<GitRoot>
├── checkpoint
    ├── city_res50deepstem_imagenet_pretrained.tar
    ├── resnet50_deepstem.pth
├── data
    ├── Cityscapes
        ├── gtFine
        ├── leftImg8bit
        ├── superpixel_seed
    ├── VOCdevkit
        ├── VOC2012
        ├── superpixels
├── ...
```

## Training

To train the model(s) in the paper, run this command:

Cityscapes
```
bash script/train_city_mul_res50.sh
```

PASCAL VOC
```
bash script/train_city_mul_res50.sh
```

## **Naive Inference (top-1 based stage 1 pseudo label generation)**

To generate pseudo labels from stage 1 model with naive argmax, run this command:

PASCAL VOC
```
checkpoint_path=checkpoint/voc_actseg_stage1_pretrained_weights/deepstem50_ppredpwr_coff12_voc_mul_pretrained_stage1
# round number from 1 to 5.
round=1
lr=0.00001
python eval_AL_voc.py -p "$checkpoint_path" \
--stage2 \
--datalist_path "$checkpoint_path"/datalist_0"$round".pkl \
--init_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--resume_checkpoint "$checkpoint_path"/checkpoint0"$round".tar \
--method eval_save_cosplbl_naive_voc \
--or_labeling \
--train_transform eval_spx_identity \
--loader eval_region_voc_all \
--trim_multihot_boundary \
--trim_kernel_size 5 \
--nseg 150 \
--model deeplabv3pluswn_resnet50deepstem \
--separable_conv \
--val_batch_size 1 \
--wandb_tags eval \
--num_workers 8 \
--plbl_type naive_argmax \
--dontlog
```

## Evaluation

To evaluate trained model, run:

Cityscapes
```
python eval_AL.py -p checkpoint/eval \
--init_checkpoint checkpoint/<your_checkpoint.tar> \
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
```

PASCAL VOC
```
[TODO]
```

To evaluate provided pre-trained models, run:

Cityscapes
```eval provided checkpoint for 5 rounds
bash script/eval_city_mul_res50.sh
```

PASCAL VOC
```eval provided checkpoint for 5 rounds
[TODO]
```

## Acknowledgements
Our source code is based on amazing repository [D2ADA: Dynamic Density-aware Active Domain Adaptation for Semantic Segmentation](https://github.com/tsunghan-wu/D2ADA/tree/main).\
Also, we modified and adapted on these great repositories:
- [Revisiting Superpixels for Active Learning in Semantic Segmentation With Realistic Annotation Costs](https://github.com/cailile/Revisiting-Superpixels-for-Active-Learning/tree/master)
- [Perturbed and Strict Mean Teachers for Semi-supervised Semantic Segmentation](https://github.com/yyliu01/PS-MT/tree/main)

If you use our model, please consider citing them as well.