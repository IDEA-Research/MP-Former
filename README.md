# MP-Former
This is the official implementation of the paper "[MP-Former: Mask-Piloted Transformer for Image Segmentation](https://arxiv.org/pdf/2303.07336.pdf)". Accepted to CVPR 2023.

## News
- 8/19/2023: Code is released.

## Install
The installation is exactly same as [Mask2Former](https://github.com/facebookresearch/Mask2Former)

## Train
The following command is for no noise and all-layer MP training setting for instance segmentation.
```shell
bash run_50ep_no_noise_all_ly.sh
```
The following command is for no noise and all-layer MP training setting for panoptic segmentation.
```shell
bash run_50ep_no_noise_all_ly_panoptic.sh
```
## Eval
You can set "weights" to the path of the checkpoint.
```shell
bash eval.sh
```
## Model
This [checkpoint](https://github.com/IDEA-Research/MP-Former/releases/download/checkpoint/model_final.pth) is the 12-epoch checkpoint (last row in Table 7 of the paper). You are expected to have AP: 40.15 if you evaluate it.
