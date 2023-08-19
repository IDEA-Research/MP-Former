#!/bin/bash
#SBATCH -J last_12ep
#SBATCH -p batch
#SBATCH --cpus-per-task=100
#SBATCH --gres=gpu:4
#SBATCH -N 1
#SBATCH -e job-%j.err
#SBATCH -o job-%j.out
out_dir=output/m2f_50ep_dn_mask_eval/
weights=model/model_final.pth
mkdir -p $out_dir
echo ${out_dir}/log
export DETECTRON2_DATASETS=dataset && python train_net.py --num-gpus 4 --eval-only  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep_DN_query.yaml OUTPUT_DIR $out_dir MODEL.WEIGHTS $weights \
	       MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME MultiScaleMaskedTransformerDecoderMaskDN  



