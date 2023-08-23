#!/bin/bash

MODEL_WEIGHT_PATH="./saved/models/MVSFormer/DTU_RESNET/model_last_4.pth"
# MODEL_WEIGHT_PATH="checkpoints/best.pth"
OUTPUT_DIR="./outputs/dtu_resnet"

CUDA_VISIBLE_DEVICES=0 python test.py --config configs/dtu_resnet.json --mode val --dataset dtu --batch_size 1 \
                                       --testpath ../DTU \
                                       --resume ${MODEL_WEIGHT_PATH} \
                                       --outdir ${OUTPUT_DIR} \
                                       --fusibile_exe_path ../fusible/fusibile \
                                       --interval_scale 1.06 --num_view 6 \
                                       --numdepth 192 --max_h 768 --max_w 768 --filter_method pcd \
                                       --disp_threshold 1.0 --num_consistent 1 --prob_threshold 0.0,0.0,0.0,0.0 \
                                       --combine_conf \
                                       --tmps 5.0,5.0,5.0,1.0\
                                       --transposed\
                                       --useresnet\
                                       --backbonepretrained
                                       # --transposed\
                                       # --useresnet\
                                       # --backbonepretrained\
                                       # --vit_only
                                   #    --refine_steps 3\
                                      # --customsoftmax\
                                     #  --transposed
                                      #  --useresnet\
                                      #  --backbonepretrained
                                   #    --customsoftmax
                                    #   --transposed\
                                       #--backbonepretrained
                                      # --backbonepretrained
                                    #    --rotated

                                    #--useresnet