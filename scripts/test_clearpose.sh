#!/bin/bash

#MODEL_WEIGHT_PATH="./saved/models/MVSFormer/TOTE_DEFAULT/model_last.pth"
MODEL_WEIGHT_PATH="checkpoints/best.pth"
OUTPUT_DIR="./outputs/clearpose_default"

CUDA_VISIBLE_DEVICES=0 python test.py --config configs/config_delftblue.json --mode val --dataset clearpose --batch_size 1 \
                                       --testpath ../ClearPose \
                                       --resume ${MODEL_WEIGHT_PATH} \
                                       --outdir ${OUTPUT_DIR} \
                                       --fusibile_exe_path ../fusible/fusibile \
                                       --interval_scale 1.06 --num_view 6 \
                                       --numdepth 192 --max_h 768 --max_w 768 --filter_method pcd \
                                       --disp_threshold 1.0 --num_consistent 1 --prob_threshold 0.0,0.0,0.0,0.0 \
                                       --combine_conf \
                                       --tmps 5.0,5.0,5.0,1.0\
                                       --transposed\
                                       --refine_steps 4
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