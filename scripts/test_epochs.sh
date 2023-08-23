#!/bin/bash

for i in {11..16}
do
   MODEL_WEIGHT_PATH="./saved/models/MVSFormer/TOTE_DEFAULT/model_last_$i.pth"
   OUTPUT_DIR="./outputs/final_default_epoch_$i"
   CUDA_VISIBLE_DEVICES=0 python test.py --config configs/config_delftblue.json --mode val --dataset tote --batch_size 1 \
                                       --testpath ../ToteMVS \
                                       --resume ${MODEL_WEIGHT_PATH} \
                                       --outdir ${OUTPUT_DIR} \
                                       --fusibile_exe_path ../fusible/fusibile \
                                       --interval_scale 1.06 --num_view 6 \
                                       --numdepth 192 --max_h 768 --max_w 768 --filter_method pcd \
                                       --disp_threshold 1.0 --num_consistent 1 --prob_threshold 0.0,0.0,0.0,0.0 \
                                       --combine_conf \
                                       --tmps 5.0,5.0,5.0,1.0\
                                       --transposed
                                   #    --use_vit
done