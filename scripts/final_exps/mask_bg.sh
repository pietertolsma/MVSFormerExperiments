#!/bin/bash

python train.py --config configs/mask_bg_exp.json \
                                         --exp_name FINAL_MASKBG \
                                         --data_path ../ToteMVS \
                                         --DDP