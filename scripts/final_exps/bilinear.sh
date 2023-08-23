#!/bin/bash

python train.py --config configs/bilinear_exp.json \
                                         --exp_name FINAL_BILINEAR  \
                                         --data_path ../ToteMVS \
                                         --DDP