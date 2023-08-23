#!/bin/bash

python train.py --config configs/softmax_exp.json \
                                         --exp_name FINAL_SOFTMAX  \
                                         --data_path ../ToteMVS \
                                         --DDP