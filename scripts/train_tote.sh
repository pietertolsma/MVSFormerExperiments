#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/config_delftblue.json \
                                         --exp_name TOTE_DEFAULT \
                                         --data_path ../ToteMVS \
                                         --DDP