#!/bin/bash

python train.py --config configs/3blocks_exp.json \
                                         --exp_name FINAL_3BLOCKS  \
                                         --data_path ../ToteMVS \
                                         --DDP