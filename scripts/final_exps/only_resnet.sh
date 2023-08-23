#!/bin/bash

python train.py --config configs/only_resnet.json \
                                         --exp_name FINAL_RESNETONLY  \
                                         --data_path ../ToteMVS \
                                         --DDP