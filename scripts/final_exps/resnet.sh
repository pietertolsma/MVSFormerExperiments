#!/bin/bash

python train.py --config configs/resnet.json \
                                         --exp_name FINAL_RESNET  \
                                         --data_path ../ToteMVS \
                                         --DDP