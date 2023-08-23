#!/bin/bash

python train.py --config configs/dtu_resnet.json \
                                         --exp_name DTU_RESNET \
                                         --data_path ../DTU \
                                         --DDP