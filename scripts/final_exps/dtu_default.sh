#!/bin/bash

python train.py --config configs/dtu_default.json \
                                         --exp_name DTU_DEFAULT \
                                         --data_path ../DTU \
                                         --DDP