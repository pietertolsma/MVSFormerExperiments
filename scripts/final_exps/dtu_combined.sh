#!/bin/bash

python train.py --config configs/dtu_combined_exp.json \
                                         --exp_name DTU_COMBINED \
                                         --data_path ../DTU \
                                         --DDP