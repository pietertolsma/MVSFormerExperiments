#!/bin/bash

python train.py --config configs/combined_exp.json \
                                         --exp_name FINAL_COMBINED_2  \
                                         --data_path ../ToteMVS \
                                         --DDP