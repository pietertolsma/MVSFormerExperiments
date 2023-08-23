#!/bin/bash

python train.py --config configs/fizyr_final.json \
                                         --exp_name FINAL_FIZYR  \
                                         --data_path ../ToteMVS \
                                         --DDP