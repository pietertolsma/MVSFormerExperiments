#!/bin/bash

python train.py --config configs/only_vit.json \
                                         --exp_name FINAL_VITONLY  \
                                         --data_path ../ToteMVS \
                                         --DDP