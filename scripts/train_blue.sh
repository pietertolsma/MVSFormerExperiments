#!/bin/bash

python train.py --config configs/config_delftblue.json \
                                         --exp_name TOTE_DEFAULT  \
                                         --data_path ../ToteMVS \
                                         --DDP