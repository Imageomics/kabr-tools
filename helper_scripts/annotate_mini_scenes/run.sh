#!/bin/bash
python annotate.py --checkpoint model/checkpoint_epoch_00075.pyth --gpu_num 1 --miniscene $1 --output $2