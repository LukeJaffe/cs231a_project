#!/bin/bash

CUDA_VISIBLE_DEVICES=0 ./main.py \
 --mode=test \
 --modality=pico \
 --load_modality=fuse \
 --partition_dir=/home/username/data/pico \
 --shuffle=1 \
 --lr=1e-3 \
 --batch_size=1 \
 --num_workers=5 \
 --resolution=112 \
 --num_frames=16
