#!/bin/bash

CUDA_VISIBLE_DEVICES=0 ./main.py \
 --mode=train \
 --modality=fuse \
 --shuffle=1 \
 --lr=1e-3 \
 --batch_size=5 \
 --num_workers=5 \
 --resolution=112 \
 --num_frames=16
