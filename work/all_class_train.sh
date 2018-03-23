#!/bin/bash

CUDA_VISIBLE_DEVICES=0 ./main.py \
 --mode=train \
 --modality=all \
 --batch_size=16
