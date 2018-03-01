#!/bin/bash

CUDA_VISIBLE_DEVICES=0 ./main.py \
 --mode=train \
 --modality=dep \
 --batch_size=32
