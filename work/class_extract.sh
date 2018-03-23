#!/bin/bash

CUDA_VISIBLE_DEVICES=0 ./main.py \
 --mode=extract \
 --modality=dep \
 --batch_size=1
