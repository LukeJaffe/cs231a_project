#!/bin/bash

CUDA_VISIBLE_DEVICES=0 ./auto_main.py \
 --resolution=28 \
 --batch_size=32
