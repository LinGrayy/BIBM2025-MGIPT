#!/bin/bash

#Please modify the following roots to yours.
dataset_root=/data/llr/Polyp/
model_root=/pretrained_model/POLYP/models/
path_save_log=/POLYP/logs/

#Dataset [BKAI, CVC-ClinicDB, ETIS-LaribPolypDB, Kvasir-SEG]
Source=BKAI

#Optimizer
optimizer=Adam
lr=0.01

#Hyperparameters
prompt_alpha=0.01
iters=3

#Command
cd POLYP
CUDA_VISIBLE_DEVICES=2 python ours.py \
--dataset_root $dataset_root --model_root $model_root --path_save_log $path_save_log \
--Source_Dataset $Source \
--optimizer $optimizer --lr $lr \
--prompt_alpha $prompt_alpha --iters $iters