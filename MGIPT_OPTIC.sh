#!/bin/bash

#Please modify the following roots to yours.
dataset_root=/data/llr/Fundus/
model_root=/pretrained_model/OPTIC/models/
path_save_log=/OPTIC/logs/

#Dataset [RIM_ONE_r3, REFUGE, ORIGA, REFUGE_Valid, Drishti_GS]
Source=RIM_ONE_r3    

#Optimizer
optimizer=Adam
lr=0.05

#Hyperparameters
prompt_alpha=0.01
iters=3

#Command
cd OPTIC
CUDA_VISIBLE_DEVICES=0 python ours.py \
--dataset_root $dataset_root --model_root $model_root --path_save_log $path_save_log \
--Source_Dataset $Source \
--optimizer $optimizer --lr $lr \
--prompt_alpha $prompt_alpha --iters $iters