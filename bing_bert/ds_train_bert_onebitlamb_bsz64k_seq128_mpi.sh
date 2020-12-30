#!/bin/bash

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=onebit_lamb_seq128_mpi
OUTPUT_DIR=${base_dir}/bert_model_outputs

mkdir -p $OUTPUT_DIR

NCCL_TREE_THRESHOLD=0 deepspeed --launcher=openmpi ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/bert_large_lamb.json \
--max_seq_length 128 \
--output_dir $OUTPUT_DIR \
--deepspeed_mpi \
--deepspeed \
--print_steps 40 \
--lr_schedule "EE" \
--lr_offset 10e-4 \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz64k_onebitlamb_config_seq128_mpi.json \
--data_path_prefix /data/bert \
--ckpt_to_save 150 \
&> $OUTPUT_DIR/${JOB_NAME}.log
#&> ${JOB_NAME}.log
