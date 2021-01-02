#! /bin/bash

# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=1

python pretrain_gpt2.py \
       --num-layers 16 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 2 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 50 \
       --save checkpoints/gpt2_345m \
       --load checkpoints/gpt2_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --cache-dir cache \
       --split 100 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .1 \
       --checkpoint-activations \


set +x
