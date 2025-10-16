#!/bin/bash

set -x

export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTHONPATH=$PYTHONPATH:.
export ASCEND_RT_VISIBLE_DEVICES=6,7,8,9,10,11,12,13

NNODES=${NNODES:=1}
# GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi --list-gpus &> /dev/null; then
  NPROC_PER_NODE=${NPROC_PER_NODE:=$(nvidia-smi --list-gpus | wc -l)}
# NPU
else
  NPROC_PER_NODE=${NPROC_PER_NODE:=$(ll /dev/davinci* | grep -v "davinci_manager" | wc -l)}
fi
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}

if [[ "$NNODES" == "1" ]]; then
  additional_args="$additional_args --standalone"
else
  additional_args="--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
fi

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs

torchrun \
  --nnodes=$NNODES \
  --nproc-per-node=$NPROC_PER_NODE \
  --node-rank=$NODE_RANK \
  $additional_args $@ 2>&1 | tee logs/train_${logfile}.log
