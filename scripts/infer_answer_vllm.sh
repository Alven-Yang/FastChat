#!/bin/bash

model_name=$1
model_id=$2
data_id=$3
GPU=$4
tensor_parallel_size=$5
output_file=$6

echo $data_id

export PYTHONPATH="/home/workspace/FastChat/fastchat:$PYTHONPATH"

cd /home/workspace/FastChat/fastchat/llm_judge

python gen_model_answer.py --model-path $model_name --model-id $model_id --bench-name $data_id --answer-file $output_file
