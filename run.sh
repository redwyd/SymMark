#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
seed=42

watermark_algorithm="KGW" # KgwExp UniEXP
watermark_type='I'  # 'S': Series 'H': Hybrid 'I': Independent 'P': Parallel
mode="yrdy" # 'train' or 'test'
attack_method="None" # None Word-D Word-S-DICT Word-S-BERT Copy-Paste Doc-P-GPT Doc-P-Dipper Doc-P-Dipper-1 Translation
text_source="generated"   # natural or generated

dataset_name="c4"
dataset_size=20
dataset_file="${dataset_name}_${dataset_size}.jsonl"
dataset_path=./data/${dataset_name}/${dataset_file}

#target_model_name="llama2-7b-chat-hf"
#target_model_name="llama3-8b-instruct"
target_model_name="opt-1.3b"
#target_model_name="opt-2.7b"
# target_model_name="opt-6.7b"
#target_model_name="gpt-j-6b"

if [[ $dataset_name =~ 't' ]]; then
  target_model_name="llama3-8b-instruct"
#  target_model_name="llama2-7b-chat-hf"
fi

target_model_path=/data/wangyidan/model/${target_model_name}

input_json_filename=${dataset_path}
output_json_filepath=./output/${watermark_type}/${watermark_algorithm}/${dataset_name}/${dataset_size}/seed_${seed}
output_json_filename=${output_json_filepath}/${target_model_name}.jsonl

mkdir -p ${output_json_filepath}

if [ $mode == 'train' ]; then
  LOG_DIR=./log/${mode}/${watermark_type}/${watermark_algorithm}/${dataset_name}/${dataset_size}/seed_${seed}
  LOG_FILE=${LOG_DIR}/${target_model_name}.log
fi

if [ $mode == 'test' ]; then
  LOG_DIR=./log/${mode}/${watermark_type}/${watermark_algorithm}/${dataset_name}/${dataset_size}/${text_source}/${attack_method}/seed_${seed}
  LOG_FILE=${LOG_DIR}/${target_model_name}.log
fi

mkdir -p "${LOG_DIR}"

nohup python -u main.py \
    --mode=${mode} \
    --seed=${seed} \
    --text_source=${text_source} \
    --watermark_algorithm=${watermark_algorithm} \
    --watermark_type=${watermark_type} \
    --attack_method=${attack_method} \
    --dataset_path=${dataset_path} \
    --dataset_name=${dataset_name} \
    --dataset_size=${dataset_size} \
    --target_model_name=${target_model_name} \
    --target_model_path=${target_model_path} \
    --input_json_filename=${input_json_filename} \
    --output_json_filename=${output_json_filename} \
> "$LOG_FILE" 2>&1 &