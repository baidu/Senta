#!/bin/bash
source ./env.sh
set -eux

epoch=1
BATCH_SIZE=1024
SAVE_STEPS=2000

MAX_LEN=512
use_fp16=False
generate_neg_sample=False
LR_RATE=2e-5
WEIGHT_DECAY=0.01
incr_every_n_steps=50
decr_every_n_nan_or_inf=3
lr_scheduler="linear_warmup_decay"
num_train_steps=200000
WARMUP_STEPS=10

model_type="roberta_skep_en"
MASKING="chunk_masking"

vocab_path="./model_files/dict/roberta_en.vocab.txt"
bpe_vocab_file="./model_files/dict/roberta_en.vocab.bpe"
bpe_json_file="./model_files/dict/roberta_en.encoder.json"
CONFIG_PATH="./model_files/config/roberta_large_en.config.json"
task_group_json="./data/en/pretraining/roberta_skep_large_en/task.json"
init_model="./model_files/roberta_skep_large_en/params"
log_dir="./log"
output_dir="./output"

rm -rf $log_dir
rm -rf $output_dir
mkdir -p $log_dir
mkdir -p $output_dir

python pretraining.py                                                 \
       --task_group_path ${task_group_json:-None}                     \
       --ernie_config_path ${CONFIG_PATH:-None}                       \
       --vocab_path ${vocab_path:-None}                               \
       --bpe_vocab_file ${bpe_vocab_file:-None}                       \
       --bpe_json_file ${bpe_json_file:-None}                         \
       --model_type ${model_type:-None}                               \
       --load_parameters  ${init_model:-""}                           \
       --masking_strategy ${MASKING:-"chunk_masking"}                 \
       --eval_step 2                                                  \
       --epoch ${epoch:-1}                                            \
       --use_fp16 ${use_fp16:-False}                                  \
       --weight_sharing "True"                                        \
       --in_tokens "True"                                             \
       --train_batch_size ${BATCH_SIZE}                               \
       --hack_old_data ${hack_old_data-"False"}                       \
       --generate_neg_sample ${generate_neg_sample-"True"}            \
       --lr_scheduler ${lr_scheduler}                                 \
       --num_train_steps ${num_train_steps}                           \
       --checkpoints ${output_dir}                                    \
       --log_dir ${log_dir}                                           \
       --save_model_step ${SAVE_STEPS}                                \
       --warmup_steps ${WARMUP_STEPS:-0}                              \
       --weight_decay ${WEIGHT_DECAY:-0}                              \
       --max_seq_len ${MAX_LEN}                                       \
       --using_spm "True"                                             \
       --do_whole_word_mask "False"                                   \
       --ngram 3                                                      \
       --train_log_step 2 
