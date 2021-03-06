#!/bin/sh
source ./env.sh
set -eux

export node_ips=`hostname -i`
export current_node_ip=`hostname -i`
selected_gpus="0"

train_json=$1

log_dir="./log"
output_dir="./output"

rm -rf $log_dir
rm -rf $output_dir
mkdir -p $log_dir
mkdir -p $output_dir

distributed_args="--node_ips ${node_ips} \
                --node_id 0 \
                --current_node_ip ${current_node_ip} \
                --nproc_per_node 1 \
                --selected_gpus ${selected_gpus} \
                --split_log_path ${log_dir} \
                --log_prefix train"
python -u ./lanch.py ${distributed_args} ./train.py  --param_path ${train_json} --log_dir ${log_dir}
