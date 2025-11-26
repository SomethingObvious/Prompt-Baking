#!/bin/bash

# Ensure the output directory exists
output_dir="results/20240722"

data_dir="data/20240722/"


# constants
batch_size=8
learning_rate="3e-4"
num_epochs=150

mkdir -p $output_dir

# File to store the commands
output_file="scripts/train_loop_commands_20240722.txt"
> $output_file  # Clear the file if it already exists

# Loop through each x0 file in the data directory
for traj_file in ${data_dir}*_x0_squad_train.jsonl; do
    # Extract the base filename without extension
    filename=$(basename -- "$traj_file")
    base_path="${filename%train.jsonl}"
	
	train_path="${data_dir}${base_path}train.jsonl"
	val_path="${data_dir}${base_path}val.jsonl"

    # Generate the training set command
    echo "python3 train_loop.py --batch_size $batch_size --learning_rate $learning_rate --data_path ${train_path} --val_path ${val_path} --num_epochs ${num_epochs} --out_dir ${output_dir}/${base_path}ep${num_epochs}" >> $output_file
done

echo "Commands have been written to $output_file"
