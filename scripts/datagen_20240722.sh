#!/bin/bash

# Ensure the output directory exists
output_dir="data/20240722"
mkdir -p $output_dir

# File to store the commands
output_file="scripts/commands_datagen_20240722.txt"
> $output_file  # Clear the file if it already exists

# Loop through each x0 file in the data directory
for x0_file in data/*_x0.md; do
    # Extract the base filename without extension
    filename=$(basename -- "$x0_file")
    xi="${filename%.md}"
    echo "Processing xi=$xi"

    # Generate the training set command
    echo "python3 generate_data.py --x0_file data/${xi}.md --question_dataset data/squad_train.jsonl --num_questions 100 --num_sequences_per_question 25 --max_sequence_length 300 --min_sequence_length 100 --temperature 2.0 --batch_size 25 --traj_out_file ${output_dir}/traj_${xi}_squad_train.jsonl" >> $output_file

    # Generate the validation set command
    # echo "# generate validation set" >> $output_file
    echo "python3 generate_data.py --x0_file data/${xi}.md --question_dataset data/squad_validation.jsonl --num_questions 25 --num_sequences_per_question 25 --max_sequence_length 300 --min_sequence_length 100 --temperature 2.0 --batch_size 32 --traj_out_file ${output_dir}/traj_${xi}_squad_val.jsonl" >> $output_file
done

echo "Commands have been written to $output_file"
