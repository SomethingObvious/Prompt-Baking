#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: executor.sh [text file location] [worker num] [total workers]"
    exit 1
fi

# Assign arguments to variables
commands_file=$1
worker_num=$2
total_workers=$3

# Ensure the worker number and total workers are integers
if ! [[ "$worker_num" =~ ^[0-9]+$ ]] || ! [[ "$total_workers" =~ ^[0-9]+$ ]]; then
    echo "Worker number and total workers must be integers."
    exit 1
fi

# Log file location
log_file="scripts/exec.log"

# Log the start of the script
echo "[$(date)] Script called with arguments: $@, PID: $$" >> $log_file

# Iterate through the commands file and execute the relevant lines
line_num=0
while IFS= read -r command; do
    # Increment line number
    
    # Check if the current line number matches the worker's responsibility
    if [ $((line_num % total_workers)) -eq $worker_num ]; then
        # Log the command execution details
        echo "[$(date)] Worker: $worker_num, Line: $line_num, Command: $command, PID: $$" >> $log_file
        
        # Prepend CUDA_VISIBLE_DEVICES to the command and execute
        CUDA_VISIBLE_DEVICES=$worker_num $command
        
        # Log the completion of the command
        echo "[$(date)] Worker: $worker_num, Line: $line_num, Command completed, PID: $$" >> $log_file
    fi

    line_num=$((line_num + 1))
done < "$commands_file"

# Log the end of the script
echo "[$(date)] Script completed for worker: $worker_num, PID: $$" >> $log_file


curl -X POST -H "Content-Type: application/json" -d "{\"script\": \"prompt_weight_equivalence/scripts/executor.sh\", \"commands_file\": \"$commands_file\", \"worker_num\": $worker_num, \"total_workers\": $total_workers}" https://maker.ifttt.com/trigger/experiment_done/json/with/key/cyKYvYe7Q2IzhoX2XpQUd1

